"""
Inference module for ARC task prediction.

Provides optimized batched generation with:
- Static KV cache with pre-allocated buffers
- Vectorized grid-state updater for position tracking
- Batched decode across all augmentations simultaneously
- torch.compile() support for kernel fusion
"""

# from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.data import (
    DIHEDRAL_TRANSFORMS,
    END_TOKEN_ID,
    IO_SEP_TOKEN_ID,
    NEWLINE_TOKEN_ID,
    START_TOKEN_ID,
    compute_positions_3d,
    decode_output_grid,
    encode_example,
    generate_color_permutations,
)
from src.transformer import TinyTransformer

# =============================================================================
# STATIC KV CACHE
# =============================================================================


class StaticKVCache:
    """
    Pre-allocated static KV cache for efficient inference.

    Allocates all memory upfront to avoid allocation overhead during generation.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.seq_len = 0

        # Pre-allocate: [n_layers, 2 (K/V), batch, heads, max_seq, head_dim]
        self.cache = torch.zeros(
            n_layers, 2, batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )

    def reset(self):
        """Reset cache for new generation."""
        self.cache.zero_()
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a layer and return full K, V up to current position."""
        new_len = k.size(2)
        end_pos = self.seq_len + new_len

        self.cache[layer_idx, 0, :, :, self.seq_len:end_pos, :] = k
        self.cache[layer_idx, 1, :, :, self.seq_len:end_pos, :] = v

        return self.cache[layer_idx, 0, :, :, :end_pos, :], self.cache[layer_idx, 1, :, :, :end_pos, :]

    def advance(self, n: int = 1):
        """Advance sequence position."""
        self.seq_len += n


# =============================================================================
# GRID STATE UPDATER
# =============================================================================


class GridStateUpdater:
    """Vectorized grid state tracker for efficient position updates."""

    def __init__(self, batch_size: int, device: torch.device, max_x: int = 30, max_y: int = 29):
        self.device = device
        self.max_x = max_x
        self.max_y = max_y
        self.x = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.y = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.in_output = torch.ones(batch_size, dtype=torch.bool, device=device)

    def reset(self):
        """Reset to initial output state."""
        self.x.zero_()
        self.y.zero_()
        self.in_output.fill_(True)

    def set_state(self, x: torch.Tensor, y: torch.Tensor, in_output: torch.Tensor) -> None:
        """Set the current grid state for batched decoding."""
        self.x = x.to(self.device, dtype=torch.long)
        self.y = y.to(self.device, dtype=torch.long)
        self.in_output = in_output.to(self.device, dtype=torch.bool)

    def update(self, tokens: torch.Tensor) -> torch.Tensor:
        """Update positions based on tokens, return positions for current step."""
        tokens = tokens.to(self.device)
        is_start = tokens == START_TOKEN_ID
        is_sep = tokens == IO_SEP_TOKEN_ID
        is_end = tokens == END_TOKEN_ID
        is_newline = tokens == NEWLINE_TOKEN_ID
        is_color = tokens < 10

        base_z = torch.where(
            self.in_output,
            torch.full_like(tokens, 3),
            torch.full_like(tokens, 1),
        )
        z = torch.where(is_start, torch.zeros_like(tokens), base_z)
        z = torch.where(is_sep, torch.full_like(tokens, 2), z)
        z = torch.where(is_end, torch.full_like(tokens, 4), z)

        x_pos = self.x.clamp(0, self.max_x)
        y_pos = self.y.clamp(0, self.max_y)
        pos_x = torch.where(is_start | is_sep | is_end | is_newline, torch.zeros_like(x_pos), x_pos)
        pos_y = torch.where(is_start | is_sep | is_end, torch.zeros_like(y_pos), y_pos)
        positions = torch.stack([pos_x, pos_y, z], dim=-1)

        reset = is_start | is_sep
        self.x = torch.where(reset, torch.zeros_like(self.x), self.x)
        self.y = torch.where(reset, torch.zeros_like(self.y), self.y)

        self.in_output = torch.where(is_start, torch.zeros_like(self.in_output), self.in_output)
        self.in_output = torch.where(is_sep, torch.ones_like(self.in_output), self.in_output)

        self.x = torch.where(is_newline, torch.zeros_like(self.x), self.x)
        self.y = torch.where(is_newline, torch.clamp(self.y + 1, max=self.max_y), self.y)

        self.x = torch.where(is_color, torch.clamp(self.x + 1, max=self.max_x), self.x)

        self.x = self.x.clamp(0, self.max_x)
        self.y = self.y.clamp(0, self.max_y)

        return positions


def _derive_initial_state_from_prompt(
    input_ids: torch.Tensor,
    positions_3d: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_x: int,
    max_y: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive grid state after the prompt for batched decoding."""
    batch_size, seq_len = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
    else:
        attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)

    lengths = attention_mask.long().sum(dim=1)
    has_tokens = lengths > 0
    last_idx = torch.clamp(lengths - 1, min=0)
    batch_idx = torch.arange(batch_size, device=input_ids.device)

    last_tokens = input_ids[batch_idx, last_idx]
    last_pos = positions_3d[batch_idx, last_idx]
    last_x = last_pos[:, 0]
    last_y = last_pos[:, 1]
    last_z = last_pos[:, 2]

    in_output = last_z >= 3
    in_output = torch.where(last_tokens == START_TOKEN_ID, torch.zeros_like(in_output), in_output)
    in_output = torch.where(last_tokens == IO_SEP_TOKEN_ID, torch.ones_like(in_output), in_output)

    next_x = last_x
    next_y = last_y

    is_start = last_tokens == START_TOKEN_ID
    is_sep = last_tokens == IO_SEP_TOKEN_ID
    is_newline = last_tokens == NEWLINE_TOKEN_ID
    is_color = last_tokens < 10

    next_x = torch.where(is_start | is_sep, torch.zeros_like(next_x), next_x)
    next_y = torch.where(is_start | is_sep, torch.zeros_like(next_y), next_y)
    next_x = torch.where(is_newline, torch.zeros_like(next_x), next_x)
    next_y = torch.where(is_newline, torch.clamp(next_y + 1, max=max_y), next_y)
    next_x = torch.where(is_color, torch.clamp(next_x + 1, max=max_x), next_x)

    next_x = next_x.clamp(0, max_x)
    next_y = next_y.clamp(0, max_y)

    next_x = torch.where(has_tokens, next_x, torch.zeros_like(next_x))
    next_y = torch.where(has_tokens, next_y, torch.zeros_like(next_y))
    in_output = torch.where(has_tokens, in_output, torch.ones_like(in_output))

    return next_x, next_y, in_output


# =============================================================================
# CORE DECODE FUNCTIONS
# =============================================================================


def _build_sdpa_mask(
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Convert a [B, S] bool mask into SDPA-compatible float mask."""
    if attention_mask is None:
        return None
    mask_bool = attention_mask.unsqueeze(1).unsqueeze(2)
    return torch.where(mask_bool, 0.0, float("-inf")).to(dtype)


def _prefill(model: TinyTransformer, input_ids: torch.Tensor, positions_3d: torch.Tensor,
             example_ids: torch.Tensor, cache: StaticKVCache,
             attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Prefill phase: process prompt and populate cache. Returns last-token logits.

    Args:
        attention_mask: [batch, seq] bool mask where True = valid token, False = padding
    """
    batch_size, seq_len = input_ids.shape

    x = model.token_embed(input_ids) + model.example_embed(example_ids).unsqueeze(1)

    # Convert attention mask to SDPA format if provided
    attn_mask = _build_sdpa_mask(attention_mask, x.dtype)

    for layer_idx, block in enumerate(model.blocks):
        x_norm = block.ln1(x)
        qkv = block.attn.qkv_proj(x_norm).reshape(batch_size, seq_len, 3, block.attn.n_heads, block.attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = block.attn.rope.apply_rotary(q, k, positions_3d)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k_full, v_full = cache.update(layer_idx, k, v)

        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask, is_causal=True)
        attn_out = block.attn.out_proj(attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1))

        x = x + attn_out
        x = x + block.ffn(block.ln2(x))

    cache.advance(seq_len)

    if attention_mask is None:
        return model.lm_head(model.ln_f(x[:, -1:, :]))

    lengths = attention_mask.long().sum(dim=1)
    last_idx = torch.clamp(lengths - 1, min=0)
    batch_idx = torch.arange(batch_size, device=x.device)
    x_last = x[batch_idx, last_idx]
    return model.lm_head(model.ln_f(x_last)).unsqueeze(1)


def _decode_step(model: TinyTransformer, token: torch.Tensor, position: torch.Tensor,
                 example_ids: torch.Tensor, cache: StaticKVCache,
                 attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Decode one token with cache. Returns logits."""
    batch_size = token.size(0)

    x = model.token_embed(token) + model.example_embed(example_ids).unsqueeze(1)

    for layer_idx, block in enumerate(model.blocks):
        x_norm = block.ln1(x)
        qkv = block.attn.qkv_proj(x_norm).reshape(batch_size, 1, 3, block.attn.n_heads, block.attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = block.attn.rope.apply_rotary(q, k, position)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k_full, v_full = cache.update(layer_idx, k, v)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.size(1) != k_full.size(2):
                attention_mask = attention_mask[:, : k_full.size(2)]
            attn_mask = _build_sdpa_mask(attention_mask, x.dtype)

        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask, is_causal=False)
        attn_out = block.attn.out_proj(attn_out.transpose(1, 2).reshape(batch_size, 1, -1))

        x = x + attn_out
        x = x + block.ffn(block.ln2(x))

    cache.advance(1)
    return model.lm_head(model.ln_f(x))


# Compiled decode step (lazily initialized)
_compiled_decode: Callable | None = None


def _get_decode_fn(use_compile: bool) -> Callable:
    """Get decode function, optionally compiled."""
    global _compiled_decode

    if not use_compile:
        return _decode_step

    if _compiled_decode is None:
        try:
            _compiled_decode = torch.compile(_decode_step, mode="reduce-overhead", fullgraph=False)
        except Exception:
            _compiled_decode = _decode_step

    return _compiled_decode


# =============================================================================
# PUBLIC API
# =============================================================================


def generate_output(
    model: TinyTransformer,
    input_ids: torch.Tensor,
    positions_3d: torch.Tensor,
    example_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 900,
    temperature: float = 0.0,
    stop_on_end: bool = True,
    use_compile: bool = False,
) -> torch.Tensor:
    """
    Generate output tokens autoregressively with static KV cache.

    Args:
        model: The transformer model
        input_ids: [batch, seq_len] prompt token IDs (left-padded for batched inference)
        positions_3d: [batch, seq_len, 3] prompt positions (x, y, z)
        example_ids: [batch] example IDs for task conditioning (None uses default 0)
        attention_mask: [batch, seq_len] bool mask (True=valid, False=padding)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        stop_on_end: If True, stop when END token is generated
        use_compile: If True, use torch.compile for decode step

    Returns:
        Full sequence [batch, seq_len + generated_len] including prompt
    """
    model.eval()
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape
    dtype = next(model.parameters()).dtype

    if example_ids is None:
        example_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

    if attention_mask is None:
        prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device)
    else:
        if attention_mask.shape != (batch_size, prompt_len):
            raise ValueError("attention_mask must have shape [batch, seq_len]")
        prompt_mask = attention_mask.to(device=device, dtype=torch.bool)

    max_total_len = min(prompt_len + max_new_tokens, model.config.max_seq_len)
    max_new_tokens = max_total_len - prompt_len
    if max_new_tokens <= 0:
        return input_ids

    # Create static cache
    cache = StaticKVCache(
        batch_size, max_total_len,
        model.config.n_layers, model.config.n_heads,
        model.config.d_model // model.config.n_heads,
        device, dtype,
    )

    grid_state = GridStateUpdater(batch_size, device)
    init_x, init_y, init_in_output = _derive_initial_state_from_prompt(
        input_ids,
        positions_3d,
        prompt_mask,
        grid_state.max_x,
        grid_state.max_y,
    )
    grid_state.set_state(init_x, init_y, init_in_output)
    decode_fn = _get_decode_fn(use_compile)
    full_attention_mask = torch.zeros(batch_size, max_total_len, dtype=torch.bool, device=device)
    full_attention_mask[:, :prompt_len] = prompt_mask

    # Prefill
    with torch.no_grad():
        logits = _prefill(
            model,
            input_ids,
            positions_3d,
            example_ids,
            cache,
            attention_mask=prompt_mask,
        )

    # Sample first token
    if temperature <= 0:
        next_token = logits.argmax(dim=-1)
    else:
        next_token = torch.multinomial(F.softmax(logits[:, 0, :] / temperature, dim=-1), 1)

    generated = [input_ids, next_token]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    full_attention_mask[:, prompt_len] = True

    if stop_on_end:
        finished = finished | (next_token.squeeze(-1) == END_TOKEN_ID)

    # Decode loop
    for step in range(1, max_new_tokens):
        if stop_on_end and finished.all():
            break

        position = grid_state.update(next_token.squeeze(-1)).unsqueeze(1)

        with torch.no_grad():
            logits = decode_fn(
                model,
                next_token,
                position,
                example_ids,
                cache,
                attention_mask=full_attention_mask[:, : cache.seq_len + 1],
            )

        if temperature <= 0:
            next_token = logits.argmax(dim=-1)
        else:
            next_token = torch.multinomial(F.softmax(logits[:, 0, :] / temperature, dim=-1), 1)

        if stop_on_end:
            finished = finished | (next_token.squeeze(-1) == END_TOKEN_ID)

        generated.append(next_token)
        full_attention_mask[:, prompt_len + step] = True

    return torch.cat(generated, dim=1)


def predict_grid(
    model: TinyTransformer,
    input_grids: list[list[list[int]]],
    output_grids: list[list[list[int]]],
    test_input: list[list[int]],
    device: torch.device,
    example_id: int = 0,
    max_output_tokens: int = 900,
    use_compile: bool = False,
) -> list[list[int]] | None:
    """
    Predict the output grid for a test input given training examples.

    Args:
        model: The transformer model
        input_grids: List of training input grids
        output_grids: List of training output grids
        test_input: The test input grid to predict output for
        device: Device to run on
        example_id: Task example ID for conditioning
        max_output_tokens: Maximum output tokens to generate
        use_compile: If True, use torch.compile

    Returns:
        Predicted output grid, or None if decoding fails
    """
    model.eval()

    # Build prompt
    all_tokens = []
    for inp, out in zip(input_grids, output_grids, strict=True):
        all_tokens.extend(encode_example(inp, out))
    all_tokens.extend(encode_example(test_input, None))

    positions = compute_positions_3d(all_tokens)

    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)
    positions_3d = torch.tensor([positions], dtype=torch.long, device=device)
    example_ids = torch.tensor([example_id], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = generate_output(
            model, input_ids, positions_3d, example_ids,
            max_new_tokens=max_output_tokens, temperature=0.0,
            stop_on_end=True, use_compile=use_compile,
        )

    tokens_list = output_ids[0].cpu().tolist()
    sep_positions = [i for i, t in enumerate(tokens_list) if t == IO_SEP_TOKEN_ID]

    if not sep_positions:
        return None

    return decode_output_grid(tokens_list[sep_positions[-1]:])


def predict_with_augmentations(
    model: TinyTransformer,
    task: dict,
    test_idx: int,
    device: torch.device,
    example_id: int = 0,
    num_color_perms: int = 8,
    max_output_tokens: int = 900,
    batched: bool = True,
    use_compile: bool = False,
) -> list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]]:
    """
    Generate predictions under all augmentations for AAIVR voting.

    Args:
        model: The transformer model
        task: Full ARC task dict with "train" and "test" keys
        test_idx: Which test example to predict
        device: Device to run on
        example_id: Task example ID for conditioning
        num_color_perms: Number of color permutations
        max_output_tokens: Maximum tokens per prediction
        batched: If True, process all augmentations in one batched pass (faster)
        use_compile: If True, use torch.compile

    Returns:
        List of (predicted_grid, dihedral_idx, color_perm_tuple) tuples.
    """
    if batched:
        return _predict_augmentations_batched(
            model, task, test_idx, device, example_id,
            num_color_perms, max_output_tokens, use_compile,
        )
    else:
        return _predict_augmentations_sequential(
            model, task, test_idx, device, example_id,
            num_color_perms, max_output_tokens, use_compile,
        )


def _predict_augmentations_batched(
    model: TinyTransformer,
    task: dict,
    test_idx: int,
    device: torch.device,
    example_id: int,
    num_color_perms: int,
    max_output_tokens: int,
    use_compile: bool,
) -> list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]]:
    """Batched augmentation prediction - all augmentations in one pass."""
    model.eval()

    train_pairs = task["train"]
    test_input = task["test"][test_idx]["input"]
    color_perms = generate_color_permutations(num_color_perms, seed=42)

    # Build all augmented prompts
    prompts, positions_list, aug_info = [], [], []

    for dihedral_idx, dihedral_fn in enumerate(DIHEDRAL_TRANSFORMS):
        aug_inputs = [dihedral_fn(p["input"]) for p in train_pairs]
        aug_outputs = [dihedral_fn(p["output"]) for p in train_pairs]
        aug_test = dihedral_fn(test_input)

        for perm_idx in range(num_color_perms):
            perm = color_perms[perm_idx]
            perm_tuple = tuple(perm.tolist()) if perm_idx > 0 else None

            if perm_idx > 0:
                p_inputs = [_apply_color_perm_grid(g, perm) for g in aug_inputs]
                p_outputs = [_apply_color_perm_grid(g, perm) for g in aug_outputs]
                p_test = _apply_color_perm_grid(aug_test, perm)
            else:
                p_inputs, p_outputs, p_test = aug_inputs, aug_outputs, aug_test

            tokens = []
            for inp, out in zip(p_inputs, p_outputs, strict=True):
                tokens.extend(encode_example(inp, out))
            tokens.extend(encode_example(p_test, None))

            prompts.append(tokens)
            positions_list.append(compute_positions_3d(tokens).tolist())
            aug_info.append((dihedral_idx, perm_tuple))

    # Left-pad and batch (upstream uses left-padding with attention mask)
    max_len = max(len(p) for p in prompts)
    total = len(prompts)
    seq_lens = [len(p) for p in prompts]

    # Left-pad: prepend padding tokens
    padded_ids = [([END_TOKEN_ID] * (max_len - len(p))) + p for p in prompts]
    padded_pos = [([[0, 0, 0]] * (max_len - len(p))) + p for p in positions_list]

    # Create attention mask: True for real tokens, False for padding
    attention_mask = torch.zeros(total, max_len, dtype=torch.bool, device=device)
    for i, seq_len in enumerate(seq_lens):
        attention_mask[i, max_len - seq_len:] = True

    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    positions_3d = torch.tensor(padded_pos, dtype=torch.long, device=device)
    example_ids = torch.full((total,), example_id, dtype=torch.long, device=device)

    # Generate all at once
    with torch.no_grad():
        output_ids = generate_output(
            model, input_ids, positions_3d, example_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_output_tokens, temperature=0.0,
            stop_on_end=True, use_compile=use_compile,
        )

    # Decode each
    predictions = []
    for i in range(total):
        tokens_list = output_ids[i].cpu().tolist()
        sep_positions = [j for j, t in enumerate(tokens_list) if t == IO_SEP_TOKEN_ID]

        if sep_positions:
            grid = decode_output_grid(tokens_list[sep_positions[-1]:])
        else:
            grid = None

        predictions.append((grid, aug_info[i][0], aug_info[i][1]))

    return predictions


def _predict_augmentations_sequential(
    model: TinyTransformer,
    task: dict,
    test_idx: int,
    device: torch.device,
    example_id: int,
    num_color_perms: int,
    max_output_tokens: int,
    use_compile: bool,
) -> list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]]:
    """Sequential augmentation prediction - one at a time (lower memory)."""
    model.eval()

    train_pairs = task["train"]
    test_input = task["test"][test_idx]["input"]
    color_perms = generate_color_permutations(num_color_perms, seed=42)
    predictions = []

    for dihedral_idx, dihedral_fn in enumerate(DIHEDRAL_TRANSFORMS):
        aug_inputs = [dihedral_fn(p["input"]) for p in train_pairs]
        aug_outputs = [dihedral_fn(p["output"]) for p in train_pairs]
        aug_test = dihedral_fn(test_input)

        for perm_idx in range(num_color_perms):
            perm = color_perms[perm_idx]
            perm_tuple = tuple(perm.tolist()) if perm_idx > 0 else None

            if perm_idx > 0:
                p_inputs = [_apply_color_perm_grid(g, perm) for g in aug_inputs]
                p_outputs = [_apply_color_perm_grid(g, perm) for g in aug_outputs]
                p_test = _apply_color_perm_grid(aug_test, perm)
            else:
                p_inputs, p_outputs, p_test = aug_inputs, aug_outputs, aug_test

            pred = predict_grid(
                model, p_inputs, p_outputs, p_test, device,
                example_id=example_id, max_output_tokens=max_output_tokens,
                use_compile=use_compile,
            )
            predictions.append((pred, dihedral_idx, perm_tuple))

    return predictions


def _apply_color_perm_grid(grid: list[list[int]], perm: np.ndarray) -> list[list[int]]:
    """Apply color permutation to a grid."""
    return [[int(perm[cell]) for cell in row] for row in grid]
