"""
Inference module for ARC task prediction.

Provides optimized batched generation with:
- Static KV cache with pre-allocated buffers
- Vectorized grid-state updater for position tracking
- Batched decode across all augmentations simultaneously
- torch.compile() support for kernel fusion
"""

from dataclasses import dataclass
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
# DECODING CONSTRAINTS
# =============================================================================


@dataclass(frozen=True)
class DecodingConstraints:
    """Optional constraints for soft-constrained decoding."""
    palette_mask: torch.Tensor | None = None  # [batch, 10] bool mask for allowed colors
    target_shape: torch.Tensor | None = None  # [batch, 2] (h, w) or -1 for no constraint
    palette_penalty: float = 2.0
    shape_penalty: float = 2.0
    disallow_tokens: tuple[int, ...] = (START_TOKEN_ID, IO_SEP_TOKEN_ID)


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
        if tokens.dim() > 1:
            tokens = tokens.squeeze(-1)
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


def _grid_shape(grid: list[list[int]] | None) -> tuple[int, int] | None:
    if not grid or not grid[0]:
        return None
    row_len = len(grid[0])
    if row_len == 0:
        return None
    for row in grid:
        if len(row) != row_len:
            return None
    return len(grid), row_len


def _infer_output_shape(task: dict) -> tuple[int, int] | None:
    shapes: list[tuple[int, int]] = []
    for pair in task.get("train", []):
        output_grid = pair.get("output")
        shape = _grid_shape(output_grid)
        if shape is None:
            return None
        shapes.append(shape)
    if not shapes:
        return None
    first = shapes[0]
    if all(s == first for s in shapes):
        return first
    return None


def _collect_palette(
    task: dict,
    test_idx: int,
    include_test_input: bool = False,
) -> set[int]:
    colors: set[int] = set()
    for pair in task.get("train", []):
        for key in ("input", "output"):
            grid = pair.get(key)
            if grid is None:
                continue
            for row in grid:
                for val in row:
                    colors.add(int(val))
    if include_test_input and task.get("test"):
        test_input = task["test"][test_idx]["input"]
        for row in test_input:
            for val in row:
                colors.add(int(val))
    return colors


def _transform_shape(
    shape: tuple[int, int] | None,
    dihedral_idx: int,
) -> tuple[int, int] | None:
    if shape is None:
        return None
    if dihedral_idx in {1, 3, 6, 7}:
        return shape[1], shape[0]
    return shape


def _apply_logit_constraints(
    logits: torch.Tensor,
    grid_state: GridStateUpdater,
    constraints: DecodingConstraints | None,
) -> torch.Tensor:
    if constraints is None:
        return logits

    batch = logits.shape[0]

    for tok in constraints.disallow_tokens:
        logits[..., tok] = float("-inf")

    palette_mask = constraints.palette_mask
    if palette_mask is not None:
        if palette_mask.device != logits.device:
            palette_mask = palette_mask.to(device=logits.device)
        if palette_mask.dim() != 2:
            palette_mask = palette_mask.reshape(-1, palette_mask.shape[-1])
        if palette_mask.shape[0] != batch:
            if palette_mask.shape[0] == 1:
                palette_mask = palette_mask.expand(batch, -1)
            else:
                palette_mask = None
        if palette_mask is not None:
            penalty = (~palette_mask).to(logits.dtype) * constraints.palette_penalty
            logits[..., :10] = logits[..., :10] - penalty.unsqueeze(1)

    target_shape = constraints.target_shape
    if target_shape is not None:
        if target_shape.device != logits.device:
            target_shape = target_shape.to(device=logits.device)
        if target_shape.dim() != 2:
            target_shape = target_shape.reshape(-1, 2)
        if target_shape.shape[0] != batch:
            if target_shape.shape[0] == 1:
                target_shape = target_shape.expand(batch, -1)
            else:
                target_shape = None
        if target_shape is not None:
            # Ensure target_shape is exactly 2D [batch, 2]
            while target_shape.dim() > 2:
                target_shape = target_shape.squeeze(-1)
            if target_shape.dim() == 1:
                target_shape = target_shape.unsqueeze(0)
            # Extract and ensure 1D
            target_h = target_shape[:, 0].contiguous().view(-1)
            target_w = target_shape[:, 1].contiguous().view(-1)
            has_shape = (target_h >= 0) & (target_w >= 0)
            if has_shape.any():
                x = grid_state.x.contiguous().view(-1)
                y = grid_state.y.contiguous().view(-1)
                # Align batch dimensions if needed
                n_batch = x.shape[0]
                n_constraint = target_w.shape[0]
                if n_batch != n_constraint:
                    if n_constraint == 1:
                        target_h = target_h.expand(n_batch)
                        target_w = target_w.expand(n_batch)
                        has_shape = has_shape.expand(n_batch)
                    else:
                        # Skip constraint if batch sizes truly incompatible
                        return logits
                row_full = has_shape & (x >= target_w)
                row_incomplete = has_shape & (x < target_w)
                past_end = has_shape & (y >= target_h)

                if row_full.any():
                    # Shape: [batch, 1, 1] to broadcast with [batch, seq, vocab]
                    penalty = row_full.to(logits.dtype)[:, None, None] * constraints.shape_penalty
                    logits[..., :10] = logits[..., :10] - penalty
                if row_incomplete.any():
                    penalty = row_incomplete.to(logits.dtype)[:, None, None] * constraints.shape_penalty
                    logits[..., NEWLINE_TOKEN_ID:NEWLINE_TOKEN_ID+1] = logits[..., NEWLINE_TOKEN_ID:NEWLINE_TOKEN_ID+1] - penalty
                if past_end.any():
                    penalty = past_end.to(logits.dtype)[:, None, None] * constraints.shape_penalty
                    logits[..., :10] = logits[..., :10] - penalty
                    logits[..., NEWLINE_TOKEN_ID:NEWLINE_TOKEN_ID+1] = logits[..., NEWLINE_TOKEN_ID:NEWLINE_TOKEN_ID+1] - penalty

    return logits


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

    x = model.token_embed(input_ids)
    if example_ids is not None and getattr(model.config, "use_example_embedding", True):
        x = x + model.example_embed(example_ids).unsqueeze(1)

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

    x = model.token_embed(token)
    if example_ids is not None and getattr(model.config, "use_example_embedding", True):
        x = x + model.example_embed(example_ids).unsqueeze(1)

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
    top_k: int | None = None,
    stop_on_end: bool = True,
    use_compile: bool = False,
    constraints: DecodingConstraints | None = None,
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
        top_k: If set and temperature > 0, sample from top-k logits
        stop_on_end: If True, stop when END token is generated
        use_compile: If True, use torch.compile for decode step
        constraints: Optional decoding constraints

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
    logits = _apply_logit_constraints(logits, grid_state, constraints)
    if temperature <= 0:
        next_token = logits.argmax(dim=-1)
    else:
        logits_t = logits[:, 0, :] / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            k = min(top_k, logits_t.size(-1))
            top_vals, top_idx = torch.topk(logits_t, k, dim=-1)
            probs = torch.zeros_like(logits_t).scatter_(1, top_idx, F.softmax(top_vals, dim=-1))
        else:
            probs = F.softmax(logits_t, dim=-1)
        next_token = torch.multinomial(probs, 1)

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

        logits = _apply_logit_constraints(logits, grid_state, constraints)
        if temperature <= 0:
            next_token = logits.argmax(dim=-1)
        else:
            logits_t = logits[:, 0, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits_t.size(-1))
                top_vals, top_idx = torch.topk(logits_t, k, dim=-1)
                probs = torch.zeros_like(logits_t).scatter_(1, top_idx, F.softmax(top_vals, dim=-1))
            else:
                probs = F.softmax(logits_t, dim=-1)
            next_token = torch.multinomial(probs, 1)

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
    temperature: float = 0.0,
    top_k: int | None = None,
    use_compile: bool = False,
    constraints: DecodingConstraints | None = None,
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
        temperature: Sampling temperature (0.0 = greedy)
        top_k: If set and temperature > 0, sample from top-k logits
        use_compile: If True, use torch.compile
        constraints: Optional decoding constraints

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
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_on_end=True,
            use_compile=use_compile,
            constraints=constraints,
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
    use_train_examples: bool = True,
    num_color_perms: int = 8,
    max_output_tokens: int = 900,
    num_samples: int = 1,
    sample_temperature: float = 0.0,
    sample_top_k: int | None = None,
    constrain_decoding: bool = False,
    palette_penalty: float = 2.0,
    shape_penalty: float = 2.0,
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
        use_train_examples: If False, prompt uses only the test input
        num_color_perms: Number of color permutations
        max_output_tokens: Maximum tokens per prediction
        num_samples: Number of samples per augmentation (self-consistency)
        sample_temperature: Sampling temperature for self-consistency
        sample_top_k: If set and temperature > 0, sample from top-k logits
        constrain_decoding: If True, apply soft palette/shape constraints
        palette_penalty: Logit penalty for disallowed colors
        shape_penalty: Logit penalty for shape rule violations
        batched: If True, process all augmentations in one batched pass (faster)
        use_compile: If True, use torch.compile

    Returns:
        List of (predicted_grid, dihedral_idx, color_perm_tuple) tuples.
    """
    num_samples = max(1, num_samples)
    base_palette = _collect_palette(task, test_idx) if constrain_decoding else None
    base_shape = _infer_output_shape(task) if constrain_decoding else None

    if batched:
        return _predict_augmentations_batched(
            model, task, test_idx, device, example_id,
            use_train_examples, num_color_perms, max_output_tokens,
            num_samples, sample_temperature, sample_top_k,
            constrain_decoding, base_palette, base_shape,
            palette_penalty, shape_penalty,
            use_compile,
        )
    else:
        return _predict_augmentations_sequential(
            model, task, test_idx, device, example_id,
            use_train_examples, num_color_perms, max_output_tokens,
            num_samples, sample_temperature, sample_top_k,
            constrain_decoding, base_palette, base_shape,
            palette_penalty, shape_penalty,
            use_compile,
        )


def _predict_augmentations_batched(
    model: TinyTransformer,
    task: dict,
    test_idx: int,
    device: torch.device,
    example_id: int,
    use_train_examples: bool,
    num_color_perms: int,
    max_output_tokens: int,
    num_samples: int,
    sample_temperature: float,
    sample_top_k: int | None,
    constrain_decoding: bool,
    base_palette: set[int] | None,
    base_shape: tuple[int, int] | None,
    palette_penalty: float,
    shape_penalty: float,
    use_compile: bool,
    max_batch_size: int = 16,  # Process augmentations in chunks to avoid OOM
) -> list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]]:
    """Batched augmentation prediction - processes in chunks to manage memory."""
    model.eval()

    train_pairs = task["train"] if use_train_examples else []
    test_input = task["test"][test_idx]["input"]
    color_perms = generate_color_permutations(num_color_perms, seed=42)

    # Build all augmented prompts
    prompts, positions_list, aug_info = [], [], []
    palettes, shapes = [], []

    for dihedral_idx, dihedral_fn in enumerate(DIHEDRAL_TRANSFORMS):
        aug_inputs = [dihedral_fn(p["input"]) for p in train_pairs]
        aug_outputs = [dihedral_fn(p["output"]) for p in train_pairs]
        aug_test = dihedral_fn(test_input)
        shape = _transform_shape(base_shape, dihedral_idx) if constrain_decoding else None

        for perm_idx in range(num_color_perms):
            perm = color_perms[perm_idx]
            perm_tuple = tuple(perm.tolist()) if perm_idx > 0 else None
            palette = None
            if base_palette is not None:
                if perm_idx > 0:
                    palette = {int(perm[c]) for c in base_palette}
                else:
                    palette = base_palette

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

            for _ in range(num_samples):
                prompts.append(tokens)
                positions_list.append(compute_positions_3d(tokens).tolist())
                aug_info.append((dihedral_idx, perm_tuple))
                palettes.append(palette)
                shapes.append(shape)

    # Process in chunks to avoid OOM
    all_predictions = []
    total = len(prompts)

    for chunk_start in range(0, total, max_batch_size):
        chunk_end = min(chunk_start + max_batch_size, total)
        chunk_prompts = prompts[chunk_start:chunk_end]
        chunk_positions = positions_list[chunk_start:chunk_end]
        chunk_aug_info = aug_info[chunk_start:chunk_end]
        chunk_size = len(chunk_prompts)
        chunk_palettes = palettes[chunk_start:chunk_end]
        chunk_shapes = shapes[chunk_start:chunk_end]

        constraints = None
        if constrain_decoding:
            palette_mask = None
            if chunk_palettes:
                mask = torch.ones((chunk_size, 10), dtype=torch.bool, device=device)
                any_palette = False
                for i, palette in enumerate(chunk_palettes):
                    if palette:
                        any_palette = True
                        mask[i].fill_(False)
                        for c in palette:
                            if 0 <= c < 10:
                                mask[i, c] = True
                if any_palette:
                    palette_mask = mask

            target_shape = None
            if chunk_shapes:
                shape_tensor = torch.full((chunk_size, 2), -1, dtype=torch.long, device=device)
                any_shape = False
                for i, shape in enumerate(chunk_shapes):
                    if shape is not None:
                        any_shape = True
                        shape_tensor[i, 0] = shape[0]
                        shape_tensor[i, 1] = shape[1]
                if any_shape:
                    target_shape = shape_tensor

            constraints = DecodingConstraints(
                palette_mask=palette_mask,
                target_shape=target_shape,
                palette_penalty=palette_penalty,
                shape_penalty=shape_penalty,
            )

        # Left-pad this chunk
        max_len = max(len(p) for p in chunk_prompts)
        seq_lens = [len(p) for p in chunk_prompts]

        padded_ids = [([END_TOKEN_ID] * (max_len - len(p))) + p for p in chunk_prompts]
        padded_pos = [([[0, 0, 0]] * (max_len - len(p))) + p for p in chunk_positions]

        attention_mask = torch.zeros(chunk_size, max_len, dtype=torch.bool, device=device)
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, max_len - seq_len:] = True

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        positions_3d = torch.tensor(padded_pos, dtype=torch.long, device=device)
        example_ids_t = torch.full((chunk_size,), example_id, dtype=torch.long, device=device)

        # Generate for this chunk
        with torch.no_grad():
            output_ids = generate_output(
                model, input_ids, positions_3d, example_ids_t,
                attention_mask=attention_mask,
                max_new_tokens=max_output_tokens,
                temperature=sample_temperature,
                top_k=sample_top_k,
                stop_on_end=True,
                use_compile=use_compile,
                constraints=constraints,
            )

        # Decode each in chunk
        for i in range(chunk_size):
            tokens_list = output_ids[i].cpu().tolist()
            sep_positions = [j for j, t in enumerate(tokens_list) if t == IO_SEP_TOKEN_ID]

            if sep_positions:
                grid = decode_output_grid(tokens_list[sep_positions[-1]:])
            else:
                grid = None

            all_predictions.append((grid, chunk_aug_info[i][0], chunk_aug_info[i][1]))

        # Free memory
        del input_ids, positions_3d, example_ids_t, attention_mask, output_ids

    return all_predictions


def _predict_augmentations_sequential(
    model: TinyTransformer,
    task: dict,
    test_idx: int,
    device: torch.device,
    example_id: int,
    use_train_examples: bool,
    num_color_perms: int,
    max_output_tokens: int,
    num_samples: int,
    sample_temperature: float,
    sample_top_k: int | None,
    constrain_decoding: bool,
    base_palette: set[int] | None,
    base_shape: tuple[int, int] | None,
    palette_penalty: float,
    shape_penalty: float,
    use_compile: bool,
) -> list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]]:
    """Sequential augmentation prediction - one at a time (lower memory)."""
    model.eval()

    train_pairs = task["train"] if use_train_examples else []
    test_input = task["test"][test_idx]["input"]
    color_perms = generate_color_permutations(num_color_perms, seed=42)
    predictions = []

    for dihedral_idx, dihedral_fn in enumerate(DIHEDRAL_TRANSFORMS):
        aug_inputs = [dihedral_fn(p["input"]) for p in train_pairs]
        aug_outputs = [dihedral_fn(p["output"]) for p in train_pairs]
        aug_test = dihedral_fn(test_input)
        shape = _transform_shape(base_shape, dihedral_idx) if constrain_decoding else None

        for perm_idx in range(num_color_perms):
            perm = color_perms[perm_idx]
            perm_tuple = tuple(perm.tolist()) if perm_idx > 0 else None
            palette = None
            if base_palette is not None:
                if perm_idx > 0:
                    palette = {int(perm[c]) for c in base_palette}
                else:
                    palette = base_palette

            if perm_idx > 0:
                p_inputs = [_apply_color_perm_grid(g, perm) for g in aug_inputs]
                p_outputs = [_apply_color_perm_grid(g, perm) for g in aug_outputs]
                p_test = _apply_color_perm_grid(aug_test, perm)
            else:
                p_inputs, p_outputs, p_test = aug_inputs, aug_outputs, aug_test

            constraints = None
            if constrain_decoding:
                palette_mask = None
                if palette:
                    mask = torch.zeros((1, 10), dtype=torch.bool, device=device)
                    for c in palette:
                        if 0 <= c < 10:
                            mask[0, c] = True
                    palette_mask = mask

                target_shape = None
                if shape is not None:
                    target_shape = torch.tensor([[shape[0], shape[1]]], dtype=torch.long, device=device)

                constraints = DecodingConstraints(
                    palette_mask=palette_mask,
                    target_shape=target_shape,
                    palette_penalty=palette_penalty,
                    shape_penalty=shape_penalty,
                )

            for _ in range(num_samples):
                pred = predict_grid(
                    model, p_inputs, p_outputs, p_test, device,
                    example_id=example_id, max_output_tokens=max_output_tokens,
                    temperature=sample_temperature, top_k=sample_top_k,
                    use_compile=use_compile, constraints=constraints,
                )
                predictions.append((pred, dihedral_idx, perm_tuple))

    return predictions


def _apply_color_perm_grid(grid: list[list[int]], perm: np.ndarray) -> list[list[int]]:
    """Apply color permutation to a grid."""
    return [[int(perm[cell]) for cell in row] for row in grid]
