"""
TinyTransformer with 3D Rotary Position Embeddings for ARC tasks.

Reproduced from scratch based on the mdlARC paper/approach.
Key innovation: 3D RoPE that handles spatial (x, y) + example (z) coordinates.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_refinement_loss(
    all_logits: list[torch.Tensor],
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    output_mask: torch.Tensor | None = None,
    weighting: Literal["uniform", "linear", "exponential"] = "linear",
    output_only: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute deep supervision loss over refinement steps.

    Args:
        all_logits: List of K logits tensors [batch, seq, vocab]
        labels: [batch, seq] target token indices
        attention_mask: [batch, seq] valid token mask (True = valid)
        output_mask: [batch, seq] output token mask (True = output)
        weighting: How to weight losses across steps
            - "uniform": all steps equal
            - "linear": later steps weighted more (1, 2, 3, ...)
            - "exponential": exponentially more weight on later steps
        output_only: If True, only compute loss on output tokens

    Returns:
        Dict with 'loss', 'input_loss', 'output_loss' and per-step losses
    """
    K = len(all_logits)
    device = all_logits[0].device
    dtype = all_logits[0].dtype

    # Compute weights based on weighting scheme
    if weighting == "uniform":
        weights = [1.0 / K] * K
    elif weighting == "linear":
        # [1, 2, 3, ...] normalized
        raw = [(k + 1) for k in range(K)]
        total = sum(raw)
        weights = [w / total for w in raw]
    elif weighting == "exponential":
        # [1, 2, 4, 8, ...] normalized
        raw = [2.0 ** k for k in range(K)]
        total = sum(raw)
        weights = [w / total for w in raw]
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Build per-token mask
    if attention_mask is not None:
        token_mask = attention_mask.float()
    else:
        token_mask = torch.ones_like(labels, dtype=dtype)

    # Optionally restrict to output tokens only
    if output_only and output_mask is not None:
        token_mask = token_mask * output_mask.float()

    # Shift for next-token prediction: logits[:, :-1] predicts labels[:, 1:]
    shift_mask = token_mask[:, 1:].contiguous()  # [B, S-1]
    shift_labels = labels[:, 1:].contiguous()    # [B, S-1]

    # Also create separate input/output masks for tracking
    if output_mask is not None:
        shift_output_mask = output_mask[:, 1:].contiguous().float()
        shift_input_mask = (1.0 - shift_output_mask) * (attention_mask[:, 1:].float() if attention_mask is not None else 1.0)
    else:
        shift_output_mask = None
        shift_input_mask = None

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_input_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_output_loss = torch.tensor(0.0, device=device, dtype=dtype)
    loss_dict = {}

    for k, logits in enumerate(all_logits):
        shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V]

        # Compute per-token loss
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.reshape(-1, vocab_size)
        flat_labels = shift_labels.reshape(-1)
        per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        per_token_loss = per_token_loss.reshape(shift_labels.shape)

        # Apply mask and compute mean
        masked_loss = per_token_loss * shift_mask
        step_loss = masked_loss.sum() / shift_mask.sum().clamp(min=1)

        # Track input/output losses separately using final step logits
        if k == K - 1 and shift_output_mask is not None:
            output_tokens = shift_output_mask.sum().clamp(min=1)
            input_tokens = shift_input_mask.sum().clamp(min=1)
            total_output_loss = (per_token_loss * shift_output_mask).sum() / output_tokens
            total_input_loss = (per_token_loss * shift_input_mask).sum() / input_tokens

        # Accumulate weighted loss
        total_loss = total_loss + weights[k] * step_loss
        loss_dict[f"loss_step_{k}"] = step_loss.detach()

    # Return dict compatible with train.py expectations
    return {
        "loss": total_loss,
        "input_loss": total_input_loss,
        "output_loss": total_output_loss,
        **loss_dict,
    }


@dataclass
class TransformerConfig:
    """Configuration for the TinyTransformer model."""
    vocab_size: int = 14          # 10 colors (0-9) + 4 special tokens (START, NEWLINE, IO_SEP, END)
    max_seq_len: int = 1863       # Maximum sequence length (matches original mdlARC)
    d_model: int = 128            # Model dimension
    n_heads: int = 4              # Number of attention heads
    d_ff: int = 320               # Feed-forward hidden dimension
    n_layers: int = 4             # Number of transformer layers
    dropout: float = 0.1         # Dropout probability
    num_examples: int = 1280      # Max number of example embeddings
    max_x: int = 32               # Max x coordinate (grid width)
    max_y: int = 32               # Max y coordinate (grid height)
    max_z: int = 8                # Max z coordinate (matches original mdlARC: 0-4 used)
    rope_base: float = 10000.0    # RoPE base frequency
    norm_eps: float = 1e-5        # RMSNorm epsilon
    rope_type: str = "3d"         # "3d", "1d", or "none"
    norm_type: str = "rmsnorm"    # "rmsnorm" or "layernorm"
    ffn_type: str = "swiglu"      # "swiglu" or "gelu"
    use_example_embedding: bool = True  # If False, ignore example IDs
    # New: 3D relative position bias (matches 3D RoPE dimensions)
    use_relative_bias: bool = False  # Add learnable relative position bias to attention
    max_relative_dist_xy: int = 30   # Maximum relative distance for x/y bias table
    max_relative_dist_z: int = 8     # Maximum relative distance for z (semantic layer) bias table
    # New: Edge and grid size encoding
    use_edge_encoding: bool = False  # Add per-token edge distance features
    use_grid_size_encoding: bool = False  # Add global grid size embedding
    # New: Refinement loop training
    num_refinement_steps: int = 1        # Number of refinement iterations (1 = standard, >1 = refinement)
    refinement_use_gating: bool = True   # Use gated updates for stability
    refinement_focus_output: bool = True # Focus attention on output tokens during refinement


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)
    This is the rotation operation used in RoPE.
    """
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding3D(nn.Module):
    """
    3D Rotary Position Embeddings.

    Unlike standard 1D RoPE, this splits the head dimension into 3 parts
    for x, y, z coordinates. This allows the model to understand spatial
    relationships in 2D grids (x, y) plus distinguish between examples (z).
    """

    def __init__(
        self,
        head_dim: int,
        max_x: int = 32,
        max_y: int = 32,
        max_z: int = 16,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.base = base

        # Split head dimension into pairs, then divide among x, y, z
        # Each dimension needs pairs of values for rotation
        n_pairs = head_dim // 2
        self.px = n_pairs // 3          # pairs for x
        self.py = n_pairs // 3          # pairs for y
        self.pz = n_pairs - self.px - self.py  # remaining for z

        # Precompute and cache cos/sin tables for each dimension
        self._build_cache()

    def _build_cache(self):
        """Precompute cos/sin lookup tables for efficiency."""
        # Build inverse frequency vectors for each dimension
        inv_freq_x = self._make_inv_freq(self.px * 2)
        inv_freq_y = self._make_inv_freq(self.py * 2)
        inv_freq_z = self._make_inv_freq(self.pz * 2)

        # Build position ranges
        pos_x = torch.arange(self.max_x, dtype=torch.float32)
        pos_y = torch.arange(self.max_y, dtype=torch.float32)
        pos_z = torch.arange(self.max_z, dtype=torch.float32)

        # Compute outer products: [max_coord, dim//2]
        freqs_x = torch.outer(pos_x, inv_freq_x)
        freqs_y = torch.outer(pos_y, inv_freq_y)
        freqs_z = torch.outer(pos_z, inv_freq_z)

        # Register as buffers (not parameters, but saved with model)
        self.register_buffer("cos_x", freqs_x.cos(), persistent=False)
        self.register_buffer("sin_x", freqs_x.sin(), persistent=False)
        self.register_buffer("cos_y", freqs_y.cos(), persistent=False)
        self.register_buffer("sin_y", freqs_y.sin(), persistent=False)
        self.register_buffer("cos_z", freqs_z.cos(), persistent=False)
        self.register_buffer("sin_z", freqs_z.sin(), persistent=False)

    def _make_inv_freq(self, dim: int) -> torch.Tensor:
        """Create inverse frequency vector for RoPE."""
        # inv_freq = 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        return inv_freq

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        return _rotate_half(x)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D rotary embeddings to queries and keys.

        Args:
            q: [batch, seq, heads, head_dim]
            k: [batch, seq, heads, head_dim]
            pos_xyz: [batch, seq, 3] - x, y, z coordinates as integers

        Returns:
            Rotated q and k tensors
        """
        # Shape is [batch, seq, heads, head_dim] - we use pos_xyz for indexing

        # Extract coordinates and clamp to cache bounds
        x_pos = pos_xyz[..., 0].clamp(0, self.max_x - 1).long()  # [B, S]
        y_pos = pos_xyz[..., 1].clamp(0, self.max_y - 1).long()
        z_pos = pos_xyz[..., 2].clamp(0, self.max_z - 1).long()

        # Gather cos/sin values for each position
        # Shape: [B, S, dim] -> need to expand for heads
        cos_x = self.cos_x[x_pos]  # [B, S, px]
        sin_x = self.sin_x[x_pos]
        cos_y = self.cos_y[y_pos]  # [B, S, py]
        sin_y = self.sin_y[y_pos]
        cos_z = self.cos_z[z_pos]  # [B, S, pz]
        sin_z = self.sin_z[z_pos]

        # Concatenate cos/sin for all dimensions
        # Double each because we need cos for both elements of each pair
        cos_full = torch.cat([
            cos_x.repeat_interleave(2, dim=-1),
            cos_y.repeat_interleave(2, dim=-1),
            cos_z.repeat_interleave(2, dim=-1),
        ], dim=-1)  # [B, S, head_dim]

        sin_full = torch.cat([
            sin_x.repeat_interleave(2, dim=-1),
            sin_y.repeat_interleave(2, dim=-1),
            sin_z.repeat_interleave(2, dim=-1),
        ], dim=-1)  # [B, S, head_dim]

        # Expand for heads: [B, S, 1, head_dim]
        cos_full = cos_full.unsqueeze(2)
        sin_full = sin_full.unsqueeze(2)

        # Apply rotation: out = x * cos + rotate_half(x) * sin
        q_rot = q * cos_full + _rotate_half(q) * sin_full
        k_rot = k * cos_full + _rotate_half(k) * sin_full

        return q_rot, k_rot


class RotaryEmbedding1D(nn.Module):
    """
    1D Rotary Position Embeddings over sequence positions.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pos: [B, S]
        pos = pos.clamp(0, self.max_seq_len - 1).long()
        cos = self.cos[pos]  # [B, S, head_dim/2]
        sin = self.sin[pos]

        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(2)

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


class RelativePositionBias3D(nn.Module):
    """
    Learnable 3D relative position bias for attention.

    Adds a learnable bias to attention scores based on the relative
    x, y, and z distances between query and key positions. This complements
    3D RoPE by providing explicit distance-based attention modulation.

    The z dimension captures semantic layer distances:
    - z_diff=0: same region (input→input or output→output)
    - z_diff=2: input↔output (the key transformation relationship)
    """

    def __init__(self, max_dist_xy: int, max_dist_z: int, n_heads: int):
        super().__init__()
        self.max_dist_xy = max_dist_xy
        self.max_dist_z = max_dist_z
        self.n_heads = n_heads

        # Learnable bias tables for x, y, and z distances
        # Range: [-max_dist, max_dist] -> 2*max_dist + 1 entries
        num_buckets_xy = 2 * max_dist_xy + 1
        num_buckets_z = 2 * max_dist_z + 1

        self.bias_x = nn.Embedding(num_buckets_xy, n_heads)
        self.bias_y = nn.Embedding(num_buckets_xy, n_heads)
        self.bias_z = nn.Embedding(num_buckets_z, n_heads)

        # Initialize with small values
        nn.init.normal_(self.bias_x.weight, std=0.02)
        nn.init.normal_(self.bias_y.weight, std=0.02)
        nn.init.normal_(self.bias_z.weight, std=0.02)

    def forward(self, pos_xyz: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D relative position bias for attention.

        Args:
            pos_xyz: [batch, seq, 3] position coordinates (x, y, z)

        Returns:
            bias: [batch, n_heads, seq, seq] attention bias
        """
        # Extract x, y, z positions
        x = pos_xyz[..., 0]  # [B, S]
        y = pos_xyz[..., 1]  # [B, S]
        z = pos_xyz[..., 2]  # [B, S]

        # Compute pairwise relative distances
        # d[i, j] = pos[i] - pos[j] (query position - key position)
        dx = x.unsqueeze(-1) - x.unsqueeze(-2)  # [B, S, S]
        dy = y.unsqueeze(-1) - y.unsqueeze(-2)  # [B, S, S]
        dz = z.unsqueeze(-1) - z.unsqueeze(-2)  # [B, S, S]

        # Clamp to valid range and shift to positive indices
        dx = dx.clamp(-self.max_dist_xy, self.max_dist_xy) + self.max_dist_xy
        dy = dy.clamp(-self.max_dist_xy, self.max_dist_xy) + self.max_dist_xy
        dz = dz.clamp(-self.max_dist_z, self.max_dist_z) + self.max_dist_z

        # Look up biases
        bias_x = self.bias_x(dx.long())  # [B, S, S, n_heads]
        bias_y = self.bias_y(dy.long())  # [B, S, S, n_heads]
        bias_z = self.bias_z(dz.long())  # [B, S, S, n_heads]

        # Combine and transpose to [B, n_heads, S, S]
        bias = (bias_x + bias_y + bias_z).permute(0, 3, 1, 2)

        return bias


def build_norm(norm_type: str, dim: int, eps: float) -> nn.Module:
    """Factory for normalization layers."""
    norm = norm_type.lower()
    if norm == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with 3D RoPE support and optional relative position bias."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_type.lower()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Rotary embeddings
        if self.rope_type == "3d":
            self.rope = RotaryEmbedding3D(
                head_dim=self.head_dim,
                max_x=config.max_x,
                max_y=config.max_y,
                max_z=config.max_z,
                base=config.rope_base,
            )
        elif self.rope_type == "1d":
            self.rope = RotaryEmbedding1D(
                head_dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                base=config.rope_base,
            )
        elif self.rope_type == "none":
            self.rope = None
        else:
            raise ValueError(f"Unknown rope_type: {config.rope_type}")

        # Optional: 3D relative position bias (complements 3D RoPE)
        self.relative_bias: RelativePositionBias3D | None = None
        if config.use_relative_bias:
            self.relative_bias = RelativePositionBias3D(
                max_dist_xy=config.max_relative_dist_xy,
                max_dist_z=config.max_relative_dist_z,
                n_heads=config.n_heads,
            )

        self.dropout = nn.Dropout(config.dropout)

        # KV cache for inference
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        pos_xyz: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Uses PyTorch's native scaled_dot_product_attention which automatically
        selects the best backend (FlashAttention, Memory-Efficient, or Math).
        When relative_bias is enabled, falls back to manual attention computation.

        Args:
            x: [batch, seq, d_model]
            pos_xyz: [batch, seq, 3] - position coordinates
            attention_mask: Optional mask for padding
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch, seq_len, _ = x.shape

        # Compute QKV
        qkv = self.qkv_proj(x)  # [B, S, 3*d_model]
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: [B, S, heads, head_dim]

        # Apply RoPE to Q and K
        if self.rope is not None:
            if self.rope_type == "3d":
                q, k = self.rope.apply_rotary(q, k, pos_xyz)
            else:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
                q, k = self.rope.apply_rotary(q, k, positions)

        # Transpose for attention: [B, heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # If using relative position bias, compute attention manually
        if self.relative_bias is not None:
            # Compute attention scores
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, S, S]

            # Add relative position bias
            rel_bias = self.relative_bias(pos_xyz)  # [B, heads, S, S]
            attn_scores = attn_scores + rel_bias

            # Apply causal mask
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=q.device, dtype=q.dtype),
                    diagonal=1
                )
                attn_scores = attn_scores + causal_mask

            # Apply padding mask if provided
            if attention_mask is not None:
                # attention_mask: [B, S] bool where True = valid, False = padding
                padding_mask = torch.where(
                    attention_mask.unsqueeze(1).unsqueeze(2),
                    0.0, float('-inf')
                ).to(q.dtype)
                attn_scores = attn_scores + padding_mask

            # Softmax and dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply to values
            attn_out = attn_weights @ v  # [B, heads, S, head_dim]

        else:
            # Use PyTorch's SDPA (auto-selects FlashAttention when possible)
            # Convert boolean mask to float attention mask for SDPA
            attn_mask = None
            use_is_causal = is_causal

            if attention_mask is not None:
                # attention_mask: [B, S] bool where True = valid, False = padding
                # Convert to [B, 1, 1, S] float where 0.0 = valid, -inf = padding
                bool_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = torch.where(bool_mask, 0.0, float('-inf')).to(q.dtype)

                # MPS and some PyTorch versions don't support is_causal=True with explicit mask
                # Build a combined causal + padding mask instead
                if is_causal:
                    # Create causal mask: [1, 1, S, S]
                    causal_mask = torch.triu(
                        torch.full((seq_len, seq_len), float('-inf'), device=q.device, dtype=q.dtype),
                        diagonal=1
                    ).unsqueeze(0).unsqueeze(0)
                    # Combine: add causal mask to padding mask
                    # attn_mask is [B, 1, 1, S], expand to [B, 1, S, S] for broadcasting
                    attn_mask = attn_mask.expand(-1, -1, seq_len, -1) + causal_mask
                    use_is_causal = False  # Causal is now baked into attn_mask

            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=use_is_causal,
            )  # [B, heads, S, head_dim]

        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.config.d_model)

        # Output projection
        return self.out_proj(attn_out)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        pos_xyz: torch.Tensor,
        cache_position: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with KV caching for efficient autoregressive generation.

        Args:
            x: [batch, 1, d_model] - single token
            pos_xyz: [batch, 1, 3] - position for this token
            cache_position: Position index in the cache
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, 1, d_model]
        """
        batch, seq_len, d_model = x.shape
        assert seq_len == 1, "Cache mode expects single token input"

        # Compute QKV for new token
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Apply RoPE
        if self.rope is not None:
            if self.rope_type == "3d":
                q, k = self.rope.apply_rotary(q, k, pos_xyz)
            else:
                positions = torch.full(
                    (batch, 1),
                    int(cache_position),
                    device=x.device,
                    dtype=torch.long,
                )
                q, k = self.rope.apply_rotary(q, k, positions)

        # Initialize cache if needed
        if self.k_cache is None or self.k_cache.size(0) != batch:
            max_len = self.config.max_seq_len
            self.k_cache = torch.zeros(
                batch, self.n_heads, max_len, self.head_dim,
                device=x.device, dtype=x.dtype
            )
            self.v_cache = torch.zeros(
                batch, self.n_heads, max_len, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        # Update cache
        k = k.transpose(1, 2)  # [B, heads, 1, head_dim]
        v = v.transpose(1, 2)
        assert self.k_cache is not None and self.v_cache is not None
        self.k_cache[:, :, cache_position:cache_position+1, :] = k
        self.v_cache[:, :, cache_position:cache_position+1, :] = v

        # Attention over cached keys/values
        q = q.transpose(1, 2)  # [B, heads, 1, head_dim]
        k_full = self.k_cache[:, :, :cache_position+1, :]
        v_full = self.v_cache[:, :, :cache_position+1, :]

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attention_mask,
            is_causal=False,  # We handle causality via cache_position
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch, 1, d_model)
        return self.out_proj(attn_out)

    def reset_cache(self):
        """Clear the KV cache."""
        self.k_cache = None
        self.v_cache = None


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc_gate = nn.Linear(config.d_model, config.d_ff)
        self.fc_up = nn.Linear(config.d_model, config.d_ff)
        self.fc_down = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.fc_gate(x))
        up = self.fc_up(x)
        x = gate * up
        x = self.fc_down(x)
        return self.dropout(x)


class FeedForwardGELU(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = build_norm(config.norm_type, config.d_model, config.norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = build_norm(config.norm_type, config.d_model, config.norm_eps)
        if config.ffn_type.lower() == "swiglu":
            self.ffn = FeedForward(config)
        elif config.ffn_type.lower() == "gelu":
            self.ffn = FeedForwardGELU(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_xyz: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), pos_xyz, attention_mask, is_causal))
        # Pre-norm FFN with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        pos_xyz: torch.Tensor,
        cache_position: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn.forward_with_cache(
            self.ln1(x), pos_xyz, cache_position, attention_mask
        )
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """
    Complete TinyTransformer for ARC tasks.

    Features:
    - 3D Rotary Position Embeddings for spatial awareness
    - Example embeddings to distinguish between train/test examples
    - Optional edge distance encoding (distance to grid boundaries)
    - Optional grid size encoding (awareness of grid dimensions)
    - KV cache for efficient inference
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Example embeddings (to distinguish different input-output pairs)
        self.example_embed = nn.Embedding(config.num_examples, config.d_model)

        # Optional: Edge distance encoding (per-token)
        # Encodes distance to left, right, top, bottom edges
        self.edge_embed: nn.Linear | None = None
        if config.use_edge_encoding:
            self.edge_embed = nn.Linear(4, config.d_model, bias=False)

        # Optional: Grid size encoding (global, added to all tokens)
        # Encodes the height and width of the grid
        self.height_embed: nn.Embedding | None = None
        self.width_embed: nn.Embedding | None = None
        if config.use_grid_size_encoding:
            self.height_embed = nn.Embedding(config.max_y + 1, config.d_model)
            self.width_embed = nn.Embedding(config.max_x + 1, config.d_model)

        # Optional: Refinement loop modules
        # Uses cross-attention + gated residual update for example embedding refinement
        self.refine_query_proj: nn.Linear | None = None
        self.refine_key_proj: nn.Linear | None = None
        self.refine_value_proj: nn.Linear | None = None
        self.refine_out_proj: nn.Linear | None = None
        self.refine_gate: nn.Linear | None = None

        if config.num_refinement_steps > 1:
            # Cross-attention projections for refinement
            self.refine_query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.refine_key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.refine_value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.refine_out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

            # Initialize output projection with small values for stable residual updates
            nn.init.normal_(self.refine_out_proj.weight, std=0.01)

            # Gated update: learns how much to incorporate the refinement
            if config.refinement_use_gating:
                self.refine_gate = nn.Linear(2 * config.d_model, config.d_model, bias=True)
                # Initialize gate bias negative so initial gate values are small
                nn.init.zeros_(self.refine_gate.weight)
                nn.init.constant_(self.refine_gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = build_norm(config.norm_type, config.d_model, config.norm_eps)

        # Language model head (tied with token embeddings optionally)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _compute_edge_distances(
        self,
        pos_xyz: torch.Tensor,
        grid_h: torch.Tensor,
        grid_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance to each edge for every token.

        Args:
            pos_xyz: [batch, seq, 3] - position coordinates
            grid_h: [batch] - grid heights
            grid_w: [batch] - grid widths

        Returns:
            edge_dist: [batch, seq, 4] - (left, right, top, bottom) distances
        """
        x = pos_xyz[..., 0].float()  # [B, S]
        y = pos_xyz[..., 1].float()  # [B, S]

        # Expand grid sizes for broadcasting
        grid_w = grid_w.float().unsqueeze(1)  # [B, 1]
        grid_h = grid_h.float().unsqueeze(1)  # [B, 1]

        # Compute distances (normalized by grid size for scale invariance)
        dist_left = x / grid_w.clamp(min=1)
        dist_right = (grid_w - 1 - x) / grid_w.clamp(min=1)
        dist_top = y / grid_h.clamp(min=1)
        dist_bottom = (grid_h - 1 - y) / grid_h.clamp(min=1)

        return torch.stack([dist_left, dist_right, dist_top, dist_bottom], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_xyz: torch.Tensor,
        example_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        grid_h: torch.Tensor | None = None,
        grid_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq] - token indices
            pos_xyz: [batch, seq, 3] - 3D position coordinates
            example_ids: [batch] - example index for each sequence
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            grid_h: [batch] - grid heights (optional, for edge/size encoding)
            grid_w: [batch] - grid widths (optional, for edge/size encoding)

        Returns:
            Logits [batch, seq, vocab_size]
        """
        # Token embeddings
        x = self.token_embed(input_ids)  # [B, S, d_model]

        # Add example embeddings if provided (THE BOTTLENECK)
        if self.config.use_example_embedding and example_ids is not None:
            ex_embed = self.example_embed(example_ids)  # [B, d_model]
            x = x + ex_embed.unsqueeze(1)  # Broadcast across sequence

        # Add edge distance encoding if enabled
        if self.edge_embed is not None:
            if grid_h is None or grid_w is None:
                # Infer grid size from positions (max x+1, max y+1)
                grid_w = pos_xyz[..., 0].max(dim=1).values + 1  # [B]
                grid_h = pos_xyz[..., 1].max(dim=1).values + 1  # [B]
            edge_dist = self._compute_edge_distances(pos_xyz, grid_h, grid_w)
            x = x + self.edge_embed(edge_dist)

        # Add grid size encoding if enabled
        if self.height_embed is not None and self.width_embed is not None:
            if grid_h is None or grid_w is None:
                # Infer grid size from positions
                grid_w = pos_xyz[..., 0].max(dim=1).values + 1  # [B]
                grid_h = pos_xyz[..., 1].max(dim=1).values + 1  # [B]
            # Clamp to valid embedding range
            grid_h_clamped = grid_h.clamp(0, self.config.max_y).long()
            grid_w_clamped = grid_w.clamp(0, self.config.max_x).long()
            size_embed = self.height_embed(grid_h_clamped) + self.width_embed(grid_w_clamped)
            x = x + size_embed.unsqueeze(1)  # Broadcast to all tokens

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, pos_xyz, attention_mask, is_causal)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def _refine_example_embed(
        self,
        ex_embed: torch.Tensor,
        hidden_states: torch.Tensor,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Refine the example embedding using cross-attention over hidden states.

        The example embedding "queries" the hidden states to find relevant
        information for refinement. This implements the TRM-style iterative
        update while preserving the bottleneck.

        Args:
            ex_embed: [batch, d_model] - current example embedding
            hidden_states: [batch, seq, d_model] - transformer hidden states
            output_mask: [batch, seq] - optional mask to focus on output tokens

        Returns:
            Updated example embedding [batch, d_model]
        """
        batch, seq_len, d_model = hidden_states.shape

        # Project for cross-attention
        assert self.refine_query_proj is not None
        assert self.refine_key_proj is not None
        assert self.refine_value_proj is not None
        assert self.refine_out_proj is not None

        query = self.refine_query_proj(ex_embed).unsqueeze(1)  # [B, 1, d_model]
        keys = self.refine_key_proj(hidden_states)              # [B, S, d_model]
        values = self.refine_value_proj(hidden_states)          # [B, S, d_model]

        # Compute attention scores
        scale = d_model ** -0.5
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) * scale  # [B, 1, S]

        # Optionally focus attention on output tokens
        if output_mask is not None and self.config.refinement_focus_output:
            # output_mask: [B, S] where True = output token
            # Mask out non-output tokens with large negative value
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(
                ~output_mask.unsqueeze(1),  # [B, 1, S]
                mask_value
            )

        # Softmax and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, S]
        context = torch.bmm(attn_weights, values).squeeze(1)  # [B, d_model]

        # Project context for update
        update = self.refine_out_proj(context)  # [B, d_model]

        # Apply gated residual update
        if self.refine_gate is not None:
            # Compute gate: how much to incorporate the update
            gate_input = torch.cat([ex_embed, context], dim=-1)  # [B, 2*d_model]
            gate = torch.sigmoid(self.refine_gate(gate_input))    # [B, d_model]
            # Gated residual: ex_embed + gate * update
            refined = ex_embed + gate * update
        else:
            # Simple residual update
            refined = ex_embed + update

        return refined

    def forward_with_refinement(
        self,
        input_ids: torch.Tensor,
        pos_xyz: torch.Tensor,
        example_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        grid_h: torch.Tensor | None = None,
        grid_w: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Forward pass with iterative refinement of the example embedding.

        Implements TRM-style refinement where:
        1. Initial forward pass with base example embedding
        2. Refine example embedding using cross-attention over hidden states
        3. Repeat for K steps
        4. Return logits at each step for deep supervision

        Args:
            input_ids: [batch, seq] - token indices
            pos_xyz: [batch, seq, 3] - 3D position coordinates
            example_ids: [batch] - example index for each sequence
            attention_mask: Optional attention mask
            output_mask: [batch, seq] - mask for output tokens (for focused refinement)
            is_causal: Whether to use causal masking
            grid_h: [batch] - grid heights (optional)
            grid_w: [batch] - grid widths (optional)
            num_steps: Override config.num_refinement_steps

        Returns:
            List of logits tensors, one per refinement step [K × (batch, seq, vocab)]
        """
        K = num_steps if num_steps is not None else self.config.num_refinement_steps

        if K <= 1 or self.refine_query_proj is None:
            # No refinement, just return regular forward pass in a list
            return [self.forward(
                input_ids, pos_xyz, example_ids, attention_mask, is_causal, grid_h, grid_w
            )]

        # Get initial example embedding
        ex_embed = self.example_embed(example_ids)  # [B, d_model]

        # Precompute edge distances if needed (avoid recomputing each step)
        edge_encoding = None
        if self.edge_embed is not None:
            if grid_h is None or grid_w is None:
                grid_w = pos_xyz[..., 0].max(dim=1).values + 1
                grid_h = pos_xyz[..., 1].max(dim=1).values + 1
            edge_dist = self._compute_edge_distances(pos_xyz, grid_h, grid_w)
            edge_encoding = self.edge_embed(edge_dist)

        # Precompute grid size embedding if needed
        size_encoding = None
        if self.height_embed is not None and self.width_embed is not None:
            if grid_h is None or grid_w is None:
                grid_w = pos_xyz[..., 0].max(dim=1).values + 1
                grid_h = pos_xyz[..., 1].max(dim=1).values + 1
            grid_h_clamped = grid_h.clamp(0, self.config.max_y).long()
            grid_w_clamped = grid_w.clamp(0, self.config.max_x).long()
            size_encoding = self.height_embed(grid_h_clamped) + self.width_embed(grid_w_clamped)

        all_logits = []

        for k in range(K):
            # Build input embeddings with current example embedding
            x = self.token_embed(input_ids)  # [B, S, d_model]
            x = x + ex_embed.unsqueeze(1)     # Add example embedding (THE BOTTLENECK)

            # Add precomputed encodings
            if edge_encoding is not None:
                x = x + edge_encoding
            if size_encoding is not None:
                x = x + size_encoding.unsqueeze(1)

            # Forward through transformer blocks
            for block in self.blocks:
                x = block(x, pos_xyz, attention_mask, is_causal)

            # Store hidden states before final norm (for refinement)
            hidden_states = x

            # Compute logits
            x = self.ln_f(x)
            logits = self.lm_head(x)
            all_logits.append(logits)

            # Refine example embedding for next iteration (except last step)
            if k < K - 1:
                ex_embed = self._refine_example_embed(ex_embed, hidden_states, output_mask)

        return all_logits

    def generate(
        self,
        input_ids: torch.Tensor,
        pos_xyz: torch.Tensor,
        max_new_tokens: int,
        example_ids: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching.

        Args:
            input_ids: [batch, prompt_len] - prompt tokens
            pos_xyz: [batch, prompt_len, 3] - prompt positions
            max_new_tokens: Maximum tokens to generate
            example_ids: Optional example indices
            temperature: Sampling temperature
            top_k: Top-k sampling
            greedy: If True, use argmax (ignores temperature/top_k)

        Returns:
            Generated sequence [batch, prompt_len + max_new_tokens]
        """
        prompt_len = input_ids.size(1)

        # Reset caches
        for block in self.blocks:
            block.attn.reset_cache()

        # Process prompt (fill cache)
        x = self.token_embed(input_ids)
        if self.config.use_example_embedding and example_ids is not None:
            x = x + self.example_embed(example_ids).unsqueeze(1)

        for block in self.blocks:
            x = block(x, pos_xyz, is_causal=True)

        # Note: Cache is filled during forward_with_cache calls below

        # Get first prediction
        x = self.ln_f(x)
        logits = self.lm_head(x[:, -1:, :])  # [B, 1, vocab]

        # Sample or greedy
        if greedy:
            next_token = logits.argmax(dim=-1)  # [B, 1]
        else:
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            if top_k is not None:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)

        # Collect generated tokens
        generated = [input_ids, next_token]

        # Continue generation
        for i in range(max_new_tokens - 1):
            # Compute next position (simplified - would need proper tracking)
            # For now, just increment z coordinate
            new_pos = pos_xyz[:, -1:, :].clone()
            new_pos[:, :, 0] = (new_pos[:, :, 0] + 1) % self.config.max_x

            # Forward single token through cache
            x = self.token_embed(next_token)
            if self.config.use_example_embedding and example_ids is not None:
                x = x + self.example_embed(example_ids).unsqueeze(1)

            cache_pos = prompt_len + i
            for block in self.blocks:
                x = block.forward_with_cache(x, new_pos, cache_pos)

            x = self.ln_f(x)
            logits = self.lm_head(x)

            if greedy:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            pos_xyz = torch.cat([pos_xyz, new_pos], dim=1)

        return torch.cat(generated, dim=1)

    def reset_cache(self):
        """Reset all KV caches."""
        for block in self.blocks:
            block.attn.reset_cache()

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    vocab_size: int = 14,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 320,
    dropout: float = 0.1,
    **kwargs,
) -> TinyTransformer:
    """Factory function to create a TinyTransformer with common defaults."""
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        **kwargs,
    )
    return TinyTransformer(config)


# Convenience functions for different model sizes
def create_tiny_model(**kwargs) -> TinyTransformer:
    """~3.6M params - for testing."""
    return create_model(d_model=256, n_heads=4, n_layers=4, d_ff=704, **kwargs)


def create_small_model(**kwargs) -> TinyTransformer:
    """~29M params - comparable to original mdlARC 28M."""
    return create_model(d_model=768, n_heads=12, n_layers=4, d_ff=2048, **kwargs)


def create_medium_model(**kwargs) -> TinyTransformer:
    """~103M params - scaled up."""
    return create_model(d_model=1024, n_heads=16, n_layers=8, d_ff=2752, **kwargs)


def create_large_model(**kwargs) -> TinyTransformer:
    """~237M params - significantly scaled."""
    return create_model(d_model=1280, n_heads=20, n_layers=12, d_ff=3392, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing TinyTransformer with 3D RoPE...")

    model = create_small_model()
    print(f"Model parameters: {model.num_parameters:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 64

    input_ids = torch.randint(0, 14, (batch_size, seq_len))
    pos_xyz = torch.stack([
        torch.arange(seq_len) % 30,  # x
        torch.arange(seq_len) // 30,  # y
        torch.zeros(seq_len),  # z
    ], dim=-1).unsqueeze(0).expand(batch_size, -1, -1).long()

    with torch.no_grad():
        logits = model(input_ids, pos_xyz)

    print(f"Input shape: {input_ids.shape}")
    print(f"Position shape: {pos_xyz.shape}")
    print(f"Output shape: {logits.shape}")
    print("Forward pass successful!")
