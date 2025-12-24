"""
TinyTransformer with 3D Rotary Position Embeddings for ARC tasks.

Reproduced from scratch based on the mdlARC paper/approach.
Key innovation: 3D RoPE that handles spatial (x, y) + example (z) coordinates.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        Rotate pairs: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)
        This is the rotation operation used in RoPE.
        """
        x1 = x[..., ::2]   # even indices
        x2 = x[..., 1::2]  # odd indices
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

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
        q_rot = q * cos_full + self._rotate_half(q) * sin_full
        k_rot = k * cos_full + self._rotate_half(k) * sin_full

        return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with 3D RoPE support."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # 3D Rotary embeddings
        self.rope = RotaryEmbedding3D(
            head_dim=self.head_dim,
            max_x=config.max_x,
            max_y=config.max_y,
            max_z=config.max_z,
            base=config.rope_base,
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

        # Apply 3D RoPE to Q and K
        q, k = self.rope.apply_rotary(q, k, pos_xyz)

        # Transpose for attention: [B, heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert boolean mask to float attention mask for SDPA
        # SDPA expects float mask where -inf masks out positions
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, S] bool where True = valid, False = padding
            # Convert to [B, 1, 1, S] float where 0.0 = valid, -inf = padding
            # Keep as bool for torch.where predicate, then cast to q.dtype
            bool_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(bool_mask, 0.0, float('-inf')).to(q.dtype)

        # Use PyTorch's SDPA (auto-selects FlashAttention when possible)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
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
        q, k = self.rope.apply_rotary(q, k, pos_xyz)

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


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FeedForward(config)
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
    - KV cache for efficient inference
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Example embeddings (to distinguish different input-output pairs)
        self.example_embed = nn.Embedding(config.num_examples, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_xyz: torch.Tensor,
        example_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq] - token indices
            pos_xyz: [batch, seq, 3] - 3D position coordinates
            example_ids: [batch] - example index for each sequence
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Logits [batch, seq, vocab_size]
        """
        # Token embeddings
        x = self.token_embed(input_ids)  # [B, S, d_model]

        # Add example embeddings if provided
        if example_ids is not None:
            ex_embed = self.example_embed(example_ids)  # [B, d_model]
            x = x + ex_embed.unsqueeze(1)  # Broadcast across sequence

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, pos_xyz, attention_mask, is_causal)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

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
        if example_ids is not None:
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
            if example_ids is not None:
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
def create_tiny_model() -> TinyTransformer:
    """~3.6M params - for testing."""
    return create_model(d_model=256, n_heads=4, n_layers=4, d_ff=704)


def create_small_model() -> TinyTransformer:
    """~29M params - comparable to original mdlARC 28M."""
    return create_model(d_model=768, n_heads=12, n_layers=4, d_ff=2048)


def create_medium_model() -> TinyTransformer:
    """~103M params - scaled up."""
    return create_model(d_model=1024, n_heads=16, n_layers=8, d_ff=2752)


def create_large_model() -> TinyTransformer:
    """~237M params - significantly scaled."""
    return create_model(d_model=1280, n_heads=20, n_layers=12, d_ff=3392)


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
