"""Tests for the 3D RoPE Transformer."""

import pytest
import torch

from src.transformer import (
    FeedForward,
    MultiHeadSelfAttention,
    RotaryEmbedding3D,
    TransformerBlock,
    TransformerConfig,
    create_small_model,
    create_tiny_model,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Small config for fast tests."""
    return TransformerConfig(
        vocab_size=14,
        max_seq_len=512,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.0,  # Disable for deterministic tests
    )


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(0, 14, (batch_size, seq_len))
    pos_xyz = torch.stack([
        torch.arange(seq_len) % 10,  # x
        torch.arange(seq_len) // 10,  # y
        torch.zeros(seq_len),  # z
    ], dim=-1).unsqueeze(0).expand(batch_size, -1, -1).long()

    return input_ids, pos_xyz


# =============================================================================
# ROTARY EMBEDDING TESTS
# =============================================================================

class TestRotaryEmbedding3D:
    """Tests for 3D Rotary Position Embeddings."""

    def test_init(self):
        """Test RoPE initialization."""
        rope = RotaryEmbedding3D(head_dim=64)

        assert rope.head_dim == 64
        assert rope.px + rope.py + rope.pz == 32  # head_dim // 2
        assert hasattr(rope, 'cos_x')
        assert hasattr(rope, 'sin_x')

    def test_cache_shapes(self):
        """Test precomputed cache shapes."""
        rope = RotaryEmbedding3D(head_dim=64, max_x=32, max_y=32, max_z=16)

        assert rope.cos_x.shape == (32, rope.px)
        assert rope.cos_y.shape == (32, rope.py)
        assert rope.cos_z.shape == (16, rope.pz)

    def test_apply_rotary(self):
        """Test rotary embedding application."""
        rope = RotaryEmbedding3D(head_dim=64)

        batch, seq, heads = 2, 16, 4
        q = torch.randn(batch, seq, heads, 64)
        k = torch.randn(batch, seq, heads, 64)
        pos = torch.randint(0, 10, (batch, seq, 3))

        q_rot, k_rot = rope.apply_rotary(q, k, pos)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        # Rotated vectors should be different
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)

    def test_position_clipping(self):
        """Test that out-of-bounds positions are clipped."""
        rope = RotaryEmbedding3D(head_dim=64, max_x=10, max_y=10, max_z=5)

        q = torch.randn(1, 4, 2, 64)
        k = torch.randn(1, 4, 2, 64)
        # Positions exceed max
        pos = torch.tensor([[[100, 100, 100]] * 4])

        # Should not raise
        q_rot, k_rot = rope.apply_rotary(q, k, pos)
        assert q_rot.shape == q.shape

    def test_deterministic(self):
        """Test that same inputs produce same outputs."""
        rope = RotaryEmbedding3D(head_dim=64)

        q = torch.randn(2, 8, 4, 64)
        k = torch.randn(2, 8, 4, 64)
        pos = torch.randint(0, 10, (2, 8, 3))

        q1, k1 = rope.apply_rotary(q.clone(), k.clone(), pos)
        q2, k2 = rope.apply_rotary(q.clone(), k.clone(), pos)

        assert torch.allclose(q1, q2)
        assert torch.allclose(k1, k2)


# =============================================================================
# ATTENTION TESTS
# =============================================================================

class TestMultiHeadSelfAttention:
    """Tests for Multi-Head Self-Attention."""

    def test_forward_shape(self, config):
        """Test attention output shape."""
        attn = MultiHeadSelfAttention(config)

        batch, seq = 2, 16
        x = torch.randn(batch, seq, config.d_model)
        pos = torch.randint(0, 10, (batch, seq, 3))

        out = attn(x, pos)
        assert out.shape == (batch, seq, config.d_model)

    def test_causal_masking(self, config):
        """Test that causal masking prevents attending to future."""
        attn = MultiHeadSelfAttention(config)
        attn.eval()

        batch, seq = 1, 8
        x = torch.randn(batch, seq, config.d_model)
        pos = torch.zeros(batch, seq, 3, dtype=torch.long)

        # With causal masking
        out_causal = attn(x, pos, is_causal=True)

        # Modify future tokens - should not affect past outputs
        x_modified = x.clone()
        x_modified[:, -1, :] = 999.0
        out_modified = attn(x_modified, pos, is_causal=True)

        # First token output should be identical
        assert torch.allclose(out_causal[:, 0, :], out_modified[:, 0, :], atol=1e-5)

    def test_kv_cache(self, config):
        """Test KV cache for inference."""
        attn = MultiHeadSelfAttention(config)
        attn.eval()

        batch = 2
        x = torch.randn(batch, 1, config.d_model)
        pos = torch.zeros(batch, 1, 3, dtype=torch.long)

        # First token
        out1 = attn.forward_with_cache(x, pos, cache_position=0)
        assert out1.shape == (batch, 1, config.d_model)
        assert attn.k_cache is not None

        # Second token
        out2 = attn.forward_with_cache(x, pos, cache_position=1)
        assert out2.shape == (batch, 1, config.d_model)

        # Reset
        attn.reset_cache()
        assert attn.k_cache is None


# =============================================================================
# FEEDFORWARD TESTS
# =============================================================================

class TestFeedForward:
    """Tests for Feed-Forward Network."""

    def test_forward_shape(self, config):
        """Test FFN output shape."""
        ffn = FeedForward(config)

        x = torch.randn(2, 16, config.d_model)
        out = ffn(x)

        assert out.shape == x.shape

    def test_hidden_dim(self, config):
        """Test that hidden dimension is correct."""
        ffn = FeedForward(config)

        assert ffn.fc_gate.out_features == config.d_ff
        assert ffn.fc_up.out_features == config.d_ff
        assert ffn.fc_down.in_features == config.d_ff


# =============================================================================
# TRANSFORMER BLOCK TESTS
# =============================================================================

class TestTransformerBlock:
    """Tests for Transformer Block."""

    def test_forward_shape(self, config):
        """Test block output shape."""
        block = TransformerBlock(config)

        batch, seq = 2, 16
        x = torch.randn(batch, seq, config.d_model)
        pos = torch.randint(0, 10, (batch, seq, 3))

        out = block(x, pos)
        assert out.shape == x.shape

    def test_residual_connection(self, config):
        """Test that residual connections work."""
        block = TransformerBlock(config)

        x = torch.randn(2, 8, config.d_model)
        pos = torch.zeros(2, 8, 3, dtype=torch.long)

        out = block(x, pos)

        # Output should not be identical to input (unless weights are zero)
        assert not torch.allclose(x, out)


# =============================================================================
# FULL MODEL TESTS
# =============================================================================

class TestTinyTransformer:
    """Tests for the full TinyTransformer model."""

    def test_forward_shape(self, sample_batch):
        """Test model output shape."""
        model = create_tiny_model()
        input_ids, pos_xyz = sample_batch

        logits = model(input_ids, pos_xyz)

        assert logits.shape == (*input_ids.shape, 14)  # vocab_size

    def test_parameter_count(self):
        """Test parameter counts for different model sizes."""
        tiny = create_tiny_model()
        small = create_small_model()

        assert tiny.num_parameters < small.num_parameters
        # Small model should be ~28-30M params
        assert 25_000_000 < small.num_parameters < 35_000_000

    def test_example_embeddings(self, sample_batch):
        """Test that example embeddings work."""
        model = create_tiny_model()
        input_ids, pos_xyz = sample_batch
        example_ids = torch.zeros(input_ids.size(0), dtype=torch.long)

        logits = model(input_ids, pos_xyz, example_ids=example_ids)
        assert logits.shape == (*input_ids.shape, 14)

    def test_gpu_forward(self, sample_batch, device):
        """Test forward pass on GPU."""
        if device.type != "cuda":
            pytest.skip("No CUDA available")

        model = create_tiny_model().to(device)
        input_ids, pos_xyz = sample_batch
        input_ids = input_ids.to(device)
        pos_xyz = pos_xyz.to(device)

        with torch.no_grad():
            logits = model(input_ids, pos_xyz)

        assert logits.device.type == "cuda"

    def test_backward_pass(self, sample_batch):
        """Test that gradients flow correctly."""
        model = create_tiny_model()
        input_ids, pos_xyz = sample_batch

        logits = model(input_ids, pos_xyz)
        loss = logits.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        grads_found = sum(1 for p in model.parameters() if p.grad is not None)
        assert grads_found > 0, "No gradients found"

        # Check key layers have gradients
        assert model.token_embed.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_eval_mode(self, sample_batch):
        """Test model in eval mode."""
        model = create_tiny_model()
        model.eval()

        input_ids, pos_xyz = sample_batch

        with torch.no_grad():
            logits1 = model(input_ids, pos_xyz)
            logits2 = model(input_ids, pos_xyz)

        # Should be deterministic in eval mode
        assert torch.allclose(logits1, logits2)


# =============================================================================
# GENERATION TESTS
# =============================================================================

class TestGeneration:
    """Tests for autoregressive generation."""

    def test_generate_shape(self):
        """Test generation output shape."""
        model = create_tiny_model()
        model.eval()

        batch_size = 2
        prompt_len = 8
        max_new = 4

        input_ids = torch.randint(0, 14, (batch_size, prompt_len))
        pos_xyz = torch.zeros(batch_size, prompt_len, 3, dtype=torch.long)

        with torch.no_grad():
            output = model.generate(input_ids, pos_xyz, max_new_tokens=max_new)

        assert output.shape == (batch_size, prompt_len + max_new)

    def test_generate_deterministic(self):
        """Test that greedy generation is deterministic."""
        model = create_tiny_model()
        model.eval()

        input_ids = torch.randint(0, 14, (1, 4))
        pos_xyz = torch.zeros(1, 4, 3, dtype=torch.long)

        with torch.no_grad():
            out1 = model.generate(input_ids, pos_xyz, max_new_tokens=8, greedy=True)
            model.reset_cache()
            out2 = model.generate(input_ids, pos_xyz, max_new_tokens=8, greedy=True)

        assert torch.equal(out1, out2)
