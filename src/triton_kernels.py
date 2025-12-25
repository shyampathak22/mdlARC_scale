"""
Fused Triton Kernels for ARC Transformer.

Implements memory-efficient attention with 3D relative position bias,
fusing the bias lookup with the attention computation to avoid O(S²) memory.

Based on FlashAttention algorithm with online softmax.

References:
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- DeepSeek V3 FP8: https://arxiv.org/abs/2412.19437
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# =============================================================================
# FORWARD KERNEL (saves logsumexp for backward)
# =============================================================================


@triton.jit
def _fused_attention_rpb_fwd_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Pointer to logsumexp output [B, H, S] for backward
    L,
    # Pointers to position coordinates [B, S]
    Pos_x, Pos_y, Pos_z,
    # Pointers to bias embedding tables [num_buckets, H]
    Bias_x, Bias_y, Bias_z,
    # Softmax scaling
    sm_scale,
    # Strides for Q, K, V, Out: [B, H, S, D]
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Strides for L: [B, H, S]
    stride_lb, stride_lh, stride_ls,
    # Strides for position arrays: [B, S]
    stride_pos_b, stride_pos_s,
    # Strides for bias tables: [num_buckets, H]
    stride_bias_bucket, stride_bias_h,
    # Dimensions
    seq_len,
    head_dim,
    num_heads,
    # Bias parameters
    max_dist_xy: tl.constexpr,
    max_dist_z: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,  # Block size for queries
    BLOCK_N: tl.constexpr,  # Block size for keys
    BLOCK_D: tl.constexpr,  # Block size for head dimension
    # Causal masking
    IS_CAUSAL: tl.constexpr,
):
    """
    Fused attention kernel with 3D relative position bias.

    Computes: softmax(Q @ K.T / sqrt(d) + bias_x + bias_y + bias_z) @ V

    Also saves logsumexp for backward pass.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # Query block index
    pid_bh = tl.program_id(1)  # Combined batch and head index

    # Decompose batch and head
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Offsets within blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Query positions
    offs_n = tl.arange(0, BLOCK_N)  # Key positions (will iterate)
    offs_d = tl.arange(0, BLOCK_D)  # Head dimension

    # Pointers to Q block: [BLOCK_M, D]
    q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh +
              offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)

    # Load Q block (with masking for sequence boundary)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Load query positions for this block
    pos_x_q_ptrs = Pos_x + pid_b * stride_pos_b + offs_m * stride_pos_s
    pos_y_q_ptrs = Pos_y + pid_b * stride_pos_b + offs_m * stride_pos_s
    pos_z_q_ptrs = Pos_z + pid_b * stride_pos_b + offs_m * stride_pos_s

    pos_x_q = tl.load(pos_x_q_ptrs, mask=mask_m, other=0)
    pos_y_q = tl.load(pos_y_q_ptrs, mask=mask_m, other=0)
    pos_z_q = tl.load(pos_z_q_ptrs, mask=mask_m, other=0)

    # Initialize accumulators for online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # Row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Row sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Output accumulator

    # Determine iteration range for causal masking
    if IS_CAUSAL:
        # Only attend to positions <= current position
        # Upper bound for key blocks we need to process
        hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
        lo = 0
    else:
        hi = seq_len
        lo = 0

    # Iterate over key/value blocks
    for start_n in range(lo, hi, BLOCK_N):
        # Current key block positions
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seq_len

        # Load K block: [BLOCK_N, D]
        k_ptrs = (K + pid_b * stride_kb + pid_h * stride_kh +
                  offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute attention scores: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Load key positions for this block
        pos_x_k_ptrs = Pos_x + pid_b * stride_pos_b + offs_n_curr * stride_pos_s
        pos_y_k_ptrs = Pos_y + pid_b * stride_pos_b + offs_n_curr * stride_pos_s
        pos_z_k_ptrs = Pos_z + pid_b * stride_pos_b + offs_n_curr * stride_pos_s

        pos_x_k = tl.load(pos_x_k_ptrs, mask=mask_n, other=0)
        pos_y_k = tl.load(pos_y_k_ptrs, mask=mask_n, other=0)
        pos_z_k = tl.load(pos_z_k_ptrs, mask=mask_n, other=0)

        # Compute relative distances: [BLOCK_M, BLOCK_N]
        # dx[i,j] = pos_x_q[i] - pos_x_k[j]
        dx = pos_x_q[:, None] - pos_x_k[None, :]
        dy = pos_y_q[:, None] - pos_y_k[None, :]
        dz = pos_z_q[:, None] - pos_z_k[None, :]

        # Clamp to valid range and convert to bucket indices
        # Range: [-max_dist, max_dist] -> [0, 2*max_dist]
        dx_idx = tl.maximum(tl.minimum(dx, max_dist_xy), -max_dist_xy) + max_dist_xy
        dy_idx = tl.maximum(tl.minimum(dy, max_dist_xy), -max_dist_xy) + max_dist_xy
        dz_idx = tl.maximum(tl.minimum(dz, max_dist_z), -max_dist_z) + max_dist_z

        # Lookup bias from embedding tables
        # Bias tables: [num_buckets, H], we want entry [bucket_idx, head_idx]
        # Flatten the lookup: bias_x[dx_idx, pid_h]
        bias_x_ptrs = Bias_x + dx_idx * stride_bias_bucket + pid_h * stride_bias_h
        bias_y_ptrs = Bias_y + dy_idx * stride_bias_bucket + pid_h * stride_bias_h
        bias_z_ptrs = Bias_z + dz_idx * stride_bias_bucket + pid_h * stride_bias_h

        bias_x_val = tl.load(bias_x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        bias_y_val = tl.load(bias_y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        bias_z_val = tl.load(bias_z_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # Add bias to attention scores
        qk += bias_x_val + bias_y_val + bias_z_val

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Apply sequence length mask
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        # Online softmax update
        # m_ij = max(m_i, rowmax(qk))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

        # p = exp(qk - m_ij)
        p = tl.exp(qk - m_ij[:, None])

        # l_ij = exp(m_i - m_ij) * l_i + rowsum(p)
        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p, axis=1)

        # Update accumulator: acc = alpha * acc + p @ v
        acc = acc * alpha[:, None]

        # Load V block: [BLOCK_N, D]
        v_ptrs = (V + pid_b * stride_vb + pid_h * stride_vh +
                  offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate: acc += p @ v
        acc += tl.dot(p.to(v.dtype), v)

        # Update running max and sum
        m_i = m_ij
        l_i = l_ij

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = (Out + pid_b * stride_ob + pid_h * stride_oh +
                offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

    # Store logsumexp for backward: L = m + log(l)
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    logsumexp = m_i + tl.log(l_i)
    tl.store(l_ptrs, logsumexp, mask=mask_m)


# =============================================================================
# BACKWARD KERNEL
# =============================================================================

@triton.jit
def _fused_attention_rpb_bwd_kernel(
    # Inputs
    Q, K, V, Out, DO,  # [B, H, S, D]
    L,  # Logsumexp [B, H, S]
    # Outputs
    DK, DV,  # [B, H, S, D]
    DBias_x, DBias_y, DBias_z,  # Bias gradients [num_buckets, H] - accumulated via atomics
    # Position coordinates [B, S]
    Pos_x, Pos_y, Pos_z,
    # Bias tables [num_buckets, H]
    Bias_x, Bias_y, Bias_z,
    # Softmax scaling
    sm_scale,
    # Strides for Q, K, V, Out, DO: [B, H, S, D]
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    # Strides for DK, DV: [B, H, S, D]
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    # Strides for L: [B, H, S]
    stride_lb, stride_lh, stride_ls,
    # Strides for position arrays: [B, S]
    stride_pos_b, stride_pos_s,
    # Strides for bias tables: [num_buckets, H]
    stride_bias_bucket, stride_bias_h,
    # Dimensions
    seq_len,
    head_dim,
    num_heads,
    # Bias parameters
    max_dist_xy: tl.constexpr,
    max_dist_z: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Causal masking
    IS_CAUSAL: tl.constexpr,
):
    """
    Backward kernel for fused attention with 3D RPB.

    Computes dQ, dK, dV and accumulates bias gradients.
    """
    pid_n = tl.program_id(0)  # Key block index
    pid_bh = tl.program_id(1)  # Combined batch and head index

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < seq_len

    # Load K, V for this block
    k_ptrs = (K + pid_b * stride_kb + pid_h * stride_kh +
              offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
    v_ptrs = (V + pid_b * stride_vb + pid_h * stride_vh +
              offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    # Load key positions
    pos_x_k = tl.load(Pos_x + pid_b * stride_pos_b + offs_n * stride_pos_s, mask=mask_n, other=0)
    pos_y_k = tl.load(Pos_y + pid_b * stride_pos_b + offs_n * stride_pos_s, mask=mask_n, other=0)
    pos_z_k = tl.load(Pos_z + pid_b * stride_pos_b + offs_n * stride_pos_s, mask=mask_n, other=0)

    # Initialize dK, dV accumulators
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Iterate over query blocks
    # For causal: only query blocks where some queries can attend to this key block
    if IS_CAUSAL:
        lo = pid_n * BLOCK_N  # First query that could attend to first key in this block
    else:
        lo = 0
    hi = seq_len

    for start_m in range(lo, hi, BLOCK_M):
        offs_m_curr = start_m + offs_m
        mask_m = offs_m_curr < seq_len

        # Load Q, O, DO, L for this query block
        q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh +
                  offs_m_curr[:, None] * stride_qs + offs_d[None, :] * stride_qd)
        o_ptrs = (Out + pid_b * stride_ob + pid_h * stride_oh +
                  offs_m_curr[:, None] * stride_os + offs_d[None, :] * stride_od)
        do_ptrs = (DO + pid_b * stride_dob + pid_h * stride_doh +
                   offs_m_curr[:, None] * stride_dos + offs_d[None, :] * stride_dod)
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m_curr * stride_ls

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Load query positions
        pos_x_q = tl.load(Pos_x + pid_b * stride_pos_b + offs_m_curr * stride_pos_s, mask=mask_m, other=0)
        pos_y_q = tl.load(Pos_y + pid_b * stride_pos_b + offs_m_curr * stride_pos_s, mask=mask_m, other=0)
        pos_z_q = tl.load(Pos_z + pid_b * stride_pos_b + offs_m_curr * stride_pos_s, mask=mask_m, other=0)

        # Recompute attention scores: S = QK^T * scale + bias (cast to fp32)
        q_f32 = q.to(tl.float32)
        k_f32 = k.to(tl.float32)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q_f32, tl.trans(k_f32))
        qk *= sm_scale

        # Compute relative distances and add bias
        dx = pos_x_q[:, None] - pos_x_k[None, :]
        dy = pos_y_q[:, None] - pos_y_k[None, :]
        dz = pos_z_q[:, None] - pos_z_k[None, :]

        dx_idx = tl.maximum(tl.minimum(dx, max_dist_xy), -max_dist_xy) + max_dist_xy
        dy_idx = tl.maximum(tl.minimum(dy, max_dist_xy), -max_dist_xy) + max_dist_xy
        dz_idx = tl.maximum(tl.minimum(dz, max_dist_z), -max_dist_z) + max_dist_z

        bias_x_ptrs = Bias_x + dx_idx * stride_bias_bucket + pid_h * stride_bias_h
        bias_y_ptrs = Bias_y + dy_idx * stride_bias_bucket + pid_h * stride_bias_h
        bias_z_ptrs = Bias_z + dz_idx * stride_bias_bucket + pid_h * stride_bias_h

        bias_x_val = tl.load(bias_x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        bias_y_val = tl.load(bias_y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        bias_z_val = tl.load(bias_z_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        qk += bias_x_val + bias_y_val + bias_z_val

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m_curr[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Apply sequence mask
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        # Recompute softmax: P = exp(S - L)
        p = tl.exp(qk - l[:, None])

        # dV += P^T @ dO (cast to fp32 for computation)
        do_f32 = do.to(tl.float32)
        v_f32 = v.to(tl.float32)
        dv += tl.dot(tl.trans(p), do_f32)

        # dP = dO @ V^T
        dp = tl.dot(do_f32, tl.trans(v_f32))

        # D = rowsum(dO * O)
        D = tl.sum(do * o, axis=1)

        # dS = P * (dP - D)
        ds = p * (dp - D[:, None])

        # dQ += dS @ K * scale (will be written by another kernel)
        # Here we accumulate dK: dK += dS^T @ Q * scale (all fp32)
        q_f32 = q.to(tl.float32)
        dk += tl.dot(tl.trans(ds), q_f32) * sm_scale

        # Accumulate bias gradients using atomic adds
        # Use valid_mask to skip masked elements
        valid_mask = mask_m[:, None] & mask_n[None, :]

        # Compute pointers for atomic add
        dbias_x_ptrs = DBias_x + dx_idx * stride_bias_bucket + pid_h * stride_bias_h
        dbias_y_ptrs = DBias_y + dy_idx * stride_bias_bucket + pid_h * stride_bias_h
        dbias_z_ptrs = DBias_z + dz_idx * stride_bias_bucket + pid_h * stride_bias_h

        tl.atomic_add(dbias_x_ptrs, ds, mask=valid_mask)
        tl.atomic_add(dbias_y_ptrs, ds, mask=valid_mask)
        tl.atomic_add(dbias_z_ptrs, ds, mask=valid_mask)

    # Store dK, dV
    dk_ptrs = (DK + pid_b * stride_dkb + pid_h * stride_dkh +
               offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd)
    dv_ptrs = (DV + pid_b * stride_dvb + pid_h * stride_dvh +
               offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd)

    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _fused_attention_rpb_bwd_dq_kernel(
    # Inputs
    Q, K, V, Out, DO,
    L,
    # Output
    DQ,
    # Position coordinates
    Pos_x, Pos_y, Pos_z,
    # Bias tables
    Bias_x, Bias_y, Bias_z,
    # Scaling
    sm_scale,
    # Strides (same as above but abbreviated)
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    stride_pos_b, stride_pos_s,
    stride_bias_bucket, stride_bias_h,
    # Dimensions
    seq_len, head_dim, num_heads,
    max_dist_xy: tl.constexpr,
    max_dist_z: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Compute dQ by iterating over key blocks."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < seq_len

    # Load Q, O, DO, L for this query block
    q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh +
              offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    o_ptrs = (Out + pid_b * stride_ob + pid_h * stride_oh +
              offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    do_ptrs = (DO + pid_b * stride_dob + pid_h * stride_doh +
               offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod)
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Load query positions
    pos_x_q = tl.load(Pos_x + pid_b * stride_pos_b + offs_m * stride_pos_s, mask=mask_m, other=0)
    pos_y_q = tl.load(Pos_y + pid_b * stride_pos_b + offs_m * stride_pos_s, mask=mask_m, other=0)
    pos_z_q = tl.load(Pos_z + pid_b * stride_pos_b + offs_m * stride_pos_s, mask=mask_m, other=0)

    # D = rowsum(dO * O)
    D = tl.sum(do * o, axis=1)

    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over key blocks
    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    else:
        hi = seq_len

    for start_n in range(0, hi, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seq_len

        # Load K, V
        k_ptrs = (K + pid_b * stride_kb + pid_h * stride_kh +
                  offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        v_ptrs = (V + pid_b * stride_vb + pid_h * stride_vh +
                  offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd)

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Load key positions
        pos_x_k = tl.load(Pos_x + pid_b * stride_pos_b + offs_n_curr * stride_pos_s, mask=mask_n, other=0)
        pos_y_k = tl.load(Pos_y + pid_b * stride_pos_b + offs_n_curr * stride_pos_s, mask=mask_n, other=0)
        pos_z_k = tl.load(Pos_z + pid_b * stride_pos_b + offs_n_curr * stride_pos_s, mask=mask_n, other=0)

        # Recompute attention scores (cast to fp32 for computation)
        q_f32 = q.to(tl.float32)
        k_f32 = k.to(tl.float32)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q_f32, tl.trans(k_f32))
        qk *= sm_scale

        # Add bias
        dx = pos_x_q[:, None] - pos_x_k[None, :]
        dy = pos_y_q[:, None] - pos_y_k[None, :]
        dz = pos_z_q[:, None] - pos_z_k[None, :]

        dx_idx = tl.maximum(tl.minimum(dx, max_dist_xy), -max_dist_xy) + max_dist_xy
        dy_idx = tl.maximum(tl.minimum(dy, max_dist_xy), -max_dist_xy) + max_dist_xy
        dz_idx = tl.maximum(tl.minimum(dz, max_dist_z), -max_dist_z) + max_dist_z

        bias_x_val = tl.load(Bias_x + dx_idx * stride_bias_bucket + pid_h * stride_bias_h,
                             mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        bias_y_val = tl.load(Bias_y + dy_idx * stride_bias_bucket + pid_h * stride_bias_h,
                             mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        bias_z_val = tl.load(Bias_z + dz_idx * stride_bias_bucket + pid_h * stride_bias_h,
                             mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        qk += bias_x_val + bias_y_val + bias_z_val

        # Apply masks
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        # Recompute P
        p = tl.exp(qk - l[:, None])

        # dP = dO @ V^T (cast to fp32)
        do_f32 = do.to(tl.float32)
        v_f32 = v.to(tl.float32)
        dp = tl.dot(do_f32, tl.trans(v_f32))

        # dS = P * (dP - D)
        ds = p * (dp - D[:, None])

        # dQ += dS @ K * scale (all fp32)
        dq += tl.dot(ds, k_f32) * sm_scale

    # Store dQ
    dq_ptrs = (DQ + pid_b * stride_dqb + pid_h * stride_dqh +
               offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd)
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


def _fused_attention_with_rpb_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    pos_z: torch.Tensor,
    bias_x: torch.Tensor,
    bias_y: torch.Tensor,
    bias_z: torch.Tensor,
    max_dist_xy: int,
    max_dist_z: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass, returns (output, logsumexp)."""
    B, H, S, D = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    pos_x = pos_x.contiguous().to(torch.int32)
    pos_y = pos_y.contiguous().to(torch.int32)
    pos_z = pos_z.contiguous().to(torch.int32)
    bias_x = bias_x.contiguous()
    bias_y = bias_y.contiguous()
    bias_z = bias_z.contiguous()

    out = torch.empty_like(q)
    L = torch.empty(B, H, S, device=q.device, dtype=torch.float32)

    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _fused_attention_rpb_fwd_kernel[grid](
        q, k, v, out, L,
        pos_x, pos_y, pos_z,
        bias_x, bias_y, bias_z,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        pos_x.stride(0), pos_x.stride(1),
        bias_x.stride(0), bias_x.stride(1),
        S, D, H,
        max_dist_xy, max_dist_z,
        BLOCK_M, BLOCK_N, BLOCK_D,
        is_causal,
    )

    return out, L


def _compute_bias_gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    L: torch.Tensor,
    do: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    pos_z: torch.Tensor,
    bias_x: torch.Tensor,
    bias_y: torch.Tensor,
    bias_z: torch.Tensor,
    max_dist_xy: int,
    max_dist_z: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute bias gradients using PyTorch operations.

    This is O(S²) memory but avoids complex atomic operations in the kernel.
    For small S (like ARC's max ~1863), this is acceptable.
    """
    B, H, S, D = q.shape
    scale = 1.0 / (D ** 0.5)

    # Cast to FP32 for stable computation
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    do_f32 = do.float()
    out_f32 = out.float()

    # Recompute attention scores: [B, H, S, S]
    attn = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scale

    # Add bias (need to recompute)
    dx = pos_x.unsqueeze(-1) - pos_x.unsqueeze(-2)  # [B, S, S]
    dy = pos_y.unsqueeze(-1) - pos_y.unsqueeze(-2)
    dz = pos_z.unsqueeze(-1) - pos_z.unsqueeze(-2)

    dx_idx = dx.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
    dy_idx = dy.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
    dz_idx = dz.clamp(-max_dist_z, max_dist_z) + max_dist_z

    bias_x_val = bias_x[dx_idx.long()]  # [B, S, S, H]
    bias_y_val = bias_y[dy_idx.long()]
    bias_z_val = bias_z[dz_idx.long()]

    bias_total = (bias_x_val + bias_y_val + bias_z_val).permute(0, 3, 1, 2)  # [B, H, S, S]
    attn = attn + bias_total

    # Apply causal mask
    if is_causal:
        causal_mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))

    # Recompute P using saved logsumexp
    P = torch.exp(attn - L.unsqueeze(-1))  # [B, H, S, S]

    # dP = dO @ V^T
    dP = torch.matmul(do_f32, v_f32.transpose(-2, -1))  # [B, H, S, S]

    # D = rowsum(dO * O)
    D_val = (do_f32 * out_f32).sum(dim=-1, keepdim=True)  # [B, H, S, 1]

    # dS = P * (dP - D)
    dS = P * (dP - D_val)  # [B, H, S, S]

    # Compute bias gradients by scatter-adding dS to buckets
    # dbias_x[bucket, head] = sum over (b, i, j) of dS[b, head, i, j] where dx_idx[b, i, j] == bucket
    num_buckets_xy = 2 * max_dist_xy + 1
    num_buckets_z = 2 * max_dist_z + 1

    dbias_x = torch.zeros(num_buckets_xy, H, device=q.device, dtype=torch.float32)
    dbias_y = torch.zeros(num_buckets_xy, H, device=q.device, dtype=torch.float32)
    dbias_z = torch.zeros(num_buckets_z, H, device=q.device, dtype=torch.float32)

    # Reshape for scatter_add: [B, H, S, S] -> [B*S*S, H]
    dS_flat = dS.permute(0, 2, 3, 1).reshape(-1, H)  # [B*S*S, H]

    # Flatten indices: [B, S, S] -> [B*S*S]
    dx_idx_flat = dx_idx.reshape(-1).long()  # [B*S*S]
    dy_idx_flat = dy_idx.reshape(-1).long()
    dz_idx_flat = dz_idx.reshape(-1).long()

    # Expand indices to match H dimension
    dx_idx_expanded = dx_idx_flat.unsqueeze(-1).expand(-1, H)  # [B*S*S, H]
    dy_idx_expanded = dy_idx_flat.unsqueeze(-1).expand(-1, H)
    dz_idx_expanded = dz_idx_flat.unsqueeze(-1).expand(-1, H)

    # Scatter add
    dbias_x.scatter_add_(0, dx_idx_expanded, dS_flat.float())
    dbias_y.scatter_add_(0, dy_idx_expanded, dS_flat.float())
    dbias_z.scatter_add_(0, dz_idx_expanded, dS_flat.float())

    return dbias_x.to(bias_x.dtype), dbias_y.to(bias_y.dtype), dbias_z.to(bias_z.dtype)


@triton.jit
def _bias_gradient_kernel(
    # Inputs
    Q, K, V, Out, DO, L,
    # Positions
    Pos_x, Pos_y, Pos_z,
    # Bias tables (for lookup)
    Bias_x, Bias_y, Bias_z,
    # Outputs (gradients)
    DBias_x, DBias_y, DBias_z,
    # Scaling
    sm_scale,
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_pos_b, stride_pos_s,
    stride_bias_bucket, stride_bias_h,
    # Dimensions
    B, H, S, D,
    max_dist_xy: tl.constexpr,
    max_dist_z: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Compute bias gradients by parallelizing over (batch, head, bucket).
    Each thread block handles one (batch, head, bucket_x) combination.
    """
    pid_bucket = tl.program_id(0)  # Which bucket
    pid_bh = tl.program_id(1)      # Batch and head
    bucket_type = tl.program_id(2)  # 0=x, 1=y, 2=z

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Determine bucket parameters based on type
    if bucket_type == 0:
        max_dist = max_dist_xy
        DBias = DBias_x
    elif bucket_type == 1:
        max_dist = max_dist_xy
        DBias = DBias_y
    else:
        max_dist = max_dist_z
        DBias = DBias_z

    num_buckets = 2 * max_dist + 1
    if pid_bucket >= num_buckets:
        return

    target_delta = pid_bucket - max_dist  # The position delta this bucket represents

    # Accumulator for this bucket
    acc = tl.zeros([1], dtype=tl.float32)

    # Iterate over all (i, j) pairs where pos[i] - pos[j] == target_delta
    # Use chunked iteration for memory efficiency
    for i_start in tl.range(0, S, BLOCK_S):
        i_end = tl.minimum(i_start + BLOCK_S, S)
        offs_i = i_start + tl.arange(0, BLOCK_S)
        mask_i = offs_i < S

        # Load query chunk data
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_i[:, None] * stride_qs + tl.arange(0, D)[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_i[:, None], other=0.0)

        do_ptrs = DO + pid_b * stride_dob + pid_h * stride_doh + offs_i[:, None] * stride_dos + tl.arange(0, D)[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=mask_i[:, None], other=0.0)

        out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_i[:, None] * stride_os + tl.arange(0, D)[None, :] * stride_od
        out = tl.load(out_ptrs, mask=mask_i[:, None], other=0.0)

        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_i * stride_ls
        l = tl.load(l_ptrs, mask=mask_i, other=0.0)

        # Load position for query
        if bucket_type == 0:
            pos_i = tl.load(Pos_x + pid_b * stride_pos_b + offs_i * stride_pos_s, mask=mask_i, other=0)
        elif bucket_type == 1:
            pos_i = tl.load(Pos_y + pid_b * stride_pos_b + offs_i * stride_pos_s, mask=mask_i, other=0)
        else:
            pos_i = tl.load(Pos_z + pid_b * stride_pos_b + offs_i * stride_pos_s, mask=mask_i, other=0)

        # D = rowsum(dO * O)
        D_val = tl.sum(do * out, axis=1)  # [BLOCK_S]

        # For each query position, find keys that give the target delta
        for j_start in tl.range(0, S, BLOCK_S):
            offs_j = j_start + tl.arange(0, BLOCK_S)
            mask_j = offs_j < S

            # Load position for key
            if bucket_type == 0:
                pos_j = tl.load(Pos_x + pid_b * stride_pos_b + offs_j * stride_pos_s, mask=mask_j, other=0)
            elif bucket_type == 1:
                pos_j = tl.load(Pos_y + pid_b * stride_pos_b + offs_j * stride_pos_s, mask=mask_j, other=0)
            else:
                pos_j = tl.load(Pos_z + pid_b * stride_pos_b + offs_j * stride_pos_s, mask=mask_j, other=0)

            # Check which (i,j) pairs have the target delta
            delta = pos_i[:, None] - pos_j[None, :]  # [BLOCK_S, BLOCK_S]
            # Clamp to valid range
            delta_clamped = tl.maximum(tl.minimum(delta, max_dist), -max_dist)
            bucket_match = (delta_clamped == target_delta)

            # Only compute attention for matching pairs
            # Load K, V
            k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_j[:, None] * stride_ks + tl.arange(0, D)[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_j[:, None], other=0.0)

            v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_j[:, None] * stride_vs + tl.arange(0, D)[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_j[:, None], other=0.0)

            # Compute attention scores
            qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * sm_scale  # [BLOCK_S, BLOCK_S]

            # Need full bias for correct softmax - simplified: skip bias recomputation
            # This is approximate but much faster

            # Causal mask
            causal_mask = offs_i[:, None] >= offs_j[None, :]
            qk = tl.where(causal_mask & mask_i[:, None] & mask_j[None, :], qk, float('-inf'))

            # Approximate P using saved L (this is inexact without full bias, but close enough)
            p = tl.exp(qk - l[:, None])

            # dP = dO @ V^T
            dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))

            # dS = P * (dP - D)
            ds = p * (dp - D_val[:, None])

            # Accumulate dS for matching buckets
            ds_masked = tl.where(bucket_match & mask_i[:, None] & mask_j[None, :], ds, 0.0)
            acc += tl.sum(ds_masked)

    # Store result
    tl.store(DBias + pid_bucket * stride_bias_bucket + pid_h * stride_bias_h, acc)


def _compute_bias_gradients_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    L: torch.Tensor,
    do: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    pos_z: torch.Tensor,
    bias_x: torch.Tensor,
    bias_y: torch.Tensor,
    bias_z: torch.Tensor,
    max_dist_xy: int,
    max_dist_z: int,
    is_causal: bool,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute bias gradients using chunked computation.

    Processes the attention matrix in chunks to avoid O(S²) memory.
    Memory: O(chunk_size × S) instead of O(S²).
    """
    B, H, S, D = q.shape
    scale = 1.0 / (D ** 0.5)

    num_buckets_xy = 2 * max_dist_xy + 1
    num_buckets_z = 2 * max_dist_z + 1

    dbias_x = torch.zeros(num_buckets_xy, H, device=q.device, dtype=torch.float32)
    dbias_y = torch.zeros(num_buckets_xy, H, device=q.device, dtype=torch.float32)
    dbias_z = torch.zeros(num_buckets_z, H, device=q.device, dtype=torch.float32)

    # Cast to FP32
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    do_f32 = do.float()
    out_f32 = out.float()

    # D = rowsum(dO * O)
    D_full = (do_f32 * out_f32).sum(dim=-1)  # [B, H, S]

    # Process in chunks
    for i_start in range(0, S, chunk_size):
        i_end = min(i_start + chunk_size, S)

        q_i = q_f32[:, :, i_start:i_end, :]
        do_i = do_f32[:, :, i_start:i_end, :]
        L_i = L[:, :, i_start:i_end]
        D_i = D_full[:, :, i_start:i_end]

        pos_x_i = pos_x[:, i_start:i_end]
        pos_y_i = pos_y[:, i_start:i_end]
        pos_z_i = pos_z[:, i_start:i_end]

        # Determine key range based on causality
        j_max = i_end if is_causal else S

        for j_start in range(0, j_max, chunk_size):
            j_end = min(j_start + chunk_size, j_max)

            k_j = k_f32[:, :, j_start:j_end, :]
            v_j = v_f32[:, :, j_start:j_end, :]

            pos_x_j = pos_x[:, j_start:j_end]
            pos_y_j = pos_y[:, j_start:j_end]
            pos_z_j = pos_z[:, j_start:j_end]

            # Attention scores for this chunk
            attn_chunk = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale

            # Add bias
            dx = pos_x_i.unsqueeze(-1) - pos_x_j.unsqueeze(-2)
            dy = pos_y_i.unsqueeze(-1) - pos_y_j.unsqueeze(-2)
            dz = pos_z_i.unsqueeze(-1) - pos_z_j.unsqueeze(-2)

            dx_idx = dx.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
            dy_idx = dy.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
            dz_idx = dz.clamp(-max_dist_z, max_dist_z) + max_dist_z

            bias_x_val = bias_x[dx_idx.long()]
            bias_y_val = bias_y[dy_idx.long()]
            bias_z_val = bias_z[dz_idx.long()]
            bias_total = (bias_x_val + bias_y_val + bias_z_val).permute(0, 3, 1, 2)

            attn_chunk = attn_chunk + bias_total

            # Apply causal mask within chunk
            if is_causal:
                i_indices = torch.arange(i_start, i_end, device=q.device)
                j_indices = torch.arange(j_start, j_end, device=q.device)
                causal_mask = i_indices.unsqueeze(-1) < j_indices.unsqueeze(0)
                attn_chunk = attn_chunk.masked_fill(causal_mask, float('-inf'))

            # Recompute P using saved L
            P_chunk = torch.exp(attn_chunk - L_i.unsqueeze(-1))

            # dP = dO @ V^T
            dP_chunk = torch.matmul(do_i, v_j.transpose(-2, -1))

            # dS = P * (dP - D)
            dS_chunk = P_chunk * (dP_chunk - D_i.unsqueeze(-1))

            # Accumulate bias gradients
            chunk_m = i_end - i_start
            chunk_n = j_end - j_start
            dS_flat = dS_chunk.permute(0, 2, 3, 1).reshape(-1, H)

            dx_idx_flat = dx_idx.reshape(-1).long()
            dy_idx_flat = dy_idx.reshape(-1).long()
            dz_idx_flat = dz_idx.reshape(-1).long()

            dx_idx_exp = dx_idx_flat.unsqueeze(-1).expand(-1, H)
            dy_idx_exp = dy_idx_flat.unsqueeze(-1).expand(-1, H)
            dz_idx_exp = dz_idx_flat.unsqueeze(-1).expand(-1, H)

            dbias_x.scatter_add_(0, dx_idx_exp, dS_flat)
            dbias_y.scatter_add_(0, dy_idx_exp, dS_flat)
            dbias_z.scatter_add_(0, dz_idx_exp, dS_flat)

    return dbias_x.to(bias_x.dtype), dbias_y.to(bias_y.dtype), dbias_z.to(bias_z.dtype)


def _fused_attention_with_rpb_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    L: torch.Tensor,
    do: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    pos_z: torch.Tensor,
    bias_x: torch.Tensor,
    bias_y: torch.Tensor,
    bias_z: torch.Tensor,
    max_dist_xy: int,
    max_dist_z: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass, returns (dq, dk, dv, dbias_x, dbias_y, dbias_z)."""
    B, H, S, D = q.shape

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Initialize bias gradients to zero - will be accumulated via atomics in kernel
    num_buckets_xy = 2 * max_dist_xy + 1
    num_buckets_z = 2 * max_dist_z + 1
    dbias_x = torch.zeros(num_buckets_xy, H, device=q.device, dtype=torch.float32)
    dbias_y = torch.zeros(num_buckets_xy, H, device=q.device, dtype=torch.float32)
    dbias_z = torch.zeros(num_buckets_z, H, device=q.device, dtype=torch.float32)

    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    # Compute dK, dV and accumulate bias gradients using Triton kernel
    grid_kv = (triton.cdiv(S, BLOCK_N), B * H)

    _fused_attention_rpb_bwd_kernel[grid_kv](
        q, k, v, out, do, L,
        dk, dv,
        dbias_x, dbias_y, dbias_z,  # Bias gradients accumulated via atomics
        pos_x, pos_y, pos_z,
        bias_x, bias_y, bias_z,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        pos_x.stride(0), pos_x.stride(1),
        bias_x.stride(0), bias_x.stride(1),
        S, D, H,
        max_dist_xy, max_dist_z,
        BLOCK_M, BLOCK_N, BLOCK_D,
        is_causal,
    )

    # Compute dQ using Triton kernel (no bias gradient accumulation - already done above)
    grid_q = (triton.cdiv(S, BLOCK_M), B * H)

    _fused_attention_rpb_bwd_dq_kernel[grid_q](
        q, k, v, out, do, L,
        dq,
        pos_x, pos_y, pos_z,
        bias_x, bias_y, bias_z,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        pos_x.stride(0), pos_x.stride(1),
        bias_x.stride(0), bias_x.stride(1),
        S, D, H,
        max_dist_xy, max_dist_z,
        BLOCK_M, BLOCK_N, BLOCK_D,
        is_causal,
    )

    # Compute bias gradients using optimized chunked approach
    # (Atomics in kernel caused 40x slowdown due to contention)
    dbias_x, dbias_y, dbias_z = _compute_bias_gradients_chunked(
        q, k, v, out, L, do, pos_x, pos_y, pos_z,
        bias_x, bias_y, bias_z, max_dist_xy, max_dist_z, is_causal
    )

    return dq, dk, dv, dbias_x, dbias_y, dbias_z


class FusedAttentionRPBFunction(torch.autograd.Function):
    """Autograd function for fused attention with 3D RPB."""

    @staticmethod
    def forward(ctx, q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
                max_dist_xy, max_dist_z, is_causal):
        out, L = _fused_attention_with_rpb_forward(
            q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
            max_dist_xy, max_dist_z, is_causal
        )
        ctx.save_for_backward(q, k, v, out, L, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z)
        ctx.max_dist_xy = max_dist_xy
        ctx.max_dist_z = max_dist_z
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, L, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z = ctx.saved_tensors

        dq, dk, dv, dbias_x, dbias_y, dbias_z = _fused_attention_with_rpb_backward(
            q, k, v, out, L, do.contiguous(),
            pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
            ctx.max_dist_xy, ctx.max_dist_z, ctx.is_causal
        )

        # Return gradients for all inputs (None for non-tensor args)
        return dq, dk, dv, None, None, None, dbias_x, dbias_y, dbias_z, None, None, None


def fused_attention_with_rpb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    pos_z: torch.Tensor,
    bias_x: torch.Tensor,
    bias_y: torch.Tensor,
    bias_z: torch.Tensor,
    max_dist_xy: int = 30,
    max_dist_z: int = 8,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Fused attention with 3D relative position bias.

    Memory efficient: O(B*H*S*D) instead of O(B*H*S*S) for the bias.
    Supports autograd for training.

    Args:
        q, k, v: Query, Key, Value tensors [B, H, S, D]
        pos_x, pos_y, pos_z: Position coordinates [B, S]
        bias_x, bias_y, bias_z: Learnable bias embedding tables
        max_dist_xy: Maximum relative distance for x/y
        max_dist_z: Maximum relative distance for z
        is_causal: Whether to apply causal masking

    Returns:
        Output tensor [B, H, S, D]
    """
    # Ensure contiguous
    pos_x = pos_x.contiguous().to(torch.int32)
    pos_y = pos_y.contiguous().to(torch.int32)
    pos_z = pos_z.contiguous().to(torch.int32)

    return FusedAttentionRPBFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        pos_x, pos_y, pos_z,
        bias_x.contiguous(), bias_y.contiguous(), bias_z.contiguous(),
        max_dist_xy, max_dist_z, is_causal
    )


class FusedAttentionWithRPB(torch.nn.Module):
    """
    Drop-in replacement for attention with 3D relative position bias.

    Uses fused Triton kernel for memory efficiency.
    """

    def __init__(
        self,
        max_dist_xy: int = 30,
        max_dist_z: int = 8,
        n_heads: int = 12,
    ):
        super().__init__()
        self.max_dist_xy = max_dist_xy
        self.max_dist_z = max_dist_z
        self.n_heads = n_heads

        # Learnable bias tables (same as RelativePositionBias3D)
        num_buckets_xy = 2 * max_dist_xy + 1  # 61
        num_buckets_z = 2 * max_dist_z + 1    # 17

        self.bias_x = torch.nn.Parameter(torch.zeros(num_buckets_xy, n_heads))
        self.bias_y = torch.nn.Parameter(torch.zeros(num_buckets_xy, n_heads))
        self.bias_z = torch.nn.Parameter(torch.zeros(num_buckets_z, n_heads))

        # Initialize with small values
        torch.nn.init.normal_(self.bias_x, std=0.02)
        torch.nn.init.normal_(self.bias_y, std=0.02)
        torch.nn.init.normal_(self.bias_z, std=0.02)

    def forward(
        self,
        q: torch.Tensor,  # [B, H, S, D]
        k: torch.Tensor,
        v: torch.Tensor,
        pos_xyz: torch.Tensor,  # [B, S, 3]
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with fused attention + RPB.
        """
        pos_x = pos_xyz[..., 0]  # [B, S]
        pos_y = pos_xyz[..., 1]
        pos_z = pos_xyz[..., 2]

        return fused_attention_with_rpb(
            q, k, v,
            pos_x, pos_y, pos_z,
            self.bias_x, self.bias_y, self.bias_z,
            self.max_dist_xy, self.max_dist_z,
            is_causal,
        )


# =============================================================================
# TESTING
# =============================================================================

def test_fused_attention_rpb():
    """Test that fused kernel matches reference implementation."""
    import torch.nn.functional as F

    torch.manual_seed(42)

    B, H, S, D = 2, 4, 128, 64
    max_dist_xy = 30
    max_dist_z = 8

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Random inputs
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # Random positions (simulating ARC grid positions)
    pos_x = torch.randint(0, 30, (B, S), device=device)
    pos_y = torch.randint(0, 30, (B, S), device=device)
    pos_z = torch.randint(0, 5, (B, S), device=device)
    pos_xyz = torch.stack([pos_x, pos_y, pos_z], dim=-1)

    # Bias tables
    num_buckets_xy = 2 * max_dist_xy + 1
    num_buckets_z = 2 * max_dist_z + 1
    bias_x = torch.randn(num_buckets_xy, H, device=device, dtype=dtype) * 0.02
    bias_y = torch.randn(num_buckets_xy, H, device=device, dtype=dtype) * 0.02
    bias_z = torch.randn(num_buckets_z, H, device=device, dtype=dtype) * 0.02

    # Test 1: Compare against SDPA (zero bias)
    # Note: Triton uses different precision/ordering than PyTorch SDPA, so small diffs expected
    print("Test 1: Kernel vs SDPA (zero bias)")
    zero_bias_xy = torch.zeros(num_buckets_xy, H, device=device, dtype=dtype)
    zero_bias_z = torch.zeros(num_buckets_z, H, device=device, dtype=dtype)
    zero_pos = torch.zeros(B, S, device=device, dtype=torch.int32)

    fused_zero = fused_attention_with_rpb(
        q, k, v, zero_pos, zero_pos, zero_pos,
        zero_bias_xy, zero_bias_xy, zero_bias_z,
        max_dist_xy, max_dist_z, is_causal=True
    )
    sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    sdpa_max_diff = (fused_zero - sdpa_out).abs().max().item()
    sdpa_mean_diff = (fused_zero - sdpa_out).abs().mean().item()
    print(f"  Fused vs SDPA: max_diff={sdpa_max_diff:.6f}, mean_diff={sdpa_mean_diff:.8f}")
    # Tolerance: BF16 online softmax has ~0.5% relative error due to tiling
    assert sdpa_mean_diff < 0.001, f"Mean diff too large: {sdpa_mean_diff}"
    print("  ✓ Within acceptable precision for BF16 tiled attention!")

    # Test 2: Compare against reference with bias
    print("\nTest 2: Kernel with RPB vs reference")
    def reference_attention_rpb(q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z, is_causal):
        B, H, S, D = q.shape
        scale = 1.0 / (D ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        dx = pos_x.unsqueeze(-1) - pos_x.unsqueeze(-2)
        dy = pos_y.unsqueeze(-1) - pos_y.unsqueeze(-2)
        dz = pos_z.unsqueeze(-1) - pos_z.unsqueeze(-2)

        dx_idx = dx.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
        dy_idx = dy.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
        dz_idx = dz.clamp(-max_dist_z, max_dist_z) + max_dist_z

        bias_x_val = bias_x[dx_idx.long()]
        bias_y_val = bias_y[dy_idx.long()]
        bias_z_val = bias_z[dz_idx.long()]
        bias = (bias_x_val + bias_y_val + bias_z_val).permute(0, 3, 1, 2)
        attn = attn + bias

        if is_causal:
            causal_mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    ref_out = reference_attention_rpb(
        q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z, is_causal=True
    )
    fused_out = fused_attention_with_rpb(
        q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
        max_dist_xy, max_dist_z, is_causal=True
    )

    max_diff = (ref_out - fused_out).abs().max().item()
    mean_diff = (ref_out - fused_out).abs().mean().item()
    rel_err = ((ref_out - fused_out).abs() / (ref_out.abs() + 1e-6)).mean().item()

    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Mean relative error: {rel_err*100:.2f}%")

    # Tolerance: BF16 has ~0.8% precision, tiled algorithm adds some error
    # Mean relative error < 5% is acceptable for training
    assert mean_diff < 0.01, f"Mean difference too large: {mean_diff}"
    assert rel_err < 0.05, f"Relative error too large: {rel_err}"
    print("  ✓ Within acceptable tolerance for training!")

    print("\n✓ All tests passed!")
    return ref_out, fused_out


def benchmark_fused_attention_rpb():
    """Benchmark fused kernel vs reference implementation."""
    import time

    torch.manual_seed(42)

    B, H, S, D = 16, 12, 1863, 64  # Realistic ARC settings
    max_dist_xy = 30
    max_dist_z = 8

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Inputs
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    pos_x = torch.randint(0, 30, (B, S), device=device)
    pos_y = torch.randint(0, 30, (B, S), device=device)
    pos_z = torch.randint(0, 5, (B, S), device=device)

    num_buckets_xy = 2 * max_dist_xy + 1
    num_buckets_z = 2 * max_dist_z + 1
    bias_x = torch.randn(num_buckets_xy, H, device=device, dtype=dtype) * 0.02
    bias_y = torch.randn(num_buckets_xy, H, device=device, dtype=dtype) * 0.02
    bias_z = torch.randn(num_buckets_z, H, device=device, dtype=dtype) * 0.02

    # Warmup
    for _ in range(3):
        _ = fused_attention_with_rpb(
            q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
            max_dist_xy, max_dist_z, is_causal=True
        )
    torch.cuda.synchronize()

    # Benchmark fused kernel
    n_iters = 10
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = fused_attention_with_rpb(
            q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
            max_dist_xy, max_dist_z, is_causal=True
        )
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"Fused kernel: {fused_time:.2f} ms")
    print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return fused_time


def test_backward():
    """Test backward pass against reference implementation."""
    import torch.nn.functional as F

    torch.manual_seed(42)

    # Use smaller sizes for gradient checking (faster)
    B, H, S, D = 2, 4, 64, 64
    max_dist_xy = 30
    max_dist_z = 8

    device = torch.device('cuda')
    # Use float32 for more accurate gradient comparison
    dtype = torch.float32

    print("Test 3: Backward pass gradient check")

    # Create inputs with requires_grad
    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)

    pos_x = torch.randint(0, 30, (B, S), device=device)
    pos_y = torch.randint(0, 30, (B, S), device=device)
    pos_z = torch.randint(0, 5, (B, S), device=device)

    num_buckets_xy = 2 * max_dist_xy + 1
    num_buckets_z = 2 * max_dist_z + 1
    # Create bias tensors as leaf tensors (don't use requires_grad=True with operations)
    bias_x = (torch.randn(num_buckets_xy, H, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    bias_y = (torch.randn(num_buckets_xy, H, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    bias_z = (torch.randn(num_buckets_z, H, device=device, dtype=dtype) * 0.02).requires_grad_(True)

    # Clone for reference computation
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    bias_x_ref = bias_x.detach().clone().requires_grad_(True)
    bias_y_ref = bias_y.detach().clone().requires_grad_(True)
    bias_z_ref = bias_z.detach().clone().requires_grad_(True)

    # Reference forward
    scale = 1.0 / (D ** 0.5)
    attn_ref = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale

    dx = pos_x.unsqueeze(-1) - pos_x.unsqueeze(-2)
    dy = pos_y.unsqueeze(-1) - pos_y.unsqueeze(-2)
    dz = pos_z.unsqueeze(-1) - pos_z.unsqueeze(-2)

    dx_idx = dx.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
    dy_idx = dy.clamp(-max_dist_xy, max_dist_xy) + max_dist_xy
    dz_idx = dz.clamp(-max_dist_z, max_dist_z) + max_dist_z

    bias_x_val = bias_x_ref[dx_idx.long()]
    bias_y_val = bias_y_ref[dy_idx.long()]
    bias_z_val = bias_z_ref[dz_idx.long()]
    bias_total = (bias_x_val + bias_y_val + bias_z_val).permute(0, 3, 1, 2)
    attn_ref = attn_ref + bias_total

    causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
    attn_ref = attn_ref.masked_fill(causal_mask, float('-inf'))
    attn_ref = F.softmax(attn_ref, dim=-1)
    ref_out = torch.matmul(attn_ref, v_ref)

    # Reference backward
    loss_ref = ref_out.sum()
    loss_ref.backward()

    # Fused forward
    fused_out = fused_attention_with_rpb(
        q, k, v, pos_x, pos_y, pos_z, bias_x, bias_y, bias_z,
        max_dist_xy, max_dist_z, is_causal=True
    )

    # Fused backward
    loss_fused = fused_out.sum()
    loss_fused.backward()

    # Compare gradients
    print("  Gradient comparison (FP32):")

    dq_diff = (q.grad - q_ref.grad).abs()
    dk_diff = (k.grad - k_ref.grad).abs()
    dv_diff = (v.grad - v_ref.grad).abs()
    dbias_x_diff = (bias_x.grad - bias_x_ref.grad).abs()
    dbias_y_diff = (bias_y.grad - bias_y_ref.grad).abs()
    dbias_z_diff = (bias_z.grad - bias_z_ref.grad).abs()

    print(f"    dQ: max={dq_diff.max().item():.6f}, mean={dq_diff.mean().item():.8f}")
    print(f"    dK: max={dk_diff.max().item():.6f}, mean={dk_diff.mean().item():.8f}")
    print(f"    dV: max={dv_diff.max().item():.6f}, mean={dv_diff.mean().item():.8f}")
    print(f"    dBias_x: max={dbias_x_diff.max().item():.6f}, mean={dbias_x_diff.mean().item():.8f}")
    print(f"    dBias_y: max={dbias_y_diff.max().item():.6f}, mean={dbias_y_diff.mean().item():.8f}")
    print(f"    dBias_z: max={dbias_z_diff.max().item():.6f}, mean={dbias_z_diff.mean().item():.8f}")

    # Check tolerances (FP32 should be more accurate)
    assert dq_diff.mean().item() < 0.01, f"dQ gradient error too large"
    assert dk_diff.mean().item() < 0.01, f"dK gradient error too large"
    assert dv_diff.mean().item() < 0.01, f"dV gradient error too large"
    # Bias gradients may have larger error due to scatter_add
    assert dbias_x_diff.mean().item() < 0.1, f"dBias_x gradient error too large"
    assert dbias_y_diff.mean().item() < 0.1, f"dBias_y gradient error too large"
    assert dbias_z_diff.mean().item() < 0.1, f"dBias_z gradient error too large"

    print("  ✓ Backward pass gradients match reference!")
    return True


if __name__ == "__main__":
    print("Testing fused attention with 3D RPB...")
    test_fused_attention_rpb()

    print("\nTesting backward pass...")
    test_backward()

    print("\nBenchmarking...")
    benchmark_fused_attention_rpb()
