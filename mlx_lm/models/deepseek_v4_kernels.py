"""Fused Metal kernels for DeepSeek V4 decode acceleration.

Eliminates ~9,000 Metal kernel dispatches per token by fusing
Hyper-Connection (HC) computations into single GPU dispatches.

Decode-only (B=1, S=1). Prefill uses the standard Python path.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Kernel 1A: Fused HC Pre-Scores
#
# Fuses: RMS norm + matmul(x, hc_fn.T) + sigmoid + softmax + Sinkhorn
# Replaces ~99 Metal dispatches with 1.
#
# Inputs:
#   x_flat [M*D] float32 -- flattened HC state (e.g. 4*4096 = 16384)
#   hc_fn  [mix_hc, M*D] float16 -- weight matrix (e.g. 24 x 16384)
#   hc_scale [3] float32
#   hc_base  [mix_hc] float32
#   dims     [4] uint32 -- [M*D, mix_hc, hc_mult, n_sinkhorn_iters]
#   eps_vals [2] float32 -- [hc_eps, norm_eps]
#
# Outputs:
#   pre  [hc_mult] float32
#   post [hc_mult] float32
#   comb [hc_mult * hc_mult] float32
# ---------------------------------------------------------------------------

_HC_PRE_SCORES_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;

    uint MD = dims[0];       // M * D (e.g. 16384)
    uint MIX_HC = dims[1];   // (2 + hc) * hc (e.g. 24)
    uint HC = dims[2];       // hc_mult (e.g. 4)
    uint N_ITERS = dims[3];  // sinkhorn iterations
    float hc_eps = eps_vals[0];
    float norm_eps = eps_vals[1];

    // --- Phase 1: RMS norm ---
    // Each of 256 threads accumulates sum-of-squares for MD/256 elements
    float local_ss = 0.0f;
    uint chunk = MD / 256;
    uint start = tid * chunk;
    for (uint i = start; i < start + chunk; i++) {
        float v = x_flat[i];
        local_ss += v * v;
    }
    // SIMD reduction within each 32-wide group
    float simd_ss = simd_sum(local_ss);

    // Cross-SIMD reduction via threadgroup shared memory
    threadgroup float shared_ss[8];
    if (simd_lane == 0) shared_ss[simd_group] = simd_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_ss = 0.0f;
    if (simd_group == 0) {
        float v = (simd_lane < 8) ? shared_ss[simd_lane] : 0.0f;
        total_ss = simd_sum(v);
    }
    threadgroup float rsqrt_shared[1];
    if (tid == 0) rsqrt_shared[0] = rsqrt(total_ss / float(MD) + norm_eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsqrt_val = rsqrt_shared[0];

    // --- Phase 2: 24 dot products (x_flat @ hc_fn.T) * rsqrt ---
    threadgroup float mixes_shared[32];  // max mix_hc = 32
    for (uint o = 0; o < MIX_HC; o++) {
        float local_dp = 0.0f;
        for (uint i = start; i < start + chunk; i++) {
            local_dp += x_flat[i] * float(hc_fn[o * MD + i]);
        }
        float simd_dp = simd_sum(local_dp);
        threadgroup float partial_dp[8];
        if (simd_lane == 0) partial_dp[simd_group] = simd_dp;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float v = (simd_lane < 8) ? partial_dp[simd_lane] : 0.0f;
            v = simd_sum(v);
            if (simd_lane == 0) mixes_shared[o] = v * rsqrt_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 3: Sigmoid, softmax, Sinkhorn (thread 0 only) ---
    if (tid == 0) {
        // Pre: sigmoid(mix * scale + base) + eps
        for (uint j = 0; j < HC; j++) {
            float v = mixes_shared[j] * hc_scale[0] + hc_base[j];
            pre[j] = 1.0f / (1.0f + exp(-v)) + hc_eps;
        }

        // Post: 2 * sigmoid(mix * scale + base)
        for (uint j = 0; j < HC; j++) {
            float v = mixes_shared[HC + j] * hc_scale[1] + hc_base[HC + j];
            post[j] = 2.0f / (1.0f + exp(-v));
        }

        // Comb: softmax(reshape, axis=-1) + eps, then Sinkhorn
        float c[16];  // max hc_mult^2 = 4*4 = 16
        // Softmax per row
        for (uint i = 0; i < HC; i++) {
            float row_max = -1e9f;
            for (uint j = 0; j < HC; j++) {
                uint idx = 2 * HC + i * HC + j;
                c[i * HC + j] = mixes_shared[idx] * hc_scale[2] + hc_base[idx];
                row_max = max(row_max, c[i * HC + j]);
            }
            float row_sum = 0.0f;
            for (uint j = 0; j < HC; j++) {
                c[i * HC + j] = exp(c[i * HC + j] - row_max);
                row_sum += c[i * HC + j];
            }
            for (uint j = 0; j < HC; j++) {
                c[i * HC + j] = c[i * HC + j] / row_sum + hc_eps;
            }
        }

        // Sinkhorn iterations
        for (uint iter = 0; iter < N_ITERS; iter++) {
            // Normalize columns (axis -2)
            for (uint j = 0; j < HC; j++) {
                float col_sum = 0.0f;
                for (uint i = 0; i < HC; i++) col_sum += c[i * HC + j];
                for (uint i = 0; i < HC; i++) c[i * HC + j] /= col_sum;
            }
            // Normalize rows (axis -1)
            for (uint i = 0; i < HC; i++) {
                float row_sum = 0.0f;
                for (uint j = 0; j < HC; j++) row_sum += c[i * HC + j];
                for (uint j = 0; j < HC; j++) c[i * HC + j] /= row_sum;
            }
        }

        // Write comb output
        for (uint k = 0; k < HC * HC; k++) comb[k] = c[k];
    }
"""

_hc_pre_scores_kernel = None


def _get_hc_pre_scores_kernel():
    global _hc_pre_scores_kernel
    if _hc_pre_scores_kernel is None:
        _hc_pre_scores_kernel = mx.fast.metal_kernel(
            name="hc_pre_scores",
            input_names=["x_flat", "hc_fn", "hc_scale", "hc_base",
                         "dims", "eps_vals"],
            output_names=["pre", "post", "comb"],
            source=_HC_PRE_SCORES_SOURCE,
        )
    return _hc_pre_scores_kernel


# ---------------------------------------------------------------------------
# Kernel 1B: Fused HC Pre Weighted Sum
#
# y[d] = sum_m(pre[m] * x[m, d])  for d in [0, D)
#
# Inputs:
#   x    [M * D] float16/32 -- HC state (M copies of hidden dim)
#   pre  [M] float32 -- weights from kernel 1A
#   dims [2] uint32 -- [M, D]
#
# Output:
#   y [D] float16/32
# ---------------------------------------------------------------------------

_HC_PRE_WSUM_SOURCE = """
    uint d = thread_position_in_grid.x;
    uint M = dims[0];
    uint D = dims[1];
    if (d < D) {
        float sum = 0.0f;
        for (uint m = 0; m < M; m++) {
            sum += pre[m] * float(x[m * D + d]);
        }
        y[d] = T(sum);
    }
"""

_hc_pre_wsum_kernel = None


def _get_hc_pre_wsum_kernel():
    global _hc_pre_wsum_kernel
    if _hc_pre_wsum_kernel is None:
        _hc_pre_wsum_kernel = mx.fast.metal_kernel(
            name="hc_pre_wsum",
            input_names=["x", "pre", "dims"],
            output_names=["y"],
            source=_HC_PRE_WSUM_SOURCE,
        )
    return _hc_pre_wsum_kernel


# ---------------------------------------------------------------------------
# Kernel 2: Fused HC Post
#
# y[i, d] = post[i] * x[d] + sum_j(comb[i, j] * residual[j, d])
#
# Inputs:
#   x        [D] float16/32 -- attention/FFN output
#   residual [M * D] float16/32 -- HC residual state
#   post     [M] float32
#   comb     [M * M] float32
#   dims     [2] uint32 -- [M, D]
#
# Output:
#   y [M * D] float16/32
# ---------------------------------------------------------------------------

_HC_POST_SOURCE = """
    uint d = thread_position_in_grid.x;
    uint i = thread_position_in_grid.y;
    uint M = dims[0];
    uint D = dims[1];
    if (d < D && i < M) {
        float val = post[i] * float(x[d]);
        for (uint j = 0; j < M; j++) {
            val += comb[j * M + i] * float(residual[j * D + d]);
        }
        y[i * D + d] = T(val);
    }
"""

_hc_post_kernel = None


def _get_hc_post_kernel():
    global _hc_post_kernel
    if _hc_post_kernel is None:
        _hc_post_kernel = mx.fast.metal_kernel(
            name="hc_post",
            input_names=["x", "residual", "post", "comb", "dims"],
            output_names=["y"],
            source=_HC_POST_SOURCE,
        )
    return _hc_post_kernel


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def fused_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, n_iters, hc_eps,
                 norm_eps):
    """Fused HC pre-computation for decode (B=1, S=1).

    Returns (y, post, comb) matching _hc_pre output shapes.
    """
    B, S, M, D = x.shape
    MD = M * D
    assert B == 1 and S == 1, "Fused kernels only support decode (B=1, S=1)"
    assert MD % 256 == 0, f"M*D must be divisible by 256, got {MD}"
    assert hc_mult <= 4, f"Fused kernel supports hc_mult <= 4, got {hc_mult}"
    x_flat = x.reshape(MD).astype(mx.float32)
    mix_hc = hc_fn.shape[0]

    dims = mx.array([MD, mix_hc, hc_mult, n_iters], dtype=mx.uint32)
    eps = mx.array([hc_eps, norm_eps], dtype=mx.float32)

    kernel = _get_hc_pre_scores_kernel()
    pre, post, comb = kernel(
        inputs=[x_flat, hc_fn, hc_scale, hc_base, dims, eps],
        output_shapes=[(hc_mult,), (hc_mult,), (hc_mult * hc_mult,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
    )

    # Weighted sum: y = sum(pre * x, axis=hc)
    wsum_dims = mx.array([M, D], dtype=mx.uint32)
    x_md = x.reshape(M, D)
    wsum_kernel = _get_hc_pre_wsum_kernel()
    (y,) = wsum_kernel(
        inputs=[x_md, pre, wsum_dims],
        output_shapes=[(D,)],
        output_dtypes=[x.dtype],
        grid=(D, 1, 1),
        threadgroup=(min(256, D), 1, 1),
        template=[("T", x.dtype)],
    )

    return (y.reshape(1, 1, D),
            post.reshape(1, 1, M),
            comb.reshape(1, 1, M, M))


def fused_hc_post(x, residual, post, comb, hc_mult):
    """Fused HC post-computation for decode (B=1, S=1).

    Returns y [1, 1, M, D].
    """
    D = x.shape[-1]
    M = hc_mult
    dims = mx.array([M, D], dtype=mx.uint32)

    kernel = _get_hc_post_kernel()
    (y,) = kernel(
        inputs=[x.reshape(D), residual.reshape(M * D),
                post.reshape(M), comb.reshape(M * M), dims],
        output_shapes=[(M * D,)],
        output_dtypes=[x.dtype],
        grid=(D, M, 1),
        threadgroup=(min(256, D), 1, 1),
        template=[("T", x.dtype)],
    )
    return y.reshape(1, 1, M, D)
