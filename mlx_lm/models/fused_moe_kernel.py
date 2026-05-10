"""Fused gate+up+SwiGLU Metal kernel for MoE decode. 4-bit and 8-bit quantized."""

import mlx.core as mx


def _make_fused_source(bits):
    """Generate the fused gate+up+SwiGLU kernel source for the given bit width."""
    if bits == 4:
        k_bytes_expr = "K / 2"
        thread_byte_off_expr = "simd_lid * VPT / 2"  # 16 nibbles = 8 bytes
        ptr_advance = "BS / 2"
        # Pre-division trick: x[i+1]/16, x[i+2]/256, x[i+3]/4096
        # so (x/D) * (raw & mask) = x * ((raw >> shift) & 0xF)
        x_load = """
        for (uint i = 0; i < 16; i += 4) {
            float x0 = float(x[x_off + k + i]);
            float x1 = float(x[x_off + k + i + 1]);
            float x2 = float(x[x_off + k + i + 2]);
            float x3 = float(x[x_off + k + i + 3]);
            xsum += x0 + x1 + x2 + x3;
            xt[i]     = x0;
            xt[i + 1] = x1 / 16.0f;
            xt[i + 2] = x2 / 256.0f;
            xt[i + 3] = x3 / 4096.0f;
        }"""
        gate_qdot = """
            const device uint16_t* gwl = (const device uint16_t*)(gw_base + row_off_bytes);
            float g_s = float(gate_s[s_base + row_off_groups]);
            float g_b = float(gate_b[s_base + row_off_groups]);
            float ga = 0;
            for (uint i = 0; i < 4; i++) {
                ga += xt[4*i]   * float(gwl[i] & 0x000fu)
                    + xt[4*i+1] * float(gwl[i] & 0x00f0u)
                    + xt[4*i+2] * float(gwl[i] & 0x0f00u)
                    + xt[4*i+3] * float(gwl[i] & 0xf000u);
            }
            gr[row] += g_s * ga + xsum * g_b;"""
        up_qdot = """
            const device uint16_t* uwl = (const device uint16_t*)(uw_base + row_off_bytes);
            float u_s = float(up_s[s_base + row_off_groups]);
            float u_b = float(up_b[s_base + row_off_groups]);
            float ua = 0;
            for (uint i = 0; i < 4; i++) {
                ua += xt[4*i]   * float(uwl[i] & 0x000fu)
                    + xt[4*i+1] * float(uwl[i] & 0x00f0u)
                    + xt[4*i+2] * float(uwl[i] & 0x0f00u)
                    + xt[4*i+3] * float(uwl[i] & 0xf000u);
            }
            ur[row] += u_s * ua + xsum * u_b;"""
    elif bits == 8:
        k_bytes_expr = "K"
        thread_byte_off_expr = "simd_lid * VPT"  # 16 bytes
        ptr_advance = "BS"
        # Pre-division trick for 8-bit: each uint16 holds 2 values
        # low byte: raw & 0xFF, high byte: (raw >> 8) & 0xFF
        # Pre-divide x[i+1] by 256 so (x/256) * (raw & 0xFF00) = x * ((raw>>8)&0xFF)
        x_load = """
        for (uint i = 0; i < 16; i += 2) {
            float x0 = float(x[x_off + k + i]);
            float x1 = float(x[x_off + k + i + 1]);
            xsum += x0 + x1;
            xt[i]     = x0;
            xt[i + 1] = x1 / 256.0f;
        }"""
        gate_qdot = """
            const device uint16_t* gwl = (const device uint16_t*)(gw_base + row_off_bytes);
            float g_s = float(gate_s[s_base + row_off_groups]);
            float g_b = float(gate_b[s_base + row_off_groups]);
            float ga = 0;
            for (uint i = 0; i < 8; i++) {
                ga += xt[2*i]   * float(gwl[i] & 0x00ffu)
                    + xt[2*i+1] * float(gwl[i] & 0xff00u);
            }
            gr[row] += g_s * ga + xsum * g_b;"""
        up_qdot = """
            const device uint16_t* uwl = (const device uint16_t*)(uw_base + row_off_bytes);
            float u_s = float(up_s[s_base + row_off_groups]);
            float u_b = float(up_b[s_base + row_off_groups]);
            float ua = 0;
            for (uint i = 0; i < 8; i++) {
                ua += xt[2*i]   * float(uwl[i] & 0x00ffu)
                    + xt[2*i+1] * float(uwl[i] & 0xff00u);
            }
            ur[row] += u_s * ua + xsum * u_b;"""
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return f"""
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint expert_id = threadgroup_position_in_grid.y;
    uint tile_id = threadgroup_position_in_grid.x;

    uint K = dims[0];
    uint N = dims[1];
    uint GS = dims[3];

    const uint VPT = 16;
    const uint RPS = 4;
    const uint BS = VPT * 32;  // 512
    const uint SST = GS / VPT;
    uint K_bytes = {k_bytes_expr};
    uint KG = K / GS;

    uint eidx = expert_indices[expert_id];
    uint out_row = tile_id * 8 + simd_gid * RPS;
    if (out_row >= N) return;

    uint expert_byte_off = eidx * N * K_bytes;
    uint row_byte_off = out_row * K_bytes;
    uint thread_byte_off = {thread_byte_off_expr};

    const device uint8_t* gw_base = ((const device uint8_t*)gate_w) + expert_byte_off + row_byte_off + thread_byte_off;
    const device uint8_t* uw_base = ((const device uint8_t*)up_w) + expert_byte_off + row_byte_off + thread_byte_off;

    uint expert_s_off = eidx * N * KG;
    uint row_s_off = out_row * KG;
    uint thread_s_off = simd_lid / SST;
    uint s_base = expert_s_off + row_s_off + thread_s_off;

    uint x_off = simd_lid * VPT;

    float gr[4] = {{0, 0, 0, 0}};
    float ur[4] = {{0, 0, 0, 0}};

    for (uint k = 0; k < K; k += BS) {{
        float xt[16];
        float xsum = 0;
{x_load}

        for (uint row = 0; row < RPS; row++) {{
            uint row_off_bytes = row * K_bytes;
            uint row_off_groups = row * KG;

            // Gate qdot
{gate_qdot}

            // Up qdot
{up_qdot}
        }}

        gw_base += {ptr_advance};
        uw_base += {ptr_advance};
        s_base += BS / GS;
    }}

    for (uint row = 0; row < RPS; row++) {{
        float g = simd_sum(gr[row]);
        float u = simd_sum(ur[row]);
        if (simd_lid == 0 && out_row + row < N) {{
            float sg = g / (1.0f + exp(-g));
            out[expert_id * N + out_row + row] = sg * u;
        }}
    }}
"""


def _make_down_source(bits):
    """Generate the fused down projection kernel source for the given bit width."""
    if bits == 4:
        k_bytes_expr = "K / 2"
        thread_byte_off_expr = "simd_lid * VPT / 2"
        ptr_advance = "BS / 2"
        x_load = """
        for (uint i = 0; i < 16; i += 4) {
            float x0 = float(h[x_off + k + i]);
            float x1 = float(h[x_off + k + i + 1]);
            float x2 = float(h[x_off + k + i + 2]);
            float x3 = float(h[x_off + k + i + 3]);
            xsum += x0 + x1 + x2 + x3;
            xt[i]     = x0;
            xt[i + 1] = x1 / 16.0f;
            xt[i + 2] = x2 / 256.0f;
            xt[i + 3] = x3 / 4096.0f;
        }"""
        qdot = """
            const device uint16_t* dwl = (const device uint16_t*)(dw_base + row * K_bytes);
            float s = float(down_s[s_base + row * KG]);
            float b = float(down_b[s_base + row * KG]);
            float a = 0;
            for (uint i = 0; i < 4; i++) {
                a += xt[4*i]   * float(dwl[i] & 0x000fu)
                   + xt[4*i+1] * float(dwl[i] & 0x00f0u)
                   + xt[4*i+2] * float(dwl[i] & 0x0f00u)
                   + xt[4*i+3] * float(dwl[i] & 0xf000u);
            }
            dr[row] += s * a + xsum * b;"""
    elif bits == 8:
        k_bytes_expr = "K"
        thread_byte_off_expr = "simd_lid * VPT"
        ptr_advance = "BS"
        x_load = """
        for (uint i = 0; i < 16; i += 2) {
            float x0 = float(h[x_off + k + i]);
            float x1 = float(h[x_off + k + i + 1]);
            xsum += x0 + x1;
            xt[i]     = x0;
            xt[i + 1] = x1 / 256.0f;
        }"""
        qdot = """
            const device uint16_t* dwl = (const device uint16_t*)(dw_base + row * K_bytes);
            float s = float(down_s[s_base + row * KG]);
            float b = float(down_b[s_base + row * KG]);
            float a = 0;
            for (uint i = 0; i < 8; i++) {
                a += xt[2*i]   * float(dwl[i] & 0x00ffu)
                   + xt[2*i+1] * float(dwl[i] & 0xff00u);
            }
            dr[row] += s * a + xsum * b;"""
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return f"""
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint expert_id = threadgroup_position_in_grid.y;
    uint tile_id = threadgroup_position_in_grid.x;

    uint K = dims[0];
    uint N = dims[1];
    uint GS = dims[3];

    const uint VPT = 16;
    const uint RPS = 4;
    const uint BS = VPT * 32;
    const uint SST = GS / VPT;
    uint K_bytes = {k_bytes_expr};
    uint KG = K / GS;

    uint eidx = expert_indices[expert_id];
    uint out_row = tile_id * 8 + simd_gid * RPS;
    if (out_row >= N) return;

    uint expert_byte_off = eidx * N * K_bytes;
    uint row_byte_off = out_row * K_bytes;
    uint thread_byte_off = {thread_byte_off_expr};

    const device uint8_t* dw_base = ((const device uint8_t*)down_w) + expert_byte_off + row_byte_off + thread_byte_off;
    uint s_base = eidx * N * KG + out_row * KG + simd_lid / SST;

    uint x_base = expert_id * K;
    uint x_off = x_base + simd_lid * VPT;

    float dr[4] = {{0, 0, 0, 0}};

    for (uint k = 0; k < K; k += BS) {{
        float xt[16];
        float xsum = 0;
{x_load}

        for (uint row = 0; row < RPS; row++) {{
{qdot}
        }}
        dw_base += {ptr_advance};
        s_base += BS / GS;
    }}

    for (uint row = 0; row < RPS; row++) {{
        float d = simd_sum(dr[row]);
        if (simd_lid == 0 && out_row + row < N)
            out[expert_id * N + out_row + row] = d;
    }}
"""


def _make_wo_source(bits):
    """Generate the fused grouped output projection kernel source for the given bit width."""
    if bits == 4:
        k_bytes_expr = "K / 2"
        thread_byte_off_expr = "simd_lid * VPT / 2"
        ptr_advance = "BS / 2"
        x_load = """
        for (uint i = 0; i < 16; i += 4) {
            float x0 = float(x[x_base + k + i]);
            float x1 = float(x[x_base + k + i + 1]);
            float x2 = float(x[x_base + k + i + 2]);
            float x3 = float(x[x_base + k + i + 3]);
            xsum += x0 + x1 + x2 + x3;
            xt[i]     = x0;
            xt[i + 1] = x1 / 16.0f;
            xt[i + 2] = x2 / 256.0f;
            xt[i + 3] = x3 / 4096.0f;
        }"""
        qdot = """
            const device uint16_t* wl = (const device uint16_t*)(w_base + row * K_bytes);
            float s = float(scales[s_off + row * KG]);
            float b = float(biases[s_off + row * KG]);
            float a = 0;
            for (uint i = 0; i < 4; i++) {
                a += xt[4*i]   * float(wl[i] & 0x000fu)
                   + xt[4*i+1] * float(wl[i] & 0x00f0u)
                   + xt[4*i+2] * float(wl[i] & 0x0f00u)
                   + xt[4*i+3] * float(wl[i] & 0xf000u);
            }
            r[row] += s * a + xsum * b;"""
    elif bits == 8:
        k_bytes_expr = "K"
        thread_byte_off_expr = "simd_lid * VPT"
        ptr_advance = "BS"
        x_load = """
        for (uint i = 0; i < 16; i += 2) {
            float x0 = float(x[x_base + k + i]);
            float x1 = float(x[x_base + k + i + 1]);
            xsum += x0 + x1;
            xt[i]     = x0;
            xt[i + 1] = x1 / 256.0f;
        }"""
        qdot = """
            const device uint16_t* wl = (const device uint16_t*)(w_base + row * K_bytes);
            float s = float(scales[s_off + row * KG]);
            float b = float(biases[s_off + row * KG]);
            float a = 0;
            for (uint i = 0; i < 8; i++) {
                a += xt[2*i]   * float(wl[i] & 0x00ffu)
                   + xt[2*i+1] * float(wl[i] & 0xff00u);
            }
            r[row] += s * a + xsum * b;"""
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return f"""
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint group_id = threadgroup_position_in_grid.y;
    uint tile_id = threadgroup_position_in_grid.x;

    uint K = dims[0];
    uint N = dims[1];
    uint n_groups = dims[2];
    uint GS = dims[3];

    const uint VPT = 16;
    const uint RPS = 4;
    const uint BS = VPT * 32;
    const uint SST = GS / VPT;
    uint K_bytes = {k_bytes_expr};
    uint KG = K / GS;

    uint out_row = tile_id * 8 + simd_gid * RPS;
    if (out_row >= N) return;

    uint w_off = group_id * N * K_bytes + out_row * K_bytes + {thread_byte_off_expr};
    uint s_off = group_id * N * KG + out_row * KG + simd_lid / SST;

    const device uint8_t* w_base = ((const device uint8_t*)w) + w_off;

    uint x_base = group_id * K + simd_lid * VPT;

    float r[4] = {{0, 0, 0, 0}};

    for (uint k = 0; k < K; k += BS) {{
        float xt[16];
        float xsum = 0;
{x_load}

        for (uint row = 0; row < RPS; row++) {{
{qdot}
        }}
        w_base += {ptr_advance};
        s_off += BS / GS;
    }}

    for (uint row = 0; row < RPS; row++) {{
        float v = simd_sum(r[row]);
        if (simd_lid == 0 && out_row + row < N)
            out[group_id * N + out_row + row] = v;
    }}
"""


# Kernel caches: keyed by bits
_kernels = {}
_down_kernels = {}
_wo_kernels = {}


def _get_kernel(bits):
    if bits not in _kernels:
        _kernels[bits] = mx.fast.metal_kernel(
            name=f"fused_gus_{bits}bit",
            input_names=["x", "gate_w", "gate_s", "gate_b",
                         "up_w", "up_s", "up_b",
                         "expert_indices", "dims"],
            output_names=["out"],
            source=_make_fused_source(bits),
        )
    return _kernels[bits]


def _get_down_kernel(bits):
    if bits not in _down_kernels:
        _down_kernels[bits] = mx.fast.metal_kernel(
            name=f"fused_down_{bits}bit",
            input_names=["h", "down_w", "down_s", "down_b",
                         "expert_indices", "dims"],
            output_names=["out"],
            source=_make_down_source(bits),
        )
    return _down_kernels[bits]


def _get_wo_kernel(bits):
    if bits not in _wo_kernels:
        _wo_kernels[bits] = mx.fast.metal_kernel(
            name=f"fused_wo_{bits}bit",
            input_names=["x", "w", "scales", "biases", "dims"],
            output_names=["out"],
            source=_make_wo_source(bits),
        )
    return _wo_kernels[bits]


def fused_gate_up_swiglu(x, gate_proj, up_proj, expert_indices):
    g, u = gate_proj, up_proj
    n_exp = expert_indices.shape[0]
    N = g.weight.shape[1]
    K = g.scales.shape[2] * g.group_size
    bits = g.bits
    assert bits in (4, 8), f"fused kernel supports 4-bit and 8-bit, got {bits}-bit"
    assert N % 8 == 0, f"fused kernel requires N divisible by 8, got {N}"
    assert K % 512 == 0, f"fused kernel requires K divisible by 512, got {K}"
    dims = mx.array([K, N, n_exp, g.group_size], dtype=mx.uint32)
    kernel = _get_kernel(bits)
    (out,) = kernel(
        inputs=[x, g.weight, g.scales, g.biases,
                u.weight, u.scales, u.biases,
                expert_indices, dims],
        output_shapes=[(n_exp * N,)],
        output_dtypes=[mx.float32],
        grid=((N // 8) * 32, n_exp * 2, 1),
        threadgroup=(32, 2, 1),
    )
    return out.reshape(n_exp, N)


def fused_down_proj(h, down_proj, expert_indices):
    """Fused down proj: all experts in one dispatch.
    h: [n_experts, hidden_dim] float32 (from fused gate+up+SwiGLU)
    Returns: [n_experts, out_dim] float32
    """
    d = down_proj
    n_exp = expert_indices.shape[0]
    N = d.weight.shape[1]  # out_dim (4096)
    K = d.scales.shape[2] * d.group_size  # hidden_dim (2048)
    bits = d.bits
    assert bits in (4, 8), f"fused kernel supports 4-bit and 8-bit, got {bits}-bit"
    assert N % 8 == 0, f"fused kernel requires N divisible by 8, got {N}"
    assert K % 512 == 0, f"fused kernel requires K divisible by 512, got {K}"
    dims = mx.array([K, N, n_exp, d.group_size], dtype=mx.uint32)
    kernel = _get_down_kernel(bits)
    (out,) = kernel(
        inputs=[h, d.weight, d.scales, d.biases,
                expert_indices, dims],
        output_shapes=[(n_exp * N,)],
        output_dtypes=[mx.float32],
        grid=((N // 8) * 32, n_exp * 2, 1),
        threadgroup=(32, 2, 1),
    )
    return out.reshape(n_exp, N)


def fused_grouped_wo(x_grouped, wo_a_list):
    """Fused 8-group wo_a projection in one dispatch.
    x_grouped: [n_groups, K] (flattened from [B, L, n_groups, heads_per_group * head_dim])
    wo_a_list: list of 8 QuantizedLinear
    Returns: [n_groups * N] float32
    """
    w0 = wo_a_list[0]
    n_groups = len(wo_a_list)
    N = w0.weight.shape[0]  # o_lora_rank
    K = w0.scales.shape[1] * w0.group_size
    bits = w0.bits
    assert bits in (4, 8), f"fused kernel supports 4-bit and 8-bit, got {bits}-bit"
    assert N % 8 == 0, f"fused kernel requires N divisible by 8, got {N}"
    assert K % 512 == 0, f"fused kernel requires K divisible by 512, got {K}"

    sw = mx.concatenate([wa.weight for wa in wo_a_list], axis=0)
    ss = mx.concatenate([wa.scales for wa in wo_a_list], axis=0)
    sb = mx.concatenate([wa.biases for wa in wo_a_list], axis=0)

    dims = mx.array([K, N, n_groups, w0.group_size], dtype=mx.uint32)
    kernel = _get_wo_kernel(bits)
    (out,) = kernel(
        inputs=[x_grouped, sw, ss, sb, dims],
        output_shapes=[(n_groups * N,)],
        output_dtypes=[mx.float32],
        grid=((N // 8) * 32, n_groups * 2, 1),
        threadgroup=(32, 2, 1),
    )
    return out.reshape(n_groups, N)
