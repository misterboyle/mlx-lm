# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation
        self._offloader = None  # Set by enable_expert_offloading()

    def __call__(self, x, indices) -> mx.array:
        # Decode: sequential per-expert for L2 cache reuse (~1.2x faster)
        if (x.shape[-2] <= 1 and indices.size <= 8
                and isinstance(self.gate_proj, QuantizedSwitchLinear)):
            return self._decode_sequential(x, indices)

        # Offloading: fall back to sequential processing during prefill
        # because gather_qmm needs ALL experts in a monolithic tensor
        if self._offloader is not None:
            return self._prefill_with_offloading(x, indices)

        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)

    def _prefill_with_offloading(self, x, indices):
        """Prefill path when expert offloading is active.

        Since gather_qmm requires the monolithic (E, O, I) tensor which
        we no longer have, we process tokens sequentially per expert.
        This is slower than the fused path but allows offloading to work
        during both prefill and decode.
        """
        offloader = self._offloader
        g = self.gate_proj
        d = self.down_proj
        u = self.up_proj

        # x: (..., seq_len, hidden) indices: (..., seq_len, top_k)
        orig_shape = x.shape
        top_k = indices.shape[-1]
        x_flat = x.reshape(-1, orig_shape[-1])  # (T, H)
        idx_flat = indices.reshape(-1, top_k)    # (T, K)
        T = x_flat.shape[0]

        results = []
        for t in range(T):
            token_x = x_flat[t:t+1]  # (1, H)
            # Ensure this token's experts are resident
            token_experts = [idx_flat[t, k].item() for k in range(top_k)]
            offloader.ensure_resident(token_experts)
            expert_outs = []
            for k in range(top_k):
                eid = token_experts[k]
                ew = offloader.get_expert_weights(eid)
                gi = mx.quantized_matmul(
                    token_x, ew.gate_w, ew.gate_s, ew.gate_b,
                    transpose=True, group_size=g.group_size, bits=g.bits,
                )
                ui = mx.quantized_matmul(
                    token_x, ew.up_w, ew.up_s, ew.up_b,
                    transpose=True, group_size=u.group_size, bits=u.bits,
                )
                hi = self.activation(ui, gi)  # (1, hidden_dim)
                oi = mx.quantized_matmul(
                    hi.astype(x.dtype), ew.down_w, ew.down_s, ew.down_b,
                    transpose=True, group_size=d.group_size, bits=d.bits,
                )
                expert_outs.append(oi.squeeze(0))
            results.append(mx.stack(expert_outs, axis=0))  # (K, out_dim)

        result = mx.stack(results, axis=0)  # (T, K, out_dim)
        # Reshape back to (..., seq_len, top_k, out_dim) then squeeze
        out_shape = list(orig_shape[:-1]) + [top_k, result.shape[-1]]
        return result.reshape(out_shape)

    def _decode_sequential(self, x, indices):
        """Fused gate+up+SwiGLU Metal kernel + per-expert down proj.
        All experts' gate+up in ONE dispatch, then sequential down."""
        flat_idx = indices.reshape(-1)
        n = flat_idx.shape[0]
        g = self.gate_proj
        d = self.down_proj

        # When offloading is active, use per-expert weights from offloader
        if self._offloader is not None:
            return self._decode_sequential_offloaded(x, flat_idx, n)

        # Fused gate+up+SwiGLU: one Metal dispatch for all experts
        if g.bits in (4, 8):
            from .fused_moe_kernel import fused_gate_up_swiglu
            h = fused_gate_up_swiglu(
                x.reshape(-1), g, self.up_proj,
                flat_idx.astype(mx.uint32))  # [n, hidden]
        else:
            # Fallback for unsupported bit widths
            u = self.up_proj
            x_2d = x.reshape(1, -1)
            hs = []
            for i in range(n):
                idx = flat_idx[i]
                gi = mx.quantized_matmul(x_2d, g.weight[idx], g.scales[idx],
                    g.biases[idx], transpose=True, group_size=g.group_size, bits=g.bits)
                ui = mx.quantized_matmul(x_2d, u.weight[idx], u.scales[idx],
                    u.biases[idx], transpose=True, group_size=u.group_size, bits=u.bits)
                hs.append(self.activation(ui, gi).squeeze(0))
            h = mx.stack(hs)

        # Down proj: fused Metal kernel (all experts, one dispatch)
        if d.bits in (4, 8):
            from .fused_moe_kernel import fused_down_proj
            result = fused_down_proj(h, d, flat_idx.astype(mx.uint32))
        else:
            outs = []
            for i in range(n):
                idx = flat_idx[i]
                oi = mx.quantized_matmul(
                    h[i:i+1].astype(x.dtype), d.weight[idx], d.scales[idx], d.biases[idx],
                    transpose=True, group_size=d.group_size, bits=d.bits)
                outs.append(oi.squeeze(0))
            result = mx.stack(outs, axis=0)
        result = result.astype(x.dtype)
        return result.reshape(list(x.shape[:-1]) + [n, -1])

    def _decode_sequential_offloaded(self, x, flat_idx, n):
        """Decode path with per-expert weights from the offloader.

        Bypasses fused Metal kernels (which need the monolithic tensor)
        and does per-expert quantized matmuls using individual weight slices.
        """
        offloader = self._offloader
        g = self.gate_proj
        u = self.up_proj
        d = self.down_proj
        x_2d = x.reshape(1, -1)

        # Ensure all active experts are loaded
        expert_ids = flat_idx.tolist()
        offloader.ensure_resident(expert_ids)

        # Gate + Up + SwiGLU per expert
        hs = []
        for i in range(n):
            eid = expert_ids[i]
            ew = offloader.get_expert_weights(eid)
            gi = mx.quantized_matmul(
                x_2d, ew.gate_w, ew.gate_s, ew.gate_b,
                transpose=True, group_size=g.group_size, bits=g.bits,
            )
            ui = mx.quantized_matmul(
                x_2d, ew.up_w, ew.up_s, ew.up_b,
                transpose=True, group_size=u.group_size, bits=u.bits,
            )
            hs.append(self.activation(ui, gi).squeeze(0))
        h = mx.stack(hs)  # (n, hidden_dim)

        # Down proj per expert
        outs = []
        for i in range(n):
            eid = expert_ids[i]
            ew = offloader.get_expert_weights(eid)
            oi = mx.quantized_matmul(
                h[i:i+1].astype(x.dtype), ew.down_w, ew.down_s, ew.down_b,
                transpose=True, group_size=d.group_size, bits=d.bits,
            )
            outs.append(oi.squeeze(0))
        result = mx.stack(outs, axis=0)

        result = result.astype(x.dtype)
        return result.reshape(list(x.shape[:-1]) + [n, -1])


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
