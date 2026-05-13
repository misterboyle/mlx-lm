"""Tests for fused MoE Metal kernels with 8-bit weights.

Covers:
- fused_gate_up_swiglu (4-bit and 8-bit) vs per-expert mx.quantized_matmul reference
- fused_down_proj (4-bit and 8-bit) vs per-expert mx.quantized_matmul reference
- fused_grouped_wo (4-bit and 8-bit) vs per-group mx.quantized_matmul reference
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.models.fused_moe_kernel import (
    fused_gate_up_swiglu,
    fused_down_proj,
    fused_grouped_wo,
)


# ---------------------------------------------------------------------------
# Shared test dims
# ---------------------------------------------------------------------------
# K must be divisible by 512 and N divisible by 8 (kernel constraints).
K = 512        # input dim
N = 256        # output dim
NUM_EXPERTS = 4
GROUP_SIZE = 64
N_GROUPS = 8   # for fused_grouped_wo


def _ref_gate_up_swiglu(x, gate, up, indices):
    """Per-expert reference computation matching fused_gate_up_swiglu."""
    refs = []
    x2 = x.reshape(1, -1)
    for i in range(indices.shape[0]):
        eid = int(indices[i].item())
        gi = mx.quantized_matmul(
            x2, gate.weight[eid], gate.scales[eid], gate.biases[eid],
            transpose=True, group_size=gate.group_size, bits=gate.bits,
        )
        ui = mx.quantized_matmul(
            x2, up.weight[eid], up.scales[eid], up.biases[eid],
            transpose=True, group_size=up.group_size, bits=up.bits,
        )
        refs.append((nn.silu(gi) * ui).squeeze(0))
    return mx.stack(refs)


def _ref_down_proj(h, down, indices):
    """Per-expert reference computation matching fused_down_proj."""
    refs = []
    for i in range(indices.shape[0]):
        eid = int(indices[i].item())
        hi = h[i:i + 1]
        oi = mx.quantized_matmul(
            hi, down.weight[eid], down.scales[eid], down.biases[eid],
            transpose=True, group_size=down.group_size, bits=down.bits,
        )
        refs.append(oi.squeeze(0))
    return mx.stack(refs)


def _ref_grouped_wo(x, wo_list):
    """Per-group reference computation matching fused_grouped_wo."""
    refs = []
    for g, wa in enumerate(wo_list):
        xi = x[g:g + 1]
        oi = mx.quantized_matmul(
            xi, wa.weight, wa.scales, wa.biases,
            transpose=True, group_size=wa.group_size, bits=wa.bits,
        )
        refs.append(oi.squeeze(0))
    return mx.stack(refs)


# ---------------------------------------------------------------------------
# 8-bit tests
# ---------------------------------------------------------------------------

class TestFusedGateUpSwiGLU(unittest.TestCase):
    """fused_gate_up_swiglu with 8-bit QuantizedSwitchLinear."""

    BITS = 8

    def setUp(self):
        mx.random.seed(0)
        self.gate = QuantizedSwitchLinear(
            K, N, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        self.up = QuantizedSwitchLinear(
            K, N, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        mx.eval(self.gate.parameters(), self.up.parameters())

    def _run(self, indices):
        x = mx.random.normal(shape=(K,), dtype=mx.float32)
        mx.eval(x)
        out = fused_gate_up_swiglu(x, self.gate, self.up, indices)
        ref = _ref_gate_up_swiglu(x, self.gate, self.up, indices)
        mx.eval(out, ref)
        return out, ref

    def test_shape_and_dtype(self):
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out, _ = self._run(indices)
        self.assertEqual(out.shape, (NUM_EXPERTS, N))
        self.assertEqual(out.dtype, mx.float32)

    def test_matches_reference(self):
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out, ref = self._run(indices)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"fused_gate_up_swiglu mismatch, max diff = {max_diff}",
        )

    def test_repeated_expert_indices(self):
        """Same expert can appear multiple times (top-k routing)."""
        indices = mx.array([0, 0, 1, 1], dtype=mx.uint32)
        out, ref = self._run(indices)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"repeated-indices mismatch, max diff = {max_diff}",
        )


class TestFusedDownProj(unittest.TestCase):
    """fused_down_proj with 8-bit QuantizedSwitchLinear."""

    BITS = 8

    def setUp(self):
        mx.random.seed(1)
        self.down = QuantizedSwitchLinear(
            K, N, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        mx.eval(self.down.parameters())

    def _run(self, indices):
        h = mx.random.normal(shape=(NUM_EXPERTS, K), dtype=mx.float32)
        mx.eval(h)
        out = fused_down_proj(h, self.down, indices)
        ref = _ref_down_proj(h, self.down, indices)
        mx.eval(out, ref)
        return out, ref

    def test_shape_and_dtype(self):
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out, _ = self._run(indices)
        self.assertEqual(out.shape, (NUM_EXPERTS, N))
        self.assertEqual(out.dtype, mx.float32)

    def test_matches_reference(self):
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out, ref = self._run(indices)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"fused_down_proj mismatch, max diff = {max_diff}",
        )

    def test_repeated_expert_indices(self):
        """Same expert appearing multiple times must still match per-expert
        reference exactly (top-k routing can pick the same expert twice)."""
        indices = mx.array([0, 0, 1, 1], dtype=mx.uint32)
        out, ref = self._run(indices)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"fused_down_proj repeated-indices mismatch, "
            f"max diff = {max_diff}",
        )


class TestFusedGroupedWO(unittest.TestCase):
    """fused_grouped_wo with 8-bit QuantizedLinear (V4 attention output)."""

    BITS = 8

    def setUp(self):
        mx.random.seed(2)
        self.wo_list = [
            nn.QuantizedLinear(K, N, bias=False, bits=self.BITS, group_size=GROUP_SIZE)
            for _ in range(N_GROUPS)
        ]
        for wa in self.wo_list:
            mx.eval(wa.parameters())

    def _run(self):
        x = mx.random.normal(shape=(N_GROUPS, K), dtype=mx.float32)
        mx.eval(x)
        out = fused_grouped_wo(x, self.wo_list)
        ref = _ref_grouped_wo(x, self.wo_list)
        mx.eval(out, ref)
        return out, ref

    def test_shape_and_dtype(self):
        out, _ = self._run()
        self.assertEqual(out.shape, (N_GROUPS, N))
        self.assertEqual(out.dtype, mx.float32)

    def test_matches_reference(self):
        out, ref = self._run()
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"fused_grouped_wo mismatch, max diff = {max_diff}",
        )

    def test_n_groups_4(self):
        """Kernel must generalize beyond N_GROUPS=8 (some V4 configs use 4)."""
        mx.random.seed(102)
        n_groups_small = 4
        wo_list = [
            nn.QuantizedLinear(
                K, N, bias=False,
                bits=self.BITS, group_size=GROUP_SIZE,
            )
            for _ in range(n_groups_small)
        ]
        for wa in wo_list:
            mx.eval(wa.parameters())

        x = mx.random.normal(shape=(n_groups_small, K), dtype=mx.float32)
        mx.eval(x)
        out = fused_grouped_wo(x, wo_list)
        ref = _ref_grouped_wo(x, wo_list)
        mx.eval(out, ref)
        self.assertEqual(out.shape, (n_groups_small, N))
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"fused_grouped_wo (N_GROUPS=4) mismatch, max diff={max_diff}",
        )


# ---------------------------------------------------------------------------
# Larger shapes (stride generalization)
# ---------------------------------------------------------------------------

class TestFusedLargerShapes(unittest.TestCase):
    """Run all three fused kernels at K=1024, N=512 to catch stride bugs
    that smaller shapes might hide (e.g. assumptions baked at K=512).
    """

    BITS = 8
    K_BIG = 1024  # divisible by 512
    N_BIG = 512   # divisible by 8

    def test_gate_up_swiglu_large(self):
        mx.random.seed(201)
        gate = QuantizedSwitchLinear(
            self.K_BIG, self.N_BIG, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        up = QuantizedSwitchLinear(
            self.K_BIG, self.N_BIG, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        mx.eval(gate.parameters(), up.parameters())

        x = mx.random.normal(shape=(self.K_BIG,), dtype=mx.float32)
        mx.eval(x)
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out = fused_gate_up_swiglu(x, gate, up, indices)
        ref = _ref_gate_up_swiglu(x, gate, up, indices)
        mx.eval(out, ref)
        self.assertEqual(out.shape, (NUM_EXPERTS, self.N_BIG))
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"large gate_up_swiglu mismatch, max diff = {max_diff}",
        )

    def test_down_proj_large(self):
        mx.random.seed(202)
        down = QuantizedSwitchLinear(
            self.K_BIG, self.N_BIG, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        mx.eval(down.parameters())

        h = mx.random.normal(
            shape=(NUM_EXPERTS, self.K_BIG), dtype=mx.float32)
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out = fused_down_proj(h, down, indices)
        ref = _ref_down_proj(h, down, indices)
        mx.eval(out, ref)
        self.assertEqual(out.shape, (NUM_EXPERTS, self.N_BIG))
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"large down_proj mismatch, max diff = {max_diff}",
        )

    def test_grouped_wo_large(self):
        mx.random.seed(203)
        wo_list = [
            nn.QuantizedLinear(
                self.K_BIG, self.N_BIG, bias=False,
                bits=self.BITS, group_size=GROUP_SIZE,
            )
            for _ in range(N_GROUPS)
        ]
        for wa in wo_list:
            mx.eval(wa.parameters())

        x = mx.random.normal(
            shape=(N_GROUPS, self.K_BIG), dtype=mx.float32)
        out = fused_grouped_wo(x, wo_list)
        ref = _ref_grouped_wo(x, wo_list)
        mx.eval(out, ref)
        self.assertEqual(out.shape, (N_GROUPS, self.N_BIG))
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"large grouped_wo mismatch, max diff = {max_diff}",
        )


# ---------------------------------------------------------------------------
# Backward compatibility: 4-bit kernels still pass
# ---------------------------------------------------------------------------

class TestBackwardCompat4bit(unittest.TestCase):
    """Same kernels run at 4 bits to verify the 4-bit path is untouched."""

    BITS = 4

    def test_gate_up_swiglu_4bit(self):
        mx.random.seed(10)
        gate = QuantizedSwitchLinear(
            K, N, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        up = QuantizedSwitchLinear(
            K, N, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        mx.eval(gate.parameters(), up.parameters())

        x = mx.random.normal(shape=(K,), dtype=mx.float32)
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out = fused_gate_up_swiglu(x, gate, up, indices)
        ref = _ref_gate_up_swiglu(x, gate, up, indices)
        mx.eval(out, ref)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertEqual(out.shape, (NUM_EXPERTS, N))
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"4-bit gate+up+swiglu regression, max diff = {max_diff}",
        )

    def test_down_proj_4bit(self):
        mx.random.seed(11)
        down = QuantizedSwitchLinear(
            K, N, num_experts=NUM_EXPERTS, bias=False,
            bits=self.BITS, group_size=GROUP_SIZE,
        )
        mx.eval(down.parameters())

        h = mx.random.normal(shape=(NUM_EXPERTS, K), dtype=mx.float32)
        indices = mx.array(list(range(NUM_EXPERTS)), dtype=mx.uint32)
        out = fused_down_proj(h, down, indices)
        ref = _ref_down_proj(h, down, indices)
        mx.eval(out, ref)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertEqual(out.shape, (NUM_EXPERTS, N))
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"4-bit down proj regression, max diff = {max_diff}",
        )

    def test_grouped_wo_4bit(self):
        mx.random.seed(12)
        wo_list = [
            nn.QuantizedLinear(K, N, bias=False, bits=self.BITS, group_size=GROUP_SIZE)
            for _ in range(N_GROUPS)
        ]
        for wa in wo_list:
            mx.eval(wa.parameters())

        x = mx.random.normal(shape=(N_GROUPS, K), dtype=mx.float32)
        out = fused_grouped_wo(x, wo_list)
        ref = _ref_grouped_wo(x, wo_list)
        mx.eval(out, ref)
        max_diff = mx.max(mx.abs(out - ref)).item()
        self.assertEqual(out.shape, (N_GROUPS, N))
        self.assertTrue(
            mx.allclose(out, ref, atol=1e-3),
            f"4-bit grouped wo regression, max diff = {max_diff}",
        )


if __name__ == "__main__":
    unittest.main()
