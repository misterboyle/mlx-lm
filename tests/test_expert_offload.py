"""Tests for MoE expert-level offloading.

Covers:
- ExpertWeights container (init + nbytes)
- ExpertOffloader registration, LRU eviction, byte tracking
- enable_expert_offloading attaches offloaders and clears monolithic weights
- Forward pass with offloading enabled (matches non-offloaded output)
"""

import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.deepseek_v4 import Model, ModelArgs
from mlx_lm.models.expert_offload import (
    ExpertOffloader,
    ExpertWeights,
    enable_expert_offloading,
)
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_args(**overrides):
    """Return a small DeepSeek V4 ModelArgs suitable for offloading tests.

    The fused MoE decode kernel requires K (hidden_size) divisible by 512,
    so we use 512 as the minimum hidden_size.
    """
    defaults = dict(
        model_type="deepseek_v4",
        vocab_size=512,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        qk_rope_head_dim=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        topk_method="noaux_tc",
        swiglu_limit=10.0,
        num_hash_layers=0,
        compress_ratios=[0, 0],
        compress_rope_theta=160000.0,
        sliding_window=8,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=4,
        num_nextn_predict_layers=0,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


def _build_quantized_model(args, seed=0):
    """Build model, quantize experts, eval params."""
    mx.random.seed(seed)
    model = Model(args)
    model._compiled = True
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda p, m: isinstance(m, SwitchLinear),
    )
    mx.eval(model.parameters())
    return model


def _save_model_weights(model, model_dir):
    """Save model weights as a single safetensors file."""
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(
        os.path.join(model_dir, "model.safetensors"),
        flat,
        metadata={"format": "mlx"},
    )


def _make_dummy_expert_weights(out_dim=128, in_dim=64, group_size=64, bits=4):
    """Construct a single ExpertWeights with random quantized arrays."""
    # Mimic shapes produced by mx.quantize: weight (O, K), scales/biases (O, K/group_size)
    K = max(in_dim // (32 // bits), 1)
    n_groups = max(in_dim // group_size, 1)
    gate_w = mx.zeros((out_dim, K), dtype=mx.uint32)
    gate_s = mx.zeros((out_dim, n_groups), dtype=mx.float16)
    gate_b = mx.zeros((out_dim, n_groups), dtype=mx.float16)
    up_w = mx.zeros((out_dim, K), dtype=mx.uint32)
    up_s = mx.zeros((out_dim, n_groups), dtype=mx.float16)
    up_b = mx.zeros((out_dim, n_groups), dtype=mx.float16)
    # down has reversed in/out dims
    down_K = max(out_dim // (32 // bits), 1)
    down_groups = max(out_dim // group_size, 1)
    down_w = mx.zeros((in_dim, down_K), dtype=mx.uint32)
    down_s = mx.zeros((in_dim, down_groups), dtype=mx.float16)
    down_b = mx.zeros((in_dim, down_groups), dtype=mx.float16)
    mx.eval(gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b)
    return ExpertWeights(
        gate_w=gate_w, gate_s=gate_s, gate_b=gate_b,
        up_w=up_w, up_s=up_s, up_b=up_b,
        down_w=down_w, down_s=down_s, down_b=down_b,
    )


# ---------------------------------------------------------------------------
# 1. ExpertWeights container
# ---------------------------------------------------------------------------

class TestExpertWeights(unittest.TestCase):

    def test_init(self):
        """Construct ExpertWeights with all arrays and verify nbytes."""
        out_dim, in_dim, group_size, bits = 128, 64, 64, 4
        ew = _make_dummy_expert_weights(out_dim, in_dim, group_size, bits)

        # Verify all attributes set
        for attr in ("gate_w", "gate_s", "gate_b",
                     "up_w", "up_s", "up_b",
                     "down_w", "down_s", "down_b"):
            self.assertIsNotNone(getattr(ew, attr), f"{attr} should be set")

        # nbytes is sum of all arrays' nbytes
        expected = sum(
            a.nbytes for a in (ew.gate_w, ew.gate_s, ew.gate_b,
                               ew.up_w, ew.up_s, ew.up_b,
                               ew.down_w, ew.down_s, ew.down_b)
        )
        self.assertEqual(ew.nbytes, expected)
        self.assertGreater(ew.nbytes, 0)

    def test_init_with_none_biases(self):
        """nbytes should skip None entries (biases may be absent)."""
        ew = _make_dummy_expert_weights()
        # Replace biases with None and rebuild
        ew2 = ExpertWeights(
            gate_w=ew.gate_w, gate_s=ew.gate_s, gate_b=None,
            up_w=ew.up_w, up_s=ew.up_s, up_b=None,
            down_w=ew.down_w, down_s=ew.down_s, down_b=None,
        )
        # nbytes excludes None
        expected = sum(
            a.nbytes for a in (ew.gate_w, ew.gate_s,
                               ew.up_w, ew.up_s,
                               ew.down_w, ew.down_s)
        )
        self.assertEqual(ew2.nbytes, expected)
        self.assertLess(ew2.nbytes, ew.nbytes)


# ---------------------------------------------------------------------------
# 2. ExpertOffloader: registration, LRU, byte tracking
# ---------------------------------------------------------------------------

class TestExpertOffloader(unittest.TestCase):

    def _make_offloader(self, max_resident=4, num_experts=8):
        # model_path is unused unless we trigger _load_expert
        return ExpertOffloader(
            layer_prefix="model.layers.0.ffn.experts",
            model_path="/tmp/nonexistent",
            max_resident_experts=max_resident,
            num_experts=num_experts,
        )

    def test_register_and_get(self):
        """Register an expert and retrieve it via get_expert_weights."""
        off = self._make_offloader(max_resident=4, num_experts=8)
        ew = _make_dummy_expert_weights()
        off.register(3, ew)

        out = off.get_expert_weights(3)
        self.assertIs(out, ew)
        self.assertEqual(off.num_resident, 1)
        self.assertEqual(off.bytes_resident, ew.nbytes)

    def test_lru_eviction(self):
        """Adding more than max_resident experts evicts the oldest."""
        N = 4
        off = self._make_offloader(max_resident=N, num_experts=8)

        weights = [_make_dummy_expert_weights() for _ in range(N + 1)]
        # Register N experts within budget
        for i in range(N):
            off.register(i, weights[i])
        self.assertEqual(off.num_resident, N)

        # Register the (N+1)-th -- now over budget
        off.register(N, weights[N])
        self.assertEqual(off.num_resident, N + 1)

        # ensure_resident with current MRU set triggers eviction
        # Touch expert N (most recent already), then evict to N
        off.ensure_resident([N])
        self.assertEqual(off.num_resident, N)

        # Oldest (id 0) should have been evicted
        with self.assertRaises(KeyError):
            off.get_expert_weights(0)
        # Most-recently registered (N) should still be there
        self.assertIs(off.get_expert_weights(N), weights[N])

    def test_lru_touch_on_access(self):
        """ensure_resident on a cached expert moves it to MRU position."""
        N = 3
        off = self._make_offloader(max_resident=N, num_experts=8)

        weights = [_make_dummy_expert_weights() for _ in range(N)]
        # Register experts 0, 1, 2 -- LRU order: 0 < 1 < 2
        for i in range(N):
            off.register(i, weights[i])

        # Touch expert 0 -- now LRU order: 1 < 2 < 0
        off.ensure_resident([0])
        self.assertEqual(off.num_resident, N)

        # Register a new expert 99 -> would push out the oldest if we ensure_resident
        new_ew = _make_dummy_expert_weights()
        off.register(99, new_ew)
        # Now over budget. Trigger eviction by ensuring an existing one
        off.ensure_resident([99])
        self.assertEqual(off.num_resident, N)

        # Expert 1 was the oldest -> should be gone
        with self.assertRaises(KeyError):
            off.get_expert_weights(1)
        # Expert 0 (was touched) should still be present
        self.assertIs(off.get_expert_weights(0), weights[0])
        # Expert 2 should still be present
        self.assertIs(off.get_expert_weights(2), weights[2])
        # New expert 99 should be present
        self.assertIs(off.get_expert_weights(99), new_ew)

    def test_bytes_tracking(self):
        """bytes_resident tracks total nbytes across registrations / evictions."""
        N = 2
        off = self._make_offloader(max_resident=N, num_experts=8)
        self.assertEqual(off.bytes_resident, 0)

        e0 = _make_dummy_expert_weights()
        e1 = _make_dummy_expert_weights()
        e2 = _make_dummy_expert_weights()

        off.register(0, e0)
        self.assertEqual(off.bytes_resident, e0.nbytes)

        off.register(1, e1)
        self.assertEqual(off.bytes_resident, e0.nbytes + e1.nbytes)

        # Push over budget
        off.register(2, e2)
        self.assertEqual(off.bytes_resident, e0.nbytes + e1.nbytes + e2.nbytes)

        # Evict back to N=2 by touching the newest
        off.ensure_resident([2])
        self.assertEqual(off.num_resident, N)
        # e0 was oldest -> evicted; bytes_resident = e1 + e2
        self.assertEqual(off.bytes_resident, e1.nbytes + e2.nbytes)
        self.assertEqual(off.total_evictions, 1)

    def test_num_resident(self):
        """num_resident reflects cache size after each op."""
        off = self._make_offloader(max_resident=10, num_experts=20)
        self.assertEqual(off.num_resident, 0)

        for i in range(5):
            off.register(i, _make_dummy_expert_weights())
        self.assertEqual(off.num_resident, 5)

        # ensure_resident with existing ids does not change count
        off.ensure_resident([0, 2, 4])
        self.assertEqual(off.num_resident, 5)

    def test_load_expert_from_disk_after_eviction(self):
        """After eviction, ensure_resident must reload weights from disk correctly.

        End-to-end check of the lazy load path: build a real model, save it,
        enable offloading with a budget that forces eviction, then explicitly
        evict expert 0, then ensure_resident([0]) and compare the reloaded
        weights byte-for-byte against the originally registered tensors.
        """
        args = _small_args(n_routed_experts=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _build_quantized_model(args, seed=123)

            # Snapshot every expert's weights BEFORE saving / offloading, so
            # we can compare a reload to ground truth. SwitchGLU lives at
            # model.layers[i].ffn.experts (the moe wrapper holds a SwitchGLU).
            gate = model.layers[0].ffn.experts.gate_proj
            up = model.layers[0].ffn.experts.up_proj
            down = model.layers[0].ffn.experts.down_proj
            mx.eval(
                gate.weight, gate.scales, gate.biases,
                up.weight, up.scales, up.biases,
                down.weight, down.scales, down.biases,
            )
            expert0_orig = {
                "gate_w": mx.array(gate.weight[0]),
                "gate_s": mx.array(gate.scales[0]),
                "gate_b": mx.array(gate.biases[0]),
                "up_w":   mx.array(up.weight[0]),
                "up_s":   mx.array(up.scales[0]),
                "up_b":   mx.array(up.biases[0]),
                "down_w": mx.array(down.weight[0]),
                "down_s": mx.array(down.scales[0]),
                "down_b": mx.array(down.biases[0]),
            }
            mx.eval(list(expert0_orig.values()))

            _save_model_weights(model, tmpdir)

            enable_expert_offloading(
                model, tmpdir, max_resident_experts=2
            )

            glu0 = model.layers[0].ffn.experts
            off = glu0._offloader
            self.assertIsNotNone(off)

            # Force-evict expert 0 if it's currently resident.
            if 0 in off._cache:
                del off._cache[0]
                # Recompute resident bytes
                off._bytes_resident = sum(ew.nbytes for ew in off._cache.values())
            self.assertNotIn(0, off._cache)

            # Now trigger a reload from disk.
            off.ensure_resident([0])
            self.assertIn(0, off._cache)

            reloaded = off.get_expert_weights(0)
            mx.eval(
                reloaded.gate_w, reloaded.gate_s, reloaded.gate_b,
                reloaded.up_w, reloaded.up_s, reloaded.up_b,
                reloaded.down_w, reloaded.down_s, reloaded.down_b,
            )

            # Every reloaded tensor must equal the original expert-0 slice.
            for name, orig in expert0_orig.items():
                got = getattr(reloaded, name)
                self.assertIsNotNone(got, f"{name} missing after reload")
                self.assertTrue(
                    mx.array_equal(got, orig),
                    f"Reloaded {name} differs from original",
                )


# ---------------------------------------------------------------------------
# 3. enable_expert_offloading: end-to-end attach + slice
# ---------------------------------------------------------------------------

class TestEnableOffloading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args()
        cls.max_resident = 4

    def test_enable_attaches_offloaders(self):
        """enable_expert_offloading attaches _offloader to each SwitchGLU,
        clears the monolithic weights, and keeps max_resident experts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _build_quantized_model(self.args)
            _save_model_weights(model, tmpdir)

            count = enable_expert_offloading(
                model, tmpdir, max_resident_experts=self.max_resident
            )
            self.assertEqual(count, self.args.num_hidden_layers)

            switchgluss = [
                m for _, m in model.named_modules()
                if isinstance(m, SwitchGLU)
            ]
            self.assertEqual(len(switchgluss), self.args.num_hidden_layers)

            for glu in switchgluss:
                # Offloader attached
                self.assertIsNotNone(glu._offloader)
                self.assertEqual(
                    glu._offloader.num_resident, self.max_resident
                )
                self.assertEqual(
                    glu._offloader.num_experts, self.args.n_routed_experts
                )
                # Monolithic weights cleared
                self.assertIsNone(glu.gate_proj.weight)
                self.assertIsNone(glu.gate_proj.scales)
                self.assertIsNone(glu.gate_proj.biases)
                self.assertIsNone(glu.up_proj.weight)
                self.assertIsNone(glu.up_proj.scales)
                self.assertIsNone(glu.up_proj.biases)
                self.assertIsNone(glu.down_proj.weight)
                self.assertIsNone(glu.down_proj.scales)
                self.assertIsNone(glu.down_proj.biases)
                # Quant params copied
                self.assertEqual(glu._offloader.group_size, 64)
                self.assertEqual(glu._offloader.bits, 4)

            # Model marker set
            self.assertTrue(getattr(model, "_expert_offloading", False))

    def test_generate_works_after_enable(self):
        """A forward pass through the offloaded model produces a finite tensor
        of the correct shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _build_quantized_model(self.args)
            _save_model_weights(model, tmpdir)
            enable_expert_offloading(
                model, tmpdir, max_resident_experts=self.max_resident
            )

            cache = model.make_cache()
            tokens = mx.zeros((1, 8), dtype=mx.int32)
            out = model(tokens, cache=cache)
            mx.eval(out)
            self.assertEqual(out.shape, (1, 8, self.args.vocab_size))
            self.assertTrue(mx.all(mx.isfinite(out)).item())


# ---------------------------------------------------------------------------
# 4. Offloaded forward equivalence vs. non-offloaded
# ---------------------------------------------------------------------------

class TestOffloadedForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args(n_routed_experts=8)
        cls.max_resident = 4

    def _build_ref_and_save(self, tmpdir):
        """Build the reference model and persist weights."""
        ref = _build_quantized_model(self.args, seed=42)
        _save_model_weights(ref, tmpdir)
        return ref

    def _build_offloaded(self, tmpdir):
        """Build an identical model and enable offloading."""
        off_model = _build_quantized_model(self.args, seed=42)
        enable_expert_offloading(
            off_model, tmpdir, max_resident_experts=self.max_resident
        )
        return off_model

    def test_prefill_no_nan_correct_shape(self):
        """Prefill with seq_len=10 produces finite output of correct shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_ref_and_save(tmpdir)
            off_model = self._build_offloaded(tmpdir)

            cache = off_model.make_cache()
            tokens = mx.zeros((1, 10), dtype=mx.int32)
            out = off_model(tokens, cache=cache)
            mx.eval(out)

            self.assertEqual(out.shape, (1, 10, self.args.vocab_size))
            self.assertTrue(mx.all(mx.isfinite(out)).item())

    def test_decode_no_nan_correct_shape(self):
        """Three decode steps produce finite outputs of correct shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._build_ref_and_save(tmpdir)
            off_model = self._build_offloaded(tmpdir)

            cache = off_model.make_cache()
            # Prefill first
            tokens = mx.zeros((1, 10), dtype=mx.int32)
            out = off_model(tokens, cache=cache)
            mx.eval(out)

            for _ in range(3):
                tok = mx.zeros((1, 1), dtype=mx.int32)
                out = off_model(tok, cache=cache)
                mx.eval(out)
                self.assertEqual(out.shape, (1, 1, self.args.vocab_size))
                self.assertTrue(mx.all(mx.isfinite(out)).item())

    def test_matches_non_offloaded(self):
        """Offloaded forward pass matches the non-offloaded one
        (per-expert mode is mathematically equivalent to gather_qmm).

        We use a small max_resident_experts (2 of 8) and random token ids so
        the router picks varying experts and LRU eviction is exercised.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_model = self._build_ref_and_save(tmpdir)
            # Build offloaded with a smaller budget than the default to
            # actually force evictions during the test.
            off_model = _build_quantized_model(self.args, seed=42)
            enable_expert_offloading(
                off_model, tmpdir, max_resident_experts=2
            )

            # Random tokens so router picks different experts per position.
            mx.random.seed(7)
            tokens = mx.random.randint(
                0, self.args.vocab_size, (1, 8), dtype=mx.int32
            )
            mx.eval(tokens)

            ref_cache = ref_model.make_cache()
            off_cache = off_model.make_cache()

            ref_out = ref_model(tokens, cache=ref_cache)
            off_out = off_model(tokens, cache=off_cache)
            mx.eval(ref_out, off_out)

            self.assertEqual(ref_out.shape, off_out.shape)
            self.assertTrue(mx.all(mx.isfinite(off_out)).item())
            self.assertTrue(
                mx.allclose(ref_out, off_out, atol=1e-3),
                f"Outputs differ: max abs diff "
                f"{mx.max(mx.abs(ref_out - off_out)).item():.6f}",
            )

            # Now decode several steps with varying tokens.
            for step in range(5):
                tok = mx.random.randint(
                    0, self.args.vocab_size, (1, 1), dtype=mx.int32
                )
                ref_d = ref_model(tok, cache=ref_cache)
                off_d = off_model(tok, cache=off_cache)
                mx.eval(ref_d, off_d)
                self.assertTrue(
                    mx.allclose(ref_d, off_d, atol=1e-3),
                    f"Decode mismatch step {step}: max abs diff "
                    f"{mx.max(mx.abs(ref_d - off_d)).item():.6f}",
                )

            # With budget=2 out of 8 experts and random routing, the
            # offloader must have evicted some experts during prefill +
            # decode -- otherwise the LRU path is silently broken.
            total_evictions = sum(
                m._offloader.total_evictions
                for _, m in off_model.named_modules()
                if isinstance(m, SwitchGLU) and m._offloader is not None
            )
            self.assertGreater(
                total_evictions, 0,
                "LRU eviction never triggered: routing or budget too lenient",
            )


# ---------------------------------------------------------------------------
# 5. enable_expert_offloading skip / no-op paths
# ---------------------------------------------------------------------------

class TestEnableOffloadingSkip(unittest.TestCase):
    """Cases where enable_expert_offloading should silently skip."""

    def test_non_quantized_model_skipped(self):
        """If the SwitchGLU experts are NOT quantized, offloading is a no-op.

        enable_expert_offloading should return 0, attach no _offloader, and
        not crash.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _small_args()
            mx.random.seed(0)
            model = Model(args)  # NOT quantized
            model._compiled = True
            mx.eval(model.parameters())
            _save_model_weights(model, tmpdir)

            count = enable_expert_offloading(
                model, tmpdir, max_resident_experts=2
            )
            self.assertEqual(count, 0)

            # No SwitchGLU should have an _offloader attached.
            for _, m in model.named_modules():
                if isinstance(m, SwitchGLU):
                    self.assertIsNone(
                        getattr(m, "_offloader", None),
                        "non-quantized SwitchGLU got an _offloader attached",
                    )
            # Marker stays unset.
            self.assertFalse(getattr(model, "_expert_offloading", False))

    def test_max_resident_geq_num_experts_skipped(self):
        """If max_resident_experts >= num_experts there's nothing to offload.

        enable_expert_offloading should return 0 and not attach offloaders.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _small_args(n_routed_experts=8)
            model = _build_quantized_model(args)
            _save_model_weights(model, tmpdir)

            count = enable_expert_offloading(
                model, tmpdir,
                max_resident_experts=args.n_routed_experts,
            )
            self.assertEqual(count, 0)

            for _, m in model.named_modules():
                if isinstance(m, SwitchGLU):
                    self.assertIsNone(
                        getattr(m, "_offloader", None),
                        "_offloader attached even though "
                        "max_resident >= num_experts",
                    )
            self.assertFalse(getattr(model, "_expert_offloading", False))


if __name__ == "__main__":
    unittest.main()
