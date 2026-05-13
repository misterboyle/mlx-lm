"""Tests for DeepSeek V4 weight sanitization.

Covers:
- _dequant_scaled_weights: FP8 e4m3 (uint8) with ue8m0 block scales -> bfloat16
- _remap_thump604: Thump604 MLX naming -> our naming (hc_attn.base, switch_mlp, etc.)
- sanitize() format detection (HF original / Thump604 / mlx-community)
"""

import os
import sys
import unittest

import mlx.core as mx

from mlx_lm.models.deepseek_v4 import Model

# Reuse the shared small ModelArgs builder from the V4 test module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_deepseek_v4 import _small_args, _build_model  # noqa: E402


def _u8_weight(shape, lo=0, hi=256):
    """Create a synthetic uint8 weight tensor."""
    return mx.random.randint(lo, hi, shape=shape).astype(mx.uint8)


def _u8_scale(shape, lo=120, hi=135):
    """Create a synthetic ue8m0 scale tensor (uint8 around the bias of 127)."""
    return mx.random.randint(lo, hi, shape=shape).astype(mx.uint8)


# ---------------------------------------------------------------------------
# 1. _dequant_scaled_weights (FP8 e4m3 + ue8m0 block scales)
# ---------------------------------------------------------------------------

class TestDequantScaled(unittest.TestCase):
    """FP8 block-scaled dequant: uint8 weight + ue8m0 scale -> bfloat16 weight."""

    def test_single_block(self):
        """128x128 weight with a single 1x1 block scale."""
        w = _u8_weight((128, 128))
        s = _u8_scale((1, 1))
        out = Model._dequant_scaled_weights({"foo.weight": w, "foo.scale": s})

        self.assertIn("foo.weight", out)
        self.assertNotIn("foo.scale", out)
        self.assertEqual(out["foo.weight"].shape, (128, 128))
        self.assertEqual(out["foo.weight"].dtype, mx.bfloat16)

    def test_multi_block(self):
        """256x384 weight with 2x3 block scales (128x128 blocks)."""
        w = _u8_weight((256, 384))
        s = _u8_scale((2, 3))
        out = Model._dequant_scaled_weights({"bar.weight": w, "bar.scale": s})

        self.assertEqual(out["bar.weight"].shape, (256, 384))
        self.assertEqual(out["bar.weight"].dtype, mx.bfloat16)

    def test_padding_required(self):
        """Non-aligned dims need padding then crop back to original shape."""
        w = _u8_weight((100, 100))
        s = _u8_scale((1, 1))
        out = Model._dequant_scaled_weights({"baz.weight": w, "baz.scale": s})

        self.assertEqual(out["baz.weight"].shape, (100, 100))
        self.assertEqual(out["baz.weight"].dtype, mx.bfloat16)

    def test_no_scale_passthrough(self):
        """Weights without a matching .scale key are left untouched."""
        plain = mx.zeros((4, 4), dtype=mx.float32)
        out = Model._dequant_scaled_weights({"keep.weight": plain})

        self.assertIn("keep.weight", out)
        self.assertEqual(out["keep.weight"].dtype, mx.float32)
        self.assertEqual(out["keep.weight"].shape, (4, 4))

    def test_orphan_scale_kept(self):
        """A .scale key with no matching .weight is kept (not dropped)."""
        s = _u8_scale((1, 1))
        out = Model._dequant_scaled_weights({"orphan.scale": s})
        self.assertIn("orphan.scale", out)

    def test_mixed_keys_only_uint8_dequanted(self):
        """Non-uint8 weights with a .scale partner are kept as-is."""
        w = mx.zeros((4, 4), dtype=mx.float32)
        s = _u8_scale((1, 1))
        out = Model._dequant_scaled_weights({"mix.weight": w, "mix.scale": s})
        self.assertIn("mix.weight", out)
        # Non-FP8 path keeps the scale too
        self.assertIn("mix.scale", out)
        self.assertEqual(out["mix.weight"].dtype, mx.float32)

    def test_dequant_known_values_e4m3_unity(self):
        """FP8 e4m3 byte 0x38 with ue8m0 scale 127 (=1.0) must dequant to 1.0.

        Cross-verified with mx.from_fp8 (the same primitive used internally).
        """
        # 128x128 to match the 128-block size, all bytes = 0x38 = 1.0
        w = mx.full((128, 128), 0x38, dtype=mx.uint8)
        s = mx.full((1, 1), 127, dtype=mx.uint8)
        out = Model._dequant_scaled_weights({"x.weight": w, "x.scale": s})
        mx.eval(out["x.weight"])
        self.assertEqual(out["x.weight"].dtype, mx.bfloat16)
        # Every element should be exactly 1.0
        diff = mx.max(mx.abs(out["x.weight"].astype(mx.float32) - 1.0)).item()
        self.assertLess(diff, 0.01, f"max abs diff from 1.0: {diff}")

    def test_dequant_known_values_e4m3_scaled(self):
        """FP8 byte 0x38 (=1.0) with ue8m0 scale 128 (=2.0) -> 2.0.

        And byte 0x40 (=2.0) with scale 127 (=1.0) -> 2.0.
        """
        # Case 1: byte = 1.0, scale = 2.0
        w1 = mx.full((128, 128), 0x38, dtype=mx.uint8)
        s1 = mx.full((1, 1), 128, dtype=mx.uint8)
        out1 = Model._dequant_scaled_weights({"a.weight": w1, "a.scale": s1})
        mx.eval(out1["a.weight"])
        diff1 = mx.max(mx.abs(
            out1["a.weight"].astype(mx.float32) - 2.0)).item()
        self.assertLess(diff1, 0.01)

        # Case 2: byte = 2.0, scale = 1.0
        w2 = mx.full((128, 128), 0x40, dtype=mx.uint8)
        s2 = mx.full((1, 1), 127, dtype=mx.uint8)
        out2 = Model._dequant_scaled_weights({"b.weight": w2, "b.scale": s2})
        mx.eval(out2["b.weight"])
        diff2 = mx.max(mx.abs(
            out2["b.weight"].astype(mx.float32) - 2.0)).item()
        self.assertLess(diff2, 0.01)

    def test_dequant_matches_from_fp8(self):
        """Dequant output (per-element) must match mx.from_fp8 * scale exactly."""
        # Random bytes, single block
        w = _u8_weight((128, 128))
        s = mx.array([[127]], dtype=mx.uint8)  # scale = 1.0
        out = Model._dequant_scaled_weights({"r.weight": w, "r.scale": s})
        mx.eval(out["r.weight"])
        # With scale = 1.0, output should equal mx.from_fp8(w) exactly
        expected = mx.from_fp8(w, dtype=mx.bfloat16)
        mx.eval(expected)
        self.assertTrue(
            mx.array_equal(out["r.weight"], expected),
            "Dequant with scale=1.0 must equal mx.from_fp8(weight)",
        )


# ---------------------------------------------------------------------------
# 2. _remap_thump604 (Thump604 MLX naming -> ours)
# ---------------------------------------------------------------------------

class TestRemapThump604(unittest.TestCase):
    """Verify Thump604-style key names are remapped correctly."""

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args(compress_ratios=[4, 0, 4, 0])
        cls.model = _build_model(cls.args)

    def _zeros(self, *shape):
        return mx.zeros(shape, dtype=mx.float32)

    def test_hc_attr_dot_to_underscore(self):
        # All nine combinations of {hc_attn, hc_ffn, hc_head} x {base, fn, scale}
        weights = {
            "layers.0.hc_attn.base":  self._zeros(24),
            "layers.0.hc_attn.fn":    self._zeros(24, 1024),
            "layers.0.hc_attn.scale": self._zeros(3),
            "layers.0.hc_ffn.base":   self._zeros(24),
            "layers.0.hc_ffn.fn":     self._zeros(24, 1024),
            "layers.0.hc_ffn.scale":  self._zeros(3),
            "layers.0.hc_head.base":  self._zeros(24),
            "layers.0.hc_head.fn":    self._zeros(24, 1024),
            "layers.0.hc_head.scale": self._zeros(3),
        }
        out = self.model._remap_thump604(weights)
        for k in (
            "layers.0.hc_attn_base",
            "layers.0.hc_attn_fn",
            "layers.0.hc_attn_scale",
            "layers.0.hc_ffn_base",
            "layers.0.hc_ffn_fn",
            "layers.0.hc_ffn_scale",
            "layers.0.hc_head_base",
            "layers.0.hc_head_fn",
            "layers.0.hc_head_scale",
        ):
            self.assertIn(k, out, f"missing {k}")
        for k in weights:
            self.assertNotIn(k, out, f"old key {k} should be gone")

    def test_layernorm_rename(self):
        weights = {
            "layers.0.input_layernorm.weight": self._zeros(256),
            "layers.0.post_attention_layernorm.weight": self._zeros(256),
        }
        out = self.model._remap_thump604(weights)
        self.assertIn("layers.0.attn_norm.weight", out)
        self.assertIn("layers.0.ffn_norm.weight", out)
        self.assertNotIn("layers.0.input_layernorm.weight", out)
        self.assertNotIn("layers.0.post_attention_layernorm.weight", out)

    def test_self_attn_to_attn(self):
        weights = {
            "layers.0.self_attn.wq_a.weight": self._zeros(128, 256),
            "layers.0.self_attn.wkv.weight": self._zeros(64, 256),
        }
        out = self.model._remap_thump604(weights)
        self.assertIn("layers.0.attn.wq_a.weight", out)
        self.assertIn("layers.0.attn.wkv.weight", out)
        self.assertNotIn("layers.0.self_attn.wq_a.weight", out)

    def test_e_score_correction_bias_to_bias(self):
        weights = {
            "layers.0.mlp.gate.e_score_correction_bias": self._zeros(4),
        }
        out = self.model._remap_thump604(weights)
        self.assertIn("layers.0.ffn.gate.bias", out)
        self.assertNotIn(
            "layers.0.mlp.gate.e_score_correction_bias", out)

    def test_switch_mlp_to_experts(self):
        weights = {
            "layers.0.mlp.switch_mlp.gate_proj.weight": self._zeros(4, 256, 256),
            "layers.0.mlp.switch_mlp.up_proj.weight": self._zeros(4, 256, 256),
            "layers.0.mlp.switch_mlp.down_proj.weight": self._zeros(4, 256, 256),
        }
        out = self.model._remap_thump604(weights)
        self.assertIn("layers.0.ffn.experts.gate_proj.weight", out)
        self.assertIn("layers.0.ffn.experts.up_proj.weight", out)
        self.assertIn("layers.0.ffn.experts.down_proj.weight", out)
        for k in weights:
            self.assertNotIn(k, out)

    def test_ffn_switch_mlp_no_double_prefix(self):
        """Regression: .ffn.switch_mlp. must become .ffn.experts., NOT .ffn.ffn.experts."""
        weights = {
            "model.layers.0.ffn.switch_mlp.gate_proj.weight":
                self._zeros(4, 256, 256),
            "model.layers.0.ffn.switch_mlp.up_proj.weight":
                self._zeros(4, 256, 256),
            "model.layers.0.ffn.switch_mlp.down_proj.weight":
                self._zeros(4, 256, 256),
        }
        out = self.model._remap_thump604(weights)
        self.assertIn(
            "model.layers.0.ffn.experts.gate_proj.weight", out)
        self.assertIn(
            "model.layers.0.ffn.experts.up_proj.weight", out)
        self.assertIn(
            "model.layers.0.ffn.experts.down_proj.weight", out)
        # The bug we fixed: ensure no double-prefix happened.
        self.assertNotIn(
            "model.layers.0.ffn.ffn.experts.gate_proj.weight", out)
        self.assertNotIn(
            "model.layers.0.ffn.ffn.experts.up_proj.weight", out)
        self.assertNotIn(
            "model.layers.0.ffn.ffn.experts.down_proj.weight", out)
        for k in weights:
            self.assertNotIn(k, out)

    def test_bare_switch_mlp_to_ffn_experts(self):
        """Bare `.switch_mlp.` (no ffn/mlp wrapper) -> .ffn.experts. ."""
        weights = {
            "layers.0.switch_mlp.gate_proj.weight":
                self._zeros(4, 256, 256),
            "layers.0.switch_mlp.up_proj.weight":
                self._zeros(4, 256, 256),
            "layers.0.switch_mlp.down_proj.weight":
                self._zeros(4, 256, 256),
        }
        out = self.model._remap_thump604(weights)
        self.assertIn("layers.0.ffn.experts.gate_proj.weight", out)
        self.assertIn("layers.0.ffn.experts.up_proj.weight", out)
        self.assertIn("layers.0.ffn.experts.down_proj.weight", out)
        # And no double prefix.
        self.assertNotIn("layers.0.ffn.ffn.experts.gate_proj.weight", out)
        for k in weights:
            self.assertNotIn(k, out)

    def test_wo_a_single_linear_replaces_list(self):
        """A bare `wo_a.weight` key (no group index) triggers wo_a -> nn.Linear.

        Thump604 stores wo_a as a single QuantizedLinear; the model class
        constructs it as a list. _remap_thump604 must rewrite the model
        attribute to a single nn.Linear so load_weights() succeeds.
        """
        import mlx.nn as nn

        # Use a fresh model (we mutate model.layers[*].attn.wo_a here)
        args = _small_args(compress_ratios=[4, 0, 4, 0])
        model = _build_model(args)

        # Before: wo_a is a list (per-group)
        self.assertIsInstance(model.layers[0].attn.wo_a, list)

        weights = {
            "layers.0.self_attn.wo_a.weight": self._zeros(128, 64),
        }
        out = model._remap_thump604(weights)

        # Output key was rewritten (self_attn -> attn)
        self.assertIn("layers.0.attn.wo_a.weight", out)
        self.assertNotIn("layers.0.self_attn.wo_a.weight", out)

        # Every layer's wo_a is now a single nn.Linear (not a list)
        for layer in model.layers:
            self.assertIsInstance(
                layer.attn.wo_a, nn.Linear,
                f"layer.attn.wo_a should be nn.Linear, "
                f"got {type(layer.attn.wo_a)}",
            )

    def test_shared_experts_rename(self):
        weights = {
            "layers.0.mlp.shared_experts.gate_proj.weight": self._zeros(256, 256),
            "layers.0.mlp.shared_experts.up_proj.weight":   self._zeros(256, 256),
            "layers.0.mlp.shared_experts.down_proj.weight": self._zeros(256, 256),
        }
        out = self.model._remap_thump604(weights)
        # mlp -> ffn, and shared_experts gate/up/down -> w1/w3/w2
        self.assertIn("layers.0.ffn.shared_experts.w1.weight", out)
        self.assertIn("layers.0.ffn.shared_experts.w3.weight", out)
        self.assertIn("layers.0.ffn.shared_experts.w2.weight", out)
        for k in weights:
            self.assertNotIn(k, out)

    def test_full_thump604_layer(self):
        """End-to-end remap of a single layer's keys."""
        weights = {
            "layers.0.hc_attn.base": self._zeros(24),
            "layers.0.hc_attn.fn": self._zeros(24, 1024),
            "layers.0.hc_attn.scale": self._zeros(3),
            "layers.0.input_layernorm.weight": self._zeros(256),
            "layers.0.post_attention_layernorm.weight": self._zeros(256),
            "layers.0.self_attn.wq_a.weight": self._zeros(128, 256),
            "layers.0.mlp.gate.weight": self._zeros(4, 256),
            "layers.0.mlp.gate.e_score_correction_bias": self._zeros(4),
            "layers.0.mlp.switch_mlp.gate_proj.weight": self._zeros(4, 256, 256),
            "layers.0.mlp.switch_mlp.up_proj.weight":   self._zeros(4, 256, 256),
            "layers.0.mlp.switch_mlp.down_proj.weight": self._zeros(4, 256, 256),
            "layers.0.mlp.shared_experts.gate_proj.weight": self._zeros(256, 256),
            "layers.0.mlp.shared_experts.up_proj.weight":   self._zeros(256, 256),
            "layers.0.mlp.shared_experts.down_proj.weight": self._zeros(256, 256),
        }
        out = self.model._remap_thump604(weights)
        expected = {
            "layers.0.attn.wq_a.weight",
            "layers.0.attn_norm.weight",
            "layers.0.ffn.experts.down_proj.weight",
            "layers.0.ffn.experts.gate_proj.weight",
            "layers.0.ffn.experts.up_proj.weight",
            "layers.0.ffn.gate.bias",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.shared_experts.w2.weight",
            "layers.0.ffn.shared_experts.w3.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.hc_attn_base",
            "layers.0.hc_attn_fn",
            "layers.0.hc_attn_scale",
        }
        self.assertEqual(set(out.keys()), expected)


# ---------------------------------------------------------------------------
# 3. Format detection in sanitize()
# ---------------------------------------------------------------------------

class TestDetectFormat(unittest.TestCase):
    """Verify that sanitize() takes the right path for each input format.

    We don't introspect the model state; instead we feed each format a tiny set
    of representative keys and verify the *output* keys reflect the format-
    specific transforms (dequant for HF, remap for Thump604, passthrough for
    mlx-community).
    """

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args(compress_ratios=[4, 0, 4, 0])
        cls.model = _build_model(cls.args)

    # --- HF original ---

    def test_hf_original_dequants_fp8(self):
        """Presence of a `.scale` key triggers FP8 dequant -> bfloat16."""
        weights = {
            "layers.0.input_layernorm.weight": mx.zeros((256,)),
            "layers.0.self_attn.wq_a.weight": _u8_weight((128, 256)),
            "layers.0.self_attn.wq_a.scale": _u8_scale((1, 2)),
        }
        out = self.model.sanitize(weights)
        # Step 1 dequanted wq_a.weight to bfloat16
        wq = out["model.layers.0.self_attn.wq_a.weight"]
        self.assertEqual(wq.dtype, mx.bfloat16)
        # The .scale key was consumed
        self.assertNotIn("model.layers.0.self_attn.wq_a.scale", out)
        self.assertNotIn("layers.0.self_attn.wq_a.scale", out)
        # Step 2 prefixed with `model.`
        self.assertIn("model.layers.0.input_layernorm.weight", out)

    def test_hf_original_drops_mtp(self):
        """`mtp.*` keys must be dropped (multi-token-prediction weights)."""
        weights = {
            "mtp.layers.0.something.weight": mx.zeros((4,)),
            "mtp.foo": mx.zeros((4,)),
            "layers.0.input_layernorm.weight": mx.zeros((256,)),
            "layers.0.fake.scale": _u8_scale((1, 1)),  # triggers HF path
            "layers.0.fake.weight": _u8_weight((128, 128)),
        }
        out = self.model.sanitize(weights)
        for k in out:
            self.assertFalse(
                k.startswith("mtp."),
                f"mtp key leaked through sanitize: {k}",
            )

    # --- Thump604 ---

    def test_thump604_remaps_hc_attn(self):
        """`hc_attn.base` should be rewritten to `hc_attn_base`."""
        weights = {
            "layers.0.hc_attn.base": mx.zeros((24,)),
            "layers.0.hc_attn.fn":   mx.zeros((24, 1024)),
            "layers.0.hc_attn.scale": mx.zeros((3,)),
        }
        out = self.model.sanitize(weights)
        self.assertIn("model.layers.0.hc_attn_base", out)
        self.assertIn("model.layers.0.hc_attn_fn", out)
        self.assertIn("model.layers.0.hc_attn_scale", out)
        self.assertNotIn("model.layers.0.hc_attn.base", out)

    def test_thump604_remaps_e_score_bias(self):
        """`e_score_correction_bias` triggers Thump604 path."""
        weights = {
            "layers.0.mlp.gate.e_score_correction_bias": mx.zeros((4,)),
            "layers.0.mlp.gate.weight": mx.zeros((4, 256)),
        }
        out = self.model.sanitize(weights)
        self.assertIn("model.layers.0.ffn.gate.bias", out)
        self.assertIn("model.layers.0.ffn.gate.weight", out)
        self.assertNotIn(
            "model.layers.0.mlp.gate.e_score_correction_bias", out)

    def test_thump604_remaps_switch_mlp(self):
        """`switch_mlp.` triggers Thump604 path."""
        weights = {
            "layers.0.mlp.switch_mlp.gate_proj.weight": mx.zeros((4, 256, 256)),
            "layers.0.mlp.switch_mlp.up_proj.weight":   mx.zeros((4, 256, 256)),
            "layers.0.mlp.switch_mlp.down_proj.weight": mx.zeros((4, 256, 256)),
        }
        out = self.model.sanitize(weights)
        self.assertIn("model.layers.0.ffn.experts.gate_proj.weight", out)
        self.assertIn("model.layers.0.ffn.experts.up_proj.weight", out)
        self.assertIn("model.layers.0.ffn.experts.down_proj.weight", out)

    # --- mlx-community (default passthrough) ---

    def test_mlx_community_passthrough(self):
        """No FP8 .scale, no Thump604 markers -> Thump604/dequant paths skipped.

        The only transform is the top-level rename (prefixing with `model.`)
        and the w1/w2/w3 expert renames already in the mlx-community format.
        """
        weights = {
            "embed.weight": mx.zeros((512, 256)),
            "head.weight":  mx.zeros((512, 256)),
            "norm.weight":  mx.zeros((256,)),
            "layers.0.attn.wq_a.weight": mx.zeros((128, 256)),
            "layers.0.attn_norm.weight": mx.zeros((256,)),
            "layers.0.ffn.gate.weight": mx.zeros((4, 256)),
            "layers.0.ffn.gate.bias":   mx.zeros((4,)),
            # Pre-stacked w1/w2/w3 (mlx-community format)
            "layers.0.ffn.experts.w1.weight": mx.zeros((4, 256, 256)),
            "layers.0.ffn.experts.w2.weight": mx.zeros((4, 256, 256)),
            "layers.0.ffn.experts.w3.weight": mx.zeros((4, 256, 256)),
        }
        out = self.model.sanitize(weights)
        # Top-level renames applied
        self.assertIn("model.embed_tokens.weight", out)
        self.assertIn("lm_head.weight", out)
        self.assertIn("model.norm.weight", out)
        # w1/w2/w3 -> gate/down/up_proj
        self.assertIn("model.layers.0.ffn.experts.gate_proj.weight", out)
        self.assertIn("model.layers.0.ffn.experts.down_proj.weight", out)
        self.assertIn("model.layers.0.ffn.experts.up_proj.weight", out)
        self.assertNotIn("model.layers.0.ffn.experts.w1.weight", out)
        # No Thump604 traces
        self.assertNotIn("model.layers.0.hc_attn.base", out)
        # No HF FP8 dequant happened: any uint8 wouldn't have been processed
        # (there are none here), but check that the gate.bias key still exists
        self.assertIn("model.layers.0.ffn.gate.bias", out)


if __name__ == "__main__":
    unittest.main()
