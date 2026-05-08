"""Tests for DeepSeek V4 model implementation.

Covers:
- Model creation with various compress_ratios and cache type selection
- Prefill + decode forward pass (shapes and cache offsets)
- Continuation prefill (chunked prefill simulation)
- Multi-turn conversation (fresh cache, no stale state)
- SparseKVCache serialization (state / from_state roundtrip)
- SparseKVCache trim (offset and sparse state invalidation)
- Compressor learned pooling (prefill shape, decode accumulation)
- Fused Metal kernels (HC pre/post, optional)
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v4 import (
    Compressor,
    Model,
    ModelArgs,
    SparseKVCache,
)
from mlx_lm.models.cache import RotatingKVCache


# ---------------------------------------------------------------------------
# Shared small-model config
# ---------------------------------------------------------------------------

def _small_args(**overrides):
    """Return a minimal ModelArgs for fast unit tests."""
    defaults = dict(
        model_type="deepseek_v4",
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=4,
        qk_rope_head_dim=64,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        topk_method="noaux_tc",
        swiglu_limit=10.0,
        num_hash_layers=0,
        compress_ratios=[],
        compress_rope_theta=160000.0,
        sliding_window=8,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=4,
        num_nextn_predict_layers=1,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


def _build_model(args):
    """Build model and initialize weights so forward pass works."""
    model = Model(args)
    # Disable mx.compile for unit-test reproducibility
    model._compiled = True
    params = model.parameters()
    mx.eval(params)
    return model


# ---------------------------------------------------------------------------
# 1. Model creation
# ---------------------------------------------------------------------------

class TestModelCreation(unittest.TestCase):

    def test_layer_count_no_compression(self):
        args = _small_args(compress_ratios=[0, 0, 0, 0])
        model = _build_model(args)
        self.assertEqual(len(model.layers), 4)

    def test_layer_count_mixed_compression(self):
        args = _small_args(compress_ratios=[4, 0, 128, 4])
        model = _build_model(args)
        self.assertEqual(len(model.layers), 4)

    def test_cache_types_no_compression(self):
        """All ratio=0 layers should get RotatingKVCache."""
        args = _small_args(compress_ratios=[0, 0, 0, 0])
        model = _build_model(args)
        caches = model.make_cache()
        self.assertEqual(len(caches), 4)
        for c in caches:
            self.assertIsInstance(c, RotatingKVCache)

    def test_cache_types_mixed(self):
        """ratio=0 -> RotatingKVCache, ratio>0 -> SparseKVCache."""
        args = _small_args(compress_ratios=[4, 0, 128, 0])
        model = _build_model(args)
        caches = model.make_cache()
        self.assertIsInstance(caches[0], SparseKVCache)
        self.assertIsInstance(caches[1], RotatingKVCache)
        self.assertIsInstance(caches[2], SparseKVCache)
        self.assertIsInstance(caches[3], RotatingKVCache)

    def test_cache_types_all_compressed(self):
        args = _small_args(compress_ratios=[4, 4, 128, 128])
        model = _build_model(args)
        caches = model.make_cache()
        for c in caches:
            self.assertIsInstance(c, SparseKVCache)

    def test_compress_ratio_attribute(self):
        args = _small_args(compress_ratios=[4, 0, 128, 0])
        model = _build_model(args)
        self.assertEqual(model.layers[0].attn.compress_ratio, 4)
        self.assertEqual(model.layers[1].attn.compress_ratio, 0)
        self.assertEqual(model.layers[2].attn.compress_ratio, 128)
        self.assertEqual(model.layers[3].attn.compress_ratio, 0)


# ---------------------------------------------------------------------------
# 2. Prefill + Decode
# ---------------------------------------------------------------------------

class TestPrefillDecode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args(compress_ratios=[4, 0, 4, 0])
        cls.model = _build_model(cls.args)

    def test_prefill_output_shape(self):
        cache = self.model.make_cache()
        tokens = mx.zeros((1, 10), dtype=mx.int32)
        out = self.model(tokens, cache=cache)
        mx.eval(out)
        self.assertTrue(mx.all(mx.isfinite(out)).item())
        self.assertEqual(out.shape, (1, 10, self.args.vocab_size))

    def test_prefill_cache_offsets(self):
        cache = self.model.make_cache()
        tokens = mx.zeros((1, 10), dtype=mx.int32)
        self.model(tokens, cache=cache)
        mx.eval(cache[0].keys if hasattr(cache[0], 'keys') and cache[0].keys is not None else mx.array(0))
        for c in cache:
            self.assertEqual(c.offset, 10)

    def test_decode_output_shape(self):
        cache = self.model.make_cache()
        tokens = mx.zeros((1, 10), dtype=mx.int32)
        out = self.model(tokens, cache=cache)
        mx.eval(out)

        for step in range(20):
            tok = mx.zeros((1, 1), dtype=mx.int32)
            out = self.model(tok, cache=cache)
            mx.eval(out)
            self.assertTrue(mx.all(mx.isfinite(out)).item())
            self.assertEqual(out.shape, (1, 1, self.args.vocab_size))

    def test_decode_cache_offsets(self):
        cache = self.model.make_cache()
        tokens = mx.zeros((1, 10), dtype=mx.int32)
        self.model(tokens, cache=cache)
        mx.eval(cache[0].keys if hasattr(cache[0], 'keys') and cache[0].keys is not None else mx.array(0))

        for step in range(20):
            tok = mx.zeros((1, 1), dtype=mx.int32)
            self.model(tok, cache=cache)
            mx.eval(cache[0].keys if hasattr(cache[0], 'keys') and cache[0].keys is not None else mx.array(0))

        for c in cache:
            self.assertEqual(c.offset, 30)


# ---------------------------------------------------------------------------
# 3. Continuation prefill (chunked prefill)
# ---------------------------------------------------------------------------

class TestContinuationPrefill(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args(compress_ratios=[4, 0, 4, 0])
        cls.model = _build_model(cls.args)

    def test_continuation_offsets(self):
        """Prefill 10, then continue with 5 more, then decode 1."""
        cache = self.model.make_cache()

        # First chunk: 10 tokens
        tokens1 = mx.zeros((1, 10), dtype=mx.int32)
        out1 = self.model(tokens1, cache=cache)
        mx.eval(out1)
        self.assertTrue(mx.all(mx.isfinite(out1)).item())
        for c in cache:
            self.assertEqual(c.offset, 10)

        # Second chunk: 5 more tokens (continuation prefill)
        tokens2 = mx.zeros((1, 5), dtype=mx.int32)
        out2 = self.model(tokens2, cache=cache)
        mx.eval(out2)
        self.assertTrue(mx.all(mx.isfinite(out2)).item())
        self.assertEqual(out2.shape, (1, 5, self.args.vocab_size))
        for c in cache:
            self.assertEqual(c.offset, 15)

        # Decode: 1 token
        tok = mx.zeros((1, 1), dtype=mx.int32)
        out3 = self.model(tok, cache=cache)
        mx.eval(out3)
        self.assertTrue(mx.all(mx.isfinite(out3)).item())
        self.assertEqual(out3.shape, (1, 1, self.args.vocab_size))
        for c in cache:
            self.assertEqual(c.offset, 16)


# ---------------------------------------------------------------------------
# 4. Second conversation (fresh cache)
# ---------------------------------------------------------------------------

class TestSecondConversation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args = _small_args(compress_ratios=[4, 0, 4, 0])
        cls.model = _build_model(cls.args)

    def test_fresh_cache_no_stale_state(self):
        """Run prefill+decode, then fresh cache, run again."""
        # First conversation
        cache1 = self.model.make_cache()
        tokens = mx.zeros((1, 10), dtype=mx.int32)
        out1 = self.model(tokens, cache=cache1)
        mx.eval(out1)
        for _ in range(5):
            tok = mx.zeros((1, 1), dtype=mx.int32)
            self.model(tok, cache=cache1)
        mx.eval(cache1[0].keys if hasattr(cache1[0], 'keys') and cache1[0].keys is not None else mx.array(0))

        # Second conversation: fresh cache
        cache2 = self.model.make_cache()
        for c in cache2:
            self.assertEqual(c.offset, 0)

        tokens2 = mx.zeros((1, 8), dtype=mx.int32)
        out2 = self.model(tokens2, cache=cache2)
        mx.eval(out2)
        self.assertEqual(out2.shape, (1, 8, self.args.vocab_size))
        for c in cache2:
            self.assertEqual(c.offset, 8)

        # Decode in second conversation
        tok = mx.zeros((1, 1), dtype=mx.int32)
        out3 = self.model(tok, cache=cache2)
        mx.eval(out3)
        self.assertEqual(out3.shape, (1, 1, self.args.vocab_size))
        for c in cache2:
            self.assertEqual(c.offset, 9)

    def test_first_conversation_cache_untouched(self):
        """First conversation caches should not be mutated by second."""
        cache1 = self.model.make_cache()
        tokens = mx.zeros((1, 10), dtype=mx.int32)
        self.model(tokens, cache=cache1)
        mx.eval(cache1[0].keys if hasattr(cache1[0], 'keys') and cache1[0].keys is not None else mx.array(0))

        offsets_after_first = [c.offset for c in cache1]

        # Second conversation
        cache2 = self.model.make_cache()
        tokens2 = mx.zeros((1, 5), dtype=mx.int32)
        self.model(tokens2, cache=cache2)
        mx.eval(cache2[0].keys if hasattr(cache2[0], 'keys') and cache2[0].keys is not None else mx.array(0))

        # First conversation offsets unchanged
        for c, expected in zip(cache1, offsets_after_first):
            self.assertEqual(c.offset, expected)


# ---------------------------------------------------------------------------
# 5. SparseKVCache serialization
# ---------------------------------------------------------------------------

class TestSparseKVCacheSerialization(unittest.TestCase):

    def _make_populated_cache(self):
        cache = SparseKVCache()
        B, n_kv, S, D = 1, 1, 10, 64
        keys = mx.random.normal(shape=(B, n_kv, S, D))
        values = mx.random.normal(shape=(B, n_kv, S, D))
        cache.update_and_fetch(keys, values)

        # Set sparse attrs to simulate real usage
        cache.win_buf = mx.random.normal(shape=(B, 8, D))
        cache.comp_buf = mx.random.normal(shape=(B, 3, D))
        cache.comp_kv_state = mx.random.normal(shape=(B, 8, D))
        cache.comp_score_state = mx.random.normal(shape=(B, 8, D))
        cache.idx_kv = mx.random.normal(shape=(B, 3, 64))
        cache.idx_comp_kv_state = mx.random.normal(shape=(B, 8, 64))
        cache.idx_comp_score_state = mx.random.normal(shape=(B, 8, 64))
        mx.eval(
            cache.keys, cache.values,
            cache.win_buf, cache.comp_buf,
            cache.comp_kv_state, cache.comp_score_state,
            cache.idx_kv, cache.idx_comp_kv_state, cache.idx_comp_score_state,
        )
        return cache

    def test_state_roundtrip(self):
        cache = self._make_populated_cache()
        state = cache.state
        meta = cache.meta_state

        restored = SparseKVCache.from_state(state, meta)

        self.assertEqual(restored.offset, cache.offset)
        # Keys and values match
        self.assertTrue(mx.array_equal(
            restored.keys[..., :restored.offset, :],
            cache.keys[..., :cache.offset, :],
        ))
        self.assertTrue(mx.array_equal(
            restored.values[..., :restored.offset, :],
            cache.values[..., :cache.offset, :],
        ))

    def test_state_sparse_attrs_preserved(self):
        cache = self._make_populated_cache()
        state = cache.state
        meta = cache.meta_state

        restored = SparseKVCache.from_state(state, meta)

        for attr in SparseKVCache._SPARSE_ATTRS:
            orig = getattr(cache, attr, None)
            rest = getattr(restored, attr, None)
            if orig is not None:
                self.assertIsNotNone(rest, f"Attr {attr} lost during restore")
                self.assertTrue(
                    mx.array_equal(orig, rest),
                    f"Attr {attr} mismatch after restore",
                )
            else:
                self.assertIsNone(rest, f"Attr {attr} appeared from nowhere")

    def test_state_empty_cache(self):
        cache = SparseKVCache()
        state = cache.state
        self.assertIsNone(state[0])
        self.assertIsNone(state[1])

    def test_meta_state_n_parts(self):
        cache = self._make_populated_cache()
        meta = cache.meta_state
        n_parts = int(meta["n_parts"])
        # 2 (keys+values) + 7 sparse attrs = 9
        self.assertEqual(n_parts, 9)


# ---------------------------------------------------------------------------
# 6. SparseKVCache trim
# ---------------------------------------------------------------------------

class TestSparseKVCacheTrim(unittest.TestCase):

    def test_trim_decrements_offset(self):
        cache = SparseKVCache()
        keys = mx.random.normal(shape=(1, 1, 20, 64))
        values = mx.random.normal(shape=(1, 1, 20, 64))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys)
        self.assertEqual(cache.offset, 20)

        trimmed = cache.trim(5)
        self.assertEqual(trimmed, 5)
        self.assertEqual(cache.offset, 15)

    def test_trim_clamps_to_offset(self):
        cache = SparseKVCache()
        keys = mx.random.normal(shape=(1, 1, 10, 64))
        values = mx.random.normal(shape=(1, 1, 10, 64))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys)

        trimmed = cache.trim(100)
        self.assertEqual(trimmed, 10)
        self.assertEqual(cache.offset, 0)

    def test_trim_invalidates_sparse_state(self):
        cache = SparseKVCache()
        keys = mx.random.normal(shape=(1, 1, 10, 64))
        values = mx.random.normal(shape=(1, 1, 10, 64))
        cache.update_and_fetch(keys, values)

        # Populate sparse attrs
        cache.win_buf = mx.ones((1, 8, 64))
        cache.comp_buf = mx.ones((1, 3, 64))
        cache.comp_kv_state = mx.ones((1, 8, 64))
        cache.comp_score_state = mx.ones((1, 8, 64))
        cache.idx_kv = mx.ones((1, 3, 64))
        cache.idx_comp_kv_state = mx.ones((1, 8, 64))
        cache.idx_comp_score_state = mx.ones((1, 8, 64))

        cache.trim(3)

        # All sparse attrs should be None after trim
        for attr in SparseKVCache._SPARSE_ATTRS:
            self.assertIsNone(
                getattr(cache, attr),
                f"Attr {attr} not invalidated after trim",
            )

    def test_is_trimmable(self):
        cache = SparseKVCache()
        self.assertTrue(cache.is_trimmable())


# ---------------------------------------------------------------------------
# 7. Compressor
# ---------------------------------------------------------------------------

class TestCompressor(unittest.TestCase):

    def setUp(self):
        self.args = _small_args(compress_ratios=[4, 0, 4, 0])
        self.ratio = 4
        self.head_dim = 64
        self.comp = Compressor(self.args, self.ratio, self.head_dim)
        mx.eval(self.comp.parameters())
        self.rope = nn.RoPE(self.args.qk_rope_head_dim, traditional=True)

    def test_prefill_shape(self):
        """16 tokens with ratio=4 -> 4 compressed tokens."""
        B = 1
        x = mx.random.normal(shape=(B, 16, self.args.hidden_size))
        out = self.comp(x, start_pos=0, rope_fn=self.rope)
        mx.eval(out)
        self.assertIsNotNone(out)
        # 16 / 4 = 4 compressed tokens
        self.assertEqual(out.shape[0], B)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(out.shape[2], self.head_dim)

    def test_prefill_short_returns_none(self):
        """Fewer tokens than ratio -> None (saved for decode)."""
        B = 1
        x = mx.random.normal(shape=(B, 2, self.args.hidden_size))
        out = self.comp(x, start_pos=0, rope_fn=self.rope)
        self.assertIsNone(out)

    def test_prefill_remainder(self):
        """17 tokens with ratio=4 -> 4 compressed (remainder=1 saved)."""
        B = 1
        x = mx.random.normal(shape=(B, 17, self.args.hidden_size))
        out = self.comp(x, start_pos=0, rope_fn=self.rope)
        mx.eval(out)
        self.assertIsNotNone(out)
        # floor(17/4) = 4 compressed tokens
        self.assertEqual(out.shape[1], 4)

    def test_decode_accumulation(self):
        """Feed ratio tokens one at a time: first ratio-1 return None,
        last one returns compressed."""
        B = 1
        # Reset state via prefill with 0 tokens equivalent
        self.comp.reset_state(B)

        results = []
        for i in range(self.ratio):
            tok = mx.random.normal(shape=(B, 1, self.args.hidden_size))
            out = self.comp(tok, start_pos=i, rope_fn=self.rope)
            if out is not None:
                mx.eval(out)
            results.append(out)

        # First ratio-1 should be None
        for i in range(self.ratio - 1):
            self.assertIsNone(results[i], f"Step {i} should return None")

        # Last one should produce 1 compressed token
        self.assertIsNotNone(results[-1])
        self.assertEqual(results[-1].shape, (B, 1, self.head_dim))

    def test_decode_multiple_compressions(self):
        """Feed 2*ratio tokens: should get 2 compressed outputs."""
        B = 1
        self.comp.reset_state(B)

        count = 0
        for i in range(2 * self.ratio):
            tok = mx.random.normal(shape=(B, 1, self.args.hidden_size))
            out = self.comp(tok, start_pos=i, rope_fn=self.rope)
            if out is not None:
                mx.eval(out)
                count += 1

        self.assertEqual(count, 2)


# ---------------------------------------------------------------------------
# 8. Fused Metal kernels (optional)
# ---------------------------------------------------------------------------

class TestFusedKernels(unittest.TestCase):

    def test_fused_hc_pre_matches_python(self):
        """Fused HC pre should match the Python _hc_pre path."""
        try:
            from mlx_lm.models.deepseek_v4_kernels import fused_hc_pre
        except (ImportError, Exception):
            self.skipTest("Fused kernels not available")

        args = _small_args(compress_ratios=[4, 0, 4, 0])
        model = _build_model(args)
        layer = model.layers[0]

        M = args.hc_mult
        D = args.hidden_size
        # Simulate decode input: [1, 1, M, D]
        x = mx.random.normal(shape=(1, 1, M, D))
        mx.eval(x)

        # Python path
        py_y, py_post, py_comb = layer._hc_pre(
            x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
        )
        mx.eval(py_y, py_post, py_comb)

        # Fused path
        n_iters = min(args.hc_sinkhorn_iters, 8)
        fu_y, fu_post, fu_comb = fused_hc_pre(
            x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            M, n_iters, args.hc_eps, args.rms_norm_eps,
        )
        mx.eval(fu_y, fu_post, fu_comb)

        self.assertTrue(
            mx.allclose(py_y, fu_y, atol=1e-2),
            f"HC pre y mismatch: max diff {mx.max(mx.abs(py_y - fu_y)).item():.6f}",
        )
        self.assertTrue(
            mx.allclose(py_post, fu_post, atol=1e-2),
            f"HC pre post mismatch",
        )
        self.assertTrue(
            mx.allclose(py_comb, fu_comb, atol=1e-2),
            f"HC pre comb mismatch",
        )

    def test_fused_hc_post_matches_python(self):
        """Fused HC post should match the Python _hc_post path."""
        try:
            from mlx_lm.models.deepseek_v4_kernels import fused_hc_post
        except (ImportError, Exception):
            self.skipTest("Fused kernels not available")

        args = _small_args(compress_ratios=[4, 0, 4, 0])
        model = _build_model(args)
        layer = model.layers[0]

        M = args.hc_mult
        D = args.hidden_size
        x_attn = mx.random.normal(shape=(1, 1, D))
        residual = mx.random.normal(shape=(1, 1, M, D))
        post = mx.random.normal(shape=(1, 1, M))
        comb = mx.random.normal(shape=(1, 1, M, M))
        mx.eval(x_attn, residual, post, comb)

        # Python path
        py_out = layer._hc_post(x_attn, residual, post, comb)
        mx.eval(py_out)

        # Fused path
        fu_out = fused_hc_post(x_attn, residual, post, comb, M)
        mx.eval(fu_out)

        self.assertTrue(
            mx.allclose(py_out, fu_out, atol=1e-2),
            f"HC post mismatch: max diff {mx.max(mx.abs(py_out - fu_out)).item():.6f}",
        )


if __name__ == "__main__":
    unittest.main()
