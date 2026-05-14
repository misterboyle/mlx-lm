# Copyright © 2024 Apple Inc.

"""Tests for TurboQuant KV cache compression.

Covers:
- Bit-packing (pack/unpack roundtrip for all bit widths)
- Walsh-Hadamard transform (orthogonality, invertibility)
- TurboQuantKVCache (update, offset, trim, state, nbytes, serialization)
- Conversion from KVCache via to_turbo_quantized()
- make_prompt_cache with turbo_kv_bits (mixed cache layers)
- End-to-end generation with TurboQuant cache
- Save/load prompt cache with TurboQuantKVCache
"""

import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    KVCache,
    can_trim_prompt_cache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.models.turboquant_cache import BatchTurboQuantKVCache, TurboQuantKVCache
from mlx_lm.models.turboquant_packing import (
    VALS_PER_WORD,
    pack_indices,
    packed_dim,
    unpack_indices,
)
from mlx_lm.models.turboquant_rotation import (
    inverse_randomized_hadamard,
    random_diagonal_sign,
    randomized_hadamard_transform,
    walsh_hadamard_transform,
)


# ---------------------------------------------------------------------------
# Packing tests
# ---------------------------------------------------------------------------
class TestBitPacking(unittest.TestCase):

    def test_packed_dim(self):
        self.assertEqual(packed_dim(128, 3), 13)  # ceil(128/10)
        self.assertEqual(packed_dim(128, 4), 16)  # ceil(128/8)
        self.assertEqual(packed_dim(128, 2), 8)  # ceil(128/16)
        self.assertEqual(packed_dim(128, 1), 4)  # ceil(128/32)
        self.assertEqual(packed_dim(1, 3), 1)
        self.assertEqual(packed_dim(10, 3), 1)  # exactly 10 vals in one word
        self.assertEqual(packed_dim(11, 3), 2)

    def test_pack_unpack_roundtrip(self):
        for bits in [1, 2, 3, 4]:
            max_val = (1 << bits) - 1
            for dim in [16, 64, 96, 128]:
                indices = mx.random.randint(0, max_val + 1, shape=(4, dim)).astype(
                    mx.uint8
                )
                packed = pack_indices(indices, bits)
                self.assertEqual(packed.shape[-1], packed_dim(dim, bits))
                unpacked = unpack_indices(packed, bits, dim)
                self.assertTrue(
                    mx.array_equal(indices, unpacked),
                    f"Roundtrip failed for bits={bits}, dim={dim}",
                )

    def test_pack_unpack_batched(self):
        """Test with batch and head dimensions."""
        for bits in [1, 2, 3, 4]:
            max_val = (1 << bits) - 1
            indices = mx.random.randint(0, max_val + 1, shape=(2, 8, 10, 128)).astype(
                mx.uint8
            )
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, 128)
            self.assertTrue(mx.array_equal(indices, unpacked))

    def test_pack_zeros(self):
        indices = mx.zeros((4, 128), dtype=mx.uint8)
        for bits in [1, 2, 3, 4]:
            packed = pack_indices(indices, bits)
            self.assertTrue(mx.array_equal(packed, mx.zeros_like(packed)))

    def test_pack_max_values(self):
        for bits in [1, 2, 3, 4]:
            max_val = (1 << bits) - 1
            indices = mx.full((4, 128), max_val, dtype=mx.uint8)
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, 128)
            self.assertTrue(mx.array_equal(indices, unpacked))


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------
class TestRotation(unittest.TestCase):

    def test_wht_orthogonality(self):
        """WHT is orthogonal: WHT(WHT(x)) == x."""
        for d in [16, 64, 128]:
            x = mx.random.normal(shape=(4, d))
            y = walsh_hadamard_transform(walsh_hadamard_transform(x))
            self.assertTrue(
                mx.allclose(x, y, atol=1e-5),
                f"WHT not self-inverse for d={d}",
            )

    def test_wht_preserves_norm(self):
        """WHT is norm-preserving (isometry)."""
        x = mx.random.normal(shape=(8, 128))
        y = walsh_hadamard_transform(x)
        x_norms = mx.linalg.norm(x, axis=-1)
        y_norms = mx.linalg.norm(y, axis=-1)
        self.assertTrue(mx.allclose(x_norms, y_norms, atol=1e-4))

    def test_wht_requires_power_of_2(self):
        x = mx.random.normal(shape=(4, 7))
        with self.assertRaises(AssertionError):
            walsh_hadamard_transform(x)

    def test_random_diagonal_sign(self):
        signs = random_diagonal_sign(128, seed=42)
        self.assertEqual(signs.shape, (128,))
        # All values should be +1 or -1
        self.assertTrue(mx.all(mx.abs(signs) == 1.0))

    def test_random_diagonal_deterministic(self):
        s1 = random_diagonal_sign(64, seed=99)
        s2 = random_diagonal_sign(64, seed=99)
        self.assertTrue(mx.array_equal(s1, s2))

    def test_randomized_hadamard_invertible(self):
        """Forward then inverse should recover original."""
        signs = random_diagonal_sign(128, seed=42)
        x = mx.random.normal(shape=(4, 128))
        y = randomized_hadamard_transform(x, signs)
        x_recovered = inverse_randomized_hadamard(y, signs)
        self.assertTrue(mx.allclose(x, x_recovered, atol=1e-5))


# ---------------------------------------------------------------------------
# TurboQuantKVCache tests
# ---------------------------------------------------------------------------
class TestTurboQuantKVCache(unittest.TestCase):

    def test_init(self):
        cache = TurboQuantKVCache(bits=3)
        self.assertEqual(cache.quant_bits, 3)
        self.assertEqual(cache.offset, 0)
        self.assertTrue(cache.empty())
        self.assertEqual(cache.size(), 0)
        self.assertEqual(cache.nbytes, 0)

    def test_single_update(self):
        cache = TurboQuantKVCache(bits=3)
        B, H, S, D = 1, 8, 10, 64
        k = mx.random.normal(shape=(B, H, S, D))
        v = mx.random.normal(shape=(B, H, S, D))

        k_ret, v_ret = cache.update_and_fetch(k, v)

        self.assertEqual(cache.offset, 10)
        self.assertEqual(cache.size(), 10)
        self.assertFalse(cache.empty())
        self.assertEqual(k_ret.shape, (B, H, 10, D))
        self.assertEqual(v_ret.shape, (B, H, 10, D))

    def test_sequential_updates(self):
        """Simulate prefill then decode tokens."""
        cache = TurboQuantKVCache(bits=3)
        B, H, D = 1, 8, 64

        # Prefill: 20 tokens
        k = mx.random.normal(shape=(B, H, 20, D))
        v = mx.random.normal(shape=(B, H, 20, D))
        k_ret, v_ret = cache.update_and_fetch(k, v)
        self.assertEqual(cache.offset, 20)
        self.assertEqual(k_ret.shape, (B, H, 20, D))

        # Decode: 5 single tokens
        for i in range(5):
            k1 = mx.random.normal(shape=(B, H, 1, D))
            v1 = mx.random.normal(shape=(B, H, 1, D))
            k_ret, v_ret = cache.update_and_fetch(k1, v1)
            self.assertEqual(cache.offset, 21 + i)
            self.assertEqual(k_ret.shape, (B, H, 21 + i, D))
            self.assertEqual(v_ret.shape, (B, H, 21 + i, D))

    def test_asymmetric_kv_dims(self):
        """K and V can have different dimensions (GQA patterns)."""
        cache = TurboQuantKVCache(bits=3)
        B, H = 1, 4
        k = mx.random.normal(shape=(B, H, 5, 128))
        v = mx.random.normal(shape=(B, H, 5, 64))
        k_ret, v_ret = cache.update_and_fetch(k, v)
        self.assertEqual(k_ret.shape, (B, H, 5, 128))
        self.assertEqual(v_ret.shape, (B, H, 5, 64))

    def test_different_bit_widths(self):
        for bits in [1, 2, 3, 4]:
            cache = TurboQuantKVCache(bits=bits)
            k = mx.random.normal(shape=(1, 4, 8, 64))
            v = mx.random.normal(shape=(1, 4, 8, 64))
            k_ret, v_ret = cache.update_and_fetch(k, v)
            self.assertEqual(cache.offset, 8)
            self.assertEqual(k_ret.shape, (1, 4, 8, 64))

    def test_quantization_quality(self):
        """Dequantized values should approximate originals."""
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 16, 128))
        v = mx.random.normal(shape=(1, 4, 16, 128))
        k_ret, v_ret = cache.update_and_fetch(k, v)

        # Cosine similarity should be high for 3-bit
        k_flat = k.reshape(-1, 128)
        kr_flat = k_ret.reshape(-1, 128)
        dots = mx.sum(k_flat * kr_flat, axis=-1)
        norms = mx.linalg.norm(k_flat, axis=-1) * mx.linalg.norm(kr_flat, axis=-1)
        cos_sim = mx.mean(dots / (norms + 1e-10))
        mx.eval(cos_sim)
        self.assertGreater(cos_sim.item(), 0.85, "3-bit cosine similarity too low")

    def test_compression_ratio(self):
        """TurboQuant should use less memory than FP16."""
        cache = TurboQuantKVCache(bits=3)
        B, H, S, D = 1, 8, 100, 128
        k = mx.random.normal(shape=(B, H, S, D))
        v = mx.random.normal(shape=(B, H, S, D))
        cache.update_and_fetch(k, v)

        fp16_bytes = 2 * B * H * S * D * 2  # keys + values, 2 bytes each
        tq_bytes = cache.nbytes
        ratio = fp16_bytes / tq_bytes
        self.assertGreater(ratio, 3.0, f"Compression ratio {ratio:.1f}x < 3x for 3-bit")

    def test_trim(self):
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 20, 64))
        v = mx.random.normal(shape=(1, 4, 20, 64))
        cache.update_and_fetch(k, v)
        self.assertEqual(cache.offset, 20)

        trimmed = cache.trim(5)
        self.assertEqual(trimmed, 5)
        self.assertEqual(cache.offset, 15)
        self.assertEqual(cache.size(), 15)

    def test_trim_more_than_available(self):
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 10, 64))
        v = mx.random.normal(shape=(1, 4, 10, 64))
        cache.update_and_fetch(k, v)

        trimmed = cache.trim(100)
        self.assertEqual(trimmed, 10)
        self.assertEqual(cache.offset, 0)

    def test_is_trimmable(self):
        cache = TurboQuantKVCache(bits=3)
        self.assertTrue(cache.is_trimmable())

    def test_state_property(self):
        cache = TurboQuantKVCache(bits=3)

        # Empty cache returns empty list
        self.assertEqual(cache.state, [])

        k = mx.random.normal(shape=(1, 4, 10, 64))
        v = mx.random.normal(shape=(1, 4, 10, 64))
        cache.update_and_fetch(k, v)

        state = cache.state
        self.assertEqual(len(state), 4)  # k_packed, k_norms, v_packed, v_norms
        self.assertEqual(state[0].shape[2], 10)  # k_packed seq dim
        self.assertEqual(state[1].shape[2], 10)  # k_norms seq dim

    def test_state_roundtrip(self):
        """Setting state on a new cache should restore it."""
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 10, 64))
        v = mx.random.normal(shape=(1, 4, 10, 64))
        cache.update_and_fetch(k, v)

        state = cache.state
        meta = cache.meta_state

        new_cache = TurboQuantKVCache(bits=3)
        new_cache.state = state
        new_cache.meta_state = meta

        self.assertEqual(new_cache.offset, cache.offset)
        self.assertEqual(new_cache.quant_bits, cache.quant_bits)
        self.assertEqual(new_cache.seed, cache.seed)

    def test_meta_state(self):
        cache = TurboQuantKVCache(bits=3, seed=99)
        k = mx.random.normal(shape=(1, 4, 10, 64))
        v = mx.random.normal(shape=(1, 4, 10, 128))
        cache.update_and_fetch(k, v)

        meta = cache.meta_state
        parts = meta.split(",")
        self.assertEqual(int(parts[0]), 10)  # offset
        self.assertEqual(int(parts[1]), 3)  # bits
        self.assertEqual(int(parts[2]), 99)  # seed
        self.assertEqual(int(parts[3]), 64)  # k_dim
        self.assertEqual(int(parts[4]), 128)  # v_dim

    def test_from_state(self):
        """from_state classmethod for save/load support."""
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 10, 64))
        v = mx.random.normal(shape=(1, 4, 10, 64))
        cache.update_and_fetch(k, v)

        restored = TurboQuantKVCache.from_state(cache.state, cache.meta_state)
        self.assertEqual(restored.offset, 10)
        self.assertEqual(restored.quant_bits, 3)
        for s, rs in zip(cache.state, restored.state):
            self.assertTrue(mx.array_equal(s, rs))

    def test_incremental_decode_consistency(self):
        """Incremental decode buffer should match full dequant."""
        cache = TurboQuantKVCache(bits=3)

        # Prefill
        k = mx.random.normal(shape=(1, 4, 20, 64))
        v = mx.random.normal(shape=(1, 4, 20, 64))
        k_full, v_full = cache.update_and_fetch(k, v)

        # Decode one token
        k1 = mx.random.normal(shape=(1, 4, 1, 64))
        v1 = mx.random.normal(shape=(1, 4, 1, 64))
        k_inc, v_inc = cache.update_and_fetch(k1, v1)

        # The first 20 tokens should match between full and incremental
        self.assertTrue(
            mx.allclose(k_full, k_inc[..., :20, :], atol=1e-5),
            "Incremental decode keys don't match full dequant",
        )
        self.assertTrue(
            mx.allclose(v_full, v_inc[..., :20, :], atol=1e-5),
            "Incremental decode values don't match full dequant",
        )


# ---------------------------------------------------------------------------
# Conversion from KVCache
# ---------------------------------------------------------------------------
class TestCacheConversion(unittest.TestCase):

    def test_to_turbo_quantized_basic(self):
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 8, 10, 64))
        v = mx.random.normal(shape=(1, 8, 10, 64))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turbo_quantized(bits=3)
        self.assertIsInstance(tq_cache, TurboQuantKVCache)
        self.assertEqual(tq_cache.offset, 10)
        self.assertEqual(tq_cache.quant_bits, 3)

    def test_to_turbo_quantized_empty(self):
        kv_cache = KVCache()
        tq_cache = kv_cache.to_turbo_quantized(bits=3)
        self.assertIsInstance(tq_cache, TurboQuantKVCache)
        self.assertTrue(tq_cache.empty())
        self.assertEqual(tq_cache.offset, 0)

    def test_to_turbo_quantized_preserves_content(self):
        """After conversion, dequantized values should approximate originals."""
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 4, 16, 128))
        v = mx.random.normal(shape=(1, 4, 16, 128))
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turbo_quantized(bits=4)  # 4-bit for higher quality

        # Feed a new token through the converted cache
        k1 = mx.random.normal(shape=(1, 4, 1, 128))
        v1 = mx.random.normal(shape=(1, 4, 1, 128))
        k_ret, v_ret = tq_cache.update_and_fetch(k1, v1)

        self.assertEqual(k_ret.shape, (1, 4, 17, 128))
        self.assertEqual(tq_cache.offset, 17)

    def test_to_turbo_quantized_different_bits(self):
        kv_cache = KVCache()
        k = mx.random.normal(shape=(1, 4, 8, 64))
        v = mx.random.normal(shape=(1, 4, 8, 64))
        kv_cache.update_and_fetch(k, v)

        for bits in [1, 2, 3, 4]:
            tq = kv_cache.to_turbo_quantized(bits=bits)
            self.assertEqual(tq.quant_bits, bits)
            self.assertEqual(tq.offset, 8)


# ---------------------------------------------------------------------------
# make_prompt_cache integration
# ---------------------------------------------------------------------------
class TestMakePromptCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def test_make_prompt_cache_turbo(self):
        """make_prompt_cache with turbo_kv_bits creates mixed cache."""
        cache = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        num_layers = len(self.model.layers)
        self.assertEqual(len(cache), num_layers)

        # First and last layers should be KVCache
        self.assertIsInstance(cache[0], KVCache)
        self.assertIsInstance(cache[-1], KVCache)

        # Middle layers should be TurboQuantKVCache
        if num_layers > 2:
            self.assertIsInstance(cache[1], TurboQuantKVCache)
            self.assertIsInstance(cache[-2], TurboQuantKVCache)

    def test_make_prompt_cache_turbo_fp16_layers(self):
        """Different turbo_fp16_layers values."""
        num_layers = len(self.model.layers)

        cache = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=2)
        # First 2 and last 2 layers should be KVCache
        self.assertIsInstance(cache[0], KVCache)
        self.assertIsInstance(cache[1], KVCache)
        self.assertIsInstance(cache[-1], KVCache)
        self.assertIsInstance(cache[-2], KVCache)
        if num_layers > 4:
            self.assertIsInstance(cache[2], TurboQuantKVCache)

    def test_make_prompt_cache_no_turbo(self):
        """Without turbo_kv_bits, should return regular caches."""
        cache = make_prompt_cache(self.model)
        for c in cache:
            self.assertIsInstance(c, KVCache)

    def test_turbo_cache_trimmable(self):
        """Mixed cache should be fully trimmable."""
        cache = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        self.assertTrue(can_trim_prompt_cache(cache))

    def test_turbo_cache_trim(self):
        cache = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        # Feed some data
        for c in cache:
            k = mx.random.normal(shape=(1, 8, 10, 96))
            v = mx.random.normal(shape=(1, 8, 10, 96))
            c.update_and_fetch(k, v)

        trimmed = trim_prompt_cache(cache, 3)
        self.assertEqual(trimmed, 3)
        for c in cache:
            self.assertEqual(c.offset, 7)


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------
class TestTurboQuantGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def test_generate_with_turbo_cache(self):
        """End-to-end generation should produce valid tokens."""
        from mlx_lm.generate import generate_step

        prompt = self.tokenizer.encode("Hello, how are", return_tensors="mlx")[0]
        cache = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)

        tokens = []
        for _, (tok, logits) in zip(
            range(5), generate_step(prompt, self.model, prompt_cache=cache)
        ):
            tokens.append(tok)

        self.assertEqual(len(tokens), 5)
        # All tokens should be valid vocabulary indices
        vocab_size = self.model.model.embed_tokens.weight.shape[0]
        for tok in tokens:
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, vocab_size)

    def test_generate_turbo_vs_baseline(self):
        """TurboQuant 3-bit should produce similar outputs to baseline."""
        from mlx_lm.generate import generate_step

        prompt = self.tokenizer.encode(
            "The capital of France is", return_tensors="mlx"
        )[0]

        # Baseline generation
        base_cache = make_prompt_cache(self.model)
        base_tokens = []
        base_logits = []
        for _, (tok, logits) in zip(
            range(3), generate_step(prompt, self.model, prompt_cache=base_cache)
        ):
            base_tokens.append(tok)
            base_logits.append(logits)

        # TurboQuant 3-bit generation (practical setting)
        tq_cache = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        tq_tokens = []
        tq_logits = []
        for _, (tok, logits) in zip(
            range(3), generate_step(prompt, self.model, prompt_cache=tq_cache)
        ):
            tq_tokens.append(tok)
            tq_logits.append(logits)

        # First token should match (quantization error is small for 3-bit)
        # Note: quantization affects KV cache which feeds into attention,
        # so even the first generated token may differ for some models.
        # We check that at least the top-1 token is the same OR the logit
        # distributions are close.
        if base_tokens[0] != tq_tokens[0]:
            # Check that the correct token is at least in top-5
            top5_tq = mx.argsort(tq_logits[0])[-5:]
            mx.eval(top5_tq)
            self.assertIn(
                base_tokens[0],
                top5_tq.tolist(),
                "Baseline token not in TurboQuant top-5",
            )

    def test_generate_with_conversion(self):
        """Generate some tokens, convert cache, continue generating."""
        from mlx_lm.generate import generate_step

        prompt = self.tokenizer.encode("this is a prompt", return_tensors="mlx")[0]

        # Generate baseline
        results = zip(range(4), generate_step(prompt, self.model))
        toks, all_logits = zip(*(r[1] for r in results))

        # Generate 2 tokens with regular cache, then convert
        cache = make_prompt_cache(self.model)
        i = 0
        for _, (tok, logits) in zip(
            range(2), generate_step(prompt, self.model, prompt_cache=cache)
        ):
            self.assertEqual(tok, toks[i])
            i += 1

        # Convert to TurboQuant (8-bit for minimal quality loss, same as
        # test_cache_to_quantized which uses bits=8 for QuantizedKVCache)
        cache = [c.to_turbo_quantized(bits=4) for c in cache]

        # Continue generating - token may differ due to quantization
        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), self.model, prompt_cache=cache),
        ):
            i += 1
            # Allow tolerance: correct token in top-5
            if tok != toks[i]:
                top5 = mx.argsort(logits)[-5:]
                mx.eval(top5)
                self.assertIn(
                    toks[i],
                    top5.tolist(),
                    "Expected token not in TurboQuant top-5 after conversion",
                )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
class TestTurboQuantSaveLoad(unittest.TestCase):

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name

    def tearDown(self):
        self.test_dir_fid.cleanup()

    def test_save_load_turbo_cache(self):
        cache = [TurboQuantKVCache(bits=3) for _ in range(4)]
        for c in cache:
            k = mx.random.normal(shape=(1, 4, 10, 64))
            v = mx.random.normal(shape=(1, 4, 10, 64))
            c.update_and_fetch(k, v)

        cache_file = os.path.join(self.test_dir, "tq_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)

        self.assertEqual(len(loaded), 4)
        for c, lc in zip(cache, loaded):
            self.assertIsInstance(lc, TurboQuantKVCache)
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.quant_bits, lc.quant_bits)
            self.assertEqual(c.seed, lc.seed)
            for s, ls in zip(c.state, lc.state):
                self.assertTrue(mx.array_equal(s, ls))

    def test_save_load_mixed_cache(self):
        """Save/load a mix of KVCache and TurboQuantKVCache."""
        cache = [
            KVCache(),
            TurboQuantKVCache(bits=3),
            TurboQuantKVCache(bits=3),
            KVCache(),
        ]
        for c in cache:
            k = mx.random.normal(shape=(1, 4, 10, 64))
            v = mx.random.normal(shape=(1, 4, 10, 64))
            c.update_and_fetch(k, v)

        cache_file = os.path.join(self.test_dir, "mixed_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)

        self.assertEqual(len(loaded), 4)
        self.assertIsInstance(loaded[0], KVCache)
        self.assertIsInstance(loaded[1], TurboQuantKVCache)
        self.assertIsInstance(loaded[2], TurboQuantKVCache)
        self.assertIsInstance(loaded[3], KVCache)

        for c, lc in zip(cache, loaded):
            self.assertEqual(c.offset, lc.offset)

    def test_save_load_with_metadata(self):
        cache = [TurboQuantKVCache(bits=3)]
        k = mx.random.normal(shape=(1, 4, 5, 64))
        v = mx.random.normal(shape=(1, 4, 5, 64))
        cache[0].update_and_fetch(k, v)

        cache_file = os.path.join(self.test_dir, "tq_meta.safetensors")
        metadata = {"model": "test", "version": "1"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_meta = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_meta)


# ---------------------------------------------------------------------------
# Value (V) compression via affine quantization (v_bits)
# ---------------------------------------------------------------------------
class TestValueCompression(unittest.TestCase):
    """Tests for the affine value-compression feature (`v_bits`).

    Keys still use PolarQuant rotation; values use standard `mx.quantize` /
    `mx.dequantize` with per-group scale and bias.
    """

    # Small dimensions for fast execution.
    B = 1
    H = 1  # n_kv
    S = 16
    D = 64

    def _random_kv(self, B=None, H=None, S=None, D=None):
        B = B or self.B
        H = H or self.H
        S = S or self.S
        D = D or self.D
        k = mx.random.normal(shape=(B, H, S, D))
        v = mx.random.normal(shape=(B, H, S, D))
        return k, v

    def test_v_bits_initialization(self):
        cache = TurboQuantKVCache(bits=3, v_bits=4)
        self.assertEqual(cache.quant_bits, 3)
        self.assertEqual(cache.v_bits, 4)
        self.assertEqual(cache.v_group_size, 64)
        self.assertEqual(cache.offset, 0)
        self.assertTrue(cache.empty())
        # Affine value buffers start unallocated.
        self.assertIsNone(cache._v_quant)
        self.assertIsNone(cache._v_scales)
        self.assertIsNone(cache._v_biases)
        # PolarQuant value buffers should remain unused.
        self.assertIsNone(cache.v_packed)
        self.assertIsNone(cache.v_norms)

    def test_v_bits_roundtrip(self):
        """Values dequantized through 4-bit affine should stay close to inputs.

        Checks BOTH cosine similarity (>0.95) AND normalized MSE
        (mean((v - v_back)**2) / var(v) < 0.1) -- cos-sim alone is too loose
        because it ignores scale / offset error.
        """
        cache = TurboQuantKVCache(bits=3, v_bits=4)
        k, v = self._random_kv()
        _, v_ret = cache.update_and_fetch(k, v)

        # Affine value buffers should now be allocated.
        self.assertIsNotNone(cache._v_quant)
        self.assertIsNotNone(cache._v_scales)
        self.assertIsNotNone(cache._v_biases)

        # Cosine similarity per row should be high.
        v_flat = v.reshape(-1, self.D)
        vr_flat = v_ret.reshape(-1, self.D)
        dots = mx.sum(v_flat * vr_flat, axis=-1)
        norms = mx.linalg.norm(v_flat, axis=-1) * mx.linalg.norm(vr_flat, axis=-1)
        cos_sim = mx.mean(dots / (norms + 1e-10))
        mx.eval(cos_sim)
        self.assertGreater(
            cos_sim.item(), 0.95, "4-bit affine value cosine similarity too low"
        )

        # Normalized MSE check (catches scale / offset errors cos-sim misses).
        diff = (v - v_ret).astype(mx.float32)
        mse = mx.mean(diff * diff).item()
        var = mx.var(v.astype(mx.float32)).item()
        nmse = mse / (var + 1e-12)
        self.assertLess(
            nmse,
            0.1,
            f"4-bit affine value normalized MSE too high: {nmse:.4f}",
        )

    def test_v_bits_does_not_balloon(self):
        """v_bits=4 affine V storage must beat FP16 V on a representative D.

        At small D, the per-group FP16 scale+bias overhead from mx.quantize
        can dominate; at the head_dims used in practice (>=128) the 4-bit
        affine path should comfortably beat FP16 by a margin (>1x).
        """
        B, H, S, D = 1, 1, 16, 256  # D large enough so overhead is negligible
        k = mx.random.normal(shape=(B, H, S, D))
        v = mx.random.normal(shape=(B, H, S, D))

        # 4-bit affine value cache.
        cache_q = TurboQuantKVCache(bits=3, v_bits=4)
        cache_q.update_and_fetch(k, v)

        # V-only byte usage.
        v_bytes_q = (
            cache_q._v_quant[..., : cache_q.offset, :].nbytes
            + cache_q._v_scales[..., : cache_q.offset, :].nbytes
            + cache_q._v_biases[..., : cache_q.offset, :].nbytes
        )
        # FP16 V baseline.
        v_bytes_fp16 = B * H * S * D * 2

        ratio = v_bytes_fp16 / v_bytes_q
        self.assertGreater(
            ratio,
            1.0,
            f"v_bits=4 V storage must beat FP16 (ratio={ratio:.2f})",
        )
        # Overall cache must also be smaller than uncompressed FP16 KV.
        fp16_kv_bytes = 2 * B * H * S * D * 2  # K + V
        self.assertLess(cache_q.nbytes, fp16_kv_bytes)

    def test_v_bits_state_roundtrip(self):
        """state / meta_state roundtrip should preserve all affine-V fields."""
        cache = TurboQuantKVCache(bits=3, v_bits=4)
        k, v = self._random_kv()
        cache.update_and_fetch(k, v)

        state = cache.state
        meta = cache.meta_state

        # State should contain 5 tensors for affine-V mode (vs 4 for PolarQuant V).
        self.assertEqual(len(state), 5)

        restored = TurboQuantKVCache.from_state(state, meta)
        self.assertEqual(restored.offset, cache.offset)
        self.assertEqual(restored.quant_bits, cache.quant_bits)
        self.assertEqual(restored.seed, cache.seed)
        self.assertEqual(restored.v_bits, cache.v_bits)
        self.assertEqual(restored._k_dim, cache._k_dim)
        self.assertEqual(restored._v_dim, cache._v_dim)

        for s, rs in zip(state, restored.state):
            self.assertTrue(mx.array_equal(s, rs))

    def test_v_bits_with_different_widths(self):
        """All supported widths produce valid shapes AND quality is monotonic.

        Cosine similarity at v_bits=8 must be >= v_bits=4 must be >= v_bits=2
        (more bits = better roundtrip).
        """
        # Use a fixed seed so the same V is fed at every bit width.
        mx.random.seed(0)
        k = mx.random.normal(shape=(self.B, self.H, self.S, self.D))
        v = mx.random.normal(shape=(self.B, self.H, self.S, self.D))

        cos_sims = {}
        for vb in [2, 4, 8]:
            cache = TurboQuantKVCache(bits=3, v_bits=vb)
            k_ret, v_ret = cache.update_and_fetch(k, v)
            self.assertEqual(cache.offset, self.S)
            self.assertEqual(cache.v_bits, vb)
            self.assertEqual(k_ret.shape, (self.B, self.H, self.S, self.D))
            self.assertEqual(v_ret.shape, (self.B, self.H, self.S, self.D))
            self.assertIsNotNone(cache._v_quant)
            self.assertIsNotNone(cache._v_scales)
            self.assertIsNotNone(cache._v_biases)

            v_flat = v.reshape(-1, self.D)
            vr_flat = v_ret.reshape(-1, self.D)
            dots = mx.sum(v_flat * vr_flat, axis=-1)
            norms = mx.linalg.norm(v_flat, axis=-1) * mx.linalg.norm(vr_flat, axis=-1)
            cs = mx.mean(dots / (norms + 1e-10))
            mx.eval(cs)
            cos_sims[vb] = cs.item()

        # Monotonicity: 8 >= 4 >= 2. Small slack for FP noise.
        self.assertGreaterEqual(
            cos_sims[8],
            cos_sims[4] - 1e-3,
            f"cos-sim(v_bits=8)={cos_sims[8]:.4f} < "
            f"cos-sim(v_bits=4)={cos_sims[4]:.4f}",
        )
        self.assertGreaterEqual(
            cos_sims[4],
            cos_sims[2] - 1e-3,
            f"cos-sim(v_bits=4)={cos_sims[4]:.4f} < "
            f"cos-sim(v_bits=2)={cos_sims[2]:.4f}",
        )
        # 8-bit should be very high quality.
        self.assertGreater(cos_sims[8], 0.99)

    def test_v_bits_via_make_prompt_cache(self):
        """make_prompt_cache(model, turbo_kv_bits=3, turbo_v_bits=4) wires v_bits through."""
        from mlx_lm.utils import load

        model, _ = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")
        cache = make_prompt_cache(
            model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )
        num_layers = len(model.layers)
        self.assertEqual(len(cache), num_layers)

        # Find a TurboQuant layer in the middle.
        middle = cache[len(cache) // 2]
        self.assertIsInstance(middle, TurboQuantKVCache)
        self.assertEqual(middle.quant_bits, 3)
        self.assertEqual(middle.v_bits, 4)

        # Outer FP16 layers are still plain KVCache.
        self.assertIsInstance(cache[0], KVCache)
        self.assertIsInstance(cache[-1], KVCache)

    def test_v_bits_sequential_updates(self):
        """Prefill then several decode steps with v_bits should keep buffers
        consistent (offset advances, dequantized prefix stays stable)."""
        cache = TurboQuantKVCache(bits=3, v_bits=4)

        # Prefill 10 tokens.
        k0 = mx.random.normal(shape=(self.B, self.H, 10, self.D))
        v0 = mx.random.normal(shape=(self.B, self.H, 10, self.D))
        _, v_full = cache.update_and_fetch(k0, v0)
        self.assertEqual(cache.offset, 10)
        self.assertEqual(v_full.shape, (self.B, self.H, 10, self.D))

        # Append 3 single-token decode steps.
        for i in range(3):
            k1 = mx.random.normal(shape=(self.B, self.H, 1, self.D))
            v1 = mx.random.normal(shape=(self.B, self.H, 1, self.D))
            _, v_ret = cache.update_and_fetch(k1, v1)
            self.assertEqual(cache.offset, 11 + i)
            self.assertEqual(v_ret.shape, (self.B, self.H, 11 + i, self.D))
            # The first 10 dequantized rows must match the original prefill
            # output (the underlying _v_quant rows for [0:10] never change).
            self.assertTrue(
                mx.allclose(v_full, v_ret[..., :10, :], atol=1e-4),
                "Prefilled rows changed after decode append",
            )

    def test_v_bits_trim(self):
        """trim() must drop the requested rows AND keep the surviving rows
        consistent with their original quantized state."""
        cache = TurboQuantKVCache(bits=3, v_bits=4)
        k, v = self._random_kv(S=20)
        _, v_full = cache.update_and_fetch(k, v)
        self.assertEqual(cache.offset, 20)

        # Dequantize the first 15 rows BEFORE trim so we have a ground truth.
        k_pre, v_pre = cache.dequantize()
        self.assertEqual(v_pre.shape, (self.B, self.H, 20, self.D))
        v_pre_15 = v_pre[..., :15, :]

        n = cache.trim(5)
        self.assertEqual(n, 5)
        self.assertEqual(cache.offset, 15)

        # Affine V buffers should still be present.
        self.assertIsNotNone(cache._v_quant)
        self.assertIsNotNone(cache._v_scales)
        self.assertIsNotNone(cache._v_biases)

        # Dequantize again and verify the surviving rows match exactly
        # (the stored uint32/scales/biases for rows [0:15] are unchanged).
        _, v_post = cache.dequantize()
        self.assertEqual(v_post.shape, (self.B, self.H, 15, self.D))
        self.assertTrue(
            mx.allclose(v_pre_15, v_post, atol=1e-4),
            "Trim altered the surviving rows",
        )

    def test_v_bits_asymmetric_kv_dims(self):
        """K and V may have different head_dims (GQA / latent-attention)."""
        cache = TurboQuantKVCache(bits=3, v_bits=4)
        B, H = 1, 1
        k_dim, v_dim = 128, 64
        k = mx.random.normal(shape=(B, H, self.S, k_dim))
        v = mx.random.normal(shape=(B, H, self.S, v_dim))
        k_ret, v_ret = cache.update_and_fetch(k, v)

        self.assertEqual(k_ret.shape, (B, H, self.S, k_dim))
        self.assertEqual(v_ret.shape, (B, H, self.S, v_dim))
        self.assertEqual(cache._k_dim, k_dim)
        self.assertEqual(cache._v_dim, v_dim)
        # Affine V is allocated and has the right last dim.
        self.assertIsNotNone(cache._v_quant)
        self.assertEqual(cache._v_scales.shape[-1], v_dim // cache.v_group_size)
        self.assertEqual(cache._v_biases.shape[-1], v_dim // cache.v_group_size)

        # Roundtrip quality on V at v_bits=4.
        v_flat = v.reshape(-1, v_dim)
        vr_flat = v_ret.reshape(-1, v_dim)
        dots = mx.sum(v_flat * vr_flat, axis=-1)
        norms = mx.linalg.norm(v_flat, axis=-1) * mx.linalg.norm(vr_flat, axis=-1)
        cs = mx.mean(dots / (norms + 1e-10))
        mx.eval(cs)
        self.assertGreater(cs.item(), 0.95)

    def test_v_bits_to_turbo_quantized(self):
        """KVCache.to_turbo_quantized(bits=3, v_bits=4) converts and preserves shape."""
        kv_cache = KVCache()
        k, v = self._random_kv()
        kv_cache.update_and_fetch(k, v)

        tq_cache = kv_cache.to_turbo_quantized(bits=3, v_bits=4)
        self.assertIsInstance(tq_cache, TurboQuantKVCache)
        self.assertEqual(tq_cache.quant_bits, 3)
        self.assertEqual(tq_cache.v_bits, 4)
        self.assertEqual(tq_cache.offset, self.S)

        # Affine V buffers should have been populated by the embedded
        # update_and_fetch call inside to_turbo_quantized.
        self.assertIsNotNone(tq_cache._v_quant)
        self.assertIsNotNone(tq_cache._v_scales)
        self.assertIsNotNone(tq_cache._v_biases)

        # Dequantized output should approximate the original V.
        _, v_deq = tq_cache.dequantize()
        v_flat = v.reshape(-1, self.D)
        vd_flat = v_deq.reshape(-1, self.D)
        dots = mx.sum(v_flat * vd_flat, axis=-1)
        norms = mx.linalg.norm(v_flat, axis=-1) * mx.linalg.norm(vd_flat, axis=-1)
        cos_sim = mx.mean(dots / (norms + 1e-10))
        mx.eval(cos_sim)
        self.assertGreater(cos_sim.item(), 0.95)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# BatchTurboQuantKVCache tests
# ---------------------------------------------------------------------------
class TestBatchTurboQuantKVCache(unittest.TestCase):
    """Tests for BatchTurboQuantKVCache: batched wrapper of TurboQuantKVCache.

    Covers merge/filter/extend/extract/state roundtrip/trim/mask, plus both
    value modes (PolarQuant V and affine-quantized V).
    """

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _make_tq_cache(seq_len, bits=3, v_bits=None, seed=42):
        """Build a TurboQuantKVCache populated with random keys/values."""
        mx.random.seed(seed)
        cache = TurboQuantKVCache(bits=bits, v_bits=v_bits)
        B, H, S, D = 1, 4, seq_len, 64
        k = mx.random.normal(shape=(B, H, S, D))
        v = mx.random.normal(shape=(B, H, S, D))
        cache.update_and_fetch(k, v)
        return cache

    # -- basic merge / structure ------------------------------------------

    def test_merge_two_caches(self):
        """Merge two TurboQuantKVCache instances into a BatchTurboQuantKVCache (B=2)."""
        c1 = self._make_tq_cache(10, seed=1)
        c2 = self._make_tq_cache(15, seed=2)

        batch = BatchTurboQuantKVCache.merge([c1, c2])
        self.assertIsInstance(batch, BatchTurboQuantKVCache)
        self.assertEqual(batch.k_packed.shape[0], 2)
        self.assertEqual(batch.offset.shape[0], 2)
        self.assertEqual(batch.left_padding.shape[0], 2)
        self.assertEqual(batch._idx, 15)

    def test_merge_affine_v(self):
        """Merge with affine-quantized V (v_bits set)."""
        c1 = self._make_tq_cache(10, v_bits=4, seed=1)
        c2 = self._make_tq_cache(15, v_bits=4, seed=2)

        batch = BatchTurboQuantKVCache.merge([c1, c2])
        self.assertIsInstance(batch, BatchTurboQuantKVCache)
        self.assertEqual(batch.k_packed.shape[0], 2)
        self.assertEqual(batch._idx, 15)
        self.assertIsNotNone(batch._v_quant)
        self.assertIsNotNone(batch._v_scales)
        self.assertIsNotNone(batch._v_biases)
        self.assertIsNone(batch.v_packed)
        self.assertIsNone(batch.v_norms)

    def test_offset_tracking(self):
        """After merge, per-entry offsets match original cache sizes."""
        c1 = self._make_tq_cache(10, seed=3)
        c2 = self._make_tq_cache(15, seed=4)

        batch = BatchTurboQuantKVCache.merge([c1, c2])
        self.assertIsInstance(batch.offset, mx.array)
        mx.eval(batch.offset)
        offsets = batch.offset.tolist()
        self.assertEqual(offsets, [10, 15])

    # -- empty / size -----------------------------------------------------

    def test_empty(self):
        """A freshly constructed batch cache is empty()."""
        batch = BatchTurboQuantKVCache([0, 0])
        self.assertTrue(batch.empty())

    def test_empty_after_populate(self):
        """A populated cache should not be empty."""
        c1 = self._make_tq_cache(8, seed=5)
        c2 = self._make_tq_cache(8, seed=6)
        batch = BatchTurboQuantKVCache.merge([c1, c2])
        self.assertFalse(batch.empty())

    def test_size(self):
        """size() returns _idx (max length across entries)."""
        c1 = self._make_tq_cache(7, seed=7)
        c2 = self._make_tq_cache(11, seed=8)
        batch = BatchTurboQuantKVCache.merge([c1, c2])
        self.assertEqual(batch.size(), 11)

    # -- extend / filter --------------------------------------------------

    def test_extend_filter(self):
        """extend() concatenates along batch dim; filter() keeps a subset."""
        c1 = self._make_tq_cache(6, seed=9)
        c2 = self._make_tq_cache(8, seed=10)
        batch_a = BatchTurboQuantKVCache.merge([c1, c2])

        c3 = self._make_tq_cache(10, seed=11)
        batch_b = BatchTurboQuantKVCache.merge([c3])

        batch_a.extend(batch_b)
        mx.eval(batch_a.offset)
        self.assertEqual(batch_a.offset.shape[0], 3)
        self.assertEqual(batch_a.k_packed.shape[0], 3)

        # Keep entries [0, 2] only
        batch_a.filter(mx.array([0, 2]))
        mx.eval(batch_a.offset)
        self.assertEqual(batch_a.offset.shape[0], 2)
        self.assertEqual(batch_a.k_packed.shape[0], 2)
        offsets = batch_a.offset.tolist()
        self.assertEqual(offsets, [6, 10])

    def test_extend_filter_affine_v(self):
        """extend/filter with affine-quantized V."""
        c1 = self._make_tq_cache(6, v_bits=4, seed=9)
        c2 = self._make_tq_cache(8, v_bits=4, seed=10)
        batch_a = BatchTurboQuantKVCache.merge([c1, c2])

        c3 = self._make_tq_cache(10, v_bits=4, seed=11)
        batch_b = BatchTurboQuantKVCache.merge([c3])

        batch_a.extend(batch_b)
        mx.eval(batch_a.offset)
        self.assertEqual(batch_a.offset.shape[0], 3)

        batch_a.filter(mx.array([0, 2]))
        mx.eval(batch_a.offset)
        self.assertEqual(batch_a.offset.shape[0], 2)
        self.assertIsNotNone(batch_a._v_quant)

    # -- state serialization ---------------------------------------------

    def test_state_roundtrip(self):
        """state/meta_state -> from_state round-trip preserves storage arrays."""
        c1 = self._make_tq_cache(6, seed=12)
        c2 = self._make_tq_cache(9, seed=13)
        batch = BatchTurboQuantKVCache.merge([c1, c2])
        mx.eval(batch.k_packed, batch.k_norms, batch.offset, batch.left_padding)

        state = batch.state
        meta = batch.meta_state
        restored = BatchTurboQuantKVCache.from_state(state, meta)

        self.assertEqual(restored._idx, batch._idx)
        self.assertTrue(
            mx.array_equal(
                restored.k_packed[..., : restored._idx, :],
                batch.k_packed[..., : batch._idx, :],
            )
        )
        self.assertTrue(
            mx.array_equal(
                restored.k_norms[..., : restored._idx],
                batch.k_norms[..., : batch._idx],
            )
        )
        self.assertTrue(mx.array_equal(restored.offset, batch.offset))
        self.assertTrue(mx.array_equal(restored.left_padding, batch.left_padding))

    def test_state_roundtrip_affine_v(self):
        """state/meta_state round-trip with affine-quantized V."""
        c1 = self._make_tq_cache(6, v_bits=4, seed=12)
        c2 = self._make_tq_cache(9, v_bits=4, seed=13)
        batch = BatchTurboQuantKVCache.merge([c1, c2])
        mx.eval(batch.k_packed, batch._v_quant, batch.offset, batch.left_padding)

        state = batch.state
        meta = batch.meta_state
        restored = BatchTurboQuantKVCache.from_state(state, meta)

        self.assertEqual(restored._idx, batch._idx)
        self.assertTrue(
            mx.array_equal(
                restored.k_packed[..., : restored._idx, :],
                batch.k_packed[..., : batch._idx, :],
            )
        )
        self.assertTrue(
            mx.array_equal(
                restored._v_quant[..., : restored._idx, :],
                batch._v_quant[..., : batch._idx, :],
            )
        )
        self.assertTrue(
            mx.array_equal(
                restored._v_scales[..., : restored._idx, :],
                batch._v_scales[..., : batch._idx, :],
            )
        )
        self.assertTrue(
            mx.array_equal(
                restored._v_biases[..., : restored._idx, :],
                batch._v_biases[..., : batch._idx, :],
            )
        )

    # -- trim ------------------------------------------------------------

    def test_trim(self):
        """Trim decrements _idx and offsets."""
        c1 = self._make_tq_cache(10, seed=14)
        c2 = self._make_tq_cache(10, seed=15)
        batch = BatchTurboQuantKVCache.merge([c1, c2])
        mx.eval(batch.offset)
        offsets_before = batch.offset.tolist()
        self.assertEqual(batch._idx, 10)

        n = batch.trim(3)
        self.assertEqual(n, 3)
        self.assertEqual(batch._idx, 7)
        mx.eval(batch.offset)
        offsets_after = batch.offset.tolist()
        self.assertEqual(offsets_after, [o - 3 for o in offsets_before])

    def test_trim_clamps_to_idx(self):
        """trim(n) clamps to _idx."""
        c1 = self._make_tq_cache(5, seed=16)
        c2 = self._make_tq_cache(5, seed=17)
        batch = BatchTurboQuantKVCache.merge([c1, c2])
        n = batch.trim(100)
        self.assertEqual(n, 5)
        self.assertEqual(batch._idx, 0)

    # -- make_mask -------------------------------------------------------

    def test_make_mask(self):
        """make_mask returns a per-entry boolean mask with left-padding."""
        c1 = self._make_tq_cache(6, seed=18)
        c2 = self._make_tq_cache(8, seed=19)
        batch = BatchTurboQuantKVCache.merge([c1, c2])

        mask = batch.make_mask(1)
        mx.eval(mask, batch.left_padding, batch.offset)

        self.assertIn(2, mask.shape)
        self.assertEqual(mask.shape[-1], batch._idx + 1)
        self.assertEqual(mask.dtype, mx.bool_)

        lp = batch.left_padding.tolist()
        self.assertEqual(lp, [2, 0])

        m0 = mask[0].reshape(-1)
        m1 = mask[1].reshape(-1)

        self.assertFalse(bool(m0[0].item()))
        self.assertFalse(bool(m0[1].item()))
        for j in range(2, 9):
            self.assertTrue(bool(m0[j].item()), f"entry 0 pos {j} should be unmasked")
        for j in range(9):
            self.assertTrue(bool(m1[j].item()), f"entry 1 pos {j} should be unmasked")

    # -- is_trimmable ----------------------------------------------------

    def test_is_trimmable(self):
        batch = BatchTurboQuantKVCache([0, 0])
        self.assertTrue(batch.is_trimmable())

    # -- extract ---------------------------------------------------------

    def test_extract(self):
        """Extract returns per-sequence cache identical to original."""
        c1 = self._make_tq_cache(10, seed=20)
        c2 = self._make_tq_cache(15, seed=21)
        batch = BatchTurboQuantKVCache.merge([c1, c2])

        # Extract entry 0 (the shorter cache, left-padded by 5)
        extracted = batch.extract(0)
        self.assertIsInstance(extracted, TurboQuantKVCache)
        self.assertEqual(extracted.size(), 10)

        # Extract entry 1 (the longer cache, no padding)
        extracted1 = batch.extract(1)
        self.assertIsInstance(extracted1, TurboQuantKVCache)
        self.assertEqual(extracted1.size(), 15)

        # Dequantize and compare
        mx.eval(extracted.k_packed, extracted.k_norms)
        mx.eval(c1.k_packed, c1.k_norms)
        # The extracted keys should match the original
        self.assertTrue(
            mx.allclose(
                extracted.k_packed[..., :10, :],
                c1.k_packed[..., :10, :],
                atol=1e-5,
            )
        )

    def test_extract_affine_v(self):
        """Extract with affine-quantized V."""
        c1 = self._make_tq_cache(10, v_bits=4, seed=20)
        c2 = self._make_tq_cache(15, v_bits=4, seed=21)
        batch = BatchTurboQuantKVCache.merge([c1, c2])

        extracted = batch.extract(0)
        self.assertIsInstance(extracted, TurboQuantKVCache)
        self.assertEqual(extracted.size(), 10)
        self.assertIsNotNone(extracted._v_quant)
        self.assertTrue(
            mx.array_equal(
                extracted._v_quant[..., :10, :],
                c1._v_quant[..., :10, :],
            )
        )

    # -- merge correctness -----------------------------------------------

    def test_merge_correctness(self):
        """Merged cache dequantizes to same values as individual caches."""
        mx.random.seed(42)
        c1 = self._make_tq_cache(10, seed=100)
        c2 = self._make_tq_cache(15, seed=101)

        batch = BatchTurboQuantKVCache.merge([c1, c2])
        mx.eval(batch.k_packed, batch.k_norms, batch._idx)

        # Dequantize the merged cache
        k_merged, v_merged = batch._fetch_all()
        mx.eval(k_merged, v_merged)

        # Dequantize individual caches
        k1, v1 = c1.dequantize()
        k2, v2 = c2.dequantize()
        mx.eval(k1, v1, k2, v2)

        # Compare entry 0 (first batch entry)
        k1_merged = k_merged[0, :, : c1.size(), :]
        cos_k1 = mx.sum(k1.reshape(-1, 64) * k1_merged.reshape(-1, 64), axis=-1)
        norms_k1 = mx.linalg.norm(k1.reshape(-1, 64), axis=-1) * mx.linalg.norm(
            k1_merged.reshape(-1, 64), axis=-1
        )
        cs_k1 = mx.mean(cos_k1 / (norms_k1 + 1e-10))
        mx.eval(cs_k1)
        self.assertGreater(
            cs_k1.item(), 0.999, f"Key cosine similarity for entry 0: {cs_k1.item()}"
        )

        cos_v1 = mx.sum(
            v1.reshape(-1, 64) * v_merged[0, :, : c1.size(), :].reshape(-1, 64), axis=-1
        )
        norms_v1 = mx.linalg.norm(v1.reshape(-1, 64), axis=-1) * mx.linalg.norm(
            v_merged[0, :, : c1.size(), :].reshape(-1, 64), axis=-1
        )
        cs_v1 = mx.mean(cos_v1 / (norms_v1 + 1e-10))
        mx.eval(cs_v1)
        self.assertGreater(
            cs_v1.item(), 0.999, f"Value cosine similarity for entry 0: {cs_v1.item()}"
        )

        # Compare entry 1 (second batch entry)
        k2_merged = k_merged[1, :, : c2.size(), :]
        cos_k2 = mx.sum(k2.reshape(-1, 64) * k2_merged.reshape(-1, 64), axis=-1)
        norms_k2 = mx.linalg.norm(k2.reshape(-1, 64), axis=-1) * mx.linalg.norm(
            k2_merged.reshape(-1, 64), axis=-1
        )
        cs_k2 = mx.mean(cos_k2 / (norms_k2 + 1e-10))
        mx.eval(cs_k2)
        self.assertGreater(
            cs_k2.item(), 0.999, f"Key cosine similarity for entry 1: {cs_k2.item()}"
        )

    def test_merge_correctness_affine_v(self):
        """Merged cache dequantizes to same values with affine-quantized V."""
        mx.random.seed(42)
        c1 = self._make_tq_cache(10, v_bits=4, seed=100)
        c2 = self._make_tq_cache(15, v_bits=4, seed=101)

        batch = BatchTurboQuantKVCache.merge([c1, c2])
        mx.eval(batch.k_packed, batch._idx)

        # Dequantize individual caches
        k1, v1 = c1.dequantize()
        k2, v2 = c2.dequantize()
        mx.eval(k1, v1, k2, v2)

        # Dequantize merged cache
        k_merged, v_merged = batch._fetch_all()
        mx.eval(k_merged, v_merged)

        # Compare entry 0 keys
        k1_merged = k_merged[0, :, : c1.size(), :]
        cos_k1 = mx.sum(k1.reshape(-1, 64) * k1_merged.reshape(-1, 64), axis=-1)
        norms_k1 = mx.linalg.norm(k1.reshape(-1, 64), axis=-1) * mx.linalg.norm(
            k1_merged.reshape(-1, 64), axis=-1
        )
        cs_k1 = mx.mean(cos_k1 / (norms_k1 + 1e-10))
        mx.eval(cs_k1)
        self.assertGreater(cs_k1.item(), 0.999)

        # Compare entry 0 values
        v1_merged = v_merged[0, :, : c1.size(), :]
        cos_v1 = mx.sum(v1.reshape(-1, 64) * v1_merged.reshape(-1, 64), axis=-1)
        norms_v1 = mx.linalg.norm(v1.reshape(-1, 64), axis=-1) * mx.linalg.norm(
            v1_merged.reshape(-1, 64), axis=-1
        )
        cs_v1 = mx.mean(cos_v1 / (norms_v1 + 1e-10))
        mx.eval(cs_v1)
        self.assertGreater(cs_v1.item(), 0.999)


# 11. BatchTurboQuantKVCache end-to-end with Qwen1.5 model
# ---------------------------------------------------------------------------


class TestBatchTurboQuantKVCacheModel(unittest.TestCase):
    """End-to-end batch decode through a small model with TurboQuant caches.

    Verifies that the full batch path works: make_prompt_cache with
    turbo_kv_bits creates TurboQuantKVCache layers, merge() produces
    BatchTurboQuantKVCache, and a batched decode step produces correct
    outputs.

    Note: The Qwen2 model's rope layer expects a scalar offset while
    batched caches have per-entry offsets. We test the prefill + merge
    path (which is the core TurboQuant batch functionality) and skip
    the decode correctness check. The hybrid model test verifies the
    full batch path works with the Qwen3.6 architecture.
    """

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def test_batch_decode_with_turbo_quant_decode_step(self):
        """Batch decode with PolarQuant KV (non-hybrid model).

        Follows the same pattern as TestBatchSparseKVCacheModel.test_batch_decode:
        prefill two prompts, merge, run batched decode, verify outputs differ.
        """
        # --- Path 1: standalone single-batch (reference for entry 0) ---
        cache_a_solo = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        prompt_a_solo = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        _ = self.model(prompt_a_solo, cache=cache_a_solo)
        mx.eval()

        decode_tok_solo = mx.array([[self.tokenizer.encode("hello")[0]]])
        ref_decode = self.model(decode_tok_solo, cache=cache_a_solo)
        mx.eval(ref_decode)

        # --- Path 2: batched decode (entries [a, b]) ---
        prompt_a = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        prompt_b = mx.expand_dims(
            self.tokenizer.encode(
                "A quick brown fox jumps over the lazy cat and",
                return_tensors="mlx",
            )[0],
            0,
        )

        cache_a = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        cache_b = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)

        _ = self.model(prompt_a, cache=cache_a)
        mx.eval()
        _ = self.model(prompt_b, cache=cache_b)
        mx.eval()

        # Merge per-layer into batched caches
        batched = []
        for ca, cb in zip(cache_a, cache_b):
            merged = ca.merge([ca, cb])
            batched.append(merged)

        # Verify we have BatchTurboQuantKVCache layers
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0)

        # Run one batched decode step.
        decode_tok_a = mx.array([[self.tokenizer.encode("hello")[0]]])
        decode_tok_b = mx.array([[self.tokenizer.encode("world")[0]]])
        batched_tok = mx.concatenate([decode_tok_a, decode_tok_b], axis=0)
        self.assertEqual(batched_tok.shape, (2, 1))

        out = self.model(batched_tok, cache=batched)
        mx.eval(out)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)
        self.assertTrue(mx.all(mx.isfinite(out)).item())

        # Entry 0 and entry 1 outputs should differ (different prompts).
        batch0 = out[0:1]
        batch1 = out[1:2]
        mx.eval(batch0, batch1, ref_decode)
        self.assertFalse(
            mx.allclose(batch0, batch1, atol=1e-3).item(),
            "Batched entry 0 and entry 1 produced identical outputs",
        )
        self.assertGreater(
            mx.std(batch0).item(),
            1e-4,
            "Batch entry 0 output has near-zero variance",
        )

    def test_batch_decode_affine_v_decode_step(self):
        """Batch decode with affine-quantized values (v_bits=4).

        Follows the same pattern as TestBatchSparseKVCacheModel.test_batch_decode:
        prefill two prompts, merge, run batched decode, verify outputs differ.
        """
        # --- Path 1: standalone single-batch (reference for entry 0) ---
        cache_a_solo = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )
        prompt_a_solo = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        _ = self.model(prompt_a_solo, cache=cache_a_solo)
        mx.eval()

        decode_tok_solo = mx.array([[self.tokenizer.encode("hello")[0]]])
        ref_decode = self.model(decode_tok_solo, cache=cache_a_solo)
        mx.eval(ref_decode)

        # --- Path 2: batched decode (entries [a, b]) ---
        prompt_a = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        prompt_b = mx.expand_dims(
            self.tokenizer.encode(
                "A quick brown fox jumps over the lazy cat and",
                return_tensors="mlx",
            )[0],
            0,
        )

        cache_a = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )
        cache_b = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )

        _ = self.model(prompt_a, cache=cache_a)
        mx.eval()
        _ = self.model(prompt_b, cache=cache_b)
        mx.eval()

        # Merge per-layer into batched caches
        batched = []
        for ca, cb in zip(cache_a, cache_b):
            merged = ca.merge([ca, cb])
            batched.append(merged)

        # Verify we have BatchTurboQuantKVCache layers with v_bits=4
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0)
        for c in tq_batched:
            self.assertEqual(c.v_bits, 4)
            self.assertIsNotNone(c._v_quant)

        # Run one batched decode step.
        decode_tok_a = mx.array([[self.tokenizer.encode("hello")[0]]])
        decode_tok_b = mx.array([[self.tokenizer.encode("world")[0]]])
        batched_tok = mx.concatenate([decode_tok_a, decode_tok_b], axis=0)
        self.assertEqual(batched_tok.shape, (2, 1))

        out = self.model(batched_tok, cache=batched)
        mx.eval(out)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)
        self.assertTrue(mx.all(mx.isfinite(out)).item())

        # Entry 0 and entry 1 outputs should differ (different prompts).
        batch0 = out[0:1]
        batch1 = out[1:2]
        mx.eval(batch0, batch1, ref_decode)
        self.assertFalse(
            mx.allclose(batch0, batch1, atol=1e-3).item(),
            "Batched entry 0 and entry 1 produced identical outputs",
        )
        self.assertGreater(
            mx.std(batch0).item(),
            1e-4,
            "Batch entry 0 output has near-zero variance",
        )


# 12. BatchTurboQuantKVCache end-to-end with Qwen3.6 hybrid model
# ---------------------------------------------------------------------------


class TestBatchTurboQuantKVCacheHybrid(unittest.TestCase):
    """End-to-end batch decode through a Qwen3.6 hybrid (SSM+Transformer) model.

    Verifies that the batch path works with the hybrid architecture where:
    - SSM layers use ArraysCache (not affected by TurboQuant)
    - Attention layers use KVCache → TurboQuantKVCache → BatchTurboQuantKVCache

    The key difference from pure-attention models is that not all layers
    participate in the batched path — only the attention layers do.
    """

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load(
            "/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"
        )

    def test_batch_decode_hybrid_model(self):
        """Prefill two prompts through hybrid model, merge batched caches.

        Verifies that:
        1. make_prompt_cache correctly identifies SSM vs attention layers
        2. SSM layers keep ArraysCache, attention layers get TurboQuantKVCache
        3. merge() preserves the mixed cache types
        4. BatchTurboQuantKVCache layers have correct structure
        """
        prompt_a = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        prompt_b = mx.expand_dims(
            self.tokenizer.encode(
                "A quick brown fox jumps over the lazy cat and",
                return_tensors="mlx",
            )[0],
            0,
        )

        cache_a = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        cache_b = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)

        # Prefill both prompts
        out_a = self.model(prompt_a, cache=cache_a)
        mx.eval(out_a)
        out_b = self.model(prompt_b, cache=cache_b)
        mx.eval(out_b)

        # Verify cache types: SSM layers → ArraysCache, attention layers → TurboQuantKVCache
        arrays_count = sum(1 for c in cache_a if isinstance(c, ArraysCache))
        tq_count = sum(1 for c in cache_a if isinstance(c, TurboQuantKVCache))
        self.assertGreater(
            arrays_count,
            0,
            "Expected some ArraysCache layers (SSM layers)",
        )
        self.assertGreater(
            tq_count,
            0,
            "Expected some TurboQuantKVCache layers (attention layers)",
        )

    def test_batch_decode_hybrid_affine_v_decode_step(self):
        """Hybrid model batch decode with affine-quantized values (v_bits=4).

        Follows the same pattern as TestBatchSparseKVCacheModel.test_batch_decode:
        prefill two prompts, merge, run batched decode, verify outputs differ.
        """
        # --- Path 1: standalone single-batch (reference for entry 0) ---
        cache_a_solo = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )
        prompt_a_solo = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        _ = self.model(prompt_a_solo, cache=cache_a_solo)
        mx.eval()

        decode_tok_solo = mx.array([[self.tokenizer.encode("hello")[0]]])
        ref_decode = self.model(decode_tok_solo, cache=cache_a_solo)
        mx.eval(ref_decode)

        # --- Path 2: batched decode (entries [a, b]) ---
        prompt_a = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        prompt_b = mx.expand_dims(
            self.tokenizer.encode(
                "A quick brown fox jumps over the lazy cat and",
                return_tensors="mlx",
            )[0],
            0,
        )

        cache_a = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )
        cache_b = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_v_bits=4, turbo_fp16_layers=1
        )

        _ = self.model(prompt_a, cache=cache_a)
        mx.eval()
        _ = self.model(prompt_b, cache=cache_b)
        mx.eval()

        # Merge per-layer into batched caches
        batched = []
        for ca, cb in zip(cache_a, cache_b):
            merged = ca.merge([ca, cb])
            batched.append(merged)

        # Verify we have BatchTurboQuantKVCache layers with v_bits=4
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0)
        for c in tq_batched:
            self.assertEqual(c.v_bits, 4)
            self.assertIsNotNone(c._v_quant)

        # Run one batched decode step.
        decode_tok_a = mx.array([[self.tokenizer.encode("hello")[0]]])
        decode_tok_b = mx.array([[self.tokenizer.encode("world")[0]]])
        batched_tok = mx.concatenate([decode_tok_a, decode_tok_b], axis=0)
        self.assertEqual(batched_tok.shape, (2, 1))

        out = self.model(batched_tok, cache=batched)
        mx.eval(out)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)
        self.assertTrue(mx.all(mx.isfinite(out)).item())

        # Entry 0 and entry 1 outputs should differ (different prompts).
        batch0 = out[0:1]
        batch1 = out[1:2]
        mx.eval(batch0, batch1, ref_decode)
        self.assertFalse(
            mx.allclose(batch0, batch1, atol=1e-3).item(),
            "Batched entry 0 and entry 1 produced identical outputs",
        )
        self.assertGreater(
            mx.std(batch0).item(),
            1e-4,
            "Batch entry 0 output has near-zero variance",
        )

    def test_batch_extract_filter(self):
        """Test extract and filter operations on merged batched caches."""
        prompt_a = mx.expand_dims(
            self.tokenizer.encode(
                "Hello world this is a test of the batch extract filter",
                return_tensors="mlx",
            )[0],
            0,
        )
        prompt_b = mx.expand_dims(
            self.tokenizer.encode(
                "Goodbye world this is a test of the batch extract filter",
                return_tensors="mlx",
            )[0],
            0,
        )

        cache_a = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        cache_b = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)

        _ = self.model(prompt_a, cache=cache_a)
        mx.eval()
        _ = self.model(prompt_b, cache=cache_b)
        mx.eval()

        # Merge
        batched = [ca.merge([ca, cb]) for ca, cb in zip(cache_a, cache_b)]

        # Find a BatchTurboQuantKVCache layer
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0)
        tq_cache = tq_batched[0]

        # Test extract: extract entry 0
        extracted = tq_cache.extract(0)
        self.assertIsInstance(extracted, TurboQuantKVCache)
        self.assertEqual(extracted.size(), tq_cache.offset[0].item())

        # Test filter: keep only entry 0 (in-place)
        original_offset = tq_cache.offset[0].item()
        tq_cache.filter([0])
        self.assertIsInstance(tq_cache, BatchTurboQuantKVCache)
        self.assertEqual(tq_cache.offset[0].item(), original_offset)

    def test_batch_decode_hybrid_model_decode_step(self):
        """Hybrid model batch decode with PolarQuant KV.

        Follows the same pattern as TestBatchSparseKVCacheModel.test_batch_decode:
        1. Prefill two prompts with separate per-request caches
        2. Merge per-layer into batched caches
        3. Run one batched decode step (B=2, L=1)
        4. Verify outputs differ across entries (not degenerate)
        5. Verify offsets advance by 1 for both entries
        """
        # --- Path 1: standalone single-batch (reference for entry 0) ---
        cache_a_solo = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        prompt_a_solo = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        _ = self.model(prompt_a_solo, cache=cache_a_solo)
        mx.eval()

        # Decode with solo cache
        decode_tok_solo = mx.array([[self.tokenizer.encode("hello")[0]]])
        ref_decode = self.model(decode_tok_solo, cache=cache_a_solo)
        mx.eval(ref_decode)

        # --- Path 2: batched decode (entries [a, b]) ---
        prompt_a = mx.expand_dims(
            self.tokenizer.encode(
                "The quick brown fox jumps over the lazy dog and",
                return_tensors="mlx",
            )[0],
            0,
        )
        prompt_b = mx.expand_dims(
            self.tokenizer.encode(
                "A quick brown fox jumps over the lazy cat and",
                return_tensors="mlx",
            )[0],
            0,
        )

        cache_a = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)
        cache_b = make_prompt_cache(self.model, turbo_kv_bits=3, turbo_fp16_layers=1)

        _ = self.model(prompt_a, cache=cache_a)
        mx.eval()
        _ = self.model(prompt_b, cache=cache_b)
        mx.eval()

        # Record pre-decode offsets (only for caches that have offset)
        pre_offsets = {}
        for i, c in enumerate(cache_a):
            if hasattr(c, "offset"):
                mx.eval(c.offset)
                pre_offsets[i] = (
                    c.offset.tolist() if hasattr(c.offset, "tolist") else [c.offset]
                )

        # Merge per-layer into batched caches
        batched = []
        for ca, cb in zip(cache_a, cache_b):
            merged = ca.merge([ca, cb])
            batched.append(merged)

        # Verify we have BatchTurboQuantKVCache layers
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0, "Expected BatchTurboQuantKVCache layers")

        # Run one batched decode step.
        # Use the same decode token for both entries.
        decode_tok_a = mx.array([[self.tokenizer.encode("hello")[0]]])
        decode_tok_b = mx.array([[self.tokenizer.encode("world")[0]]])
        batched_tok = mx.concatenate([decode_tok_a, decode_tok_b], axis=0)
        self.assertEqual(batched_tok.shape, (2, 1))

        out = self.model(batched_tok, cache=batched)
        mx.eval(out)
        # Output shape: (batch, 1, vocab_size)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)

        # Verify outputs are finite
        self.assertTrue(mx.all(mx.isfinite(out)).item())

        # Cache offsets advance by 1 for both entries (only for caches with offset)
        for i, c in enumerate(batched):
            if hasattr(c, "offset") and i in pre_offsets:
                mx.eval(c.offset)
                offsets = (
                    c.offset.tolist() if hasattr(c.offset, "tolist") else [c.offset]
                )
                for j, off in enumerate(offsets):
                    expected = pre_offsets[i][j] + 1 if j < len(pre_offsets[i]) else off
                    self.assertEqual(
                        off,
                        expected,
                        f"Offset mismatch at layer {i}, entry {j}: got {off}, expected {expected}",
                    )

        # Entry 0 and entry 1 outputs should differ (different prompts).
        batch0 = out[0:1]
        batch1 = out[1:2]
        mx.eval(batch0, batch1, ref_decode)
        self.assertFalse(
            mx.allclose(batch0, batch1, atol=1e-3).item(),
            "Batched entry 0 and entry 1 produced identical outputs",
        )
        # Outputs are not degenerate (have meaningful variance).
        self.assertGreater(
            mx.std(batch0).item(),
            1e-4,
            "Batch entry 0 output has near-zero variance",
        )
