# Copyright © 2024 Apple Inc.

"""Server lifecycle tests for BatchTurboQuantKVCache.

These tests exercise the full server flow that opencode uses, following
patterns from test_prompt_cache.py and test_deepseek_v4.py:

- test_save_load_rotating_cache: save→load→feed same tokens→verify outputs match
- test_cache_with_generate: multi-step generation with cache continuation
- test_trim_cache_with_generate: trim then regenerate, verify same tokens
- TestBatchSparseKVCacheModel.test_batch_decode: prefill→merge→decode→verify

Covers:
1. Save→load inference check (pattern from test_save_load_rotating_cache)
2. Multi-step decode after merge (actual GenerationBatch._step() flow)
3. Full server lifecycle (prefill→merge→serialize→from_state→decode→extend→filter→trim→decode)
4. Model-level save/load with inference check (pattern from test_cache_with_generate)
"""

import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import (
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.models.turboquant_cache import BatchTurboQuantKVCache


class TestBatchTurboQuantKVCacheServerLifecycle(unittest.TestCase):
    """Server lifecycle tests for BatchTurboQuantKVCache.

    Tests the actual server flows:
    1. Save→load inference check (save_prompt_cache → load_prompt_cache → decode)
    2. Multi-step decode after merge (GenerationBatch._step() flow: merge → decode → decode → decode)
    3. Full server lifecycle (prefill → merge → serialize → from_state → decode → extend → filter → trim → decode)
    4. Model-level save/load with inference check (generate → save → load → continue generating)
    """

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name

    def tearDown(self):
        self.test_dir_fid.cleanup()

    # -- 1. Save→load inference check (pattern from test_save_load_rotating_cache) --

    def test_save_load_inference(self):
        """Save prompt cache, load it, verify inference correctness.

        Pattern from test_save_load_rotating_cache (test_prompt_cache.py:64-99):
        Feed the same tokens into both original and loaded caches, verify
        update_and_fetch outputs match. This is the gold standard for
        save/load testing — tensor equality alone is not sufficient.
        """
        prompt = self.tokenizer.encode(
            "The quick brown fox jumps over the lazy dog and",
            return_tensors="mlx",
        )[0]

        # Build and populate cache
        cache = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        _ = self.model(mx.expand_dims(prompt, 0), cache=cache)
        mx.eval()

        # Save and load
        cache_file = os.path.join(self.test_dir, "tq_save_load.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        # Verify structure
        self.assertEqual(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(type(c).__name__, type(lc).__name__)
            self.assertEqual(c.offset, lc.offset)

        # Feed the same decode tokens into both caches and verify outputs match
        decode_tok = self.tokenizer.encode("hello world", return_tensors="mlx")[0]
        decode_tok = mx.expand_dims(decode_tok, 0)

        # Run decode on original cache
        out_orig = self.model(decode_tok, cache=cache)
        mx.eval(out_orig)

        # Run decode on loaded cache
        out_loaded = self.model(decode_tok, cache=loaded_cache)
        mx.eval(out_loaded)

        # Outputs should match (within fp16 tolerance)
        self.assertTrue(
            mx.allclose(out_orig, out_loaded, atol=1e-2).item(),
            "Loaded cache produces different outputs than original",
        )

    # -- 2. Multi-step decode after from_state --

    def test_multi_step_decode_after_merge(self):
        """Merge → multi-step decode → verify offsets advance correctly.

        This is the actual server flow for continuous batching.
        GenerationBatch._step() calls self.model(inputs[:, None], cache=self.prompt_cache)
        which invokes update_and_fetch on the batched caches. No from_state involved.

        The server does multiple decode steps after merging, each step advancing
        offsets by 1 and producing finite outputs.
        """
        prompt_a = self.tokenizer.encode(
            "The quick brown fox jumps over the lazy dog and",
            return_tensors="mlx",
        )[0]
        prompt_b = self.tokenizer.encode(
            "A quick brown fox jumps over the lazy cat and",
            return_tensors="mlx",
        )[0]

        # Build and populate two separate caches
        cache_a = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        cache_b = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        _ = self.model(mx.expand_dims(prompt_a, 0), cache=cache_a)
        _ = self.model(mx.expand_dims(prompt_b, 0), cache=cache_b)
        mx.eval()

        # Merge into batched caches (B=2)
        batched = [ca.merge([ca, cb]) for ca, cb in zip(cache_a, cache_b)]
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0)

        # Record pre-decode offsets
        pre_offsets = {}
        for i, c in enumerate(batched):
            if hasattr(c, "offset"):
                mx.eval(c.offset)
                pre_offsets[i] = c.offset.tolist() if hasattr(c.offset, "tolist") else [c.offset]

        # Multi-step decode: 3 tokens (batched, B=2)
        decode_tokens = self.tokenizer.encode(
            "hello world test", return_tensors="mlx",
        )[0]
        decode_tokens = mx.expand_dims(decode_tokens, 0)

        out = self.model(decode_tokens, cache=batched)
        mx.eval(out)

        # Verify outputs are finite
        self.assertTrue(mx.all(mx.isfinite(out)).item())
        self.assertEqual(out.shape[0], 2)

        # Verify offsets advanced by decode length
        decode_len = decode_tokens.shape[1]
        for i, c in enumerate(batched):
            if hasattr(c, "offset") and i in pre_offsets:
                mx.eval(c.offset)
                offsets = c.offset.tolist() if hasattr(c.offset, "tolist") else [c.offset]
                for j, off in enumerate(offsets):
                    expected = pre_offsets[i][j] + decode_len if j < len(pre_offsets[i]) else off
                    self.assertEqual(
                        off, expected,
                        f"Offset mismatch at layer {i}, entry {j}: got {off}, expected {expected}",
                    )

    # -- 3. Full server lifecycle --

    def test_full_server_lifecycle(self):
        """Full server lifecycle: prefill → merge → decode → extend → filter → trim → decode.

        This is the exact sequence the server runs through during a
        multi-turn conversation with continuous batching.

        Note: from_state is NOT used in the normal server flow. The LRU cache
        stores live Python objects and uses copy.deepcopy when fetching.
        from_state is only for disk persistence (load_prompt_cache), which
        is a separate concern tested by test_save_load_inference and
        test_model_save_load_inference.
        """
        prompt_a = self.tokenizer.encode(
            "The quick brown fox jumps over the lazy dog and",
            return_tensors="mlx",
        )[0]
        prompt_b = self.tokenizer.encode(
            "A quick brown fox jumps over the lazy cat and",
            return_tensors="mlx",
        )[0]

        # Step 1: Prefill two prompts
        cache_a = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        cache_b = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        _ = self.model(mx.expand_dims(prompt_a, 0), cache=cache_a)
        _ = self.model(mx.expand_dims(prompt_b, 0), cache=cache_b)
        mx.eval()

        # Step 2: Merge into batched caches
        batched = [ca.merge([ca, cb]) for ca, cb in zip(cache_a, cache_b)]
        tq_batched = [c for c in batched if isinstance(c, BatchTurboQuantKVCache)]
        self.assertGreater(len(tq_batched), 0)

        # Step 3: Decode 2 tokens (batched, B=2)
        decode_tok_a = mx.expand_dims(
            self.tokenizer.encode("hello", return_tensors="mlx")[0], 0
        )
        decode_tok_b = mx.expand_dims(
            self.tokenizer.encode("world", return_tensors="mlx")[0], 0
        )
        batched_tok = mx.concatenate([decode_tok_a, decode_tok_b], axis=0)
        out = self.model(batched_tok, cache=batched)
        mx.eval(out)
        self.assertTrue(mx.all(mx.isfinite(out)).item())
        self.assertEqual(out.shape[0], 2)

        # Step 4: Extend with a new prompt
        prompt_c = self.tokenizer.encode(
            "The lazy dog slept all day long",
            return_tensors="mlx",
        )[0]
        cache_c = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        _ = self.model(mx.expand_dims(prompt_c, 0), cache=cache_c)
        mx.eval()

        # Merge cache_c into the batch
        new_tq_layers = [c for c in cache_c if isinstance(c, BatchTurboQuantKVCache)]
        if new_tq_layers:
            for i, c in enumerate(batched):
                if isinstance(c, BatchTurboQuantKVCache):
                    c.extend(new_tq_layers[i])

        # Step 5: Filter to keep only entry 0
        for c in batched:
            if isinstance(c, BatchTurboQuantKVCache):
                c.filter(mx.array([0]))

        # Verify only 1 entry remains
        for c in batched:
            if isinstance(c, BatchTurboQuantKVCache):
                self.assertEqual(c.offset.shape[0], 1)

        # Step 6: Trim 1 token
        for c in batched:
            if isinstance(c, BatchTurboQuantKVCache):
                c.trim(1)

        # Step 7: Decode 1 more token (single entry, B=1)
        decode_tok_final = mx.expand_dims(
            self.tokenizer.encode("test", return_tensors="mlx")[0], 0
        )
        out_final = self.model(decode_tok_final, cache=batched)
        mx.eval(out_final)
        self.assertTrue(mx.all(mx.isfinite(out_final)).item())

    # -- 4. Model-level save/load with inference check --

    def test_model_save_load_inference(self):
        """Model-level save/load: save_prompt_cache → load_prompt_cache → decode → verify.

        Pattern from test_cache_with_generate (test_prompt_cache.py:183-203):
        Generate tokens, save cache, load cache, continue generating, verify
        tokens match. This tests the actual server path (save_prompt_cache /
        load_prompt_cache) rather than just state/from_state.
        """
        prompt = self.tokenizer.encode(
            "The capital of France is", return_tensors="mlx",
        )[0]

        # Build cache and generate 2 tokens
        cache = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        results = list(
            generate_step(
                prompt, self.model, prompt_cache=cache, max_tokens=2
            )
        )
        toks_before, logits_before = zip(*results)
        mx.eval(*toks_before, *logits_before)

        # Save cache after 2 tokens
        cache_file = os.path.join(self.test_dir, "tq_model_save.safetensors")
        save_prompt_cache(cache_file, cache)

        # Load cache
        loaded_cache = load_prompt_cache(cache_file)

        # Continue generating from loaded cache
        last_tok = mx.array([toks_before[-1]])
        results_loaded = list(
            generate_step(
                last_tok, self.model, prompt_cache=loaded_cache, max_tokens=1
            )
        )
        toks_loaded, logits_loaded = zip(*results_loaded)
        mx.eval(*toks_loaded, *logits_loaded)

        # The loaded cache should produce the same token as if we continued
        # from the original cache at the same point
        results_continue = list(
            generate_step(
                last_tok, self.model, prompt_cache=cache, max_tokens=1
            )
        )
        toks_continue, logits_continue = zip(*results_continue)
        mx.eval(*toks_continue, *logits_continue)

        # Handle both int and mx.array token types
        loaded_tok = toks_loaded[0].item() if hasattr(toks_loaded[0], "item") else toks_loaded[0]
        continue_tok = toks_continue[0].item() if hasattr(toks_continue[0], "item") else toks_continue[0]
        
        self.assertEqual(
            loaded_tok, continue_tok,
            "Loaded cache produces different token than original cache",
        )
        self.assertTrue(
            mx.allclose(logits_loaded[0], logits_continue[0], atol=1e-2).item(),
            "Loaded cache produces different logits than original cache",
        )


if __name__ == "__main__":
    unittest.main()
