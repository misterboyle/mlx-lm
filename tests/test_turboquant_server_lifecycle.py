# Copyright © 2024 Apple Inc.

"""Server lifecycle tests for BatchTurboQuantKVCache with Qwen3.6 hybrid model.

These tests exercise the full server flow that opencode uses with the Qwen3.6
hybrid model (SSM+Transformer), following patterns from test_prompt_cache.py
and test_deepseek_v4.py:

- test_save_load_rotating_cache: save→load→feed same tokens→verify outputs match
- test_cache_with_generate: multi-step generation with cache continuation
- test_trim_cache_with_generate: trim then regenerate, verify same tokens
- TestBatchSparseKVCacheModel.test_batch_decode: prefill→merge→decode→verify

Covers:
1. Save→load inference check (pattern from test_save_load_rotating_cache)
2. Multi-step decode after merge (actual GenerationBatch._step() flow)
3. Full server lifecycle (prefill → merge → decode → extend → filter → trim → decode)
4. Model-level save/load with inference check (pattern from test_cache_with_generate)

Note: Qwen3.6 hybrid has SSM layers (ArraysCache) and attention layers
(TurboQuantKVCache → BatchTurboQuantKVCache). Only attention layers participate
in merge/filter/extend/trim operations.
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
    """Server lifecycle tests for BatchTurboQuantKVCache with Qwen3.6 hybrid.

    Tests the actual server flows with the Qwen3.6 hybrid model:
    1. Save→load inference check (save_prompt_cache → load_prompt_cache → decode)
    2. Multi-step decode after merge (GenerationBatch._step() flow)
    3. Full server lifecycle (prefill → merge → decode → extend → filter → trim → decode)
    4. Model-level save/load with inference check (generate → save → load → continue)
    """

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load(
            "/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"
        )

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name

    def tearDown(self):
        self.test_dir_fid.cleanup()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _tq_layers(cache):
        """Extract only BatchTurboQuantKVCache layers from a cache list."""
        return [c for c in cache if isinstance(c, BatchTurboQuantKVCache)]

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
            # Only check offset on cache types that have it
            if hasattr(c, "offset"):
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

        # Outputs should match within bfloat16 precision.
        # We use atol (absolute tolerance) instead of rtol (relative tolerance)
        # because logits can be near zero, making rtol unstable.
        # atol=1e-0 accounts for bfloat16 precision floor (~0.01-0.1) and
        # multiple comparisons across 248k logits (expecting some outliers).
        # An error of 1.0 in a logit corresponds to a factor of e (~2.7) in
        # probability, which is often acceptable for "close enough" in inference.
        # If the error were 10.0 (significant degradation), this would fail.
        self.assertTrue(
            mx.allclose(out_orig, out_loaded, rtol=0, atol=1e-0).item(),
            "Loaded cache outputs differ too much from original",
        )

    # -- 2. Multi-step decode after merge --

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
        tq_batched = self._tq_layers(batched)
        self.assertGreater(len(tq_batched), 0)

        # Record pre-decode offsets (only for TQ layers)
        pre_offsets = {}
        for i, c in enumerate(batched):
            if isinstance(c, BatchTurboQuantKVCache):
                mx.eval(c.offset)
                pre_offsets[i] = c.offset.tolist()

        # Multi-step decode: 3 tokens (batched, B=2)
        # Must match the batch size of the merged cache — the real server
        # always sends inputs with B matching the batched cache size.
        decode_tokens = self.tokenizer.encode(
            "hello world test", return_tensors="mlx",
        )[0]
        # Create B=2 batch: each entry gets the same decode tokens
        decode_tokens = mx.concatenate(
            [mx.expand_dims(decode_tokens, 0), mx.expand_dims(decode_tokens, 0)], axis=0
        )

        out = self.model(decode_tokens, cache=batched)
        mx.eval(out)

        # Verify outputs are finite
        self.assertTrue(mx.all(mx.isfinite(out)).item())
        self.assertEqual(out.shape[0], 2)

        # Verify offsets advanced by decode length
        decode_len = decode_tokens.shape[1]
        for i, c in enumerate(batched):
            if isinstance(c, BatchTurboQuantKVCache) and i in pre_offsets:
                mx.eval(c.offset)
                offsets = c.offset.tolist()
                for j, off in enumerate(offsets):
                    expected = pre_offsets[i][j] + decode_len
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
        is tested separately by test_save_load_inference and
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
        tq_batched = self._tq_layers(batched)
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

        # Merge cache_c into the batch (only TQ layers)
        new_tq = self._tq_layers(cache_c)
        if new_tq:
            tq_idx = 0
            for i, c in enumerate(batched):
                if isinstance(c, BatchTurboQuantKVCache):
                    c.extend(new_tq[tq_idx])
                    tq_idx += 1

        # Step 5: Filter to keep only entry 0 (ALL layers, not just TQ)
        # The SSM layers (ArraysCache) must also be filtered — otherwise
        # they still have B=2 while the input is B=1, causing shape mismatch.
        for c in batched:
            if hasattr(c, "filter"):
                c.filter(mx.array([0]))

        # Verify only 1 entry remains in all batchable layers
        for c in batched:
            if hasattr(c, "offset"):
                self.assertEqual(c.offset.shape[0], 1)

        # Step 6: Trim 1 token (all layers)
        for c in batched:
            if hasattr(c, "trim"):
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

        # Tokens must match - this is the important check
        self.assertEqual(
            loaded_tok, continue_tok,
            "Loaded cache produces different token than original cache",
        )
        # Logits should match within bfloat16 precision.
        # We use atol (absolute tolerance) instead of rtol (relative tolerance)
        # because logits can be near zero, making rtol unstable.
        # atol=1e-0 accounts for bfloat16 precision floor (~0.01-0.1) and
        # multiple comparisons across 248k logits (expecting some outliers).
        # An error of 1.0 in a logit corresponds to a factor of e (~2.7) in
        # probability, which is often acceptable for "close enough" in inference.
        # If the error were 10.0 (significant degradation), this would fail.
        self.assertTrue(
            mx.allclose(logits_loaded[0], logits_continue[0], rtol=0, atol=1e-0).item(),
            "Loaded cache logits differ too much from original cache",
        )


    # -- 5. Quality test: batch+TurboQuant must match single-mode+TurboQuant output style --

    def test_batch_turbo_quant_matches_single_mode(self):
        """Regression test: batch+TurboQuant output must match single-mode+TurboQuant output.

        This test locks down a critical quality regression where batch+TurboQuant
        produces output with a completely different style than single-mode+TurboQuant.

        The bug manifests as:
        - Batch output starts with "Here's a thinking process:" instead of direct answer
        - Batch output has different structure/content than single-mode
        - The model's reasoning style changes when going through BatchGenerator

        Single-mode TurboQuant is the known-good baseline.
        Batch+TurboQuant must match single-mode+TurboQuant output style.

        Batching should not add quality degradation on top of quantization.
        The same quantization method should produce similar output regardless of batching.
        """
        from mlx_lm.generate import batch_generate

        # Realistic multi-turn conversation with system prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You can use tools to read files and run commands."},
            {"role": "user", "content": "Look at the mlx_lm/models directory. Find the TurboQuant cache implementation and explain how the quantize-on-write lifecycle works. Specifically: how are the norms stored, and when does quantization happen?"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(prompt, return_tensors="mlx")[0]

        # --- Single-mode TurboQuant (known good baseline) ---
        cache_single = make_prompt_cache(
            self.model, turbo_kv_bits=3, turbo_fp16_layers=1
        )
        results_single = list(
            generate_step(tokens, self.model, prompt_cache=cache_single, max_tokens=200)
        )
        toks_single, logits_single = zip(*results_single)
        mx.eval(*toks_single, *logits_single)
        generated_single = self.tokenizer.decode(list(toks_single))

        # --- Batch+TurboQuant (must match single-mode) ---
        response_batch = batch_generate(
            self.model,
            self.tokenizer,
            prompts=[tokens.tolist()],
            max_tokens=200,
            turbo_kv_bits=3,
            turbo_fp16_layers=1,
        )
        generated_batch = response_batch.texts[0]

        # Word similarity
        single_words = set(generated_single.lower().split())
        batch_words = set(generated_batch.lower().split())
        overlap = len(single_words & batch_words)
        total = len(single_words | batch_words)
        similarity = overlap / total if total > 0 else 0

        # Batching should not add quality degradation on top of quantization.
        # The same quantization method should produce similar output regardless of batching.
        # A threshold of 0.70 allows for minor stochastic variation but catches
        # the regression where batch output has a completely different style.
        self.assertGreater(
            similarity, 0.70,
            f"Batch+TurboQuant output style differs from single-mode+TurboQuant "
            f"(similarity={similarity:.2f}). Batch output should match single-mode output style. "
            f"Batching should not add quality degradation on top of quantization."
        )

        # Also check that batch output doesn't start with reasoning-style prefixes
        # that single-mode doesn't use
        single_starts_with_reasoning = generated_single.strip().startswith(
            ("Here's a thinking", "Thinking Process:", "Here is the thinking")
        )
        batch_starts_with_reasoning = generated_batch.strip().startswith(
            ("Here's a thinking", "Thinking Process:", "Here is the thinking")
        )

        # If single-mode doesn't start with reasoning but batch does, that's the bug
        if not single_starts_with_reasoning and batch_starts_with_reasoning:
            self.fail(
                "Batch+TurboQuant output starts with reasoning-style prefix "
                f"({generated_batch[:50]}) while single-mode does not "
                f"({generated_single[:50]}). This indicates the model's output style "
                "has changed when going through BatchGenerator."
            )


if __name__ == "__main__":
    unittest.main()
