"""TurboQuantKVCache: PolarQuant KV cache compression with fused Metal kernels.

Implements TurboQuant (arXiv 2504.19874, ICLR 2026) for MLX KV cache compression.
4.6x compression via randomized Hadamard rotation + Lloyd-Max quantization.
Bit-packed uint32 storage with fused Metal quantize/dequantize kernels.
"""

import mlx.core as mx
import math
from mlx_lm.models.turboquant_rotation import random_diagonal_sign
from mlx_lm.models.turboquant_packing import (
    pack_indices,
    unpack_indices,
    packed_dim,
    VALS_PER_WORD,
)
from mlx_lm.models.turboquant_metal import fused_quantize, dequant_fp16
from mlx_lm.models.turboquant_kernels import packed_dequantize


def _compute_gaussian_codebook(bits):
    codebooks = {
        1: [-0.7979, 0.7979],
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520],
        4: [
            -2.7326,
            -2.0690,
            -1.6180,
            -1.2562,
            -0.9423,
            -0.6568,
            -0.3881,
            -0.1284,
            0.1284,
            0.3881,
            0.6568,
            0.9423,
            1.2562,
            1.6180,
            2.0690,
            2.7326,
        ],
    }
    return mx.array(codebooks[bits], dtype=mx.float32)


def _compute_boundaries(centroids):
    return (centroids[:-1] + centroids[1:]) / 2.0


class _Quantizer:
    def __init__(self, dim, bits, seed):
        self.dim = dim
        self.bits = bits
        self.signs = random_diagonal_sign(dim, seed=seed)
        self.centroids = _compute_gaussian_codebook(bits)
        self.boundaries = _compute_boundaries(self.centroids)


class TurboQuantKVCache:
    """TurboQuant KV cache — drop-in replacement for KVCache.

    Compresses keys using PolarQuant (Hadamard rotation + Lloyd-Max codebook
    quantization). Stores bit-packed indices in uint32 + float32 norms.

    Values can be compressed either with PolarQuant (default) or with standard
    affine quantization (when ``v_bits`` is set). Affine quantization is simpler,
    faster, and values tolerate it well without rotation.

    Uses fused Metal kernels for quantize and dequantize operations.
    Maintains an incremental decode buffer for O(1) per-step dequantization.
    """

    step = 256

    def __init__(self, bits: int = 3, seed: int = 42, v_bits=None, min_tokens_before_quant: int = 128):
        self.quant_bits = bits
        self.seed = seed
        self.v_bits = v_bits
        self.v_group_size = 64
        self.min_tokens_before_quant = min_tokens_before_quant
        self.offset = 0

        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None

        # Affine-quantized value storage (used when v_bits is set)
        self._v_quant = None  # quantized uint32 data
        self._v_scales = None  # per-group scales
        self._v_biases = None  # per-group biases

        # FP16 prefix storage for attention sinks (tokens before min_tokens_before_quant)
        self._k_prefix = None  # raw fp16 keys for prefix tokens
        self._v_prefix = None  # raw fp16 values for prefix tokens

        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0

        self._k_q = None
        self._v_q = None
        self._k_dim = None
        self._v_dim = None
        self._k_pdim = None
        self._v_pdim = None
        self._dtype = None

    def _ensure_quantizer(self, k_dim, v_dim):
        if self._k_q is None:
            self._k_q = _Quantizer(k_dim, self.quant_bits, self.seed)
            self._k_dim = k_dim
            self._k_pdim = packed_dim(k_dim, self.quant_bits)
        if self._v_q is None and self.v_bits is None:
            self._v_q = _Quantizer(v_dim, self.quant_bits, self.seed + 1)
            self._v_dim = v_dim
            self._v_pdim = packed_dim(v_dim, self.quant_bits)
        elif self._v_dim is None:
            self._v_dim = v_dim

    def _ensure_storage(self, B, H, num_new):
        prev = self.offset
        needed = prev + num_new
        if self.k_packed is None or needed > self.k_packed.shape[2]:
            n = ((needed + self.step - 1) // self.step) * self.step
            if self.k_packed is not None:
                # Allocate new buffer and copy old data into it
                new_kp = mx.zeros((B, H, n, self._k_pdim), dtype=mx.uint32)
                new_kn = mx.zeros((B, H, n), dtype=mx.float32)
                new_kp[..., :prev, :] = self.k_packed[..., :prev, :]
                new_kn[..., :prev] = self.k_norms[..., :prev]
                self.k_packed, self.k_norms = new_kp, new_kn

                if self.v_bits is not None:
                    # Affine-quantized values
                    el_per_int = 8 * mx.uint32.size // self.v_bits
                    v_qdim = self._v_dim // el_per_int
                    v_sdim = self._v_dim // self.v_group_size
                    new_vq = mx.zeros((B, H, n, v_qdim), dtype=mx.uint32)
                    new_vs = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                    new_vb = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                    new_vq[..., :prev, :] = self._v_quant[..., :prev, :]
                    new_vs[..., :prev, :] = self._v_scales[..., :prev, :]
                    new_vb[..., :prev, :] = self._v_biases[..., :prev, :]
                    self._v_quant, self._v_scales, self._v_biases = (
                        new_vq,
                        new_vs,
                        new_vb,
                    )
                else:
                    new_vp = mx.zeros((B, H, n, self._v_pdim), dtype=mx.uint32)
                    new_vn = mx.zeros((B, H, n), dtype=mx.float32)
                    new_vp[..., :prev, :] = self.v_packed[..., :prev, :]
                    new_vn[..., :prev] = self.v_norms[..., :prev]
                    self.v_packed, self.v_norms = new_vp, new_vn
            else:
                self.k_packed = mx.zeros((B, H, n, self._k_pdim), dtype=mx.uint32)
                self.k_norms = mx.zeros((B, H, n), dtype=mx.float32)

                if self.v_bits is not None:
                    el_per_int = 8 * mx.uint32.size // self.v_bits
                    v_qdim = self._v_dim // el_per_int
                    v_sdim = self._v_dim // self.v_group_size
                    self._v_quant = mx.zeros((B, H, n, v_qdim), dtype=mx.uint32)
                    self._v_scales = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                    self._v_biases = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                else:
                    self.v_packed = mx.zeros((B, H, n, self._v_pdim), dtype=mx.uint32)
                    self.v_norms = mx.zeros((B, H, n), dtype=mx.float32)

    def _full_dequant(self, packed, norms, q, dim, B, H, total, dtype):
        flat_p = packed[..., :total, :].reshape(-1, packed.shape[-1])
        flat_n = norms[..., :total].reshape(-1)
        out = packed_dequantize(
            flat_p, flat_n, q.centroids, q.signs, dim, self.quant_bits
        )
        return out.reshape(B, H, total, dim).astype(dtype)

    def _dequantize_affine_values(self, B, H, total, dtype):
        """Dequantize affine-quantized values from _v_quant/scales/biases."""
        vq = self._v_quant[..., :total, :]
        vs = self._v_scales[..., :total, :]
        vb = self._v_biases[..., :total, :]
        return mx.dequantize(
            vq,
            vs,
            vb,
            group_size=self.v_group_size,
            bits=self.v_bits,
        ).astype(dtype)

    def update_and_fetch(self, keys, values):
        B, H, S, k_dim = keys.shape
        v_dim = values.shape[3]
        self._dtype = keys.dtype
        self._ensure_quantizer(k_dim, v_dim)
        self._ensure_storage(B, H, S)
        prev = self.offset

        # Determine how many tokens are in the prefix (fp16) vs quantized region
        prefix_end = self.min_tokens_before_quant
        prefix_tokens = max(0, prefix_end - prev)
        quant_tokens = S - prefix_tokens
        actual_prefix = min(prefix_tokens, S)

        # Store prefix tokens in fp16 (attention sinks)
        if prefix_tokens > 0:
            # Ensure prefix storage is allocated
            if self._k_prefix is None:
                alloc = ((prefix_end + self.step - 1) // self.step) * self.step
                self._k_prefix = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
                self._v_prefix = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
            # Copy only the actual prefix tokens (may be less than prefix_tokens if S < prefix_tokens)
            self._k_prefix[..., prev : prev + actual_prefix, :] = keys[..., :actual_prefix, :]
            self._v_prefix[..., prev : prev + actual_prefix, :] = values[..., :actual_prefix, :]

        # Quantize tokens beyond the prefix threshold
        if quant_tokens > 0:
            q_start = prefix_end
            q_keys = keys[..., prefix_tokens:, :] if prefix_tokens > 0 else keys
            q_values = values[..., prefix_tokens:, :] if prefix_tokens > 0 else values
            q_S = quant_tokens

            # Fused Metal quantize for keys (PolarQuant)
            k_pk, k_nrm = fused_quantize(
                q_keys.reshape(-1, k_dim),
                self._k_q.signs,
                self._k_q.boundaries,
                k_dim,
                self.quant_bits,
            )
            k_pk = k_pk.reshape(B, H, q_S, self._k_pdim)

            self.k_packed[..., q_start : q_start + q_S, :] = k_pk
            self.k_norms[..., q_start : q_start + q_S] = k_nrm.reshape(B, H, q_S)

            if self.v_bits is not None:
                # Affine quantize values with mx.quantize
                vq, vs, vb = mx.quantize(
                    q_values, group_size=self.v_group_size, bits=self.v_bits
                )
                self._v_quant[..., q_start : q_start + q_S, :] = vq
                self._v_scales[..., q_start : q_start + q_S, :] = vs
                self._v_biases[..., q_start : q_start + q_S, :] = vb
            else:
                # PolarQuant for values
                v_pk, v_nrm = fused_quantize(
                    q_values.reshape(-1, v_dim),
                    self._v_q.signs,
                    self._v_q.boundaries,
                    v_dim,
                    self.quant_bits,
                )
                v_pk = v_pk.reshape(B, H, q_S, self._v_pdim)
                self.v_packed[..., q_start : q_start + q_S, :] = v_pk
                self.v_norms[..., q_start : q_start + q_S] = v_nrm.reshape(B, H, q_S)

        self.offset += S
        total = self.offset

        # Build dequantized output: prefix tokens from fp16 storage, rest from dequant
        if self._k_deq_buf is None or total > self._deq_alloc:
            alloc = ((total + self.step - 1) // self.step) * self.step
            self._k_deq_buf = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
            self._v_deq_buf = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
            self._deq_alloc = alloc

        # Copy prefix tokens from fp16 storage
        if prefix_tokens > 0:
            self._k_deq_buf[..., prev : prev + actual_prefix, :] = self._k_prefix[..., prev : prev + actual_prefix, :]
            self._v_deq_buf[..., prev : prev + actual_prefix, :] = self._v_prefix[..., prev : prev + actual_prefix, :]

        # Dequant quantized tokens
        if quant_tokens > 0:
            q_start = prev + actual_prefix
            all_k = self._full_dequant(
                self.k_packed[..., q_start:total, :],
                self.k_norms[..., q_start:total],
                self._k_q, k_dim, B, H, quant_tokens, keys.dtype
            )
            if self.v_bits is not None:
                all_v = self._dequantize_affine_values(B, H, quant_tokens, values.dtype)
                # Slice to only the quantized portion
                all_v = all_v[..., :quant_tokens, :]
            else:
                all_v = self._full_dequant(
                    self.v_packed[..., q_start:total, :],
                    self.v_norms[..., q_start:total],
                    self._v_q, v_dim, B, H, quant_tokens, values.dtype
                )
            self._k_deq_buf[..., q_start:total, :] = all_k
            self._v_deq_buf[..., q_start:total, :] = all_v

        self._deq_offset = total
        return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

    def empty(self):
        return self.k_packed is None

    @property
    def nbytes(self):
        if self.k_packed is None:
            return 0
        total = (
            self.k_packed[..., : self.offset, :].nbytes
            + self.k_norms[..., : self.offset].nbytes
        )
        if self.v_bits is not None:
            total += (
                self._v_quant[..., : self.offset, :].nbytes
                + self._v_scales[..., : self.offset, :].nbytes
                + self._v_biases[..., : self.offset, :].nbytes
            )
        else:
            total += (
                self.v_packed[..., : self.offset, :].nbytes
                + self.v_norms[..., : self.offset, :].nbytes
            )
        # Add prefix storage if present
        if self._k_prefix is not None:
            total += self._k_prefix.nbytes
            total += self._v_prefix.nbytes
        return total

    @property
    def state(self):
        if self.k_packed is None:
            return []
        if self.v_bits is not None:
            state = [
                self.k_packed[..., : self.offset, :],
                self.k_norms[..., : self.offset],
                self._v_quant[..., : self.offset, :],
                self._v_scales[..., : self.offset, :],
                self._v_biases[..., : self.offset, :],
            ]
        else:
            state = [
                self.k_packed[..., : self.offset, :],
                self.k_norms[..., : self.offset],
                self.v_packed[..., : self.offset, :],
                self.v_norms[..., : self.offset],
            ]
        # Include prefix storage if present
        if self._k_prefix is not None:
            state.append(self._k_prefix)
            state.append(self._v_prefix)
        return state

    @state.setter
    def state(self, v):
        if not v:
            return
        if self.v_bits is not None:
            (
                self.k_packed,
                self.k_norms,
                self._v_quant,
                self._v_scales,
                self._v_biases,
            ) = v[:5]
            # Restore prefix storage if present
            if len(v) > 5:
                self._k_prefix = v[5]
                self._v_prefix = v[6]
        else:
            self.k_packed, self.k_norms, self.v_packed, self.v_norms = v[:4]
            # Restore prefix storage if present
            if len(v) > 4:
                self._k_prefix = v[4]
                self._v_prefix = v[5]
        self.offset = self.k_packed.shape[2]

    _DTYPE_MAP = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    _DTYPE_NAME = {v: k for k, v in _DTYPE_MAP.items()}

    @property
    def meta_state(self):
        dtype_str = self._DTYPE_NAME.get(self._dtype, "float16")
        v_bits_str = str(self.v_bits) if self.v_bits is not None else "0"
        return f"{self.offset},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0},{dtype_str},{v_bits_str},{self.min_tokens_before_quant}"

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self.offset, self.quant_bits, self.seed = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
        )
        self._k_dim = int(parts[3]) or None
        self._v_dim = int(parts[4]) or None
        if len(parts) > 5:
            self._dtype = self._DTYPE_MAP.get(parts[5], mx.float16)
        else:
            self._dtype = mx.float16
        if len(parts) > 6:
            vb = int(parts[6])
            self.v_bits = vb if vb > 0 else None
        else:
            self.v_bits = None
        if len(parts) > 7:
            self.min_tokens_before_quant = int(parts[7])
        else:
            self.min_tokens_before_quant = 128

    def dequantize(self):
        """Return full dequantized (keys, values) as dense arrays."""
        if self.k_packed is None:
            return None, None
        B, H = self.k_packed.shape[:2]
        dtype = self._dtype if self._dtype is not None else mx.float16
        self._ensure_quantizer(self._k_dim, self._v_dim)
        k = self._full_dequant(
            self.k_packed,
            self.k_norms,
            self._k_q,
            self._k_dim,
            B,
            H,
            self.offset,
            dtype,
        )
        if self.v_bits is not None:
            v = self._dequantize_affine_values(B, H, self.offset, dtype)
        else:
            v = self._full_dequant(
                self.v_packed,
                self.v_norms,
                self._v_q,
                self._v_dim,
                B,
                H,
                self.offset,
                dtype,
            )
        return k, v

    def copy(self):
        """Return a shallow copy with independent offset and invalidated decode buffers."""
        import copy as _copy

        c = _copy.copy(self)
        c._k_deq_buf = None
        c._v_deq_buf = None
        c._deq_offset = 0
        c._deq_alloc = 0
        # Prefix storage is shared (shallow copy) — acceptable since it's immutable after write
        return c

    def __getstate__(self):
        """Serialize _dtype as string (mlx.core.Dtype is not pickle-able)."""
        state = self.__dict__.copy()
        dtype_val = state.get("_dtype")
        if dtype_val is not None:
            state["_dtype"] = TurboQuantKVCache._DTYPE_NAME.get(dtype_val, "float16")
        return state

    def __setstate__(self, state):
        """Restore _dtype from string back to mlx.core.Dtype."""
        dtype_val = state.get("_dtype")
        if isinstance(dtype_val, str):
            state["_dtype"] = TurboQuantKVCache._DTYPE_MAP.get(dtype_val, mx.float16)
        self.__dict__.update(state)
        # Ensure min_tokens_before_quant exists for backward compatibility
        if "min_tokens_before_quant" not in state:
            self.min_tokens_before_quant = 128

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0
        # Clear prefix storage if trimmed past it
        if self.offset < self.min_tokens_before_quant:
            self._k_prefix = None
            self._v_prefix = None
        return n

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.k_packed = None
        obj.k_norms = None
        obj.v_packed = None
        obj.v_norms = None
        obj._v_quant = None
        obj._v_scales = None
        obj._v_biases = None
        obj._k_prefix = None
        obj._v_prefix = None
        obj._k_deq_buf = None
        obj._v_deq_buf = None
        obj._deq_offset = 0
        obj._deq_alloc = 0
        obj._k_q = None
        obj._v_q = None
        obj._k_dim = None
        obj._v_dim = None
        obj._k_pdim = None
        obj._v_pdim = None
        obj._dtype = None
        obj.v_bits = None
        obj.v_group_size = 64
        obj.min_tokens_before_quant = 128
        obj.meta_state = meta_state
        obj.state = state
        return obj

    @classmethod
    def merge(cls, caches):
        """Merge per-sequence TurboQuantKVCache instances into a BatchTurboQuantKVCache."""
        return BatchTurboQuantKVCache.merge(caches)


class BatchTurboQuantKVCache:
    """Batched TurboQuant KV cache for continuous batching with history.

    Similar to BatchKVCache but stores packed indices + norms instead of
    dense keys/values. Each batch element is left-padded so that token i
    of sequence b appears at position (left_padding[b] + i) in the
    batched buffer.
    """

    step = 256

    def __init__(self, left_padding, bits=3, seed=42, v_bits=None, min_tokens_before_quant=128):
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self._prefix_len = 0  # cumulative prefix tokens written
        self.quant_bits = bits
        self.seed = seed
        self.v_bits = v_bits
        self.v_group_size = 64
        self.min_tokens_before_quant = min_tokens_before_quant
        self._right_padding = None

        # Packed key storage: (B, H, max_len, packed_dim)
        self.k_packed = None
        self.k_norms = None
        # Packed value storage (PolarQuant)
        self.v_packed = None
        self.v_norms = None
        # Affine-quantized value storage
        self._v_quant = None
        self._v_scales = None
        self._v_biases = None

        # FP16 prefix storage for attention sinks
        self._k_prefix = None
        self._v_prefix = None

        # Quantizer metadata (initialized on first update)
        self._k_q = None
        self._v_q = None
        self._k_dim = None
        self._v_dim = None
        self._k_pdim = None
        self._v_pdim = None
        self._dtype = None

    @classmethod
    def merge(cls, caches):
        """Create a BatchTurboQuantKVCache from per-sequence caches."""
        if not caches:
            return cls([0], bits=3)

        # Extract metadata from first non-empty cache
        first = next((c for c in caches if not c.empty()), None)
        if first is None:
            # All empty — return single batch cache with zero padding
            B = len(caches)
            return cls(
                [0] * B,
                bits=caches[0].quant_bits,
                seed=caches[0].seed,
                v_bits=caches[0].v_bits,
                min_tokens_before_quant=caches[0].min_tokens_before_quant,
            )

        B = len(caches)
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]

        # Get quantizer params from first cache
        bits = first.quant_bits
        seed = first.seed
        v_bits = first.v_bits
        min_tokens = first.min_tokens_before_quant
        k_dim = first._k_dim
        v_dim = first._v_dim
        k_pdim = first._k_pdim
        v_pdim = first._v_pdim
        dtype = first._dtype if first._dtype is not None else mx.float16

        # Get head count and dtype from first non-empty cache
        H = first.k_packed.shape[1] if first.k_packed is not None else 8
        dt = first._dtype if first._dtype is not None else mx.float16

        # Recompute packed dims if not yet set (cache restored from state)
        if k_pdim is None and k_dim is not None:
            k_pdim = packed_dim(k_dim, bits)
        if v_pdim is None and v_dim is not None and v_bits is None:
            v_pdim = packed_dim(v_dim, bits)

        # Allocate batched storage
        cache = cls(padding, bits=bits, seed=seed, v_bits=v_bits, min_tokens_before_quant=min_tokens)
        cache._k_dim = k_dim
        cache._v_dim = v_dim
        cache._k_pdim = k_pdim
        cache._v_pdim = v_pdim
        cache._dtype = dtype

        if max_length > 0:
            cache.k_packed = mx.zeros((B, H, max_length, k_pdim), dtype=mx.uint32)
            cache.k_norms = mx.zeros((B, H, max_length), dtype=mx.float32)

            if v_bits is not None:
                el_per_int = 8 * mx.uint32.size // v_bits
                v_qdim = v_dim // el_per_int
                v_sdim = v_dim // 64
                cache._v_quant = mx.zeros((B, H, max_length, v_qdim), dtype=mx.uint32)
                cache._v_scales = mx.zeros((B, H, max_length, v_sdim), dtype=mx.float16)
                cache._v_biases = mx.zeros((B, H, max_length, v_sdim), dtype=mx.float16)
            else:
                cache.v_packed = mx.zeros((B, H, max_length, v_pdim), dtype=mx.uint32)
                cache.v_norms = mx.zeros((B, H, max_length), dtype=mx.float32)

            # Copy data from each sequence
            for i, (p, c) in enumerate(zip(padding, caches)):
                if c.empty():
                    continue
                total = c.size()
                cache.k_packed[i : i + 1, :, p : p + total, :] = c.k_packed[
                    ..., :total, :
                ]
                cache.k_norms[i : i + 1, :, p : p + total] = c.k_norms[..., :total]
                if v_bits is not None:
                    cache._v_quant[i : i + 1, :, p : p + total, :] = c._v_quant[
                        ..., :total, :
                    ]
                    cache._v_scales[i : i + 1, :, p : p + total, :] = c._v_scales[
                        ..., :total, :
                    ]
                    cache._v_biases[i : i + 1, :, p : p + total, :] = c._v_biases[
                        ..., :total, :
                    ]
                else:
                    cache.v_packed[i : i + 1, :, p : p + total, :] = c.v_packed[
                        ..., :total, :
                    ]
                    cache.v_norms[i : i + 1, :, p : p + total] = c.v_norms[..., :total]

        cache._idx = max_length
        return cache

    def _ensure_quantizer(self, k_dim, v_dim):
        if self._k_q is None:
            from mlx_lm.models.turboquant_cache import _Quantizer

            self._k_q = _Quantizer(k_dim, self.quant_bits, self.seed)
            self._k_dim = k_dim
            self._k_pdim = packed_dim(k_dim, self.quant_bits)
        if self._v_q is None and self.v_bits is None:
            from mlx_lm.models.turboquant_cache import _Quantizer

            self._v_q = _Quantizer(v_dim, self.quant_bits, self.seed + 1)
            self._v_dim = v_dim
            self._v_pdim = packed_dim(v_dim, self.quant_bits)
        elif self._v_dim is None:
            self._v_dim = v_dim

    def _ensure_storage(self, B, H, num_new):
        prev = self._idx
        needed = prev + num_new
        if self.k_packed is None or needed > self.k_packed.shape[2]:
            n = ((needed + self.step - 1) // self.step) * self.step
            if self.k_packed is not None:
                new_kp = mx.zeros((B, H, n, self._k_pdim), dtype=mx.uint32)
                new_kn = mx.zeros((B, H, n), dtype=mx.float32)
                new_kp[..., :prev, :] = self.k_packed[..., :prev, :]
                new_kn[..., :prev] = self.k_norms[..., :prev]
                self.k_packed, self.k_norms = new_kp, new_kn

                if self.v_bits is not None:
                    el_per_int = 8 * mx.uint32.size // self.v_bits
                    v_qdim = self._v_dim // el_per_int
                    v_sdim = self._v_dim // self.v_group_size
                    new_vq = mx.zeros((B, H, n, v_qdim), dtype=mx.uint32)
                    new_vs = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                    new_vb = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                    new_vq[..., :prev, :] = self._v_quant[..., :prev, :]
                    new_vs[..., :prev, :] = self._v_scales[..., :prev, :]
                    new_vb[..., :prev, :] = self._v_biases[..., :prev, :]
                    self._v_quant, self._v_scales, self._v_biases = (
                        new_vq,
                        new_vs,
                        new_vb,
                    )
                else:
                    new_vp = mx.zeros((B, H, n, self._v_pdim), dtype=mx.uint32)
                    new_vn = mx.zeros((B, H, n), dtype=mx.float32)
                    new_vp[..., :prev, :] = self.v_packed[..., :prev, :]
                    new_vn[..., :prev] = self.v_norms[..., :prev]
                    self.v_packed, self.v_norms = new_vp, new_vn
            else:
                self.k_packed = mx.zeros((B, H, n, self._k_pdim), dtype=mx.uint32)
                self.k_norms = mx.zeros((B, H, n), dtype=mx.float32)

                if self.v_bits is not None:
                    el_per_int = 8 * mx.uint32.size // self.v_bits
                    v_qdim = self._v_dim // el_per_int
                    v_sdim = self._v_dim // self.v_group_size
                    self._v_quant = mx.zeros((B, H, n, v_qdim), dtype=mx.uint32)
                    self._v_scales = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                    self._v_biases = mx.zeros((B, H, n, v_sdim), dtype=mx.float16)
                else:
                    self.v_packed = mx.zeros((B, H, n, self._v_pdim), dtype=mx.uint32)
                    self.v_norms = mx.zeros((B, H, n), dtype=mx.float32)

    def _full_dequant(self, packed, norms, q, dim, B, H, total, dtype):
        flat_p = packed[..., :total, :].reshape(-1, packed.shape[-1])
        flat_n = norms[..., :total].reshape(-1)
        out = packed_dequantize(
            flat_p, flat_n, q.centroids, q.signs, dim, self.quant_bits
        )
        return out.reshape(B, H, total, dim).astype(dtype)

    def _dequantize_affine_values(self, B, H, total, dtype):
        vq = self._v_quant[..., :total, :]
        vs = self._v_scales[..., :total, :]
        vb = self._v_biases[..., :total, :]
        return mx.dequantize(
            vq,
            vs,
            vb,
            group_size=self.v_group_size,
            bits=self.v_bits,
        ).astype(dtype)

    def update_and_fetch(self, keys, values):
        """Update with batched keys/values. keys/values shape: (B, H, S, D)."""
        B, H, S, k_dim = keys.shape
        v_dim = values.shape[3]
        self._dtype = keys.dtype
        self._ensure_quantizer(k_dim, v_dim)
        self._ensure_storage(B, H, S)

        import os
        if os.environ.get('TQ_DEBUG'):
            print(f"[TQ-DEBUG] update_and_fetch: _idx={self._idx}, _prefix_len={self._prefix_len}, S={S}, min_tokens_before_quant={self.min_tokens_before_quant}")

        # Determine prefix boundary (global across all sequences)
        prefix_end = self.min_tokens_before_quant
        quant_start = max(0, prefix_end - self._idx)
        quant_len = S - quant_start

        # Store prefix tokens in fp16 (attention sinks)
        if quant_start > 0:
            # Ensure prefix storage is allocated
            if self._k_prefix is None:
                alloc = ((prefix_end + self.step - 1) // self.step) * self.step
                self._k_prefix = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
                self._v_prefix = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
            # Copy only the actual prefix tokens (may be less than quant_start if S < quant_start)
            actual_prefix = min(quant_start, S)
            prefix_keys = keys[..., :actual_prefix, :]
            prefix_values = values[..., :actual_prefix, :]
            self._k_prefix[..., self._idx : self._idx + actual_prefix, :] = prefix_keys
            self._v_prefix[..., self._idx : self._idx + actual_prefix, :] = prefix_values
            # Track cumulative prefix length for replacement
            self._prefix_len += actual_prefix

        # Quantize tokens beyond the prefix threshold
        if quant_len > 0:
            q_keys = keys[..., quant_start:, :] if quant_start > 0 else keys
            q_values = values[..., quant_start:, :] if quant_start > 0 else values

            # Fused Metal quantize for keys
            k_pk, k_nrm = fused_quantize(
                q_keys.reshape(-1, k_dim),
                self._k_q.signs,
                self._k_q.boundaries,
                k_dim,
                self.quant_bits,
            )
            k_pk = k_pk.reshape(B, H, quant_len, self._k_pdim)
            k_nrm = k_nrm.reshape(B, H, quant_len)

            # Write to batched buffer with left-padding
            for i in range(B):
                p = self.left_padding[i].item()
                self.k_packed[i : i + 1, :, p + quant_start : p + S, :] = k_pk[i : i + 1]
                self.k_norms[i : i + 1, :, p + quant_start : p + S] = k_nrm[i : i + 1]

            if self.v_bits is not None:
                vq, vs, vb = mx.quantize(
                    q_values, group_size=self.v_group_size, bits=self.v_bits
                )
                for i in range(B):
                    p = self.left_padding[i].item()
                    self._v_quant[i : i + 1, :, p + quant_start : p + S, :] = vq[i : i + 1]
                    self._v_scales[i : i + 1, :, p + quant_start : p + S, :] = vs[i : i + 1]
                    self._v_biases[i : i + 1, :, p + quant_start : p + S, :] = vb[i : i + 1]
            else:
                v_pk, v_nrm = fused_quantize(
                    q_values.reshape(-1, v_dim),
                    self._v_q.signs,
                    self._v_q.boundaries,
                    v_dim,
                    self.quant_bits,
                )
                v_pk = v_pk.reshape(B, H, quant_len, self._v_pdim)
                v_nrm = v_nrm.reshape(B, H, quant_len)
                for i in range(B):
                    p = self.left_padding[i].item()
                    self.v_packed[i : i + 1, :, p + quant_start : p + S, :] = v_pk[i : i + 1]
                    self.v_norms[i : i + 1, :, p + quant_start : p + S] = v_nrm[i : i + 1]

        self._idx += S
        total = self._idx

        # Build dequantized output: prefix tokens from fp16 storage, rest from dequant
        all_k = self._full_dequant(
            self.k_packed, self.k_norms, self._k_q, k_dim, B, H, total, keys.dtype
        )
        if self.v_bits is not None:
            all_v = self._dequantize_affine_values(B, H, total, values.dtype)
        else:
            all_v = self._full_dequant(
                self.v_packed, self.v_norms, self._v_q, v_dim, B, H, total, values.dtype
            )

        # Replace quantized prefix tokens with raw fp16 values
        if self._prefix_len > 0:
            import os
            if os.environ.get('TQ_DEBUG'):
                print(f"[TQ-DEBUG] replace: _prefix_len={self._prefix_len}, _idx={self._idx}, total={total}, all_k.shape={all_k.shape}")
            all_k[..., :self._prefix_len, :] = self._k_prefix[..., self._idx - self._prefix_len : self._idx, :]
            all_v[..., :self._prefix_len, :] = self._v_prefix[..., self._idx - self._prefix_len : self._idx, :]

        return all_k, all_v

    def empty(self):
        return self.k_packed is None

    @property
    def nbytes(self):
        if self.k_packed is None:
            return 0
        total = (
            self.k_packed[..., : self._idx, :].nbytes
            + self.k_norms[..., : self._idx].nbytes
        )
        if self.v_bits is not None:
            total += (
                self._v_quant[..., : self._idx, :].nbytes
                + self._v_scales[..., : self._idx, :].nbytes
                + self._v_biases[..., : self._idx, :].nbytes
            )
        else:
            total += (
                self.v_packed[..., : self._idx, :].nbytes
                + self.v_norms[..., : self._idx].nbytes
            )
        # Add prefix storage if present
        if self._k_prefix is not None:
            total += self._k_prefix.nbytes
            total += self._v_prefix.nbytes
        return total

    @property
    def state(self):
        if self.k_packed is None:
            return []
        if self.v_bits is not None:
            state = [
                self.k_packed[..., : self._idx, :],
                self.k_norms[..., : self._idx],
                self._v_quant[..., : self._idx, :],
                self._v_scales[..., : self._idx, :],
                self._v_biases[..., : self._idx, :],
                self.left_padding,
                self.offset,
            ]
        else:
            state = [
                self.k_packed[..., : self._idx, :],
                self.k_norms[..., : self._idx],
                self.v_packed[..., : self._idx, :],
                self.v_norms[..., : self._idx],
                self.left_padding,
                self.offset,
            ]
        # Include prefix storage if present
        if self._k_prefix is not None:
            state.append(self._k_prefix)
            state.append(self._v_prefix)
            state.append(self._prefix_len)
        return state

    @state.setter
    def state(self, v):
        if not v:
            return
        if self.v_bits is not None:
            (
                self.k_packed,
                self.k_norms,
                self._v_quant,
                self._v_scales,
                self._v_biases,
            ) = v[:5]
            self.left_padding, self.offset = v[5], v[6]
            # Restore prefix storage if present
            if len(v) > 7:
                self._k_prefix = v[7]
                self._v_prefix = v[8]
                if len(v) > 9:
                    self._prefix_len = int(v[9])
        else:
            self.k_packed, self.k_norms, self.v_packed, self.v_norms = v[:4]
            self.left_padding, self.offset = v[4], v[5]
            # Restore prefix storage if present
            if len(v) > 6:
                self._k_prefix = v[6]
                self._v_prefix = v[7]
                if len(v) > 8:
                    self._prefix_len = int(v[8])
        self._idx = self.k_packed.shape[2]

    @property
    def meta_state(self):
        dtype_str = TurboQuantKVCache._DTYPE_NAME.get(self._dtype, "float16")
        v_bits_str = str(self.v_bits) if self.v_bits is not None else "0"
        return f"{self._idx},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0},{dtype_str},{v_bits_str},{self.min_tokens_before_quant}"

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self._idx = int(parts[0])
        self.quant_bits = int(parts[1])
        self.seed = int(parts[2])
        self._k_dim = int(parts[3]) or None
        self._v_dim = int(parts[4]) or None
        if len(parts) > 5:
            self._dtype = TurboQuantKVCache._DTYPE_MAP.get(parts[5], mx.float16)
        else:
            self._dtype = mx.float16
        if len(parts) > 6:
            vb = int(parts[6])
            self.v_bits = vb if vb > 0 else None
        else:
            self.v_bits = None
        if len(parts) > 7:
            self.min_tokens_before_quant = int(parts[7])
        else:
            self.min_tokens_before_quant = 128

    def size(self):
        return self._idx

    def empty(self):
        return self.k_packed is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        import os
        if os.environ.get('TQ_DEBUG'):
            print(f"[TQ-DEBUG] trim: n={n}, _idx={self._idx}, _prefix_len={self._prefix_len}")
        n = min(self._idx, n)
        self._idx -= n
        # Reset prefix tracking — trimmed tokens may have included prefix tokens
        # and the cumulative counter is no longer valid
        self._prefix_len = 0
        if os.environ.get('TQ_DEBUG'):
            print(f"[TQ-DEBUG] trim after: _idx={self._idx}, _prefix_len={self._prefix_len}")
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self._idx, **kwargs)

    def filter(self, batch_indices):
        """In-place filter to keep just the given indices in the cache.

        Mirrors BatchKVCache.filter: selects rows by index, then shifts
        left to reduce padding.
        """
        # Convert to mx.array with proper dtype for indexing
        if isinstance(batch_indices, list):
            batch_indices = mx.array(batch_indices, dtype=mx.int32)

        if len(batch_indices) == 0:
            # Empty filter — clear all data
            self.k_packed = None
            self.k_norms = None
            self._v_quant = None
            self._v_scales = None
            self._v_biases = None
            self.v_packed = None
            self.v_norms = None
            self.offset = mx.array([], dtype=mx.int32)
            self.left_padding = mx.array([], dtype=mx.int32)
            self._idx = 0
            return

        if self.k_packed is not None:
            self.k_packed = self.k_packed[batch_indices]
            self.k_norms = self.k_norms[batch_indices]
            if self.v_bits is not None:
                self._v_quant = self._v_quant[batch_indices]
                self._v_scales = self._v_scales[batch_indices]
                self._v_biases = self._v_biases[batch_indices]
            else:
                self.v_packed = self.v_packed[batch_indices]
                self.v_norms = self.v_norms[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = int(self.left_padding.min().item())
        if min_left_pad > 0:
            if self.k_packed is not None:
                self.k_packed = self.k_packed[..., min_left_pad:, :]
                self.k_norms = self.k_norms[..., min_left_pad:]
                if self.v_bits is not None:
                    self._v_quant = self._v_quant[..., min_left_pad:, :]
                    self._v_scales = self._v_scales[..., min_left_pad:, :]
                    self._v_biases = self._v_biases[..., min_left_pad:, :]
                else:
                    self.v_packed = self.v_packed[..., min_left_pad:, :]
                    self.v_norms = self.v_norms[..., min_left_pad:]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad
            # Prefix storage positions shifted — reset cumulative tracking
            self._prefix_len = 0

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        """Prepare cache for right-padded prompt processing.

        Mirrors BatchKVCache.prepare: stores right-padding so finalize()
        can roll data back to left-padded layout.
        """
        if left_padding is not None:
            if self.k_packed is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding = self.left_padding + left_padding
            self.offset = self.offset - left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
        else:
            self._right_padding = None

    def finalize(self):
        """Finalize right-padded prompt processing.

        Mirrors BatchKVCache.finalize: rolls data left by the right-padding
        amount, converting right-padding into left-padding.
        """
        if self._right_padding is None:
            return

        padding = self._right_padding
        # Roll each sequence's data left by its right-padding amount
        for i in range(self.left_padding.shape[0]):
            shift = int(padding[i].item())
            if shift == 0:
                continue
            slice_i = slice(i, i + 1)
            if self.k_packed is not None:
                self.k_packed[slice_i] = mx.roll(
                    self.k_packed[slice_i], -shift, axis=2
                )
                self.k_norms[slice_i] = mx.roll(
                    self.k_norms[slice_i], -shift, axis=1
                )
                if self.v_bits is not None:
                    self._v_quant[slice_i] = mx.roll(
                        self._v_quant[slice_i], -shift, axis=2
                    )
                    self._v_scales[slice_i] = mx.roll(
                        self._v_scales[slice_i], -shift, axis=2
                    )
                    self._v_biases[slice_i] = mx.roll(
                        self._v_biases[slice_i], -shift, axis=2
                    )
                else:
                    self.v_packed[slice_i] = mx.roll(
                        self.v_packed[slice_i], -shift, axis=2
                    )
                    self.v_norms[slice_i] = mx.roll(
                        self.v_norms[slice_i], -shift, axis=1
                    )
        self.offset = self.offset - padding
        self.left_padding = self.left_padding + padding
        self._right_padding = None

    def extend(self, other):
        """In-place extend this cache with another batched cache.

        Mirrors BatchKVCache.extend: right-justifies both caches to the
        same length, then concatenates them vertically.
        """
        if self.k_packed is None and other.k_packed is None:
            self.left_padding = mx.concatenate(
                [self.left_padding, other.left_padding]
            )
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        D = M = 0
        if self.k_packed is not None:
            B, H, L1, D = self.k_packed.shape
            M = self._v_quant.shape[3] if self.v_bits is not None else self.v_packed.shape[3]
        if other.k_packed is not None:
            B, H, L2, D = other.k_packed.shape
            M = other._v_quant.shape[3] if other.v_bits is not None else other.v_packed.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            kp = c.k_packed
            kn = c.k_norms
            # Initialize value storage variables for both quant modes
            kv_quant = kv_scales = kv_biases = None
            kv_packed = kv_norms = None
            if kp is None:
                Bc = c.offset.shape[0]
                kp = mx.array([]).reshape(Bc, H, 0, D)
                kn = mx.array([]).reshape(Bc, H, 0)
                # Create placeholder value storage arrays using dimensions
                # from the non-empty cache (D for keys, M for values)
                if c.v_bits is not None:
                    el_per_int = 8 * mx.uint32.size // c.v_bits
                    v_qdim = M // el_per_int if M else 1
                    v_sdim = c._v_dim // 64 if c._v_dim else 1
                    kv_quant = mx.zeros((Bc, H, 0, v_qdim), dtype=mx.uint32)
                    kv_scales = mx.zeros((Bc, H, 0, v_sdim), dtype=mx.float16)
                    kv_biases = mx.zeros((Bc, H, 0, v_sdim), dtype=mx.float16)
                else:
                    kv_packed = mx.zeros((Bc, H, 0, M), dtype=mx.uint32)
                    kv_norms = mx.zeros((Bc, H, 0), dtype=mx.float32)
            else:
                if c.v_bits is not None:
                    kv_quant = c._v_quant
                    kv_scales = c._v_scales
                    kv_biases = c._v_biases
                else:
                    kv_packed = c.v_packed
                    kv_norms = c.v_norms

            # Compute padding for both empty and non-empty caches
            left = max_idx - c._idx
            right = max_size - kp.shape[2] - left
            if right < 0:
                kp = kp[..., :right, :]
                kn = kn[..., :right]
                if c.v_bits is not None:
                    kv_quant = kv_quant[..., :right, :]
                    kv_scales = kv_scales[..., :right, :]
                    kv_biases = kv_biases[..., :right, :]
                else:
                    kv_packed = kv_packed[..., :right, :]
                    kv_norms = kv_norms[..., :right]
                right = 0
            if left != 0 or right != 0:
                pad_k = [(0, 0), (0, 0), (left, right), (0, 0)]
                kp = mx.pad(kp, pad_k)
                pad_n = [(0, 0), (0, 0), (left, right)]
                kn = mx.pad(kn, pad_n)
                if c.v_bits is not None:
                    pad_vq = [(0, 0), (0, 0), (left, right), (0, 0)]
                    kv_quant = mx.pad(kv_quant, pad_vq)
                    pad_vs = [(0, 0), (0, 0), (left, right), (0, 0)]
                    kv_scales = mx.pad(kv_scales, pad_vs)
                    pad_vb = [(0, 0), (0, 0), (left, right), (0, 0)]
                    kv_biases = mx.pad(kv_biases, pad_vb)
                else:
                    pad_vp = [(0, 0), (0, 0), (left, right), (0, 0)]
                    kv_packed = mx.pad(kv_packed, pad_vp)
                    pad_vn = [(0, 0), (0, 0), (left, right)]
                    kv_norms = mx.pad(kv_norms, pad_vn)
            left_padding = c.left_padding + left
            return kp, kn, kv_quant, kv_scales, kv_biases, kv_packed, kv_norms, c.offset, left_padding

        s_kp, s_kn, s_vq, s_vs, s_vb, s_vp, s_vn, s_off, s_lp = pad(self)
        o_kp, o_kn, o_vq, o_vs, o_vb, o_vp, o_vn, o_off, o_lp = pad(other)

        self.k_packed = mx.concatenate([s_kp, o_kp], axis=0)
        self.k_norms = mx.concatenate([s_kn, o_kn], axis=0)
        self.offset = mx.concatenate([s_off, o_off])
        self.left_padding = mx.concatenate([s_lp, o_lp])

        if self.v_bits is not None:
            self._v_quant = mx.concatenate([s_vq, o_vq], axis=0)
            self._v_scales = mx.concatenate([s_vs, o_vs], axis=0)
            self._v_biases = mx.concatenate([s_vb, o_vb], axis=0)
        else:
            self.v_packed = mx.concatenate([s_vp, o_vp], axis=0)
            self.v_norms = mx.concatenate([s_vn, o_vn], axis=0)

        self._idx = max_idx

    def extract(self, idx: int) -> "TurboQuantKVCache":
        """Extract per-sequence TurboQuantKVCache from batched buffer.

        Mirrors BatchKVCache.extract: pulls the left-padded slice for
        sequence *idx* out of the batched packed storage and returns a
        standalone TurboQuantKVCache with matching metadata.
        """
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache

        padding = int(self.left_padding[idx])
        seq_len = self._idx - padding

        if self.k_packed is None or seq_len == 0:
            # Empty cache — return a fresh empty instance with metadata
            cache = TurboQuantKVCache(bits=self.quant_bits, v_bits=self.v_bits)
            cache._k_dim = self._k_dim
            cache._v_dim = self._v_dim
            cache._dtype = self._dtype
            cache.quant_bits = self.quant_bits
            cache.seed = self.seed
            return cache

        # Extract packed data for this sequence
        k_packed = self.k_packed[idx : idx + 1, :, padding : padding + seq_len, :]
        k_norms = self.k_norms[idx : idx + 1, :, padding : padding + seq_len]

        if self.v_bits is not None:
            v_quant = self._v_quant[idx : idx + 1, :, padding : padding + seq_len, :]
            v_scales = self._v_scales[idx : idx + 1, :, padding : padding + seq_len, :]
            v_biases = self._v_biases[idx : idx + 1, :, padding : padding + seq_len, :]
            state = [k_packed, k_norms, v_quant, v_scales, v_biases]
        else:
            v_packed = self.v_packed[idx : idx + 1, :, padding : padding + seq_len, :]
            v_norms = self.v_norms[idx : idx + 1, :, padding : padding + seq_len]
            state = [k_packed, k_norms, v_packed, v_norms]

        # Build meta_state string matching TurboQuantKVCache format
        dtype_str = TurboQuantKVCache._DTYPE_NAME.get(self._dtype, "float16")
        v_bits_str = str(self.v_bits) if self.v_bits is not None else "0"
        meta_state = (
            f"{seq_len},{self.quant_bits},{self.seed},"
            f"{self._k_dim or 0},{self._v_dim or 0},"
            f"{dtype_str},{v_bits_str}"
        )

        return TurboQuantKVCache.from_state(state, meta_state)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.k_packed = None
        obj.k_norms = None
        obj.v_packed = None
        obj.v_norms = None
        obj._v_quant = None
        obj._v_scales = None
        obj._v_biases = None
        obj._k_q = None
        obj._v_q = None
        obj._k_dim = None
        obj._v_dim = None
        obj._k_pdim = None
        obj._v_pdim = None
        obj._dtype = None
        obj.v_bits = None
        obj.v_group_size = 64
        obj.left_padding = None
        obj.offset = None
        obj._idx = 0
        obj._prefix_len = 0
        obj.meta_state = meta_state
        obj.state = state
        return obj

    def __getstate__(self):
        """Serialize _dtype as string (mlx.core.Dtype is not pickle-able)."""
        state = self.__dict__.copy()
        dtype_val = state.get("_dtype")
        if dtype_val is not None:
            state["_dtype"] = TurboQuantKVCache._DTYPE_NAME.get(dtype_val, "float16")
        return state

    def __setstate__(self, state):
        """Restore _dtype from string back to mlx.core.Dtype."""
        dtype_val = state.get("_dtype")
        if isinstance(dtype_val, str):
            state["_dtype"] = TurboQuantKVCache._DTYPE_MAP.get(dtype_val, mx.float16)
        self.__dict__.update(state)
