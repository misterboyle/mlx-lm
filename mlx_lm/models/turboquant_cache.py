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

    def __init__(self, bits: int = 3, seed: int = 42, v_bits=None):
        self.quant_bits = bits
        self.seed = seed
        self.v_bits = v_bits
        self.v_group_size = 64
        self.offset = 0

        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None

        # Affine-quantized value storage (used when v_bits is set)
        self._v_quant = None  # quantized uint32 data
        self._v_scales = None  # per-group scales
        self._v_biases = None  # per-group biases

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

        # Fused Metal quantize for keys (PolarQuant)
        k_pk, k_nrm = fused_quantize(
            keys.reshape(-1, k_dim),
            self._k_q.signs,
            self._k_q.boundaries,
            k_dim,
            self.quant_bits,
        )
        k_pk = k_pk.reshape(B, H, S, self._k_pdim)

        self.k_packed[..., prev : prev + S, :] = k_pk
        self.k_norms[..., prev : prev + S] = k_nrm.reshape(B, H, S)

        if self.v_bits is not None:
            # Affine quantize values with mx.quantize
            vq, vs, vb = mx.quantize(
                values, group_size=self.v_group_size, bits=self.v_bits
            )
            self._v_quant[..., prev : prev + S, :] = vq
            self._v_scales[..., prev : prev + S, :] = vs
            self._v_biases[..., prev : prev + S, :] = vb
        else:
            # PolarQuant for values
            v_pk, v_nrm = fused_quantize(
                values.reshape(-1, v_dim),
                self._v_q.signs,
                self._v_q.boundaries,
                v_dim,
                self.quant_bits,
            )
            v_pk = v_pk.reshape(B, H, S, self._v_pdim)
            self.v_packed[..., prev : prev + S, :] = v_pk
            self.v_norms[..., prev : prev + S] = v_nrm.reshape(B, H, S)

        self.offset += S
        total = self.offset

        # Incremental decode
        if S <= 4 and self._v_deq_buf is not None and self._deq_offset == prev:
            if total > self._deq_alloc:
                na = ((total + self.step - 1) // self.step) * self.step
                self._k_deq_buf = mx.concatenate(
                    [
                        self._k_deq_buf[..., : self._deq_offset, :],
                        mx.zeros((B, H, na - self._deq_alloc, k_dim), dtype=keys.dtype),
                    ],
                    axis=2,
                )
                self._v_deq_buf = mx.concatenate(
                    [
                        self._v_deq_buf[..., : self._deq_offset, :],
                        mx.zeros(
                            (B, H, na - self._deq_alloc, v_dim), dtype=values.dtype
                        ),
                    ],
                    axis=2,
                )
                self._deq_alloc = na

            nk = dequant_fp16(
                k_pk.reshape(-1, self._k_pdim),
                k_nrm,
                self._k_q.centroids,
                self._k_q.signs,
                k_dim,
                self.quant_bits,
            ).reshape(B, H, S, k_dim)
            if self.v_bits is not None:
                nv = mx.dequantize(
                    vq, vs, vb, group_size=self.v_group_size, bits=self.v_bits
                ).astype(values.dtype)
            else:
                nv = dequant_fp16(
                    v_pk.reshape(-1, self._v_pdim),
                    v_nrm,
                    self._v_q.centroids,
                    self._v_q.signs,
                    v_dim,
                    self.quant_bits,
                ).reshape(B, H, S, v_dim)
            self._k_deq_buf[..., prev:total, :] = nk
            self._v_deq_buf[..., prev:total, :] = nv
            self._deq_offset = total
            return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

        # Full dequant (prefill)
        all_k = self._full_dequant(
            self.k_packed, self.k_norms, self._k_q, k_dim, B, H, total, keys.dtype
        )
        if self.v_bits is not None:
            all_v = self._dequantize_affine_values(B, H, total, values.dtype)
        else:
            all_v = self._full_dequant(
                self.v_packed, self.v_norms, self._v_q, v_dim, B, H, total, values.dtype
            )
        alloc = ((total + self.step - 1) // self.step) * self.step
        self._k_deq_buf = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
        self._v_deq_buf = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
        self._k_deq_buf[..., :total, :] = all_k
        self._v_deq_buf[..., :total, :] = all_v
        self._deq_offset = total
        self._deq_alloc = alloc
        return all_k, all_v

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
                + self.v_norms[..., : self.offset].nbytes
            )
        return total

    @property
    def state(self):
        if self.k_packed is None:
            return []
        if self.v_bits is not None:
            return [
                self.k_packed[..., : self.offset, :],
                self.k_norms[..., : self.offset],
                self._v_quant[..., : self.offset, :],
                self._v_scales[..., : self.offset, :],
                self._v_biases[..., : self.offset, :],
            ]
        return [
            self.k_packed[..., : self.offset, :],
            self.k_norms[..., : self.offset],
            self.v_packed[..., : self.offset, :],
            self.v_norms[..., : self.offset],
        ]

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
            ) = v
        else:
            self.k_packed, self.k_norms, self.v_packed, self.v_norms = v
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
        return f"{self.offset},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0},{dtype_str},{v_bits_str}"

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
        return c

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0
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

    def __init__(self, left_padding, bits=3, seed=42, v_bits=None):
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self.quant_bits = bits
        self.seed = seed
        self.v_bits = v_bits
        self.v_group_size = 64

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
            )

        B = len(caches)
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]

        # Get quantizer params from first cache
        bits = first.quant_bits
        seed = first.seed
        v_bits = first.v_bits
        k_dim = first._k_dim
        v_dim = first._v_dim
        k_pdim = first._k_pdim
        v_pdim = first._v_pdim
        dtype = first._dtype if first._dtype is not None else mx.float16

        # Get head count and dtype from first non-empty cache
        H = first.k_packed.shape[1] if first.k_packed is not None else 8
        dt = first._dtype if first._dtype is not None else mx.float16

        # Allocate batched storage
        cache = cls(padding, bits=bits, seed=seed, v_bits=v_bits)
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

        # Fused Metal quantize for keys
        k_pk, k_nrm = fused_quantize(
            keys.reshape(-1, k_dim),
            self._k_q.signs,
            self._k_q.boundaries,
            k_dim,
            self.quant_bits,
        )
        k_pk = k_pk.reshape(B, H, S, self._k_pdim)
        k_nrm = k_nrm.reshape(B, H, S)

        # Write to batched buffer with left-padding
        for i in range(B):
            p = self.left_padding[i].item()
            self.k_packed[i : i + 1, :, p : p + S, :] = k_pk[i : i + 1]
            self.k_norms[i : i + 1, :, p : p + S] = k_nrm[i : i + 1]

        if self.v_bits is not None:
            vq, vs, vb = mx.quantize(
                values, group_size=self.v_group_size, bits=self.v_bits
            )
            for i in range(B):
                p = self.left_padding[i].item()
                self._v_quant[i : i + 1, :, p : p + S, :] = vq[i : i + 1]
                self._v_scales[i : i + 1, :, p : p + S, :] = vs[i : i + 1]
                self._v_biases[i : i + 1, :, p : p + S, :] = vb[i : i + 1]
        else:
            v_pk, v_nrm = fused_quantize(
                values.reshape(-1, v_dim),
                self._v_q.signs,
                self._v_q.boundaries,
                v_dim,
                self.quant_bits,
            )
            v_pk = v_pk.reshape(B, H, S, self._v_pdim)
            v_nrm = v_nrm.reshape(B, H, S)
            for i in range(B):
                p = self.left_padding[i].item()
                self.v_packed[i : i + 1, :, p : p + S, :] = v_pk[i : i + 1]
                self.v_norms[i : i + 1, :, p : p + S] = v_nrm[i : i + 1]

        self._idx += S
        total = self._idx

        # Dequantize and return
        all_k = self._full_dequant(
            self.k_packed, self.k_norms, self._k_q, k_dim, B, H, total, keys.dtype
        )
        if self.v_bits is not None:
            all_v = self._dequantize_affine_values(B, H, total, values.dtype)
        else:
            all_v = self._full_dequant(
                self.v_packed, self.v_norms, self._v_q, v_dim, B, H, total, values.dtype
            )
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
        return total

    @property
    def state(self):
        if self.k_packed is None:
            return []
        if self.v_bits is not None:
            return [
                self.k_packed[..., : self._idx, :],
                self.k_norms[..., : self._idx],
                self._v_quant[..., : self._idx, :],
                self._v_scales[..., : self._idx, :],
                self._v_biases[..., : self._idx, :],
                self.left_padding,
                self.offset,
            ]
        return [
            self.k_packed[..., : self._idx, :],
            self.k_norms[..., : self._idx],
            self.v_packed[..., : self._idx, :],
            self.v_norms[..., : self._idx],
            self.left_padding,
            self.offset,
        ]

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
        else:
            self.k_packed, self.k_norms, self.v_packed, self.v_norms = v[:4]
            self.left_padding, self.offset = v[4], v[5]
        self._idx = self.k_packed.shape[2]

    @property
    def meta_state(self):
        dtype_str = TurboQuantKVCache._DTYPE_NAME.get(self._dtype, "float16")
        v_bits_str = str(self.v_bits) if self.v_bits is not None else "0"
        return f"{self._idx},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0},{dtype_str},{v_bits_str}"

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self._idx, self.quant_bits, self.seed = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
        )
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

    def size(self):
        return self._idx

    def empty(self):
        return self.k_packed is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self._idx, **kwargs)

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
        obj.meta_state = meta_state
        obj.state = state
        return obj
