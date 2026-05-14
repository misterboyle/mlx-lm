"""TurboQuantKVCache: PolarQuant KV cache compression with fused Metal kernels.

Implements TurboQuant (arXiv 2504.19874, ICLR 2026) for MLX KV cache compression.
4.6x compression via randomized Hadamard rotation + Lloyd-Max quantization.
Bit-packed uint32 storage with fused Metal quantize/dequantize kernels.
"""

import math

import mlx.core as mx

from mlx_lm.models.turboquant_kernels import packed_dequantize
from mlx_lm.models.turboquant_metal import dequant_fp16, fused_quantize
from mlx_lm.models.turboquant_packing import (
    VALS_PER_WORD,
    pack_indices,
    packed_dim,
    unpack_indices,
)
from mlx_lm.models.turboquant_rotation import random_diagonal_sign


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
        self._dtype_str = None  # stored as string to avoid mx.Dtype deepcopy issues

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
        self._dtype_str = self._DTYPE_NAME.get(keys.dtype, "float16")
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
        dtype_str = self._dtype_str or "float16"
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
            self._dtype_str = parts[5]
        else:
            self._dtype_str = "float16"
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
        dtype = (
            self._DTYPE_MAP.get(self._dtype_str, mx.float16)
            if self._dtype_str
            else mx.float16
        )
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
    def merge(cls, caches):
        from .turboquant_cache import BatchTurboQuantKVCache

        return BatchTurboQuantKVCache.merge(caches)

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
        obj._dtype_str = None
        obj.v_bits = None
        obj.v_group_size = 64
        obj.meta_state = meta_state
        obj.state = state
        return obj


class BatchTurboQuantKVCache:
    """Batched version of TurboQuantKVCache for concurrent request handling.

    Wraps multiple TurboQuantKVCache entries into a single batched cache
    with left-padding support.  Supports both PolarQuant V (``v_bits=None``)
    and affine-quantized V (``v_bits`` set) value modes.
    """

    step = 256

    def __init__(self, left_padding):
        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None
        self._v_quant = None
        self._v_scales = None
        self._v_biases = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        self._dtype_str = None
        self.quant_bits = None
        self.seed = None
        self.v_bits = None
        self._k_dim = None
        self._v_dim = None
        self._k_pdim = None
        self._v_pdim = None
        self.v_group_size = 64
        # Quantizer parameters (stored as arrays, not regenerated)
        self._k_signs = None
        self._k_centroids = None
        self._v_signs = None
        self._v_centroids = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys, values):
        B, H, S, k_dim = keys.shape
        B, H = int(B), int(H)
        v_dim = values.shape[3]
        self._dtype_str = TurboQuantKVCache._DTYPE_NAME.get(keys.dtype, "float16")
        if self.quant_bits is None:
            self.quant_bits = 3
            self.seed = 42

        # Regenerate quantizer params from seed when missing (e.g. from_state).
        # merge() sets these from the individual caches; from_state does not.
        if self._k_signs is None:
            k_q = _Quantizer(k_dim, self.quant_bits, self.seed)
            self._k_signs = k_q.signs
            self._k_centroids = k_q.centroids
            self._k_dim = k_dim
            self._k_pdim = packed_dim(k_dim, self.quant_bits)
        if self._v_signs is None and self.v_bits is None:
            v_q = _Quantizer(v_dim, self.quant_bits, self.seed + 1)
            self._v_signs = v_q.signs
            self._v_centroids = v_q.centroids
            self._v_dim = v_dim
            self._v_pdim = packed_dim(v_dim, self.quant_bits)
        elif self._v_dim is None:
            self._v_dim = v_dim
            if self._v_pdim is None:
                self._v_pdim = packed_dim(v_dim, self.quant_bits)

        prev = self._idx
        if self.k_packed is None or (prev + S) > self.k_packed.shape[2]:
            B0, H0, L0, Dk = (
                self.k_packed.shape if self.k_packed is not None else (B, H, 0, k_dim)
            )
            n = ((prev + S + self.step - 1) // self.step) * self.step
            if self.k_packed is not None:
                new_kp = mx.zeros((B0, H0, n, self._k_pdim), dtype=mx.uint32)
                new_kn = mx.zeros((B0, H0, n), dtype=mx.float32)
                new_kp[..., :prev, :] = self.k_packed[..., :prev, :]
                new_kn[..., :prev] = self.k_norms[..., :prev]
                self.k_packed, self.k_norms = new_kp, new_kn
                if self.v_bits is not None:
                    v_qdim = self._v_dim // (8 * mx.uint32.size // self.v_bits)
                    v_sdim = self._v_dim // self.v_group_size
                    new_vq = mx.zeros((B0, H0, n, v_qdim), dtype=mx.uint32)
                    new_vs = mx.zeros((B0, H0, n, v_sdim), dtype=mx.float16)
                    new_vb = mx.zeros((B0, H0, n, v_sdim), dtype=mx.float16)
                    new_vq[..., :prev, :] = self._v_quant[..., :prev, :]
                    new_vs[..., :prev, :] = self._v_scales[..., :prev, :]
                    new_vb[..., :prev, :] = self._v_biases[..., :prev, :]
                    self._v_quant, self._v_scales, self._v_biases = (
                        new_vq,
                        new_vs,
                        new_vb,
                    )
                else:
                    new_vp = mx.zeros((B0, H0, n, self._v_pdim), dtype=mx.uint32)
                    new_vn = mx.zeros((B0, H0, n), dtype=mx.float32)
                    new_vp[..., :prev, :] = self.v_packed
                    new_vn[..., :prev] = self.v_norms
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

        # Write to padded positions
        for i in range(B):
            pad = int(self.left_padding[i])
            end = pad + S
            key_slice = keys[i : i + 1, :, :, :]
            # Quantize keys using PolarQuant (Hadamard rotation + codebook)
            k_pk, k_nrm = self._quantize_key_packed(key_slice)
            # Squeeze batch dim: k_pk is (1, H, S, k_pdim), target is (H, S, k_pdim)
            self.k_packed[i, :, pad:end, :] = k_pk[0]
            self.k_norms[i, :, pad:end] = k_nrm[0]

            # Quantize values
            val_slice = values[i : i + 1, :, :, :]
            if self.v_bits is not None:
                # Affine quantize values with mx.quantize
                vq, vs, vb = mx.quantize(
                    val_slice, group_size=self.v_group_size, bits=self.v_bits
                )
                self._v_quant[i, :, pad:end, :] = vq[0]
                self._v_scales[i, :, pad:end, :] = vs[0]
                self._v_biases[i, :, pad:end, :] = vb[0]
            else:
                # PolarQuant for values
                v_pk, v_nrm = self._quantize_value_packed(val_slice)
                self.v_packed[i, :, pad:end, :] = v_pk[0]
                self.v_norms[i, :, pad:end] = v_nrm[0]

        self._idx += S
        self.offset += S
        return self._fetch_all()

    def _quantize_key(self, keys):
        """Quantize keys using PolarQuant (Hadamard rotation + codebook).
        Returns only the norms (k_nrm)."""
        from .turboquant_metal import fused_quantize

        B, H, S, k_dim = keys.shape
        k_pk, k_nrm = fused_quantize(
            keys.reshape(-1, k_dim),
            self._get_k_signs(),
            self._get_k_boundaries(),
            k_dim,
            self.quant_bits,
        )
        return k_nrm.reshape(B, H, S)

    def _quantize_key_packed(self, keys):
        """Quantize keys using PolarQuant. Returns (k_pk, k_nrm)."""
        from .turboquant_metal import fused_quantize

        B, H, S, k_dim = keys.shape
        k_pk, k_nrm = fused_quantize(
            keys.reshape(-1, k_dim),
            self._get_k_signs(),
            self._get_k_boundaries(),
            k_dim,
            self.quant_bits,
        )
        k_pk = k_pk.reshape(B, H, S, self._k_pdim)
        return k_pk, k_nrm

    def _quantize_value_packed(self, values):
        """Quantize values using PolarQuant. Returns (v_pk, v_nrm)."""
        from .turboquant_metal import fused_quantize

        B, H, S, v_dim = values.shape
        v_pk, v_nrm = fused_quantize(
            values.reshape(-1, v_dim),
            self._get_v_signs(),
            self._get_v_boundaries(),
            v_dim,
            self.quant_bits,
        )
        v_pk = v_pk.reshape(B, H, S, self._v_pdim)
        return v_pk, v_nrm

    def _get_k_signs(self):
        """Return key quantizer signs (stored from merge)."""
        return self._k_signs

    def _get_k_boundaries(self):
        """Return key quantizer centroids (stored from merge)."""
        return self._k_centroids

    def _fetch_all(self):
        """Return dequantized (keys, values) for the active range.

        Accounts for per-entry left-padding: each entry's data starts at
        left_padding[i] and has length offset[i].
        """
        B, H = self.k_packed.shape[:2]
        dtype = TurboQuantKVCache._DTYPE_MAP.get(self._dtype_str, mx.float16)
        mx.eval(self.left_padding, self.offset)
        lp = self.left_padding.tolist()
        off = self.offset.tolist()

        # Dequantize keys
        k_out = mx.zeros((B, H, max(off), self._k_dim), dtype=dtype)
        for i in range(B):
            start = lp[i]
            length = off[i]
            if length == 0:
                continue
            flat_p = self.k_packed[i : i + 1, :, start : start + length, :].reshape(
                -1, self._k_pdim
            )
            flat_n = self.k_norms[i : i + 1, :, start : start + length].reshape(-1)
            deq = packed_dequantize(
                flat_p,
                flat_n,
                self._k_centroids,
                self._get_k_signs(),
                self._k_dim,
                self.quant_bits,
            )
            k_out[i : i + 1, :, :length, :] = deq.reshape(
                1, H, length, self._k_dim
            ).astype(dtype)

        # Dequantize values
        max_v = max(off) if off else 0
        if self.v_bits is not None:
            v_out = mx.zeros((B, H, max_v, self._v_dim), dtype=dtype)
            for i in range(B):
                start = lp[i]
                length = off[i]
                if length == 0:
                    continue
                vq = self._v_quant[i : i + 1, :, start : start + length, :]
                vs = self._v_scales[i : i + 1, :, start : start + length, :]
                vb = self._v_biases[i : i + 1, :, start : start + length, :]
                v_out[i : i + 1, :, :length, :] = mx.dequantize(
                    vq,
                    vs,
                    vb,
                    group_size=self.v_group_size,
                    bits=self.v_bits,
                ).astype(dtype)
        else:
            v_out = mx.zeros((B, H, max_v, self._v_dim), dtype=dtype)
            for i in range(B):
                start = lp[i]
                length = off[i]
                if length == 0:
                    continue
                flat_vp = self.v_packed[
                    i : i + 1, :, start : start + length, :
                ].reshape(-1, self._v_pdim)
                flat_vn = self.v_norms[i : i + 1, :, start : start + length].reshape(-1)
                deq = packed_dequantize(
                    flat_vp,
                    flat_vn,
                    self._v_centroids,
                    self._get_v_signs(),
                    self._v_dim,
                    self.quant_bits,
                )
                v_out[i : i + 1, :, :length, :] = deq.reshape(
                    1, H, length, self._v_dim
                ).astype(dtype)

        return k_out, v_out

    def _get_v_signs(self):
        """Return value quantizer signs (stored from merge)."""
        return self._v_signs

    def _get_v_boundaries(self):
        """Return value quantizer centroids (stored from merge)."""
        return self._v_centroids

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------

    @classmethod
    def merge(cls, caches):
        """Merge multiple TurboQuantKVCache instances into a BatchTurboQuantKVCache."""
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        if max_length == 0:
            return cls([0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)

        # Get dimensions from first non-empty cache
        first = next(c for c in caches if c.size() > 0)
        quant_bits = first.quant_bits
        seed = first.seed
        v_bits = first.v_bits
        k_dim = first._k_dim
        v_dim = first._v_dim
        k_pdim = first._k_pdim
        v_pdim = first._v_pdim
        dtype_str = first._dtype_str or "float16"
        v_group_size = first.v_group_size

        # Allocate batched storage
        kp = mx.zeros((B, first.k_packed.shape[1], max_length, k_pdim), dtype=mx.uint32)
        kn = mx.zeros((B, first.k_packed.shape[1], max_length), dtype=mx.float32)

        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.size() == 0:
                continue
            kp[i : i + 1, :, p : p + c.offset, :] = c.k_packed[..., : c.offset, :]
            kn[i : i + 1, :, p : p + c.offset] = c.k_norms[..., : c.offset]

        batch_cache = cls(padding)
        batch_cache.k_packed = kp
        batch_cache.k_norms = kn
        batch_cache._dtype_str = dtype_str
        batch_cache.quant_bits = quant_bits
        batch_cache.seed = seed
        batch_cache.v_bits = v_bits
        batch_cache._k_dim = k_dim
        batch_cache._v_dim = v_dim
        batch_cache._k_pdim = k_pdim
        batch_cache._v_pdim = v_pdim
        batch_cache.v_group_size = v_group_size
        # Store quantizer parameters for correct dequantization.
        # Regenerate from seed if missing (e.g. cache loaded via from_state).
        if first._k_q is not None:
            batch_cache._k_signs = first._k_q.signs
            batch_cache._k_centroids = first._k_q.centroids
        else:
            k_q = _Quantizer(k_dim, quant_bits, seed)
            batch_cache._k_signs = k_q.signs
            batch_cache._k_centroids = k_q.centroids
        if v_bits is None and first._v_q is not None:
            batch_cache._v_signs = first._v_q.signs
            batch_cache._v_centroids = first._v_q.centroids
        elif v_bits is None:
            v_q = _Quantizer(v_dim, quant_bits, seed + 1)
            batch_cache._v_signs = v_q.signs
            batch_cache._v_centroids = v_q.centroids
        batch_cache._idx = max_length
        batch_cache.offset += max_length

        if v_bits is not None:
            el_per_int = 8 * mx.uint32.size // v_bits
            v_qdim = v_dim // el_per_int
            v_sdim = v_dim // v_group_size
            vq = mx.zeros(
                (B, first.k_packed.shape[1], max_length, v_qdim), dtype=mx.uint32
            )
            vs = mx.zeros(
                (B, first.k_packed.shape[1], max_length, v_sdim), dtype=mx.float16
            )
            vb = mx.zeros(
                (B, first.k_packed.shape[1], max_length, v_sdim), dtype=mx.float16
            )
            for i, (p, c) in enumerate(zip(padding, caches)):
                if c.size() == 0:
                    continue
                vq[i : i + 1, :, p : p + c.offset, :] = c._v_quant[..., : c.offset, :]
                vs[i : i + 1, :, p : p + c.offset, :] = c._v_scales[..., : c.offset, :]
                vb[i : i + 1, :, p : p + c.offset, :] = c._v_biases[..., : c.offset, :]
            batch_cache._v_quant = vq
            batch_cache._v_scales = vs
            batch_cache._v_biases = vb
        else:
            vp = mx.zeros(
                (B, first.k_packed.shape[1], max_length, v_pdim), dtype=mx.uint32
            )
            vn = mx.zeros((B, first.k_packed.shape[1], max_length), dtype=mx.float32)
            for i, (p, c) in enumerate(zip(padding, caches)):
                if c.size() == 0:
                    continue
                vp[i : i + 1, :, p : p + c.offset, :] = c.v_packed[..., : c.offset, :]
                vn[i : i + 1, :, p : p + c.offset] = c.v_norms[..., : c.offset]
            batch_cache.v_packed = vp
            batch_cache.v_norms = vn

        return batch_cache

    # ------------------------------------------------------------------
    # filter
    # ------------------------------------------------------------------

    def filter(self, batch_indices):
        """In-place filter to keep just the given indices in the cache."""
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
        self.left_padding = self.left_padding[batch_indices]
        self.offset = self.offset[batch_indices]

        # Shift left
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
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

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(self, idx):
        """Extract a single cache entry back to a TurboQuantKVCache."""
        mx.eval(self.left_padding)
        padding = max(0, self.left_padding.tolist()[idx])
        cache = TurboQuantKVCache()
        cache.quant_bits = self.quant_bits
        cache.seed = self.seed
        cache.v_bits = self.v_bits
        cache.v_group_size = self.v_group_size
        cache._k_dim = self._k_dim
        cache._v_dim = self._v_dim
        cache._k_pdim = self._k_pdim
        cache._v_pdim = self._v_pdim
        cache._dtype_str = self._dtype_str

        cache.k_packed = mx.contiguous(
            self.k_packed[idx : idx + 1, :, padding : self._idx, :]
        )
        cache.k_norms = mx.contiguous(
            self.k_norms[idx : idx + 1, :, padding : self._idx]
        )
        if self.v_bits is not None:
            cache._v_quant = mx.contiguous(
                self._v_quant[idx : idx + 1, :, padding : self._idx, :]
            )
            cache._v_scales = mx.contiguous(
                self._v_scales[idx : idx + 1, :, padding : self._idx, :]
            )
            cache._v_biases = mx.contiguous(
                self._v_biases[idx : idx + 1, :, padding : self._idx, :]
            )
        else:
            cache.v_packed = mx.contiguous(
                self.v_packed[idx : idx + 1, :, padding : self._idx, :]
            )
            cache.v_norms = mx.contiguous(
                self.v_norms[idx : idx + 1, :, padding : self._idx]
            )
        cache.offset = cache.k_packed.shape[2]
        return cache

    # ------------------------------------------------------------------
    # extend
    # ------------------------------------------------------------------

    def extend(self, other):
        """In-place extend this cache with another BatchTurboQuantKVCache."""
        if self.k_packed is None and other.k_packed is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.k_packed is not None:
            L1 = self.k_packed.shape[2]
        if other.k_packed is not None:
            L2 = other.k_packed.shape[2]
        max_size = max(L1, L2)

        def pad_cache(c):
            Bc = c.left_padding.shape[0]
            Hc = c.k_packed.shape[1] if c.k_packed is not None else 0
            left = max_idx - c._idx
            right = (
                max_size - c.k_packed.shape[2] - left if c.k_packed is not None else 0
            )
            if right < 0:
                right = 0

            def _pad_4d(arr, left_p, right_p):
                if arr is None:
                    return None
                pad_widths = [(0, 0), (0, 0), (left_p, right_p), (0, 0)]
                return mx.pad(arr, pad_widths)

            def _pad_3d(arr, left_p, right_p):
                if arr is None:
                    return None
                pad_widths = [(0, 0), (0, 0), (left_p, right_p)]
                return mx.pad(arr, pad_widths)

            new_kp = _pad_4d(c.k_packed, left, right)
            new_kn = _pad_3d(c.k_norms, left, right)
            new_lp = c.left_padding + left
            new_off = c.offset  # offset stays as-is (data length, not padded position)

            if c.v_bits is not None:
                new_vq = _pad_4d(c._v_quant, left, right)
                new_vs = _pad_4d(c._v_scales, left, right)
                new_vb = _pad_4d(c._v_biases, left, right)
            else:
                new_vq = _pad_4d(c.v_packed, left, right)
                new_vs = _pad_3d(c.v_norms, left, right)
                new_vb = None

            return new_kp, new_kn, new_vq, new_vs, new_vb, new_lp, new_off

        r1 = pad_cache(self)
        r2 = pad_cache(other)

        self.k_packed = mx.concatenate([r1[0], r2[0]], axis=0)
        self.k_norms = mx.concatenate([r1[1], r2[1]], axis=0)
        if self.v_bits is not None:
            self._v_quant = mx.concatenate([r1[2], r2[2]], axis=0)
            self._v_scales = mx.concatenate([r1[3], r2[3]], axis=0)
            self._v_biases = mx.concatenate([r1[4], r2[4]], axis=0)
        else:
            self.v_packed = mx.concatenate([r1[2], r2[2]], axis=0)
            self.v_norms = mx.concatenate([r1[3], r2[3]], axis=0)
        self.left_padding = mx.concatenate([r1[5], r2[5]])
        self.offset = mx.concatenate([r1[6], r2[6]])
        self._idx = max_idx

    # ------------------------------------------------------------------
    # trim
    # ------------------------------------------------------------------

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    # ------------------------------------------------------------------
    # state / meta_state / from_state
    # ------------------------------------------------------------------

    @property
    def state(self):
        if self.k_packed is None:
            return []
        total = self._idx
        parts = [
            self.k_packed[..., :total, :],
            self.k_norms[..., :total],
        ]
        if self.v_bits is not None:
            parts.extend(
                [
                    self._v_quant[..., :total, :],
                    self._v_scales[..., :total, :],
                    self._v_biases[..., :total, :],
                ]
            )
        else:
            parts.extend(
                [
                    self.v_packed[..., :total, :],
                    self.v_norms[..., :total],
                ]
            )
        # Include left_padding and offset for full serialization
        parts.append(self.left_padding)
        parts.append(self.offset)
        return parts

    @state.setter
    def state(self, v):
        if not v:
            return
        total = v[0].shape[2]
        self.k_packed = v[0]
        self.k_norms = v[1]
        if self.v_bits is not None:
            self._v_quant = v[2]
            self._v_scales = v[3]
            self._v_biases = v[4]
            self.left_padding = v[5]
            self.offset = v[6]
        else:
            self.v_packed = v[2]
            self.v_norms = v[3]
            self.left_padding = v[4]
            self.offset = v[5]
        self._idx = total

    @property
    def meta_state(self):
        dtype_str = self._dtype_str or "float16"
        v_bits_str = str(self.v_bits) if self.v_bits is not None else "0"
        return f"{self._idx},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0},{dtype_str},{v_bits_str},{self.v_group_size}"

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self._idx = int(parts[0])
        self.quant_bits = int(parts[1])
        self.seed = int(parts[2])
        self._k_dim = int(parts[3]) or None
        self._v_dim = int(parts[4]) or None
        if len(parts) > 5:
            self._dtype_str = parts[5]
        else:
            self._dtype_str = "float16"
        if len(parts) > 6:
            vb = int(parts[6])
            self.v_bits = vb if vb > 0 else None
        else:
            self.v_bits = None
        if len(parts) > 7:
            self.v_group_size = int(parts[7])
        else:
            self.v_group_size = 64

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
        obj._right_padding = None
        obj.offset = None
        obj.left_padding = None
        obj._dtype_str = None
        obj.quant_bits = None
        obj.seed = None
        obj.v_bits = None
        obj._k_dim = None
        obj._v_dim = None
        obj._k_pdim = None
        obj._v_pdim = None
        obj.v_group_size = 64
        obj._k_signs = None
        obj._k_centroids = None
        obj._v_signs = None
        obj._v_centroids = None
        obj.meta_state = meta_state
        obj.state = state
        return obj

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        from .base import create_causal_mask

        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    # ------------------------------------------------------------------
    # query methods
    # ------------------------------------------------------------------

    def empty(self):
        return self.k_packed is None

    def size(self):
        return self._idx

    @property
    def nbytes(self):
        total = 0
        if self.k_packed is not None:
            total += self.k_packed.nbytes + self.k_norms.nbytes
            if self.v_bits is not None:
                total += (
                    self._v_quant.nbytes + self._v_scales.nbytes + self._v_biases.nbytes
                )
            else:
                total += self.v_packed.nbytes + self.v_norms.nbytes
        return total

    def is_trimmable(self):
        return True

    # ------------------------------------------------------------------
    # prepare / finalize (right-padding support)
    # ------------------------------------------------------------------

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.k_packed is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            # Right-padding support: shift data right by per-entry amounts
            # and update left_padding accordingly.
            # Simplified: just accumulate into left_padding for now.
            self.left_padding += self._right_padding
            self.offset -= self._right_padding
            self._right_padding = None
