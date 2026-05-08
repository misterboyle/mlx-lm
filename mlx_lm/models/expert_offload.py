"""MoE expert-level offloading for models larger than available RAM.

Keeps only a subset of experts resident in memory (LRU eviction) and
lazily reloads cold experts from the safetensors files on disk.  This
lets you run models like DeepSeek V4 (256 experts, 6 active per token)
on machines that cannot fit all expert weights simultaneously.

Usage:
    After loading the model, call ``enable_expert_offloading(model, model_path)``
    to split monolithic expert tensors into per-expert slices and attach an
    ``ExpertOffloader`` to every ``SwitchGLU`` layer.
"""

import glob
import logging
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class ExpertWeights:
    """Lightweight container for a single expert's weight arrays."""

    __slots__ = ("gate_w", "gate_s", "gate_b",
                 "up_w", "up_s", "up_b",
                 "down_w", "down_s", "down_b",
                 "nbytes")

    def __init__(self, gate_w, gate_s, gate_b,
                 up_w, up_s, up_b,
                 down_w, down_s, down_b):
        self.gate_w = gate_w
        self.gate_s = gate_s
        self.gate_b = gate_b
        self.up_w = up_w
        self.up_s = up_s
        self.up_b = up_b
        self.down_w = down_w
        self.down_s = down_s
        self.down_b = down_b
        self.nbytes = sum(
            a.nbytes for a in (gate_w, gate_s, gate_b,
                               up_w, up_s, up_b,
                               down_w, down_s, down_b)
            if a is not None
        )


class ExpertOffloader:
    """Manages per-expert weight residency with LRU eviction.

    Args:
        layer_prefix: weight-key prefix for this layer, e.g.
            ``"model.layers.3.ffn.experts"`` -- used to reload from disk.
        model_path: path to directory containing model safetensors.
        max_resident_experts: how many experts to keep in RAM.
        num_experts: total number of experts in this layer.
    """

    def __init__(
        self,
        layer_prefix: str,
        model_path: str,
        max_resident_experts: int,
        num_experts: int,
    ):
        self.layer_prefix = layer_prefix
        self.model_path = Path(model_path)
        self.max_resident = max_resident_experts
        self.num_experts = num_experts

        # expert_id -> ExpertWeights, ordered by access time (LRU at front)
        self._cache: OrderedDict[int, ExpertWeights] = OrderedDict()

        # Quantization params (set during register)
        self.group_size: int = 64
        self.bits: int = 4

        # Stats
        self.total_evictions: int = 0
        self.total_loads: int = 0
        self._bytes_resident: int = 0

        # Lazy-built index: safetensors file -> set of weight keys it contains
        self._file_index = None

    # ------------------------------------------------------------------
    # Registration (called once during setup)
    # ------------------------------------------------------------------

    def register(self, expert_id: int, weights: ExpertWeights):
        """Register an expert that is already in memory."""
        self._cache[expert_id] = weights
        self._bytes_resident += weights.nbytes

    def set_quant_params(self, group_size: int, bits: int):
        self.group_size = group_size
        self.bits = bits

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def ensure_resident(self, expert_ids: list):
        """Make sure every expert in *expert_ids* is in RAM.

        Evicts least-recently-used experts when the cache exceeds
        ``max_resident_experts``.
        """
        unique_ids = list(dict.fromkeys(expert_ids))  # dedupe, preserve order

        # Touch / load each required expert
        for eid in unique_ids:
            if eid in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(eid)
            else:
                self._load_expert(eid)

        # Evict if over budget
        self._evict_to(self.max_resident)

    def get_expert_weights(self, expert_id: int) -> ExpertWeights:
        """Return the ExpertWeights for *expert_id* (must be resident)."""
        return self._cache[expert_id]

    @property
    def bytes_resident(self) -> int:
        return self._bytes_resident

    @property
    def num_resident(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal: eviction
    # ------------------------------------------------------------------

    def _evict_to(self, target: int):
        """Evict LRU experts until at most *target* are resident."""
        while len(self._cache) > target:
            eid, ew = self._cache.popitem(last=False)  # pop oldest
            self._bytes_resident -= ew.nbytes
            self.total_evictions += 1
            # Explicitly delete arrays so MLX can reclaim memory
            del ew
        mx.clear_cache()

    # ------------------------------------------------------------------
    # Internal: lazy reloading from safetensors
    # ------------------------------------------------------------------

    def _build_file_index(self):
        """Build a mapping from weight key -> safetensors file path."""
        self._file_index = {}
        for sf in sorted(glob.glob(str(self.model_path / "model*.safetensors"))):
            # mx.load with a safetensors file returns a lazy dict-like
            header = mx.load(sf, return_metadata=False)
            for key in header:
                self._file_index[key] = sf

    def _find_weight_file(self, key: str) -> str:
        """Return the safetensors path that contains *key*."""
        if self._file_index is None:
            self._build_file_index()
        return self._file_index.get(key, None)

    def _load_expert(self, expert_id: int):
        """Load one expert from disk into the cache."""
        prefix = self.layer_prefix

        # The monolithic tensors were split, so on disk the weights are stored
        # as the original (E, O, I) tensor.  We need to load the whole tensor
        # and slice, or -- if per-expert keys exist -- load them directly.
        #
        # Strategy: try per-expert keys first (HF format), then fall back to
        # loading the monolithic tensor and slicing.

        needed = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for arr in ("weight", "scales", "biases"):
                needed[f"{proj}.{arr}"] = f"{prefix}.{proj}.{arr}"

        loaded = {}
        for local_key, full_key in needed.items():
            fpath = self._find_weight_file(full_key)
            if fpath is None:
                # Key not found -- may not exist (e.g. biases might be absent)
                loaded[local_key] = None
                continue
            data = mx.load(fpath)
            tensor = data[full_key]
            # tensor is (num_experts, ...) -- slice out our expert
            loaded[local_key] = tensor[expert_id]
            # Evaluate to force the load and drop the reference to the full tensor
            mx.eval(loaded[local_key])

        ew = ExpertWeights(
            gate_w=loaded["gate_proj.weight"],
            gate_s=loaded["gate_proj.scales"],
            gate_b=loaded.get("gate_proj.biases"),
            up_w=loaded["up_proj.weight"],
            up_s=loaded["up_proj.scales"],
            up_b=loaded.get("up_proj.biases"),
            down_w=loaded["down_proj.weight"],
            down_s=loaded["down_proj.scales"],
            down_b=loaded.get("down_proj.biases"),
        )
        self._cache[expert_id] = ew
        self._cache.move_to_end(expert_id)
        self._bytes_resident += ew.nbytes
        self.total_loads += 1
        logger.debug(
            "Loaded expert %d (%.1f MB), %d resident, %d total loads",
            expert_id,
            ew.nbytes / 1e6,
            len(self._cache),
            self.total_loads,
        )


# ======================================================================
# Public helper: attach offloaders to all SwitchGLU layers in a model
# ======================================================================

def _find_switchglu_layers(model):
    """Yield (weight_prefix, SwitchGLU_module) for every SwitchGLU in the model."""
    from .switch_layers import SwitchGLU

    for name, mod in model.named_modules():
        if isinstance(mod, SwitchGLU):
            yield name, mod


def enable_expert_offloading(
    model: nn.Module,
    model_path: str,
    max_resident_experts: int = 32,
):
    """Split monolithic expert tensors and attach offloaders.

    Call this *after* ``load()`` but *before* generation.

    Args:
        model: the loaded nn.Module (e.g. DeepSeekV4ForCausalLM).
        model_path: path to the directory with model safetensors.
        max_resident_experts: how many experts to keep in RAM per layer.
    """
    from .switch_layers import QuantizedSwitchLinear, SwitchGLU

    count = 0
    for prefix, glu in _find_switchglu_layers(model):
        gate = glu.gate_proj
        up = glu.up_proj
        down = glu.down_proj

        # Only quantized experts are supported for offloading
        if not isinstance(gate, QuantizedSwitchLinear):
            logger.info(
                "Skipping non-quantized SwitchGLU at %s", prefix
            )
            continue

        num_experts = gate.num_experts
        if max_resident_experts >= num_experts:
            logger.info(
                "max_resident_experts (%d) >= num_experts (%d) at %s, skipping",
                max_resident_experts, num_experts, prefix,
            )
            continue

        offloader = ExpertOffloader(
            layer_prefix=prefix,
            model_path=model_path,
            max_resident_experts=max_resident_experts,
            num_experts=num_experts,
        )
        offloader.set_quant_params(gate.group_size, gate.bits)

        # Split the monolithic (E, O, I) tensors into per-expert slices
        for e in range(num_experts):
            ew = ExpertWeights(
                gate_w=gate.weight[e],
                gate_s=gate.scales[e],
                gate_b=gate.biases[e] if gate.biases is not None else None,
                up_w=up.weight[e],
                up_s=up.scales[e],
                up_b=up.biases[e] if up.biases is not None else None,
                down_w=down.weight[e],
                down_s=down.scales[e],
                down_b=down.biases[e] if down.biases is not None else None,
            )
            offloader.register(e, ew)

        # Evaluate all the slices so they are concrete arrays
        mx.eval([
            arr
            for eid in range(num_experts)
            for arr in (
                offloader._cache[eid].gate_w,
                offloader._cache[eid].gate_s,
                offloader._cache[eid].up_w,
                offloader._cache[eid].up_s,
                offloader._cache[eid].down_w,
                offloader._cache[eid].down_s,
            )
        ])

        # Delete the monolithic tensors to free memory
        gate.weight = None
        gate.scales = None
        gate.biases = None
        up.weight = None
        up.scales = None
        up.biases = None
        down.weight = None
        down.scales = None
        down.biases = None
        mx.clear_cache()

        # Now evict to the target count -- keeps only the most recently
        # registered (which is the highest-numbered experts; arbitrary but
        # fine since LRU will sort itself out during inference).
        offloader._evict_to(max_resident_experts)

        # Attach the offloader to the SwitchGLU module
        glu._offloader = offloader

        count += 1
        logger.info(
            "Enabled expert offloading at %s: %d experts, %d resident, "
            "%.1f MB resident",
            prefix,
            num_experts,
            offloader.num_resident,
            offloader.bytes_resident / 1e6,
        )

    # Disable mx.compile on the model -- the offloaded path uses Python-level
    # LRU cache mutations and lazy disk I/O which are incompatible with
    # compiled graph tracing.
    if count > 0:
        model._expert_offloading = True

    if count == 0:
        logger.warning("No SwitchGLU layers found for expert offloading")
    else:
        logger.info(
            "Expert offloading enabled on %d layers "
            "(max %d resident per layer)",
            count,
            max_resident_experts,
        )
    return count
