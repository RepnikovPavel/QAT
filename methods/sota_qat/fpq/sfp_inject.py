"""Wire SFP + CSD feature hooks onto Q^2 ``QYOLOv5`` (arXiv:2503.11159).

SFP (Eq. 9, 14) is applied on each quantized conv's *activation* input using
that layer's LSQ step ``s``. CSD (Eq. 16–17) compares student vs FP teacher
block outputs after channel-wise standardization.

Does not modify ``methods/q2`` sources; wraps modules after model build.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from fpq import FeatureHook, stochastic_feature_perturb  # script path
except ImportError:  # package import: methods.sota_qat.fpq.sfp_inject
    from .fpq import FeatureHook, stochastic_feature_perturb

# yolov5s top-level block indices used for CSD (C3 / SPPF stage outputs).
# Full per-conv CSD is possible but heavy; stage features match Algorithm 1
# intent (layer-wise feature alignment) at detection scale.
DEFAULT_CSD_LAYERS: Tuple[int, ...] = (4, 6, 9, 13, 17, 20, 23)


class SFPActWrapper(nn.Module):
    """Activation path: optional SFP (Eq. 14) then fake-quant (LSQ/PACT/…).

    Parameters
    ----------
    a_quant :
        Existing activation quantizer (e.g. :class:`qat.quantizers.LSQ`).
    p :
        Bernoulli probability of injecting noise (paper Tab. 3 peak ~0.1).
    """

    def __init__(self, a_quant: nn.Module, p: float = 0.1) -> None:
        super().__init__()
        self.a_quant = a_quant
        self.p = float(p)

    def _step_size(self) -> torch.Tensor:
        """Current quant step ``s`` for noise scale (Eq. 9)."""
        aq = self.a_quant
        if hasattr(aq, "step"):
            s = aq.step.detach().abs()
            # guard against uninitialised / zero step
            if torch.is_tensor(s):
                return s.clamp(min=1e-8)
        return torch.tensor(1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0:
            s = self._step_size()
            # broadcast scalar/per-tensor step to activation NCHW
            if s.numel() == 1:
                step = s
            else:
                # unexpected per-channel act quant: mean step as scalar scale
                step = s.mean()
            x = stochastic_feature_perturb(x, step, p=self.p, training=True)
        return self.a_quant(x)


def enable_sfp(model: nn.Module, p: float = 0.1) -> int:
    """Wrap every quantized-conv activation quantizer with :class:`SFPActWrapper`.

    Returns the number of modules wrapped. Idempotent if already wrapped.
    """
    n = 0
    for m in model.modules():
        # QYOLOv5._QuantizedConv has both a_quant and w_quant
        if not (hasattr(m, "a_quant") and hasattr(m, "w_quant")):
            continue
        aq = getattr(m, "a_quant", None)
        if aq is None:
            continue
        if isinstance(aq, SFPActWrapper):
            aq.p = float(p)
            n += 1
            continue
        m.a_quant = SFPActWrapper(aq, p=p)
        n += 1
    return n


def disable_sfp(model: nn.Module) -> int:
    """Unwrap SFP wrappers (eval / export). Returns number unwrapped."""
    n = 0
    for m in model.modules():
        if not hasattr(m, "a_quant"):
            continue
        aq = getattr(m, "a_quant", None)
        if isinstance(aq, SFPActWrapper):
            m.a_quant = aq.a_quant
            n += 1
    return n


def _layer_modules(
    qyolo: nn.Module,
    indices: Sequence[int],
) -> List[nn.Module]:
    """Resolve top-level ``det.model[i]`` modules for CSD hooks."""
    layers = qyolo.model  # property → det.model (nn.Sequential / ModuleList)
    out: List[nn.Module] = []
    for i in indices:
        if 0 <= i < len(layers):
            # Skip Detect head (last layer) — no spatial feature map for CSD
            tname = type(layers[i]).__name__
            if tname == "Detect":
                continue
            out.append(layers[i])
    return out


def attach_csd_hooks(
    student: nn.Module,
    teacher: nn.Module,
    layer_indices: Optional[Sequence[int]] = None,
) -> Tuple[FeatureHook, FeatureHook, List[int]]:
    """Register forward hooks on matching student/teacher stages for CSD.

    Returns ``(student_hook, teacher_hook, resolved_indices)``.
    """
    idxs = list(layer_indices if layer_indices is not None else DEFAULT_CSD_LAYERS)
    s_mods = _layer_modules(student, idxs)
    t_mods = _layer_modules(teacher, idxs)
    if len(s_mods) != len(t_mods):
        raise RuntimeError(
            f"CSD layer count mismatch student={len(s_mods)} teacher={len(t_mods)}"
        )
    if not s_mods:
        raise RuntimeError("no CSD layers resolved; check layer_indices")
    # keep only indices that actually resolved (Detect skipped)
    resolved = []
    layers = student.model
    for i in idxs:
        if 0 <= i < len(layers) and type(layers[i]).__name__ != "Detect":
            resolved.append(i)
    return FeatureHook(s_mods), FeatureHook(t_mods), resolved


def freeze_teacher(teacher: nn.Module) -> None:
    """Put FP teacher in eval mode and freeze all parameters."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)


def copy_body_weights(student: nn.Module, teacher: nn.Module) -> int:
    """Copy matching tensors from student → teacher (same pretrained body).

    Teacher is FP (no quantizer params); only shared ``det.*`` conv/bn weights
    transfer. Returns number of tensors copied.
    """
    s_sd = student.state_dict()
    t_sd = teacher.state_dict()
    new = {}
    n = 0
    for k, v in t_sd.items():
        if k in s_sd and s_sd[k].shape == v.shape:
            new[k] = s_sd[k].detach().clone()
            n += 1
        else:
            new[k] = v
    teacher.load_state_dict(new, strict=False)
    return n
