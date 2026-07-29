"""YOLOv5s with Binary Normalized mid-backbone convs (arXiv:2509.07025).

Adaptation for 2D detection (paper itself has no detection experiments):
  * stem (layer 0 ``Conv``) stays FP32
  * Detect head stays FP32
  * every other YOLOv5 ``Conv`` (standalone and inside C3/SPPF/Bottleneck)
    is replaced by :class:`BinaryNormalizedConv2d` (W1 {0,1} + STE +
    per-example Normalize + original SiLU)

Built on the official ultralytics/yolov5 ``DetectionModel`` — architecture is
not reimplemented (same pattern as ``methods/q2/qat/models/yolov5.py``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .layers import BinaryNormalizedConv2d


def _yolov5s_yaml_path() -> str:
    import os
    import yolov5

    return os.path.join(os.path.dirname(yolov5.__file__), "models", "yolov5s.yaml")


def _as_pair(v: Union[int, Tuple[int, ...]]) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    if len(v) == 1:
        return (v[0], v[0])
    return (int(v[0]), int(v[1]))


def conv_module_to_bnl(conv_module: nn.Module) -> BinaryNormalizedConv2d:
    """Convert an official YOLOv5 ``Conv`` (Conv2d-BN-SiLU) to BNCVL.

    BN is dropped: paper Alg. 3 uses fixed per-example Normalize instead of
    learnable BN after the binary convolution. SiLU is kept as the activation.
    """
    c: nn.Conv2d = conv_module.conv  # type: ignore[attr-defined]
    act = getattr(conv_module, "act", None)
    if isinstance(act, nn.Identity):
        act = None

    k = _as_pair(c.kernel_size)
    s = _as_pair(c.stride)
    # padding may be int or tuple; BinaryNormalizedConv2d accepts both
    p = c.padding
    if isinstance(p, tuple) and len(p) == 2:
        padding: Union[int, Tuple[int, int]] = (int(p[0]), int(p[1]))
    else:
        padding = int(p) if not isinstance(p, tuple) else int(p[0])

    bnl = BinaryNormalizedConv2d(
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        kernel_size=k,
        stride=s,
        padding=padding,
        dilation=_as_pair(c.dilation),
        groups=c.groups,
        bias=True,  # paper quantizes bias to {0,1} as well
        activation=act,
    )
    # Seed full-precision shadow weights from the original conv (useful when
    # converting a pretrained FP body before STE binary training).
    with torch.no_grad():
        bnl.weight.copy_(c.weight.detach())
        if c.bias is not None and bnl.bias is not None:
            bnl.bias.copy_(c.bias.detach())
    return bnl


def _copy_route_meta(src: nn.Module, dst: nn.Module) -> None:
    """Preserve yolov5 parse_model routing attrs (i, f, type, np)."""
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def count_bnl_convs(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, BinaryNormalizedConv2d))


class BNLYOLOv5(nn.Module):
    """YOLOv5s with binary-normalized body (stem + Detect head FP32).

    Args:
        nc: number of classes (20 for VOC).
        keep_stem_fp: if True (default), leave layer-0 stem Conv in FP32.
        binary_body: if False, leave the official model fully FP32 (baseline).
    """

    def __init__(
        self,
        nc: int = 20,
        keep_stem_fp: bool = True,
        binary_body: bool = True,
    ) -> None:
        super().__init__()
        from yolov5.models.yolo import DetectionModel

        self.nc = nc
        self.keep_stem_fp = keep_stem_fp
        self.binary_body = binary_body

        self.det = DetectionModel(cfg=_yolov5s_yaml_path(), ch=3, nc=nc)
        # yolov5 ComputeLoss reads model.hyp
        self.det.hyp = {
            "box": 0.05,
            "cls": 0.5,
            "cls_pw": 1.0,
            "obj": 1.0,
            "obj_pw": 1.0,
            "label_smoothing": 0.0,
            "fl_gamma": 0.0,
            "anchor_t": 4.0,
        }

        self.n_bnl = 0
        if binary_body:
            self.n_bnl = self._inject_bnl(keep_stem_fp=keep_stem_fp)

    # ---------------------------------------------------------- accessors
    @property
    def model(self):
        """Official DetectionModel layer list (``det.model``)."""
        return self.det.model

    @property
    def save(self):
        return self.det.save

    # ---------------------------------------------------------- injection
    def _inject_bnl(self, keep_stem_fp: bool = True) -> int:
        """Replace mid-body Conv modules with BinaryNormalizedConv2d.

        Returns the number of replaced convs.
        """
        replaced = 0
        for i, layer in enumerate(self.model):
            tname = type(layer).__name__
            if tname == "Detect":
                # Detection head stays full precision
                continue
            if tname == "Conv":
                if keep_stem_fp and i == 0:
                    continue
                new = conv_module_to_bnl(layer)
                _copy_route_meta(layer, new)
                self.model[i] = new
                replaced += 1
            elif tname in ("C3", "SPPF", "Bottleneck", "C3x"):
                replaced += self._replace_recurrent(layer)
        return replaced

    def _replace_recurrent(self, module: nn.Module) -> int:
        """Recursively replace every YOLOv5 ``Conv`` child with BNL."""
        n = 0
        for name, child in list(module.named_children()):
            if type(child).__name__ == "Conv":
                setattr(module, name, conv_module_to_bnl(child))
                n += 1
            else:
                n += self._replace_recurrent(child)
        return n

    # ---------------------------------------------------------- helpers
    def bnl_modules(self) -> List[BinaryNormalizedConv2d]:
        return [m for m in self.modules() if isinstance(m, BinaryNormalizedConv2d)]

    def fp_stem_and_head(self) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
        """Return (stem layer 0, Detect head) for inspection."""
        stem = self.model[0] if len(self.model) else None
        head = None
        for layer in self.model:
            if type(layer).__name__ == "Detect":
                head = layer
                break
        return stem, head

    # ---------------------------------------------------------- forward
    def forward(self, x):
        return self.det._forward_once(x)
