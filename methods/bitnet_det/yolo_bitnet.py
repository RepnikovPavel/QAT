"""YOLOv5s with BitNet b1.58 ternary mid-backbone (W1.58 A8).

Adaptation for 2D detection:
  * stem (layer 0 ``Conv``) stays FP32
  * Detect head stays FP32
  * every other YOLOv5 ``Conv`` (standalone and inside C3/SPPF/Bottleneck)
    is replaced by :class:`BitConv2d` (ternary absmean weights + absmax acts +
    built-in channel RMSNorm + original SiLU)

Built on the official ultralytics/yolov5 ``DetectionModel`` — architecture is
not reimplemented (same pattern as ``methods/q2`` and ``methods/bnl``).

Optional ``mode='xnor'`` injects :class:`XNORConv2d` (W1 A1 binary) instead.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from bitlinear import BitConv2d
from xnor_conv import XNORConv2d


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


def conv_module_to_bitconv(conv_module: nn.Module) -> BitConv2d:
    """Convert an official YOLOv5 ``Conv`` (Conv2d-BN-SiLU) to BitConv2d.

    BN is dropped: BitNet folds normalization into the layer (channel RMSNorm)
    before ternary/A8 quant. SiLU is kept as the post-conv activation.
    """
    c: nn.Conv2d = conv_module.conv  # type: ignore[attr-defined]
    act = getattr(conv_module, "act", None)
    if isinstance(act, nn.Identity):
        act = None

    k = _as_pair(c.kernel_size)
    s = _as_pair(c.stride)
    p = c.padding
    if isinstance(p, tuple) and len(p) == 2:
        padding: Union[int, Tuple[int, int]] = (int(p[0]), int(p[1]))
    else:
        padding = int(p) if not isinstance(p, tuple) else int(p[0])

    bit = BitConv2d(
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        kernel_size=k,
        stride=s,
        padding=padding,
        dilation=_as_pair(c.dilation),
        groups=c.groups,
        bias=c.bias is not None,
        activation=act,
    )
    with torch.no_grad():
        bit.weight.copy_(c.weight.detach())
        if c.bias is not None and bit.bias is not None:
            bit.bias.copy_(c.bias.detach())
    return bit


def conv_module_to_xnor(conv_module: nn.Module) -> XNORConv2d:
    """Convert YOLOv5 ``Conv`` to XNOR-Net binary conv."""
    c: nn.Conv2d = conv_module.conv  # type: ignore[attr-defined]
    act = getattr(conv_module, "act", None)
    if isinstance(act, nn.Identity):
        act = None

    k = _as_pair(c.kernel_size)
    s = _as_pair(c.stride)
    p = c.padding
    if isinstance(p, tuple) and len(p) == 2:
        padding: Union[int, Tuple[int, int]] = (int(p[0]), int(p[1]))
    else:
        padding = int(p) if not isinstance(p, tuple) else int(p[0])

    xnor = XNORConv2d(
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        kernel_size=k,
        stride=s,
        padding=padding,
        dilation=_as_pair(c.dilation),
        groups=c.groups,
        bias=c.bias is not None,
        activation=act,
    )
    with torch.no_grad():
        xnor.weight.copy_(c.weight.detach())
        if c.bias is not None and xnor.bias is not None:
            xnor.bias.copy_(c.bias.detach())
    return xnor


def _copy_route_meta(src: nn.Module, dst: nn.Module) -> None:
    """Preserve yolov5 parse_model routing attrs (i, f, type, np)."""
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def count_bit_convs(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, (BitConv2d, XNORConv2d)))


class BitNetYOLOv5(nn.Module):
    """YOLOv5s with BitNet ternary (or XNOR binary) body; stem + Detect FP32.

    Args:
        nc: number of classes (20 for VOC).
        keep_stem_fp: if True (default), leave layer-0 stem Conv in FP32.
        binary_body: if False, leave the official model fully FP32 (baseline).
        mode: ``'bitnet'`` → BitConv2d (W1.58A8); ``'xnor'`` → XNORConv2d (W1A1).
    """

    def __init__(
        self,
        nc: int = 20,
        keep_stem_fp: bool = True,
        binary_body: bool = True,
        mode: str = "bitnet",
    ) -> None:
        super().__init__()
        if mode not in ("bitnet", "xnor"):
            raise ValueError(f"mode must be 'bitnet' or 'xnor', got {mode!r}")
        from yolov5.models.yolo import DetectionModel

        self.nc = nc
        self.keep_stem_fp = keep_stem_fp
        self.binary_body = binary_body
        self.mode = mode

        self.det = DetectionModel(cfg=_yolov5s_yaml_path(), ch=3, nc=nc)
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

        self.n_bit = 0
        if binary_body:
            self.n_bit = self._inject(keep_stem_fp=keep_stem_fp)

    @property
    def model(self):
        return self.det.model

    @property
    def save(self):
        return self.det.save

    def _converter(self):
        return conv_module_to_bitconv if self.mode == "bitnet" else conv_module_to_xnor

    def _inject(self, keep_stem_fp: bool = True) -> int:
        convert = self._converter()
        replaced = 0
        for i, layer in enumerate(self.model):
            tname = type(layer).__name__
            if tname == "Detect":
                continue
            if tname == "Conv":
                if keep_stem_fp and i == 0:
                    continue
                new = convert(layer)
                _copy_route_meta(layer, new)
                self.model[i] = new
                replaced += 1
            elif tname in ("C3", "SPPF", "Bottleneck", "C3x"):
                replaced += self._replace_recurrent(layer, convert)
        return replaced

    def _replace_recurrent(self, module: nn.Module, convert) -> int:
        n = 0
        for name, child in list(module.named_children()):
            if type(child).__name__ == "Conv":
                setattr(module, name, convert(child))
                n += 1
            else:
                n += self._replace_recurrent(child, convert)
        return n

    def bit_modules(self) -> List[nn.Module]:
        return [m for m in self.modules() if isinstance(m, (BitConv2d, XNORConv2d))]

    def fp_stem_and_head(self) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
        stem = self.model[0] if len(self.model) else None
        head = None
        for layer in self.model:
            if type(layer).__name__ == "Detect":
                head = layer
                break
        return stem, head

    def forward(self, x):
        return self.det._forward_once(x)
