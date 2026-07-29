"""Quantization-aware YOLOv5 built on the *official* ultralytics/yolov5 code.

We deliberately avoid re-implementing the architecture by hand (error-prone).
Instead we instantiate the official ``DetectionModel`` from ``yolov5s.yaml``
and inject Q^2 components by walking the parsed layer list:

* every ``Conv`` (Conv2d-BN-SiLU) gets fake-quantized weights and a
  fake-quantized input activation;
* every ``Concat`` (the 4 PANet feature-fusion nodes at layers 12/16/19/22 of
  yolov5s) is replaced by :class:`qat.qgbfusion.QGBFusion` when ``use_qgb``.

The official ``Detect`` head and loss (``yolov5.utils.loss.ComputeLoss``) are
reused as-is; Q-ADA is added on top via feature hooks at the concat inputs.

Reference structure of yolov5s (parsed, width_multiple=0.5):
  layer  type   channels
  0  Conv   3->32        1  Conv   32->64
  2  C3     64->64       3  Conv   64->128
  4  C3     128->128     5  Conv   128->256
  6  C3     256->256     7  Conv   256->512
  8  C3     512->512     9  SPPF   512->512
  10 Conv   512->256     11 Upsample
  12 Concat -> 512       13 C3     512->256
  14 Conv   256->128     15 Upsample
  16 Concat -> 256       17 C3     256->128
  18 Conv   128->128     19 Concat -> 256
  20 C3     256->256     21 Conv   256->256
  22 Concat -> 512       23 C3     512->512
  24 Detect
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..qgbfusion import QGBFusion
from ..quantizers.base import QuantizerBase


def _yolov5s_yaml_path() -> str:
    import os
    import yolov5
    return os.path.join(os.path.dirname(yolov5.__file__), "models", "yolov5s.yaml")


# Total fused channels at each Concat of yolov5s (see reference table above).
_CONCAT_TOTAL_CHANNELS = {12: 512, 16: 256, 19: 256, 22: 512}


def _quant_factory(name: Optional[str], bit_width: int, per_channel: bool):
    if name is None:
        return None
    from ..quantizers import LSQ, PACT, N2UQ
    cls = {"lsq": LSQ, "pact": PACT, "n2uq": N2UQ}[name.lower()]
    return cls(bit_width=bit_width, signed=True,
               per_channel=per_channel, channel_dim=0)


class _QuantizedConv(nn.Module):
    """Wraps an official YOLOv5 Conv to fake-quantize its weight & input.

    Forward mirrors ``yolov5.models.common.Conv.forward`` but routes the conv
    weight and input through the (optional) quantizers. BN/act are unchanged.
    """

    def __init__(self, conv_module, w_quant: Optional[QuantizerBase],
                 a_quant: Optional[QuantizerBase]) -> None:
        super().__init__()
        self.conv = conv_module.conv
        self.bn = conv_module.bn
        self.act = conv_module.act
        self.w_quant = w_quant
        self.a_quant = a_quant

    def forward(self, x):
        if self.a_quant is not None:
            x = self.a_quant(x)
        w = self.conv.weight
        if self.w_quant is not None:
            w = self.w_quant(w)
        c = self.conv
        return self.act(self.bn(nn.functional.conv2d(
            x, w, c.bias, c.stride, c.padding, c.dilation, c.groups,
        )))

    @property
    def weight(self):
        return self.conv.weight


class QYOLOv5(nn.Module):
    """Q^2-augmented YOLOv5s.

    Args:
        nc: number of classes (20 for VOC).
        quant: weight/act quantizer name in {None,'lsq','pact','n2uq'}.
        wbits, abits: weight / activation bit-widths.
        use_qgb: replace PANet Concat nodes with Q-GBFusion.
        quantize_head: if False (default), leave the Detect head convs in FP.
    """

    def __init__(
        self,
        nc: int = 20,
        quant: Optional[str] = "lsq",
        wbits: int = 4,
        abits: int = 4,
        use_qgb: bool = False,
        quantize_head: bool = False,
    ) -> None:
        super().__init__()
        from yolov5.models.yolo import DetectionModel

        self.nc = nc
        self.use_qgb = use_qgb
        self.quant = quant

        # Build the OFFICIAL model (no hand-rolled architecture).
        self.det = DetectionModel(cfg=_yolov5s_yaml_path(), ch=3, nc=nc)
        # yolov5's ComputeLoss reads model.hyp; attach the default hyp dict so
        # the official loss works without the full train.py plumbing.
        self.det.hyp = {
            "box": 0.05, "cls": 0.5, "cls_pw": 1.0, "obj": 1.0, "obj_pw": 1.0,
            "label_smoothing": 0.0, "fl_gamma": 0.0,
            "anchor_t": 4.0,
        }
        # NOTE: do NOT register self.model as an attribute (it would duplicate
        # every weight in state_dict as both `model.*` and `det.model.*`).
        # Access the layer list via the `model` property below.

        # Inject quantization into Conv layers (skip head if requested).
        self.qgb_nodes: Dict[int, QGBFusion] = {}
        self._inject(quant, wbits, abits, quantize_head, use_qgb)

    # ---------------------------------------------------------- accessors
    @property
    def model(self):
        """The official DetectionModel layer list (det.model)."""
        return self.det.model

    @property
    def save(self):
        return self.det.save

        # Inject quantization into Conv layers (skip head if requested).
        self.qgb_nodes: Dict[int, QGBFusion] = {}
        self._inject(quant, wbits, abits, quantize_head, use_qgb)

    # ---------------------------------------------------------- injection
    def _inject(self, quant, wbits, abits, quantize_head, use_qgb):
        if quant is None and not use_qgb:
            return

        for i, layer in enumerate(self.model):
            tname = type(layer).__name__
            new = None
            if tname == "Conv" and quant is not None:
                if not quantize_head and i >= len(self.model) - 1:
                    continue
                wq = _quant_factory(quant, wbits, per_channel=True)
                aq = _quant_factory(quant, abits, per_channel=False)
                new = _QuantizedConv(layer, wq, aq)
            elif tname in ("C3", "SPPF") and quant is not None:
                self._quantize_recurrent(layer, quant, wbits, abits)
            elif tname == "Concat" and use_qgb:
                total_c = _CONCAT_TOTAL_CHANNELS.get(i)
                if total_c is None:
                    continue  # not a known fusion node; leave plain
                self.qgb_nodes[i] = QGBFusion(num_branches=2, num_channels=total_c)
                new = _QGBConcat(self.qgb_nodes[i])

            if new is not None:
                # Preserve the routing metadata that yolov5's parse_model
                # attached to the original layer (i, f, type, np).
                for attr in ("i", "f", "type", "np"):
                    if hasattr(layer, attr):
                        setattr(new, attr, getattr(layer, attr))
                self.model[i] = new

    def _quantize_recurrent(self, module, quant, wbits, abits):
        """Replace every Conv inside C3/SPPF with a quantized wrapper."""
        for name, child in module.named_children():
            if type(child).__name__ == "Conv":
                wq = _quant_factory(quant, wbits, per_channel=True)
                aq = _quant_factory(quant, abits, per_channel=False)
                setattr(module, name, _QuantizedConv(child, wq, aq))
            else:
                self._quantize_recurrent(child, quant, wbits, abits)
                if type(child).__name__ == "Conv":
                    pass  # already handled above

    # ---------------------------------------------------------- helpers
    def qgbfusion_modules(self) -> List[QGBFusion]:
        return list(self.qgb_nodes.values())

    def step_qgb(self):
        """Run the closed-loop dual update on every Q-GBFusion node.

        Call after ``loss.backward()`` and before ``optimizer.step()``.
        """
        energies = {}
        for idx, fuser in self.qgb_nodes.items():
            energies[idx] = fuser.update_dual()
        return energies

    # ---------------------------------------------------------- forward
    def forward(self, x):
        # Reuse the official DetectionModel._forward_once routing (handles the
        # `save` layer indices, upsample/concat topology). We only replaced
        # modules in-place, so the original forward logic still applies.
        return self.det._forward_once(x)


class _QGBConcat(nn.Module):
    """Adapter so a QGBFusion node can sit where a yolov5 Concat was.

    YOLOv5 feeds Concat a list of branch tensors; QGBFusion expects the same.
    """

    def __init__(self, fuser: QGBFusion) -> None:
        super().__init__()
        self.fuser = fuser
        self.d = 1  # mimic Concat.d for yaml introspection

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return self.fuser(list(x))
        return x
