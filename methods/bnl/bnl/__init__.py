"""Binary Normalized Layers (Cabral et al., arXiv:2509.07025)."""

from .quant import mean_threshold_quantize, ste_quantize
from .normalize import per_example_normalize
from .layers import BinaryNormalizedLinear, BinaryNormalizedConv2d

# yolo_bnl imports yolov5 lazily inside BNLYOLOv5.__init__; re-export the
# converter helpers that do not need yolov5 at import time.
from .yolo_bnl import conv_module_to_bnl, count_bnl_convs  # noqa: E402

__all__ = [
    "mean_threshold_quantize",
    "ste_quantize",
    "per_example_normalize",
    "BinaryNormalizedLinear",
    "BinaryNormalizedConv2d",
    "conv_module_to_bnl",
    "count_bnl_convs",
]

