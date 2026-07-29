# BitNet-style ternary detection (W1.58 A8)

Port of Microsoft BitNet b1.58 training BitLinear into YOLOv5s for 2D detection,
plus an XNOR-Net binary (W1A1) variant with a popcount reference kernel.

## Quick start

```sh
# unit tests
PYTHONPATH=methods/bitnet_det python -m pytest methods/bitnet_det/tests/ -q

# VOC train (GPU server + Docker) — see recipes/bitnet_voc.md
```

## Layout

| File | Description |
| --- | --- |
| `bitlinear.py` | Official FAQ Fig. 3 BitLinear + detector BitConv2d |
| `xnor_conv.py` | XNOR+popcount binary conv (fake-quant + ref) |
| `yolo_bitnet.py` | YOLOv5s mid-body injection (stem + Detect FP) |
| `train_detect.py` | VOC training |
| `METHOD.md` | formulas and source mapping |
| `tests/` | ternary property, popcount bit-exactness, YOLO inject |

## References

- BitNet b1.58: arXiv:2402.17764
- Training code FAQ: microsoft/unilm `bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf`
- Inference: https://github.com/microsoft/BitNet
- XNOR-Net: arXiv:1603.05279
