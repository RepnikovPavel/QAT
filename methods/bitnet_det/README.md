# BitNet-style binary detection (XNOR + popcount)

Goal: bring 1-bit / 1.58-bit weight quantization (BitNet b1.58, XNOR-popcount
convolution) from microsoft/BitNet into 2D object detection and benchmark on
PASCAL VOC + COCO.

## Plan

1. Port the BitLinear layer (1.58-bit ternary weights, per-token LayNorm,
   absmax activations) from microsoft/BitNet into a detector-friendly module.
2. Replace YOLOv5 conv blocks with BitLinear where applicable (keep first/last
   and detection head in higher precision).
3. Train from scratch / fine-tune on VOC + COCO, measure mAP@0.5 and inference
   cost (XNOR+popcount throughput vs FP conv).

## References

- microsoft/BitNet (b1.58): https://github.com/microsoft/BitNet
- XNOR-Net (original binary conv): arXiv:1603.05279

Code lands here as the port matures.
