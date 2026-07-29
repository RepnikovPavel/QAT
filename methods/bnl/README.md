# Binary Normalized Layers (BNL)

Port of Cabral et al. arXiv:2509.07025 binary normalized layers for use in
2D object detection QAT experiments.

- Paper parse: `papers/2509.07025/document.md`
- Method map + decision: `METHOD.md`
- Numbers: `RESULTS.md`
- Core: `bnl/` (`BinaryNormalizedLinear`, `BinaryNormalizedConv2d`)

```sh
# unit tests (local)
cd methods/bnl && python -m pytest tests/ -q
```
