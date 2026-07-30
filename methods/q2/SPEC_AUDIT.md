# Q² spec audit — paper (arXiv:2511.05898v2) vs our code

Line-by-line reconciliation against the ocrc-parsed paper
(`papers/2511.05898/document.md`). Everything in the "paper" column is a verbatim
citation; the "ours" column is the current code state. Discrepancies flagged with
⚠ are candidates for the mAP gap and the next round of fixes.

## 1. Environment & scale (Sec 4.1 line 227, Appendix 8 line 433)

| Item | paper (verbatim) | ours | note |
| --- | --- | --- | --- |
| HW | "8 × NVIDIA GeForce RTX 4090 GPUs" | 1× RTX 4090 | ⚠ 1/8 of the GPUs |
| batch | "a batch size of 64" (8.1, line 437) | 32 | ⚠ half the effective batch |
| OS | "Ubuntu 20.04" (line 433) | ubuntu:22.04 in image | cosmetically different |
| torch | "PyTorch 2.3.1 and CUDA 11.8" (line 433) | torch 2.13+cu126 | ⚠ major torch/CUDA delta |
| workers | "4 data loader workers" (line 437) | 4 | ✓ |
| seed | "fixed random seed of 0" (line 437) | 0 | ✓ |

## 2. Optimizer / schedule (Appendix 8.1, line 437)

| Item | paper (verbatim) | ours | note |
| --- | --- | --- | --- |
| optim | "we use SGD" | SGD Nesterov | ✓ (Nesterov is an addition; verify) |
| init | "models initialized from full-precision pretrained checkpoints" | COCO yolov5s body, **head skipped** | ⚠ **HEAD IS COLD — biggest gap** |
| lr | "0.00334" | 0.00334 | ✓ |
| schedule | "OneCycleLR with a final ratio of 0.15135" | OneCycleLR, final_div_factor=1/0.15135 | ✓ |
| momentum | "0.74832" | 0.74832 | ✓ |
| weight decay | "0.00025" | 0.00025 (0 on quant params) | ✓ for weights; 0 on step is our fix |
| Q-ADA weight | "loss weight of 0.01" (line 433) | 0.01 | ✓ |

## 3. FP reference point (Table 1, line 237)

| Item | paper | ours |
| --- | --- | --- |
| YOLOv5s FP on VOC | **85.9%** ("YOLOv5s (FP: 85.9%)") | ⚠ not measured; this is the QAT **starting point** the authors mean by "full-precision pretrained checkpoint". Our COCO-pretrained body + random VOC head is NOT this checkpoint. |
| LSQ W4A4 baseline | 76.9% | 44.9% (30 ep, bs32, 1 GPU) |

**Reading of "initialized from full-precision pretrained checkpoints":** the
authors first train/finetune a full-precision YOLOv5s **on VOC** (reaches 85.9%),
then start QAT from THAT checkpoint. We start from COCO-pretrained weights with a
randomly-init'd VOC head, so the head must learn detection from scratch under
quantization — explaining most of the 76.9→44.9 gap.

## 4. Q-GBFusion (Sec 3.2, Eq 5–10)

| Eq | paper (verbatim) | ours (`qgbfusion.py`) | note |
| --- | --- | --- | --- |
| 5 | α=Softmax(λ); F'_i=α_i·F̃_i | softmax(λ); rescale | ✓ |
| 6 | per-channel LN over fused feat | per-channel LN | ✓ |
| 8 | Ḡ_i←(1−β)Ḡ_i+βG_i | EMA | ✓ |
| 9 | e_i=log(Ḡ_i+ε)−mean_j log(Ḡ_j+ε)−τ_i | same | ✓ |
| 10 | λ_i←λ_i−ηe_i | same | ✓ |
| β, η, τ | **not given in the paper** | η=0.05, β=0.1, τ=0 (defaults) | unspecified — our choice; τ=0 matches "default τ_i=0" (line 109) |
| ε | "ε>0 numerical-stability constant" | ε in code | ✓ qualitatively |

## 5. Q-ADA (Sec 3.3, Eq 13–16)

| Eq | paper (verbatim) | ours (`qada.py`) | note |
| --- | --- | --- | --- |
| 13 | Δ=|X−X̂| | same | ✓ |
| 14 | z=|X−μ_c|/(σ_c+κ); r=Δ/(s_c+κ) | same | ✓ |
| 15 | S=log(1+z²)+γ log(1+r²) | same | ✓ |
| 16 | JS divergence of P vs R (P=A^f/Σ, R=A^q/Σ) | JS (KL option) | ✓ |
| γ, κ | **not given in the paper** | our defaults | unspecified |
| attention weight | Ã=Sigmoid(S) | softmax(S) **for the distribution** | ⚠ we deviate; see METHOD.md note — softmax(S) vs softmax(sigmoid(S)). Paper says Ã=Sigmoid(S) then normalises to P; we softmax the raw S. Revisit. |
| teacher | "frozen pretrained FP model" (line 291) | **not implemented** — qada_targets always None | ⚠ Q-ADA is OFF in training |

## 6. Other protocol details NOT in the paper

The paper does **not** state (so any value is our assumption, to flag honestly):
- number of QAT epochs (Table 4 uses time-to-convergence / early stopping, not fixed epochs)
- image size (we use 640, YOLOv5 default)
- Q-GBFusion η, β; Q-ADA γ, κ
- which fusion nodes get Q-GBFusion (we use the 4 PANet Concat layers 12/16/19/22)
- whether the first/last layers stay 8-bit (LSQ paper does this; we quant all Conv except the Detect head)

## Priority fixes to close the gap (in order)

1. **FP-VOC checkpoint** (biggest): train full-precision YOLOv5s on VOC to ~85.9%,
   then init QAT from it (head warm). This is what "full-precision pretrained
   checkpoint" means.
2. **batch 64** (DDP across the 2× 4090, or grad-accum=2).
3. **Q-ADA teacher pass** (Sec 5, line 291): frozen FP teacher + feature hooks; the
   current training never builds teacher features, so Q-ADA contributes 0.
4. **Q-ADA attention distribution**: align with Eq 16 (normalise Ã=Sigmoid(S)),
   not softmax(S).
5. Match torch 2.3.1 + cu118 if exactness matters (likely a minor effect vs 1–4).
