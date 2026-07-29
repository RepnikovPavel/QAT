# Paper arXiv:2603.05964

Source: arXiv:2603.05964 (pdftotext first-pass; ocrc parse supersedes when ready).

```
                                             QATMA: Quantization-Aware Training with
                                             Multimodal Alignment for Open-Vocabulary
                                                         Object Detection

                                             Jinyeong Park1 , Donghwa Kang2 , Seunghwan An1 , Insoo Kim1 , Brent
                                                   ByungHoon Kang2 , Hyeongboo Baek†3 , and Jibum Kim‡1
                                                        1
                                                         Incheon National University, Incheon, South Korea
                                                                 2
                                                                   KAIST, Daejeon, South Korea
                                                            3
                                                              University of Seoul, Seoul, South Korea
                                                 Corresponding authors: † hbbaek@uos.ac.kr, ‡ jibumkim@inu.ac.kr
arXiv:2603.05964v3 [cs.CV] 23 Jul 2026




                                               Abstract. Quantizing open-vocabulary object detection (OVOD) mod-
                                               els reduces their memory and computational costs, but extremely low-bit
                                               quantization severely degrades both cross-modal (region-text) and intra-
                                               modal (region-region) alignments. This multimodal degradation is a
                                               unique challenge that prior quantization methods for closed-vocabulary
                                               detectors fail to resolve. To overcome this, we propose Quantization-Aware
                                               Training with Multimodal Alignment (QATMA), the first multimodal-
                                               aware and architecture-agnostic QAT framework tailored for OVOD.
                                               QATMA integrates two key components: (i) Curriculum QAT, which
                                               partitions the detector by functional roles and progressively expands
                                               the quantization scope to suppress error accumulation and ensure sta-
                                               ble optimization; and (ii) Text-anchored Pairwise Similarity Distillation,
                                               which transfers both region-text and region-region alignments from a
                                               full-precision teacher model via pairwise cosine similarities in the joint
                                               embedding space. Experimental results on LVIS and COCO zero-shot
                                               benchmarks demonstrate that QATMA significantly outperforms existing
                                               QAT baselines under extremely low-bit settings, achieving gains of up to
                                               4.3 and 7.6 AP, respectively.

                                               Keywords: Open-vocabulary object detection · Modality Alignment ·
                                               Quantization-aware training · Knowledge distillation


                                         1   Introduction
                                         The advent of vision-language models (VLMs) [22, 39] has shifted the object
                                         detection paradigm toward open-vocabulary object detection (OVOD) [48], which
                                         detects novel categories beyond predefined sets. However, OVOD models [14, 25,
                                         29, 50] rely on heavy ViT-based backbones and text encoders, incurring massive
                                         computational overhead. Although lightweight, real-time OVOD models such
                                         as YOLO-World [7] have been proposed, their overhead remains substantial,
                                         so they still require further compression such as pruning [17, 18], knowledge
                                         distillation [5, 19], or quantization [11, 21] for edge deployment.
2          J. Park et al.

    FP32         PTQ         QAT          Ours        0.89
                                                      0.45                     0.100                      PTQ




                                                             Region-text MAE
                                                                               0.075
    P=0.56      P=0.00      P=0.20       P=0.43
                                                      0.00                                                         QAT
                                                                               0.050     Ours
        (a) Region-text alignment: category “Train”
                                                                               0.025
    FP32         PTQ         QAT          Ours        1.00                                FP32
                                                                               0.000
                                                                                       0.00 0.03 0.06 0.09 0.12 0.15
                                                      0.71                                Region-region MAE
 reference      r = 0.68    r = 0.19    r = 0.98
                                                      0.43                       (c) Embedding distortion

                (b) Region-region alignment

Fig. 1: Impact of 4-bit quantization on YOLO-World [7] ( Objects365v2 [44]). (a)
Confidence scores from the similarity between each region embedding and the “Train”
text embedding (P̄ : mean over positive regions). (b) Pairwise cosine similarity matrix
among positive region embeddings within the same category (r: Pearson correlation
with FP32). (c) Quantitative comparison of embedding distortion relative to FP32;
closer to the origin indicates less distortion.



    Quantization replaces floating-point operations with low-bit integer arithmetic
without architectural changes. It is broadly categorized into post-training quanti-
zation (PTQ), which calibrates a pretrained model without retraining [9, 34, 35],
and quantization-aware training (QAT), which simulates quantization during
training [2, 11, 21]. While quantization has been explored for object detec-
tion [6, 16, 26, 47], existing methods address only closed-vocabulary settings,
leaving a critical gap for OVOD, which depends heavily on precise vision-language
alignment.
    We investigate this gap by analyzing extremely low-bit (4-bit) quantization
on YOLO-World with the Objects365v2 [44] dataset. Specifically, we examine
how quantization affects the fine-grained cross-modal and intra-modal alignment
in the joint embedding space. The former is region-text alignment, which is the
cosine similarity between region and text embeddings. The latter is region-region
alignment, which is the pairwise cosine similarity among region embeddings
assigned to the same text query. As shown in Fig. 1(a), PTQ destroys region-text
alignment, and naive QAT achieves only partial recovery. Furthermore, naive
QAT fails to preserve region-region alignment (Fig. 1(b)).
    A quantitative comparison (Fig. 1(c)) further reveals that naive QAT fails to
jointly minimize the distortions of both region-text and region-region alignments,
each measured as mean absolute error (MAE) from FP32. This suggests that,
under extreme capacity constraints, the model overfits to ground-truth matching
scores while sacrificing intra-modal alignment. Task loss alone is therefore insuffi-
cient to preserve both alignments. In contrast, our method jointly minimizes both
distortions, most closely approaching the FP32 reference (Fig. 1(c)). Knowledge
distillation (KD) is a natural candidate for addressing this limitation, transfer-
         QATMA: Quantization-Aware Training with Multimodal Alignment            3

ring knowledge from a full-precision teacher to a quantized student. However,
at extremely low bit-widths, single-stage QAT is unstable [51], and the large
teacher–student capacity gap further hinders distillation [33].
    In this context, we propose Quantization-Aware Training with Multimodal
Alignment (QATMA), a novel framework that synergizes stage-by-stage opti-
mization with module-specific knowledge distillation. QATMA is architecture-
agnostic: it applies to any OVOD detector with the backbone-neck-head structure
and region-text matching common to such models. We design curriculum QAT
(CQAT) to enable effective distillation by suppressing error accumulation. Instead
of quantizing the whole detector at once, CQAT partitions it into functional
modules and progressively expands the quantization scope, keeping the remaining
modules frozen to isolate errors.
    Building on this curriculum, we design a module-specific KD strategy tailored
to each component’s functional role. We apply feature distillation to the task-
agnostic backbone to recover its representation capability. In contrast, the task-
relevant neck-head governs the cross-modal and intra-modal alignment that
conventional, closed-vocabulary detection KD does not address. We therefore
propose Text-anchored Pairwise Similarity Distillation (TPSD), which jointly
transfers region-text and region-region alignment via pairwise cosine similarities
in the joint embedding space.
    Our main contributions are summarized as follows:
 – To the best of our knowledge, this is the first work to tackle extremely low-bit
   quantization of OVOD models. We propose QATMA, the first multimodal-
   aware and architecture-agnostic QAT framework, in which CQAT suppresses
   error accumulation by progressively quantizing the detector’s functional
   modules, providing a stable optimization foundation that single-stage QAT
   lacks.
 – We propose TPSD, a distillation method for the cross-modal and intra-modal
   alignment central to OVOD, which conventional detection KD overlooks.
   TPSD distills both alignments from the teacher through text-anchored pair-
   wise cosine similarities in the joint embedding space.
 – Extensive experiments demonstrate that QATMA significantly outperforms
   existing QAT baselines under extremely low-bit settings, with gains of up to
   4.3 AP on LVIS and 7.6 AP on COCO. Beyond YOLO-World, QATMA also
   proves effective on the DETR-based OmDet-Turbo, confirming its generality
   across OVOD architectures.


2     Related Work
2.1   Open-Vocabulary Object Detection
Object detection has advanced through two-stage [13, 42], one-stage [27, 30, 45],
transformer-based [4,49,58], and YOLO-series [23,40,41,46] paradigms, but all rely
on a fixed set of training categories [28, 44]. This closed-vocabulary assumption
prevents novel category detection, restricting their open-world applicability.
4       J. Park et al.

    To overcome this, OVOD was introduced to detect novel categories via ar-
bitrary text queries. Early work like OVR-CNN [48] established the standard
OVOD setting. Leveraging large-scale VLMs [22,39], ViLD [14], RegionCLIP [53],
and OWL-ViT [32] transferred CLIP’s visual-semantic representations, while
GLIP [25, 50] and Grounding DINO [29] reformulated detection as region-text
matching. These methods achieve remarkable generalization, yet most rely on
heavy ViT backbones (e.g., Swin-T [31]) and complex cross-modal fusion, incur-
ring massive computational costs and memory footprints that hinder real-time
deployment.
    Recently, real-time OVOD detectors have emerged across diverse architectures,
pre-computing text embeddings offline to remove the inference-time text encoder.
For instance, YOLO-World [7] builds on a lightweight YOLOv8 [23] backbone,
while OmDet-Turbo [52] adopts DETR with an efficient fusion head. Despite this
progress, such lightweight models still retain a substantial number of parameters
and involve intensive cross-modal operations, necessitating further compression
such as quantization for edge deployment. It remains a critical challenge to design
an effective quantization technique that preserves the fine-grained vision-language
alignment of OVOD models.

2.2   Quantization for Object Detection
Network quantization reduces memory and computational overhead by repre-
senting models in low-bit precision. Broadly categorized into PTQ [9, 34, 35]
and QAT [2, 8, 11, 21, 55], it has achieved notable success in image classification.
While object detection quantization has been explored, it remains less investi-
gated. Early methods [21, 26, 59] were limited to 8-bit precision, struggled with
training instability, or imposed architectural constraints. To achieve accurate,
fully quantized object detection, subsequent works proposed multi-level batch
normalization [6] and weight oscillation mitigation [16, 36].
    Since naive QAT struggles to recover performance in extremely low-bit set-
tings, integrating knowledge distillation (KD) from a full-precision teacher has
become prevalent [24, 38]. For dense prediction, task-specific designs are more
effective, such as FPN-level feature distillation for one-stage detectors [57], distri-
bution rectification distillation for DETR-based detectors [47], and view-guided
distillation for BEV-based 3D detection [51].
    However, all aforementioned studies are strictly confined to closed-vocabulary
settings. The quantization of OVOD models, which heavily rely on fine-grained
vision-language alignment, remains largely unexplored. To bridge this gap, our
work proposes QATMA, a low-bit quantization framework tailored for OVOD
models.

3     Preliminaries
In this section, we briefly review the fundamental concepts of network quantization
and open-vocabulary object detection, which form the basis of our proposed
framework.
                QATMA: Quantization-Aware Training with Multimodal Alignment                                                                             5

3.1        Quantization

We assume uniform quantization for weights and activations. For a parameter
w ∈ R, b-bit uniform quantization maps it to a discrete level. Given a scale s > 0,
a zero-point z ∈ Z, and a range [l, u] defined by the bit-width b, the process is
defined as:
                             \bar {w} = \left \lfloor \text {clip}\!\left (\frac {w}{s} + z,\; l,\; u\right ) \right \rceil , \label {eq:quantize}  (1)

where ⌊·⌉ denotes round-to-nearest, and clip(x, l, u) clips the value of x to the
interval [l, u]. The range is [−2b−1 , 2b−1 − 1] for signed and [0, 2b − 1] for unsigned
integers. The dequantized parameter ŵ is reconstructed as:

                                                                           \hat {w} = (\bar {w} - z) \cdot s. \label {eq:dequantize}                    (2)

This scheme is termed symmetric quantization when z = 0, and asymmetric
otherwise.
    QAT simulates these operations during training. In the forward pass, w
undergoes quantize-dequantize steps (Eqs. (1) and (2)) to compute the loss L.
During backpropagation, the straight-through estimator (STE) [1] approximates
gradients through the non-differentiable rounding operator. Furthermore, s and
z can be jointly optimized as learnable parameters [2, 11], allowing the model to
actively compensate for quantization errors, thereby achieving significant gains
over post-training quantization in low-bit settings.


3.2        Open-Vocabulary Object Detection

Closed-vocabulary detectors rely on a parameterized classifier tailored to a
fixed set of training categories Cbase . Consequently, they cannot detect novel
categories in open-world scenarios. OVOD transcends this by detecting unseen
categories Cnovel (Cbase ∩ Cnovel = ∅) during inference, despite training solely on
Cbase annotations [48].
    A key idea behind OVOD is to replace the fixed classifier with text embeddings.
Specifically, OVOD models [7, 25, 50] employ dual encoders, where a text encoder
encodes each category query into a text embedding t ∈ RD and an image encoder
extracts a region embedding v ∈ RD for each object candidate from multi-scale
visual features.
    Final classification relies on region-text similarity scores, computed as the
cosine similarity between the region embedding v and the text embedding t.
Leveraging text encoders pretrained on massive datasets [39], this region-text
matching seamlessly extends detection to both Cbase and Cnovel simply by altering
the query set.
                                                                         Ngt
    Given an input image x and ground-truth annotations y = {(βi , ci )}i=1  , where
Ngt is the number of ground-truth objects with bounding boxes βi and base
categories ci ∈ Cbase , the model optimizes:

                                            \mathcal {L}_{\text {task}}(x, y) = \mathcal {L}_{\text {cls}}(x, y) + \mathcal {L}_{\text {loc}}(x, y),    (3)
6       J. Park et al.

where Lcls and Lloc denote the classification loss (e.g., focal loss [27]) on region-
text similarity scores and the localization loss (e.g., IoU-based losses [43, 45]),
respectively.
    Crucially, the open-vocabulary detection capability relies heavily on the
fine-grained alignment between region and text embeddings. Extremely low-bit
quantization distorts this embedding space, degrading both cross-modal and
intra-modal alignments and consequently impairing classification. Preserving
these alignments under such quantization is the central motivation for QATMA.


4     Method

QATMA is an integrated framework that mitigates quantization error accumula-
tion and restores both cross-modal and intra-modal alignments. It consists of two
synergistic components: CQAT for stable, stage-by-stage optimization, and TPSD
tailored for task-relevant modules to reconstruct region-text and region-region
alignment.


4.1   Curriculum QAT

Single-stage QAT quantizes all layers simultaneously to optimize the task loss
Ltask . However, in extremely low-bit settings, quantization errors originating
from early layers accumulate rapidly and propagate through subsequent layers.
This results in severe degradation of representation capability, making recovery
challenging [51].
    To overcome this, CQAT partitions the model into K functional modules
{M1 , M2 , . . . , MK } and progressively expands the quantization scope along the
data flow. Unlike prior progressive quantization, which proceeds at the level
of individual weights [54] or layers [37], CQAT instead defines the progression
over functional modules grouped by their role in the detection pipeline, letting
each stage be paired with a module-specific distillation objective (Sec. 4.2). Let
wk be the weights and ϕk be the quantization parameters (e.g., scaling factor,
zero-point) of module Mk . At stage k, we apply QAT to the first k modules while
freezing the remaining modules (Mk+1 , . . . , MK ) in full precision. The task loss
at stage k is formulated as:

               \begin {aligned} \arg \min _{\{\mathbf {w}_i, \phi _i\}_{i=1}^{k}} &\quad \mathcal {L}_{\text {task}}(x,y) \\ \text {s.t.} &\quad M_{k+1},\ldots ,M_K \text { remain in full precision}. \end {aligned} 
                                                                                                                                                                                                                          (4)


    The core principles underlying this design are error isolation and sequential
recovery. Error isolation prevents premature noise propagation, allowing unquan-
tized modules to maintain stable intermediate feature processing and gradient
flow. Meanwhile, sequential recovery ensures that each module is optimized on top
of quantization-adapted inputs from preceding modules, enabling it to primarily
compensate for its own quantization error.
            QATMA: Quantization-Aware Training with Multimodal Alignment                                                   7


  (a) Stage 1: Task-agnostic module quantization
   Text queries             Task-agnostic module (𝑀! )
      person
      truck                     Backbone 🔥                             Neck                Head                    ℒ$#%&
      airplane
                                                     ℒ!"#$
                                                                     🔥 Quantized & lernable            Forward pass
                                Backbone
                                                                        Full precision & frozen        Supervision pass


  (b) Stage 2: End-to-end quantization
   Text queries             Task-agnostic module (𝑀! )                    Task-relevant module (𝑀" )
      person
      truck                     Backbone 🔥                             Neck 🔥             Head 🔥                   ℒ$#%&
      airplane
                                                     ℒ!"#$                                             ℒ'()*
                                Backbone                               Neck               Head



  (c) Feature distillation (ℒ𝐟𝐞𝐚𝐭 )         (d) Text-anchored pairwise similairty distillation (ℒ'()* )
  Preserve task-agnostic representations   Preserve cross-modal & intra-modal alignment
                                           ① Text & region embeddings ② Unified pairwise similairty matrix
  𝑓!" 𝑥                                                                         Teacher 𝐒#'                  Student 𝐒#(
                                                             𝐯#,!             𝐭 # 𝐯#,! ⋯ 𝐯#,%!
                                                    𝐭#                     𝐭#                   Region-text
                                                                         𝐯#,!
                                                              𝐯#,"
  𝑓!# 𝑥                                                                    ⋮                   Region-region
                                            𝐯#,*       𝐯#,)
                                                                        𝐯#,%!


Fig. 2: Overview of the proposed QATMA framework. (Red ) blocks denote quantized
and learnable modules, and (blue) blocks denote full-precision and frozen modules. (a)
Stage 1: the backbone (M1 ) is quantized with Lfeat supervision from the full-precision
teacher, while the neck-head remains frozen for error isolation. (b) Stage 2: the neck-
head (M2 ) is additionally quantized, supervised by both Lfeat and LTPSD . (c) Feature
distillation aligns the student’s multi-scale backbone features f1S (x) to those of the
teacher f1T (x). (d) TPSD groups region embeddings by text query tc and constructs a
unified pairwise similarity matrix Sc to transfer region-text and region-region alignment.



    For widely-adopted OVOD architectures (e.g., GLIP [25], YOLO-World [7]),
we instantiate a two-stage curriculum (K = 2). As illustrated in Fig. 2, Stage 1
(Fig. 2(a)) quantizes the task-agnostic backbone (M1 ) to recover from multi-scale
feature degradation, utilizing the frozen neck-head (M2 ) as an error isolator.
Stage 2 (Fig. 2(b)) subsequently quantizes the neck-head to complete end-to-end
optimization. We treat the neck and head as a single task-relevant module, as
both carry out cross-modal fusion and region-text matching—the core multimodal
operations in OVOD. We validate this grouping in Sec. 5.


4.2       Text-anchored Pairwise Similarity Distillation

While CQAT structurally mitigates error accumulation, the information lost due
to severe bit-width reduction remains substantial. To enhance sequential recovery,
8            J. Park et al.

QATMA integrates module-specific KD into each curriculum stage. By leveraging
the stable optimization foundation of CQAT, KD can be effectively applied
even in extremely low-bit settings. Accordingly, we design a module-specific KD
strategy based on the functional role of each module.
    Given a full-precision teacher, the objective at stage k is:


                                                        \mathcal {L}_{\text {stage}\;k} = \mathcal {L}_{\text {task}} + \sum _{i=1}^{k} \lambda _i \mathcal {L}_{\text {KD}}^{(i)},              (5)


              (i)
where LKD is the KD loss specifically designed for module Mi , and λi is its
balancing weight. As the curriculum progresses, KD losses from previous stages
are retained as regularizers to prevent re-degradation.


Backbone (M1 ): Feature Mimicking. As a task-agnostic feature extractor,
the backbone produces features whose quality fundamentally dictates the per-
formance of all subsequent modules. Thus, we directly mimic the multi-scale
features of the teacher’s backbone (Fig. 2(c)) using a feature distillation loss
Lfeat :
                            \mathcal {L}_{\text {KD}}^{(1)} = \mathcal {L}_{\text {feat}}\!\left (f_1^S(x),\; f_1^T(x)\right ),  (6)
where f1S (x) and f1T (x) denote the backbone output features of the student and
teacher, respectively. In practice, we adopt PKD [3] as Lfeat .


Neck-head (M2 ): TPSD. The task-relevant neck-head governs fine-grained
vision-language alignment. Extremely low-bit quantization severely degrades both
cross-modal and intra-modal alignments. Conventional KD, however, transfers
intermediate features or output logits, rather than the alignment in the joint
embedding space. To address this, we propose TPSD, which uses text embeddings
as anchors to capture both alignments within a unified pairwise similarity matrix
(Fig. 2(d)).
    For each text query c, TPSD constructs a matrix comprising its text embedding
tc and Nc assigned region embeddings {vc,n }N   c
                                              n=1 :


                                    \mathbf {X}_c = \begin {bmatrix} \mathbf {t}_c & \mathbf {v}_{c,1} & \cdots & \mathbf {v}_{c,N_c} \end {bmatrix}^\top \in \mathbb {R}^{(1+N_c) \times D},    (7)

where D is the embedding dimension. The region embeddings assigned to each
text query (i.e., the positive regions for category c) are determined by the label
assignment scheme of the base detector (Sec. 5).
    After row-wise L2-normalization to obtain X̂c , we compute the pairwise cosine
similarity matrix:
                            \mathbf {S}_c = \hat {\mathbf {X}}_c \hat {\mathbf {X}}_c^\top \in \mathbb {R}^{(1+N_c) \times (1+N_c)}.  (8)
The first row and column of this matrix explicitly encode the region-text alignment,
while the remaining internal blocks capture the region-region alignment.
         QATMA: Quantization-Aware Training with Multimodal Alignment                                                                                                                                                                                     9

   We compute this matrix for both the teacher and the student, denoted STc
and SSc , respectively. By minimizing their discrepancy, the student retains the
teacher’s cross-modal and intra-modal alignment:


            \mathcal {L}_{\text {KD}}^{(2)} = \mathcal {L}_{\text {TPSD}} = \frac {1}{|\mathcal {C}|} \sum _{c \in \mathcal {C}} \frac {1}{(1+N_c)^2} \sum _{i=1}^{1+N_c} \sum _{j=1}^{1+N_c} \ell _\delta \!\left (S^S_{c,ij},\; S^T_{c,ij}\right ),    (9)


where C is the set of text queries in the image, and ℓδ denotes the smooth L1
loss. Since object frequencies are long-tailed, frequent or large-object categories
yield far more assigned regions Nc , which would dominate a single global average.
We therefore adopt a text-query-balanced average: normalizing within each text
query before averaging across queries, so that every text query contributes equally
regardless of Nc .


5     Experiments
QATMA is architecture-agnostic, making it directly applicable to any OVOD
detector configured with a backbone-neck-head structure and region-text match-
ing. We primarily evaluate our framework on YOLO-World [7]—a representative,
lightweight, real-time OVOD detector—under an extremely low-bit (4-4-8) quan-
tization setting for zero-shot detection, alongside comprehensive ablation studies.
Additionally, we evaluate QATMA on OmDet-Turbo [52] to verify its architectural
generality.

5.1   Experimental Settings
Datasets and Evaluation. For QAT training, we use the Objects365v2 [44] train
split and GQA [20]. Following YOLO-World, zero-shot evaluation is conducted
on LVIS [15] minival with the Fixed AP [10] and COCO [28] val2017.

Quantization Settings. We use official pre-trained YOLO-World-M/L/X
checkpoints. The text encoder (CLIP [39]) is excluded from quantization, as it is
removed at inference by pre-computing text embeddings offline [7]. We apply sym-
metric quantization for weights and asymmetric for activations and attention. Our
main results adopt a 4-4-8 bit-width configuration (weight-activation-attention)
with Ch-T-H granularity (per-channel weight, per-tensor activation, per-head
attention). The first and last layers remain unquantized, following common prac-
tice [8, 55]. For calibration, we use 256 samples from the Objects365v2 validation
set.

Training Details. We adopt the YOLO-World pretraining configuration, except
for the learning rate of 1.5 × 10−5 and the batch size of 48 for QAT. The learning
rate scheduler is disabled as we fine-tune for 1 epoch. The text encoder remains
frozen. Quantization parameters are learned via LSQ [2, 11] with a learning rate
10       J. Park et al.

Table 1: LVIS minival zero-shot evaluation with 4-4-8 quantization (Ch-T-H granu-
larity). Ch = per-channel, T = per-tensor, H = per-head. Evaluation follows the LVIS
fixed AP protocol [10].

Model              Bits    Size(MB) BOPs(T) Method           AP   APr APc APf
                   FP32     110.88   51.82    -             31.0 23.8 29.2 33.9
                                              QAT         13.8    10.7   11.6 16.3
YOLO-World-M
                                              EMA+QC [16] 13.7    11.8   12.2 15.4
                   4-4-8    15.11     7.10
                                              QFD [57]    13.0     7.1   11.0 15.8
                                              Ours        16.3    11.4   14.5 18.8
                   FP32     181.4    98.39    -             35.4 27.6 34.1 38.0
                                              QAT         12.8 8.2 11.2 15.0
YOLO-World-L
                                              EMA+QC [16] 13.1 9.0 11.3 15.3
                   4-4-8    24.32     8.18
                                              QFD [57]    13.0 9.5 11.9 14.7
                                              Ours        16.1 13.4 14.3 18.2
                   FP32     281.11   149.08   -             36.6 29.4 35.0 39.4
                                              QAT         12.7 7.4 11.7 14.5
YOLO-World-X
                                              EMA+QC [16] 12.0 11.7 9.4 14.4
                   4-4-8    37.19     9.32
                                              QFD [57]    13.2 9.7 11.6 15.3
                                              Ours        17.0 14.3 15.1 19.1



of 0.1× the base rate. Curriculum stage 1 (backbone) uses the first 1/3 of the
data, while stage 2 (neck-head) uses the remaining 2/3. The KD loss weight λi is
6.0 for both feature distillation and TPSD. For TPSD, regions assigned to each
text query are positive samples determined by the teacher’s task-aligned label
assignment (TAL) [12].

5.2     Main Results
The 4-4-8 (W4A4) configuration with per-tensor activation quantization is highly
aggressive. PTQ suffers severe performance collapse (AP 0.0) across all models on
both LVIS and COCO regardless of calibration strategy (MinMax, Percentile [26],
OMSE [9]), demonstrating that low-bit quantization of OVOD models is infea-
sible without training. The QAT baseline simultaneously quantizes all layers
and trains with only the task loss Ltask , using LSQ [2, 11], a widely adopted
learnable quantization scheme. For fair comparison, the baseline shares the same
training configuration, total data, and number of iterations as QATMA. We
additionally compare against two closed-vocabulary detection QAT baselines: (i)
EMA+QC [16], which mitigates weight and activation oscillations during low-bit
YOLO QAT, and (ii) QFD [57], a KD-based QAT that quantizes a full-precision
teacher’s FPN-level features and distills them to the student. Under this setting,
4-4-8 quantization reduces model size by up to 7.6× and bit operations (BOPs)
by up to 33.4× (reported per benchmark in Tabs. 1 and 2, as BOPs vary with
the number of text queries).
         QATMA: Quantization-Aware Training with Multimodal Alignment             11

Table 2: COCO val2017 zero-shot evaluation with 4-4-8 quantization (Ch-T-H granu-
larity)

Model            Bits    BOPs(T) Method      AP     AP50 AP75     APs    APm    APl
                 FP32     44.56    -         41.9   57.0   45.5   26.7   46.1   55.0
YOLO-World-M
                                   QAT      22.0 32.1      23.6  7.8 23.5 32.4
                 4-4-8     2.05
                                   Ours     26.1 38.1      28.1 11.0 28.1 37.9
                 FP32     90.60    -         45.1   60.7   48.9   29.8   49.8   57.5
YOLO-World-L
                                   QAT      18.4 27.2      19.6  7.5 20.1 26.9
                 4-4-8     3.09
                                   Ours     25.2 36.7      27.0 10.5 27.5 37.7
                 FP32     140.66   -         46.7   62.8   50.9   31.8   51.2   60.8
YOLO-World-X
                                   QAT      18.6 26.8      19.9  7.8 20.5 27.0
                 4-4-8     4.21
                                   Ours     26.2 38.2      28.1 12.2 28.7 37.2



LVIS MiniVal Zero-Shot. LVIS [15] contains 1,203 object categories, each
classified by annotation frequency into rare (APr ), common (APc ), and frequent
(APf ). The large number of categories and the high proportion of rare categories
make it a benchmark that directly reflects the difficulty of open-vocabulary
detection. Table 1 reports results for YOLO-World-M/L/X. While QAT recovers
from the complete failure of PTQ, a large gap relative to FP32 remains, with
drops of −17.2 AP on YOLO-World-M and −22.6 on YOLO-World-L. QATMA
consistently outperforms QAT across all models, with improvements of +2.5,
+3.3, and +4.3 AP on YOLO-World-M, L, and X, respectively. Notably, QAT
degradation grows with model scale, and our improvements grow accordingly,
indicating that our method becomes more beneficial as the capacity gap widens.
The gains are particularly pronounced in APr , which measures rare category
detection and serves as a key indicator of open-vocabulary capability [7, 25].
Since rare categories heavily depend on vision-language alignment quality, the
substantial APr gains of +5.2 on YOLO-World-L and +6.9 on YOLO-World-X,
corresponding to 63.4% and 93.2% relative improvements, demonstrate that
QATMA effectively restores the fine-grained alignment degraded by quantization.
Moreover, QATMA consistently outperforms not only naive QAT but also the
closed-vocabulary detection QAT baselines EMA+QC and QFD across all model
scales. Since these baselines do not target the vision-language alignment that
underlies OVOD zero-shot detection, they fail to recover the alignment distorted
under 4-bit quantization.


COCO Val2017 Zero-Shot. COCO [28] contains 80 categories with scale-wise
evaluation across small (APs ), medium (APm ), and large (APl ), making it suitable
for assessing scale robustness. As shown in Tab. 2, PTQ again completely fails with
AP 0.0, and QAT leaves a large gap from FP32. QATMA consistently outperforms
QAT, with improvements of +4.1 and +6.8 AP on YOLO-World-M and YOLO-
12      J. Park et al.

Table 3: Ablations on QATMA. KD in-          Table 4: TPSD component analysis. All
cludes feature distillation and TPSD. LVIS   variants use CQAT with feature distilla-
zero-shot AP is reported.                    tion. LVIS zero-shot AP is reported.

          Curriculum KD AP                         TPSD variant           AP
               ✗         ✗ 12.8                    w/o TPSD (feat only) 14.5
               ✓         ✗ 14.0                    region-text          15.0
               ✗         ✓ 13.1                    region-region        15.4
               ✓         ✓ 16.1                    full TPSD             16.1



World-L, and the gain further increases to +7.6 AP (40.9%) on YOLO-World-X,
again confirming larger gains at greater model scales. These improvements also
hold across all object scales, as illustrated by YOLO-World-M where APs , APm ,
and APl improve by +3.2, +4.6, and +5.5, respectively. The +7.4 gain in AP75 on
YOLO-World-L further demonstrates effective recovery of localization precision,
confirming that QATMA restores not only classification alignment but also fine-
grained localization capability. QATMA similarly outperforms EMA+QC and
QFD on COCO, as detailed in the supplementary material.


5.3   Ablation Studies

Ablations on QATMA. Table 3 analyzes the individual contributions of the
curriculum strategy and KD on YOLO-World-L with 4-4-8 (Ch-T-H) quantization.
Applying only the curriculum (CQAT) to QAT (AP 12.8) yields AP 14.0 (+1.2),
while applying KD alone yields only AP 13.1 (+0.3). Combining both achieves
AP 16.1 (+3.3), far exceeding the sum of their individual gains (+1.5). This
strong synergy confirms that the curriculum is an essential foundation for KD to
operate effectively.


Ablations on TPSD. Table 4 decomposes TPSD into its two components:
region-text alignment and region-region alignment. Using only CQAT with back-
bone feature distillation (i.e., conventional KD without TPSD) yields AP 14.5,
only +0.5 over CQAT alone (14.0). Distilling only region-text alignment yields AP
15.0 (+0.5), while distilling only region-region alignment yields AP 15.4 (+0.9).
Combining both, full TPSD achieves the best AP of 16.1, a further +1.6 that
far exceeds the +0.5 from conventional feature distillation. This confirms that
preserving cross-modal and intra-modal alignment is critical to low-bit OVOD
accuracy.


Effect of Curriculum Stages. Table 5 compares two-stage (backbone→neck-
head) and three-stage (backbone→neck→head) curriculum strategies. Two-stage
is consistently superior, both without KD (14.0 vs. 11.0) and with KD (16.1 vs.
15.3). The neck and head jointly perform cross-modal fusion and region-text
           QATMA: Quantization-Aware Training with Multimodal Alignment                                                      13


Table 5: Effect of curriculum stages. 3-                                          0.8




                                              Confidence-space distortion (MAE)
stage uses 1/3 data per stage. LVIS zero-                                                     QAT
shot AP is reported.
                                                                                              Ours
                                                                                  0.6
    Curriculum stages          KD AP
                                ✗ 14.0
    2 (backbone→neck-head)                                                        0.4
                                ✓ 16.1
                                ✗ 11.0
    3 (backbone→neck→head)
                                ✓ 15.3                                            0.2

Table 6: Generalization of QATMA to                                               0.0
OmDet-Turbo [52] under 4-4-8 (Ch-T-H)                                                   0.0     0.2     0.4      0.6     0.8
quantization. LVIS zero-shot AP is re-                                                    Embedding-space distortion (MAE)
ported.
                                                   Fig. 3: Embedding-space vs. confidence-
       Method AP APr APc APf                       space distortion (MAE of the region-region
       FP32     29.1 22.5 28.2 31.1                similarity relative to FP32). Each point
                                                   is an (image, category) group with ≥10
       QAT      12.8 9.0 11.1 14.9
                                                   anchors.
       Ours     17.6 10.9 16.9 19.5



matching, forming a natural functional unit. Notably, our KD improves both
partitions substantially (+2.1 for two-stage and +4.3 for three-stage) and narrows
their gap from 3.0 to 0.8. This demonstrates that our KD is robust to the choice
of curriculum partition.


5.4     Analysis

Architecture Generality. We apply QATMA to OmDet-Turbo [52]4 , a real-
time OVOD detector whose DETR-family architecture is fundamentally distinct
from the one-stage anchor-free YOLO-World. TPSD is applied identically, except
that positive samples are the top-K decoder queries per instance rather than
dense TAL anchors. Table 6 reports zero-shot results on LVIS minival under 4-
4-8 (Ch-T-H) quantization after one epoch of QAT. QATMA improves over QAT
by +4.8 AP, confirming generalization across heterogeneous OVOD architectures.


Embedding-to-Confidence Alignment Transfer. To verify that preserving
intra-modal alignment also benefits the predictions, we measure how quantization
distorts the region-region pairwise similarity in two spaces on 512 LVIS samples
(Fig. 3): among region embeddings and among their confidence score vectors.
Preserving the latter is known to retain a model’s discriminative knowledge
and output-distribution fidelity [56]. The two distortions are strongly correlated
4
    It releases only checkpoints; we re-implement its training and evaluation pipeline.
14      J. Park et al.

                                             FP32                                                                QAT                                                                     Ours
               shower curtain:
                         hand 0.58
                               towel: 0.48                                           curtain:
                                                                            toothbrush:
                                                                              toothbrush:   0.470.38
                                                                                          0.31                                                                                                                              cupboard: 0.33
                                                                   bottle: 0.45                                                          water jug: 0.39                                                   bottle: 0.43
                                                    faucet: 0.35             coaster: 0.40                                faucet:faucet:
                                                                                                                                  0.63 0.61                                                        faucet: 0.40      clutch bag: 0.37
                                                                                                                faucet: 0.67
                                       sink: 0.82                                                            sink: 0.91                                                              sink: 0.80


                  cabinet:
                   drawer: 0.55
                           0.63       drawer: 0.63              drawer: 0.59                                                                                     drawer: 0.44       drawer: 0.45           drawer: 0.43
                                                  combination lock: 0.33
                     drawer: 0.36                                  drawer: 0.36                                cupboard: 0.31                                      drawer: 0.31                           drawer: 0.38


                       drawer: 0.47                                drawer: 0.38                                                                                      drawer: 0.45




                                                                                  0.2                  0.4                      0.6                        0.8                      1.0

Fig. 4: Qualitative comparison on YOLO-World-L (4-4-8, Ch-T-H). (Top) Detection
results. (Bottom) Region-region similarity heatmap of average pairwise cosine similarity
among same-class anchors. QAT distorts both detection and similarity patterns of FP32,
whereas QATMA restores them.


(Spearman correlation of 0.76), showing that embedding-space distortion prop-
agates directly to the predictions. QATMA reduces both far more than QAT
(MAE 0.1455 → 0.0494 and 0.2766 → 0.1813) because TPSD explicitly distills
this alignment, which QAT’s task-only objective leaves distorted.

Qualitative Analysis. Figure 4 presents a qualitative comparison of zero-shot
detection results on an LVIS image. In the detection results (top), QAT shows
a significant reduction in the number of detections compared to FP32, missing
fine-grained objects, including multiple drawers. QATMA substantially recovers
these detections. For region-region similarity (bottom), we visualize the average
pairwise cosine similarity of each anchor with other anchors of the same class as a
heatmap. We observe that QATMA exhibits similarity patterns close to those of
FP32, whereas QAT displays notably different patterns, qualitatively supporting
the effectiveness of QATMA in preserving region-region (intra-modal) alignment.


6    Conclusion
In this paper, we propose QATMA to address the severe degradation of cross-
modal and intra-modal alignment inherent in extremely low-bit OVOD. To
suppress the accumulation of quantization errors, we formulate CQAT, which
provides a stable foundation for effective KD via stage-by-stage network parti-
tioning. Concurrently, TPSD explicitly transfers the teacher’s cross-modal and
intra-modal alignment through text-anchored pairwise cosine similarities in the
joint embedding space. Extensive evaluations on LVIS and COCO zero-shot
benchmarks demonstrate that QATMA effectively restores the degraded align-
ment, significantly outperforming existing QAT baselines under extremely low-bit
settings with improvements of up to 4.3 AP and 7.6 AP, respectively.
          QATMA: Quantization-Aware Training with Multimodal Alignment                   15

References
 1. Bengio, Y., Léonard, N., Courville, A.: Estimating or propagating gradients through
    stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432
    (2013)
 2. Bhalgat, Y., Lee, J., Nagel, M., Blankevoort, T., Kwak, N.: LSQ+: Improving
    low-bit quantization through learnable offsets and better initialization. In: CVPRW.
    pp. 696–697 (2020)
 3. Cao, W., Zhang, Y., Gao, J., Cheng, A., Cheng, K., Cheng, J.: PKD: General
    distillation framework for object detectors via pearson correlation coefficient. In:
    NeurIPS. pp. 15394–15406 (2022)
 4. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.:
    End-to-end object detection with transformers. In: ECCV. pp. 213–229 (2020)
 5. Chen, G., Choi, W., Yu, X., Han, T., Chandraker, M.: Learning efficient object
    detection models with knowledge distillation. In: NeurIPS. vol. 30 (2017)
 6. Chen, P., Liu, J., Zhuang, B., Tan, M., Shen, C.: AQD: Towards accurate quantized
    object detection. In: CVPR. pp. 104–113 (2021)
 7. Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., Shan, Y.: YOLO-World: Real-time
    open-vocabulary object detection. In: CVPR. pp. 16901–16911 (2024)
 8. Choi, J., Wang, Z., Venkataramani, S., Chuang, P.I.J., Srinivasan, V., Gopalakrish-
    nan, K.: PACT: Parameterized clipping activation for quantized neural networks.
    arXiv preprint arXiv:1805.06085 (2018)
 9. Choukroun, Y., Kravchik, E., Yang, F., Kisilev, P.: Low-bit quantization of neural
    networks for efficient inference. In: ICCV Workshops. pp. 3009–3018 (2019)
10. Dave, A., Dollár, P., Ramanan, D., Kirillov, A., Girshick, R.: Evaluating
    large-vocabulary object detectors: The devil is in the details. arXiv preprint
    arXiv:2102.01066 (2021)
11. Esser, S.K., McKinstry, J.L., Bablani, D., Appuswamy, R., Modha, D.S.: Learned
    step size quantization. In: ICLR (2020)
12. Feng, C., Zhong, Y., Gao, Y., Scott, M.R., Huang, W.: TOOD: Task-aligned
    one-stage object detection. In: ICCV. pp. 3490–3499 (2021)
13. Girshick, R., Donahue, J., Darrell, T., Malik, J.: Rich feature hierarchies for accurate
    object detection and semantic segmentation. In: CVPR. pp. 580–587 (2014)
14. Gu, X., Lin, T., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision
    and language knowledge distillation. In: ICLR (2022)
15. Gupta, A., Dollár, P., Girshick, R.: LVIS: A dataset for large vocabulary instance
    segmentation. In: CVPR. pp. 5356–5364 (2019)
16. Gupta, K., Asthana, A.: Reducing the side-effects of oscillations in training of
    quantized YOLO networks. In: WACV. pp. 2452–2461 (2024)
17. Han, S., Pool, J., Tran, J., Dally, W.: Learning both weights and connections for
    efficient neural network. In: NeurIPS. vol. 28 (2015)
18. He, Y., Zhang, X., Sun, J.: Channel pruning for accelerating very deep neural
    networks. In: ICCV. pp. 1389–1397 (2017)
19. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531 (2015)
20. Hudson, D.A., Manning, C.D.: GQA: A new dataset for real-world visual reasoning
    and compositional question answering. In: CVPR. pp. 6700–6709 (2019)
21. Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H.,
    Kalenichenko, D.: Quantization and training of neural networks for efficient integer-
    arithmetic-only inference. In: CVPR. pp. 2704–2713 (2018)
16      J. Park et al.

22. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H.,
    Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning
    with noisy text supervision. In: ICML. pp. 4904–4916 (2021)
23. Jocher, G., Chaurasia, A., Qiu, J.: Ultralytics YOLOv8. https://github.com/
    ultralytics/ultralytics (2023)
24. Kim, J., Bhalgat, Y., Lee, J., Patel, C., Kwak, N.: QKD: Quantization-aware
    knowledge distillation. arXiv preprint arXiv:1911.12491 (2019)
25. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L.,
    Zhang, L., Hwang, J., Chang, K., Gao, J.: Grounded language-image pre-training.
    In: CVPR. pp. 10955–10965 (2022)
26. Li, R., Wang, Y., Liang, F., Qin, H., Yan, J., Fan, R.: Fully quantized network for
    object detection. In: CVPR. pp. 2810–2819 (2019)
27. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object
    detection. In: ICCV. pp. 2999–3007 (2017)
28. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár,
    P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV. pp.
    740–755 (2014)
29. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang,
    J., Su, H., Zhu, J., Zhang, L.: Grounding DINO: Marrying DINO with grounded
    pre-training for open-set object detection. In: ECCV. pp. 38–55 (2024)
30. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., Berg, A.C.: SSD:
    Single shot multibox detector. In: ECCV. pp. 21–37 (2016)
31. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin
    transformer: Hierarchical vision transformer using shifted windows. In: ICCV. pp.
    10012–10022 (2021)
32. Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Doso-
    vitskiy, A., Mahendran, A., Arnab, A., Dehghani, M., Shen, Z., et al.: Simple
    open-vocabulary object detection. In: ECCV. pp. 728–755 (2022)
33. Mirzadeh, S.I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., Ghasemzadeh,
    H.: Improved knowledge distillation via teacher assistant. In: AAAI. vol. 34, pp.
    5191–5198 (2020)
34. Nagel, M., Amjad, R.A., Van Baalen, M., Louizos, C., Blankevoort, T.: Up or down?
    adaptive rounding for post-training quantization. In: ICML. pp. 7197–7206 (2020)
35. Nagel, M., Baalen, M.v., Blankevoort, T., Welling, M.: Data-free quantization
    through weight equalization and bias correction. In: ICCV. pp. 1325–1334 (2019)
36. Nagel, M., Fournarakis, M., Bondarenko, Y., Blankevoort, T.: Overcoming oscilla-
    tions in quantization-aware training. In: ICML. pp. 16318–16330 (2022)
37. Park, E., Yoo, S.: PROFIT: A novel training method for sub-4-bit MobileNet
    models. In: ECCV. pp. 430–446 (2020)
38. Polino, A., Pascanu, R., Alistarh, D.: Model compression via distillation and
    quantization. In: ICLR (2018)
39. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
    Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
    natural language supervision. In: ICML. pp. 8748–8763 (2021)
40. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified,
    real-time object detection. In: CVPR. pp. 779–788 (2016)
41. Redmon, J., Farhadi, A.: YOLOv3: An incremental improvement. arXiv preprint
    arXiv:1804.02767 (2018)
42. Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object
    detection with region proposal networks. In: NeurIPS. pp. 91–99 (2015)
         QATMA: Quantization-Aware Training with Multimodal Alignment                   17

43. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Generalized
    intersection over union: A metric and a loss for bounding box regression. In: CVPR.
    pp. 658–666 (2019)
44. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365:
    A large-scale, high-quality dataset for object detection. In: ICCV. pp. 8430–8439
    (2019)
45. Tian, Z., Shen, C., Chen, H., He, T.: FCOS: Fully convolutional one-stage object
    detection. In: ICCV. pp. 9627–9636 (2019)
46. Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M.: YOLOv7: Trainable bag-of-freebies
    sets new state-of-the-art for real-time object detectors. In: CVPR. pp. 7464–7475
    (2023)
47. Xu, S., Li, Y., Lin, M., Gao, P., Guo, G., Lü, J., Zhang, B.: Q-DETR: An efficient
    low-bit quantized detection transformer. In: CVPR. pp. 3842–3851 (2023)
48. Zareian, A., Rosa, K.D., Hu, D.H., Chang, S.F.: Open-vocabulary object detection
    using captions. In: CVPR. pp. 14393–14402 (2021)
49. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: DINO:
    DETR with improved denoising anchor boxes for end-to-end object detection. In:
    ICLR (2023)
50. Zhang, H., Zhang, P., Hu, X., Chen, Y.C., Li, L., Dai, X., Wang, L., Yuan, L., Hwang,
    J.N., Gao, J.: GLIPv2: Unifying localization and vision-language understanding. In:
    NeurIPS. pp. 36067–36080 (2022)
51. Zhang, Y., Dong, Z., Yang, H., Lu, M., Tseng, C.C., Du, Y., Keutzer, K., Du, L.,
    Zhang, S.: QD-BEV: Quantization-aware view-guided distillation for multi-view 3D
    object detection. In: ICCV. pp. 3825–3835 (2023)
52. Zhao, T., Liu, P., He, X., Zhang, L., Lee, K.: Real-time transformer-based open-
    vocabulary detection with efficient fusion head. arXiv preprint arXiv:2403.06892
    (2024)
53. Zhong, Y., Yang, J., Zhang, P., Li, C., Codella, N., Li, L.H., Zhou, L., Dai, X.,
    Yuan, L., Li, Y., Gao, J.: RegionCLIP: Region-based language-image pretraining.
    In: CVPR. pp. 16793–16803 (2022)
54. Zhou, A., Yao, A., Guo, Y., Xu, L., Chen, Y.: Incremental network quantization:
    Towards lossless CNNs with low-precision weights. In: ICLR (2017)
55. Zhou, S., Ni, Z., Zhou, X., Wen, H., Wu, Y., Zou, Y.: DoReFa-Net: Training low
    bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint
    arXiv:1606.06160 (2016)
56. Zhou, Z., Shen, Y., Shao, S., Gong, L., Lin, S.: Rethinking centered kernel alignment
    in knowledge distillation. In: IJCAI (2024)
57. Zhu, K., He, Y.Y., Wu, J.: Quantized feature distillation for network quantization.
    In: AAAI. vol. 37, pp. 11452–11460 (2023)
58. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable DETR: Deformable
    transformers for end-to-end object detection. In: ICLR (2021)
59. Zhuang, B., Liu, L., Tan, M., Shen, C., Reid, I.: Training quantized neural networks
    with a full-precision auxiliary module. In: CVPR. pp. 1488–1497 (2020)
18      J. Park et al.

A     More Experimental Results

A.1     Full COCO Zero-Shot Comparison

Table 7 reports the complete COCO val2017 zero-shot results, including the
EMA+QC [16] and QFD [57] closed-vocabulary detection QAT baselines omitted
from the main paper for brevity. Across all model scales and object sizes, QATMA
consistently outperforms both baselines as well as naive QAT (LSQ [2, 11]),
whereas EMA+QC and QFD perform comparably to naive QAT.
    Under our zero-shot protocol, each category is recognized by aligning its text
query with region embeddings in the joint embedding space. The detector is
trained on Objects365v2 [44] and GQA [20], never on COCO. EMA+QC and
QFD, by contrast, were validated for closed-vocabulary detection QAT—even on
COCO—yet neither targets this alignment. EMA+QC mitigates weight and acti-
vation oscillations during low-bit QAT, a modality-agnostic stability mechanism.
QFD distills only the teacher’s FPN-level feature maps, a visual-feature-level
supervision that lies outside the joint embedding space. Both therefore leave the
cross-modal and intra-modal alignment distorted and degrade even on COCO.
While effective in the closed-vocabulary setting they were designed for, these
techniques do not extend to open-vocabulary detection.


Table 7: Full COCO val2017 zero-shot comparison, including the EMA+QC [16] and
QFD [57] closed-vocabulary detection QAT baselines (4-4-8, Ch-T-H granularity).

Model            Bits    BOPs(T) Method         AP AP50 AP75 APs APm APl
                 FP32     44.56    -            41.9 57.0   45.5 26.7 46.1 55.0
                                   QAT         22.0 32.1    23.6 7.8 23.5 32.4
YOLO-World-M
                                   EMA+QC [16] 21.2 31.1    22.7 8.3 22.7 31.3
                 4-4-8     2.05
                                   QFD [57]    21.6 31.4    23.3 7.9 23.7 31.8
                                   Ours        26.1 38.1    28.1 11.0 28.1 37.9
                 FP32     90.60    -            45.1 60.7   48.9 29.8 49.8 57.5
                                   QAT         18.4 27.2    19.6 7.5 20.1 26.9
YOLO-World-L
                                   EMA+QC [16] 18.4 27.2    19.6 7.6 20.1 27.2
                 4-4-8     3.09
                                   QFD [57]    18.2 26.8    19.2 7.0 19.6 27.0
                                   Ours        25.2 36.7    27.0 10.5 27.5 37.7
                 FP32     140.66   -            46.7 62.8   50.9 31.8 51.2 60.8
                                   QAT         18.6 26.8    19.9 7.8 20.5 27.0
YOLO-World-X
                                   EMA+QC [16] 17.7 25.8    19.0 7.7 18.6 25.6
                 4-4-8     4.21
                                   QFD [57]    19.2 27.9    20.8 8.8 21.5 28.0
                                   Ours        26.2 38.2    28.1 12.2 28.7 37.2
         QATMA: Quantization-Aware Training with Multimodal Alignment           19

A.2    Effect of Quantization Granularity
Table 8 reports LVIS and COCO zero-shot performance on YOLO-World-L
across different quantization granularities and bit-widths. Relaxing the activation
bit-width to 5-bit (Ch-T-H 4-5-8) raises QAT AP to 23.0 on LVIS and 36.2 on
COCO, and QATMA surpasses both, reaching 23.8 and 37.6, respectively. With
per-channel activation granularity (Ch-Ch-H), even the extreme 3-bit setting
(3-3-8) remains feasible, where QATMA still outperforms QAT on both LVIS
(20.0 vs. 18.5) and COCO (32.6 vs. 30.7). These results confirm that QATMA is
consistently robust with respect to granularity and bit-width changes on both
benchmarks.

Table 8: Effect of quantization granularity and bit-width on YOLO-World-L. Ch-Ch-H
= per-channel activations. LVIS and COCO zero-shot AP is reported.

                      Granularity Bits Method LVIS COCO
                                          QAT    23.0 36.2
                      Ch-T-H      4-5-8
                                          Ours   23.8 37.6
                                          QAT    18.5 30.7
                      Ch-Ch-H     3-3-8
                                          Ours   20.0 32.6




A.3    Data Split
A natural question is why Curriculum QAT allocates only 1/3 of the training
data to Stage 1 (backbone) and the remaining 2/3 to Stage 2 (neck-head). This
split follows from the differing roles of the two modules. The backbone is a
task-agnostic extractor of generic image features, whereas the neck-head is the
task-relevant module that performs the fine-grained vision-language alignment—
through cross-modal attention and region-text matching—and thus dominates the
final detection accuracy. Beyond its functional role, the neck-head also contains
more parameters than the backbone (27.7M versus 19.8M in YOLO-World-L,
excluding the text encoder), further motivating a larger training budget for
Stage 2. We therefore avoid over-investing in backbone stabilization: Stage 1
needs only to restore an adequate feature quality, so that the larger share of
the data budget can be devoted to the harder and more critical recovery of the
neck-head in Stage 2. Table 9 supports this design: under identical conditions, an
even 1/2 : 1/2 split reaches only 15.0 AP, whereas the proposed 1/3 : 2/3 split
attains 16.1 AP in LVIS zero-shot detection. Allocating more data to Stage 2 is
thus clearly beneficial.

A.4    Hyperparameter Analysis
To isolate the effect of TPSD, we adopt a head-only W4A4 setting: the backbone
and neck remain in full precision, only the head is quantized to W4A4, and
20      J. Park et al.

Table 9: Effect of the curriculum data split between Stage 1 (backbone) and Stage 2
(neck-head) on YOLO-World-L (4-4-8, Ch-T-H). LVIS zero-shot AP is reported.

                    Stage 1 (backbone) Stage 2 (neck-head) AP
                           1/3                    2/3           16.1
                           1/2                    1/2           15.0



the whole network is trained. Quantizing the head alone is already severe: a
simple MinMax PTQ on it collapses the LVIS zero-shot AP to 4.8. We then
study the sensitivity to the TPSD loss weight λ2 on YOLO-World-L, trained
on the Objects365v2 [44] validation split (80K images), where λ2 = 0 reduces
to naive QAT. As shown in Tab. 10, TPSD improves over naive QAT across all
λ2 , and the AP stays stable (within 0.4 among non-zero weights), confirming its
robustness to this hyperparameter. PKD [3] is also known to be robust to its loss
weight, so we set λ1 = λ2 = 6.


Table 10: TPSD loss weight λ2 sensitivity on YOLO-World-L (head-only W4A4). LVIS
zero-shot AP is reported.

                          λ2     0   3     6     12   18   48
                          AP 30.4 30.8 31.2 31.2 31.0 30.9




A.5    Training Cost
QATMA incurs no additional inference cost, since the curriculum and distillation
affect only training. Its two-stage schedule also keeps the training overhead modest.
As reported in Tab. 11, on YOLO-World-L QATMA increases the per-iteration
compute from 27.1k to 35.1k GFLOPs (+30%) and the total training time from
95.2 to 112.4 GPU-hours (+18%) relative to plain QAT. This overhead stays
small because Stage 1 is lightweight—the neck-head is frozen and the teacher is
forwarded only through the backbone.


Table 11: Training cost on YOLO-World-L. QATMA adds no inference cost, and its
two-stage training overhead over plain QAT is modest.

                         Method GFLOPs/iter GPU-hours
                         QAT             27.1k        95.2
                         Ours            35.1k        112.4
         QATMA: Quantization-Aware Training with Multimodal Alignment            21

B     Visualization

B.1    LVIS Zero-Shot Detection
Figure 5 presents qualitative LVIS zero-shot detection results on YOLO-World-L
(4-4-8, Ch-T-H). Naive QAT frequently misses or misclassifies objects. In the first
row, for example, it labels a mouse (red) as a speaker (purple) and misses the
laptop computer (green); in the second row it confuses a bathtub (yellow) with a
sink (pink); and in the third row it classifies an elephant (red) as a horse (pink).
Such errors stem from a failure to account for the context of neighboring regions,
underscoring the importance of the cross-modal and intra-modal alignment. By
preserving this alignment, QATMA corrects these mistakes and yields detections
that closely match the FP32 reference.
22     J. Park et al.




                                    FP32                                                                                QAT                                                                             Ours
         desk: 0.62      computer0.83
                laptop computer:  keyboard: 0.49                                                                                                                                laptop computer: 0.74
                                                                                                                                                                          desk: 0.71
                                    mousepad: 0.64



            computer keyboard: 0.68                                  mouse: 0.85              computer keyboard: 0.33                                 speaker:0.63
                                                                                                                                                      mouse:   0.44             computer keyboard: 0.81                                 mouse: 0.86




                                                         mirror: 0.78       lampshade:   0.39
                                                                            lightbulb: 0.41                                              mirror: 0.58        lamp: 0.39                                                      mirror: 0.46       fan: 0.46 0.31
                                                                                                                                                                                                                                                reflector:
                                                         vent: 0.39
                                                              lampshade:
                                                              reflector:    0.49
                                                                         0.47                                                                                                                                                    lampshade: 0.49

                  lightbulb: 0.34                                                                                                                              mirror: 0.46                                                                      mirror: 0.62
                                                                                                                                                                                                            mirror: 0.46




                                                                        pencil sharpener:
                                                                        wall socket: 0.33 0.34
                                                                  coat  hanger:                                                                         faucet: 0.44
                                                                  coatrack:   0.390.47
                                                                 hand
                                                                 towel:
                                                               wall    towel:
                                                                    socket:    0.77
                                                                         0.740.81                                                                 hand
                                                                                                                                                  paper
                                                                                                                                                 wall    towel:
                                                                                                                                                          towel:
                                                                                                                                                      socket:   0.55
                                                                                                                                                                 0.70
                                                                                                                                                              0.35                                                                 paper
                                                                                                                                                                                                                                   hand
                                                                                                                                                                                                                                  wall     towel:
                                                                                                                                                                                                                                          towel:
                                                                                                                                                                                                                                       socket:    0.56
                                                                                                                                                                                                                                                 0.50
                                                                                                                                                                                                                                               0.77
                                                                                faucet: 0.56                                                                     faucet: 0.53                                                                     faucet: 0.35
                                toilet:
                      saltshaker:
                      faucet: 0.49      0.86 0.70
                                  0.41tank:                                                                         toilet: 0.61                                                                    toilet:tank:
                                                                                                                                                                                          faucet: 0.64      0.82 0.31
                                                                  nailfile:
                                                                     sink: 0.54
                                                                            0.66                                                                        sink: 0.470.31
                                                                                                                                                        hotplate:                                                                       sink: 0.81
          bathtub: 0.79                                                                    sink: 0.66                                                                         bathtub: 0.45
                                                          cabinet: 0.53                                                                                                                                                      cabinet: 0.38
                                                               handle: 0.30                                                                                  cupboard: 0.32




              elephant: 0.57                                                                  elephant:
                                                                                             cow:   0.56 0.68
                                                                                                  0.58
                                                                                             horse:                                                                               elephant: 0.68




         elephant: 0.35                                                    trunk: 0.43                                                                                    elephant: 0.38

                                                                      tambourine: 0.52                                                                 frisbee:
                                                                                                                                             wooden leg:soccer  0.43
                                                                                                                                                         0.41 ball: 0.64 0.74
                                                                                                                                                                 handbag:                                                                            backpack: 0.59
                               elephant: 0.41                                                                                                                                                      elephant: 0.46



         goat: 0.33

                   monitor: 0.80                    monitor: 0.71                                       monitor: 0.60              television set: 0.53
                                                                                                                                   monitor: 0.64                                       monitor: 0.73                   monitor: 0.74


                                                                                  bottle: 0.48
         laptop computer: 0.65                                                           laptop computer: 0.69
                                                                               bottle: 0.50                                                                               monitor: 0.33 0.82
                                                                                                                                                                          laptop computer:                                                           bottle: 0.45

                                         cigar box: 0.63            beer
                                                                    jar: can: 0.64
                                                                    can:0.66
                                                                    saltshaker:
                                                                         0.33   0.35                                                                  canister: 0.68      desk: 0.78                                                    beer can:
                                                                                                                                                                                                                                        milk can: 0.32
                                                                                                                                                                                                                                                  0.64
         desk: 0.78                                                                      desk: 0.70
                                                                          packet: 0.36
                                    computer keyboard: 0.66                                                                                                                                           computer keyboard: 0.31
                                                                              wallet: 0.44
                                            headset: 0.43                                                                      plate: 0.42
                       napkin: 0.80                                   mouse: 0.78                                                                       mouse: 0.89                                                                         mouse: 0.80




         dress suit:
         blazer: 0.370.38   choker: 0.49                            chandelier: 0.55      person: 0.69                                                pennant:0.54
                                                                                                                                                     cymbal:    0.55      person: 0.66


                                       necktie:
                                       bow-tie: 0.47
                                                0.34                                                                                                                                                    necktie: 0.34
                                                                                                                         suspenders: 0.36




                                                                                                                                                                                                                                       person: 0.35
                                                                                                                                                    blouse: 0.37 0.35
                                                                                                                                                            0.30
                                                                                                                                                    trench coat:
                                                                    chair: 0.64
                                                                              dining table: 0.64                                                     chair: 0.58                                                                       chair: 0.47




                                                                               chair: 0.49                                                                                                                                                           clothes hamper: 0.47
                                                                                                                                                                                                                                                     chair: 0.33




                                       ring: 0.64                                                                                                                                                         ring: 0.54




Fig. 5: Qualitative comparison of LVIS zero-shot detection on YOLO-World-L (4-4-8,
Ch-T-H)
```
