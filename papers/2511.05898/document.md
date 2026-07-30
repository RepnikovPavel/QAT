Q²: Quantization-Aware Gradient Balancing and Attention Alignment for Low-Bit Quantization

Zhaoyang Wang¹ and Dong Wang¹

¹ Institute of Information Science, Beijing Jiaotong University, No.3 Shangyuan Village, Xizhimenwai, Haidian District, Beijing, China, 100044
24125200@bjtu.edu.cn
² wangdong@bjtu.edu.cn

arXiv:2511.05898v2 [cs.CV] 26 Feb 2026

**Abstract.** Quantization-aware training (QAT) has achieved remarkable success in low-bit (≤4-bit) quantization for classification networks. However, when applied to more complex visual tasks such as object detection and image segmentation, performance still suffers significant degradation. A key cause of this limitation has been largely overlooked in the literature. In this work, we revisit this phenomenon from a new perspective and identify a major failure factor: gradient imbalance at feature fusion stages, induced by accumulated quantization errors. This imbalance biases the optimization trajectory and impedes convergence under low-bit quantization. Based on this diagnosis, we propose Q², a two-pronged framework comprising: (1) Quantization-aware Gradient Balancing Fusion (Q-GBFusion), a closed-loop mechanism that dynamically rebalances gradient contributions during feature fusion; and (2) Quantization-aware Attention Distribution Alignment (Q-ADA), a parameter-free supervision strategy that reconstructs the supervision distribution using semantic relevance and quantization sensitivity, yielding more stable and reliable supervision to stabilize training and accelerate convergence. Extensive experiments show that our method, as a plug-and-play and general strategy, can be integrated into various state-of-the-art QAT pipelines, achieving an average +2.5% mAP gain on object detection and a +3.7% mDICE improvement on image segmentation. Notably, it is applied only during training and introduces no inference-time overhead, making it highly practical for real-world deployment.

Keywords:

Model Quantization · Complex Visual Tasks

# 1 Introduction

Model quantization reduces memory footprint and computational cost by approximating full-precision values with low-bit integers, and has become a foundational technique for neural network compression[21]. Existing quantization methods mainly fall into Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). PTQ minimizes quantization error on a small calibration set. QAT trains with quantization in the loop (fine-tuning or from scratch) and usually yields higher accuracy, particularly at ≤4-bit. State-of-the-art QAT

Z. Wang and D. Wang

![](images/task_4c3e0ecb40d1_page_1_pic_0.png)

**Fig. 1.** (a) Comparison of the accuracy drop of three representative QAT schemes. YOLOv5 is used as the baseline model with 4-bit quantization setting. (b) Measured average gradient magnitude in the Concat Layer of the YOLOv5 model shown in Fig. 2. To verify that this is a quantization-induced and general phenomenon, the corresponding measurements on other models are reported in the **Appendix 6**.

approaches have demonstrated remarkable performance; in some cases, quantized models even outperform their full-precision counterparts [19,13,15]. For instance, the recent N2UQ method [19] achieves a Top-1 accuracy of 78% on a 4-bit ResNet-50, surpassing the full-precision baseline by 1%.

However, applying existing QAT methods at 4-bit precision to more complex vision tasks such as object detection and image segmentation remains highly challenging. Only a limited number of related studies have been reported in the literature[10,12,24], and the achieved performance still falls short of that attained by quantized classification models. To better illustrate the performance gap, we apply three mostly cited schemes, including PACT[4], LSQ[8] and N2UQ[19], to the YOLO and compare the accuracy of the quantized model in Fig. 1(a). It is clear that even with the most powerful non-uniform quantization method N2UQ, the quantized model still suffers a 3.8% accuracy loss.

These observations suggest that a quantizer-centric explanation is insufficient to account for low-bit degradation in complex visual tasks. Although models such as ResNet and YOLO are quantized with the same convolutional operator in implementation, their performance drops differ substantially under low-bit settings. This discrepancy indicates that quantization efficacy is shaped not only by the quantizer itself, but also by architectural characteristics. Existing studies mainly focus on quantization representations themselves: either by redesigning network architectures to improve quantization-friendliness [6,5], or by further optimizing quantizers [10,27,25], while implicitly assuming that the optimization path itself remains reliable. However, in detection and segmentation networks with feature fusion structures, this assumption does not always hold.

In this work, we revisit this problem from the perspective of optimization dynamics at feature-fusion stages. Taking YOLO as a representative detection architecture, as shown in Fig. 2, its backbone and neck jointly support localization and recognition through multi-scale feature fusion [18]. In this process, shallow features (Branch-0) preserve fine-grained spatial details [11], whereas

Q² for Low-Bit Quantization

deep features (Branch-1) carry more abstract semantic information [23]. Unlike classification networks that mainly rely on final-layer high-level features, detection and segmentation models depend heavily on this multi-scale fusion mechanism for precise prediction.

We observe that under low-bit QAT, quantization errors progressively accumulate with network depth, leading to mismatched quantization-induced perturbation strength across different branches. When these branches are fused at fusion nodes, the backpropagated signal exhibits clear gradient imbalance. To quantify this phenomenon, based on the feature-gradient principle of Grad-CAM [22], we perform a feature gradient-flow analysis at the Concat layer (Fig. 2) by tracking the average gradient magnitude. The results are shown in Fig. 1(b). We find a significant discrepancy between the two branches: QAT tends to disproportionately prioritize deeper branches (Branch-1) while relatively under-optimizing shallower ones (Branch-0), ultimately causing biased gradient updates at fusion points and degraded quantization performance. This effect becomes especially pronounced under ultra-low-bit settings (e.g., ≤4-bit).

![](images/task_4c3e0ecb40d1_page_2_pic_0.png)

**Fig. 2.** Applying the proposed feature fusion strategy to the YOLO network. $\alpha_i$ denotes the *regulation factors* introduced by Q-GBFusion.

Notably, the branch-wise gradient imbalance is non-stationary: its magnitude varies across fusion-layer locations, training stages, and quantization perturbations. Therefore, a fixed balancing coefficient is prone to under- or over-compensation during training, motivating an online feedback-based closed-loop regulation. To address this, we propose **Quantization-aware Gradient Balancing Fusion (Q-GBFusion)**, which introduces branch-wise regulation factors $\alpha_i$ (Fig. 2) at fusion nodes and performs closed-loop adjustment based on

Z. Wang and D. Wang

gradient-energy feedback $G_i$ to balance the optimization of different branches, while applying post-fusion normalization to stabilize gradient propagation under low-bit quantization noise. During inference deployment, the closed-loop feedback update is disabled, and the related modules can be folded into parameters, thus introducing no additional runtime overhead.

Moreover, this imbalance is often accompanied by unstable QAT optimization and slower convergence to well-calibrated quantization parameters, since conventional QAT losses primarily minimize numerical discrepancies to ground-truth targets. However, for tasks such as object localization, predictions rely heavily on fine-grained semantic cues (e.g., shape, texture, and other feature representations), rather than numerical fidelity alone. Unlike existing methods that directly match feature tensors to enforce feature fidelity [32,26,3], our goal is to preserve not only semantic saliency but also quantization sensitivity; otherwise, distillation may amplify unreliable regions. Based on this analysis, we propose a quantization-distortion-aware attention distillation objective, termed **Quantization-aware Attention Distribution Alignment (Q-ADA)**. It aligns the full-precision teacher and quantized student by matching quantization-sensitive saliency-feature distributions via a saliency statistics rule, with greater emphasis on distortion-prone regions. This encourages the student to preserve fine-grained structural cues critical for downstream tasks, improving stability and accelerating convergence. Our contributions are summarized as follows:

- **Mechanism-driven diagnosis.** We provide the first in-depth analysis showing that the performance degradation of low-bit quantization on complex vision tasks arises from a previously underexplored optimization pathology at feature-fusion stages, namely *branch-wise gradient imbalance* caused by accumulated quantization errors, which further biases training optimization and leads to slow convergence.
- **Methodological contribution.** Guided by the above diagnosis, we propose $Q^2$, a two-pronged framework consisting of two complementary components: **Q-GBFusion**, which performs online feedback control of branch gradient allocation at feature-fusion stages to balance gradients, and **Q-ADA**, which enforces quantization-aware alignment of salient feature distributions to accelerate convergence. $Q^2$ is training-time only and introduces no extra inference cost.
- **Empirical contribution.** Extensive experiments across different architectures (CNNs and Transformers), tasks (object detection and image segmentation), and QAT pipelines show that the proposed method can be consistently integrated into diverse QAT methods with sustained performance gains, achieving an average +2.5% mAP improvement on object detection and a +3.7% mDICE improvement on image segmentation.

Q² for Low-Bit Quantization

## 2 Related Work

### 2.1 Quantization-Aware Training

Existing QAT approaches can be broadly categorized into two complementary research directions. The first focuses on quantizer design, aiming to improve the fidelity of the quantization mapping itself, i.e., how floating-point weights and activations are optimally converted to low-bit fixed-point representations. Representative efforts include learnable scaling factors (e.g., LSQ[8]), adaptive clipping ranges (e.g., PACT[4]), non-uniform quantization schemes (e.g., N2UQ[19]), and compress–expand asymmetric quantization methods (e.g., LCQ[28]). These methods seek to minimize quantization-induced distortion at the operator level. The second line of work addresses the optimization dynamics of QAT, recognizing that even with an accurate quantizer, standard training may converge to suboptimal solutions due to gradient mismatch, representation collapse, or loss of task-relevant features. To mitigate this, researchers have introduced enhanced supervision strategies such as transfer-rate scheduling and adaptive learning rate adjustment to improve QAT stability [15], as well as delayed QAT activation with bit-width-controlled regularization for domain generalization [14]. These approaches treat quantization not merely as a numerical approximation problem but as a representation learning challenge.

### 2.2 Quantization of Complex Visual Tasks

Unlike traditional classification tasks, object detection and image segmentation rely on more complex network architectures that can achieve near full-precision performance at 8-bit or higher precision [24], yet still suffer significant degradation at 4-bit (or lower) precision. Recent studies have attempted to narrow this gap. For example, HQOD [12] introduces task-correlated losses to balance regression optimization across IoU levels; however, it does not fully close the gap, and substantial degradation remains under 4-bit quantization (e.g., nearly a 7% mAP drop on YOLO). Some methods further design quantization strategies for specific architectures (e.g., YOLO or DETR) [10,27,25]. While effective in targeted settings, they still leave a substantial gap under very low-bit quantization and have limited general applicability. This indicates that existing methods have not explicitly identified the shared structural bottleneck in complex vision networks, namely feature fusion stages. Some works recognize that architecture itself plays a critical role in quantization robustness and improve compatibility through architectural modifications [6,5]. For example, [6] replaces YOLO’s CSP backbone with a more “quantization-friendly” ResNet variant. However, such changes often compromise YOLO’s original design principles, leading to degraded baseline performance and reduced practical utility.

These observations suggest the need for a more general, architecture-agnostic solution that targets the shared structural bottleneck in complex vision networks—namely, feature fusion stages—while preserving inference efficiency.

Z. Wang and D. Wang

3 Proposed Method

3.1 Problem Formulation

During quantization-aware training, each layer introduces quantization error. Let $x_l$ denote the full-precision output of the $l$-th layer and $\tilde{x}_l$ its quantized counterpart, with $\delta_l = \tilde{x}_l - x_l$. Since the input to layer $l$ is also quantized, $\tilde{x}_{l-1} = x_{l-1} + \delta_{l-1}$, the quantized output can be written as $\tilde{x}_l = f_l(\tilde{x}_{l-1}) + \epsilon_l \approx f_l(x_{l-1}+\delta_{l-1})+\epsilon_l$, where $f_l(\cdot)$ is the full-precision transformation of the $l$-th layer and $\epsilon_l$ is the layer-wise quantization noise. Under a first-order approximation,

$$
\delta_l \approx J_l \delta_{l-1} + \epsilon_l, \quad (1)
$$

where $J_l = \partial f_l / \partial x_{l-1}$ is the Jacobian evaluated at $x_{l-1}$. Eq. (1) shows that quantization disturbances propagate and accumulate with depth, making deep representations more sensitive under low-bit settings.

Modern detectors/segmentors rely on multi-scale feature fusion across multiple branches. Consider a fusion node with $K$ quantized branch features $\{\tilde{\mathbf{F}}_i\}_{i=1}^K$ (e.g., concatenation), where $\tilde{\mathbf{F}}_i$ denotes the vectorized feature of the $i$-th branch and $\tilde{\mathbf{F}}_i = \mathbf{F}_i + \delta_i$, with $\mathbf{F}_i$ the corresponding full-precision feature and $\delta_i$ the feature-level quantization error. Because different branches generally have different effective depths and receptive fields, the magnitudes of $\delta_i$ can vary substantially under low-bit quantization, and the disparity becomes more pronounced as bit-width decreases.

To analyze how quantization affects localization, we adopt a local linear proxy of the regression head around the fusion bottleneck:

$$
\hat{\mathbf{t}} = \sum_{i=1}^{K} \mathbf{W}_i \tilde{\mathbf{F}}_i = \sum_{i=1}^{K} \mathbf{W}_i (\mathbf{F}_i + \delta_i), \quad (2)
$$

where $\hat{\mathbf{t}} \in \mathbb{R}^m$ denotes the localization output vector, $\mathbf{t} \in \mathbb{R}^m$ is the corresponding ground-truth target, and $\mathbf{W}_i \in \mathbb{R}^{m \times n_i}$ is the effective regression weight of the $i$-th branch with $\tilde{\mathbf{F}}_i \in \mathbb{R}^{n_i}$. For CIoU-style [31] regression, we approximate the loss sensitivity by squared error: $\mathcal{L}_{\text{CIoU}} \propto \|\hat{\mathbf{t}} - \mathbf{t}\|_2^2$, which expands as

$$
\hat{\mathbf{t}} - \mathbf{t} = \sum_{i=1}^{K} \mathbf{W}_i \delta_i + \sum_{i=1}^{K} \mathbf{W}_i \mathbf{F}_i - \mathbf{t}. \quad (3)
$$

A key observation in low-bit QAT is that, due to the depth-wise accumulation in Eq. (1), some branches can incur larger effective disturbances $\mathbf{W}_i \delta_i$ at fusion bottlenecks, yielding biased backpropagated gradient flow. To make Eq. (3) actionable for optimization, we summarize each branch by its gradient energy $G_i \triangleq \|\partial \mathcal{L} / \partial \tilde{\mathbf{F}}_i\|_2^2$. When $\mathbf{W}_i \delta_i$ dominates Eq. (3) for a subset of branches, their $G_i$ becomes disproportionately large, causing preferential optimization of those branches while others are under-updated; we therefore impose a multi-branch log-energy balance constraint.

Q² for Low-Bit Quantization

Let $\mathbf{g}_i = \partial\mathcal{L}/\partial\tilde{\mathbf{F}}_i$ denote the per-mini-batch gradient of the $i$-th branch at a fusion node. We measure branch gradient energy by $\ell_2$ norms and enforce log-domain balancing across all $K$ branches:

$$
\mathbb{E}_{\mathcal{B}}[\log (\|\mathbf{g}_i\|_2 + \epsilon)] - \frac{1}{K} \sum_{j=1}^{K} \mathbb{E}_{\mathcal{B}}[\log (\|\mathbf{g}_j\|_2 + \epsilon)] = \tau_i, \quad \forall i \in \{1, \dots, K\}, \quad (4)
$$

where $\mathbb{E}_{\mathcal{B}}[\cdot]$ denotes expectation over mini-batches (approximated online during training), $\epsilon > 0$ is a numerical-stability constant inside the logarithm, and $\tau_i$ is a target offset (default $\tau_i = 0$ for all $i$ to encourage uniform balancing).³ In this work, our goal is to design a *plug-and-play* fusion mechanism that enforces Eq. (4) online during QAT without altering the original network topology or incurring inference overhead.

## 3.2 Quantization-Aware Gradient Balancing Fusion

As analyzed above, branch-wise gradient imbalance at fusion bottlenecks under low-bit QAT is inherently non-stationary, varying across layers, training stages, and quantization perturbations. Therefore, gradient balancing should be treated as a dynamic optimization problem rather than addressed by a fixed coefficient. A static allocation may violate the desired gradient-energy constraint over time, leading to biased updates. This motivates a feedback-driven regulation mechanism that adaptively enforces the multi-branch gradient-energy constraint throughout training. Empirical evidence with fixed coefficients is provided in the **Appendix 10.2**. We propose **Quantization-aware Gradient Balancing Fusion (Q-GBFusion)**, a plug-and-play module that enforces the multi-branch gradient-energy constraint in Eq. (4) at fusion bottlenecks. Q-GBFusion is *training-only controllable*: it requires no second-order gradients, does not modify the backbone topology, and introduces no inference-time overhead.

Consider a fusion node with $K$ quantized branch features $\{\tilde{\mathbf{F}}_i\}_{i=1}^K$. We maintain an unconstrained internal state (dual logits) $\boldsymbol{\lambda} \in \mathbb{R}^K$ and obtain a simplex-constrained allocation vector by a softmax projection:

$$
\boldsymbol{\alpha} = \text{Softmax}(\boldsymbol{\lambda}), \quad \mathbf{F}'_i = \alpha_i \cdot \tilde{\mathbf{F}}_i, \quad i = 1, \dots, K, \qquad (5)
$$

followed by the original fusion operator, e.g., $\mathbf{F}'_{\text{cat}} = [\mathbf{F}'_1; \dots; \mathbf{F}'_K]$. This parameterization ensures $\alpha_i \ge 0$ and $\sum_i \alpha_i = 1$, and admits a closed-loop interpretation: $\boldsymbol{\lambda}$ acts as a dual control state whose projection determines the fusion allocation among branches.

Immediately after fusion, we insert a LayerNorm [2] module applied across channels at each spatial location:

$$
\text{LN}(h) = \frac{h - \mu_h}{\sigma_h}, \qquad h = \mathbf{F}'_{\text{fuse}}(i, j), \qquad (6)
$$

³ Equivalently, Eq. (4) constrains the deviations of per-branch log-energies from their mean; setting all $\tau_i = 0$ minimizes inter-branch imbalance in the log-energy domain.

Z. Wang and D. Wang

where $\mathbf{F}'_{\text{fuse}}$ denotes the fused feature (e.g., $\mathbf{F}'_{\text{cat}}$), $(i, j)$ indexes a spatial location, and $\mu_h, \sigma_h$ are computed over the channel dimension (per sample and per location).

**Training-time: closed-loop gradient-energy balancing.** Let $\mathbf{g}_i = \partial\mathcal{L}/\partial\bar{\mathbf{F}}_i$ denote the per-mini-batch branch gradient at this fusion node, and define the corresponding gradient energy $G_i = \|\mathbf{g}_i\|_2$. By the chain rule applied to Eq. (5), we have $\mathbf{g}_i = \alpha_i \partial\mathcal{L}/\partial\mathbf{F}'_i$. Taking log-norms yields a per-branch decomposition

$$
\log(G_i + \epsilon) = \log(\alpha_i + \epsilon) + \log\left(\left\|\frac{\partial\mathcal{L}}{\partial\mathbf{F}'_i}\right\|_2 + \epsilon\right), \quad (7)
$$

which separates the controllable allocation term $\log(\alpha_i + \epsilon)$ from a residual term that captures gradient statistics after gating and post-fusion conditioning. Therefore, balancing branch energies in the log domain can be achieved by regulating $\alpha$ (hence $\lambda$) using gradient-energy feedback.

In practice, we estimate the expectations in Eq. (4) using Exponential Moving Average (EMA) statistics $\{\bar{G}_i\}_{i=1}^K$ of $\{G_i\}_{i=1}^K$:

$$
\bar{G}_i \leftarrow (1 - \beta)\bar{G}_i + \beta G_i, \quad i = 1, \dots, K, \quad (8)
$$

where $\beta \in (0, 1)$ is the EMA momentum. We then form log-energy deviations from the mean:

$$
e_i = \log(\bar{G}_i + \epsilon) - \frac{1}{K} \sum_{j=1}^{K} \log(\bar{G}_j + \epsilon) - \tau_i, \quad i = 1, \dots, K, \quad (9)
$$

where $\tau_i$ is a target offset (default $\tau_i = 0$ for uniform balancing). Finally, we update the dual logits by a simple first-order feedback law:

$$
\lambda_i \leftarrow \lambda_i - \eta e_i, \quad i = 1, \dots, K, \quad (10)
$$

with step size $\eta > 0$.

During training, the network weights are optimized by standard backpropagation, while the fusion allocation is adjusted by gradient-energy feedback to reduce biased updates across branches. After training, the learned allocation $\alpha$ can be fixed, and the closed-loop update in Eqs. (8)–(10) is disabled, hence introducing no extra operators or inference overhead.

**Deployment-time: LayerNorm-free inference.** LayerNorm (LN) is generally viewed as "non-removable" because it is input-dependent: the statistics $\mu_h$ and $\sigma_h$ vary with $h$, so replacing them by fixed constants can cause a noticeable mismatch. However, in low-bit quantization the feature space is severely range-limited, making the per-sample LN statistics much less variant. In this work, we propose a "removable" approach: compute calibration statistics on a small held-out set and approximate LN by a fixed affine transform:

$$
\text{LN}(h) = \frac{h - \mu_h}{\sigma_h} \approx \frac{h - \bar{\mu}}{\bar{\sigma}}, \quad (11)
$$

Q² for Low-Bit Quantization

where $(\bar{\mu}, \bar{\sigma})$ are calibration estimates of $(\mu_h, \sigma_h)$.

Let a downstream linear map be $y = Wh_{\text{LN}} + b$ with $h_{\text{LN}} = \text{LN}(h)$. Using Eq. (11), we obtain an equivalent re-parameterization

$$
y \approx W'h + b', \quad W' = \frac{1}{\bar{\sigma}}W, \quad b' = b - \frac{\bar{\mu}}{\bar{\sigma}}W1, \qquad (12)
$$

where $1$ is an all-ones vector with compatible dimension (capturing that $\bar{\mu}$ is subtracted uniformly across channels at each spatial location). Thus, LN can be safely removed at deployment by (i) calibrating $(\bar{\mu}, \bar{\sigma})$ and (ii) folding Eq. (12) into the following layer. Detailed LayerNorm-removal derivations for Eq. (11) are provided in the **Appendix 7.1**, and empirical accuracy validations are reported in Sec. 4.5.

3.3 Quantization-Aware Attention Distribution Alignment

Conventional quantization methods typically supervise training end-to-end using task-level losses against ground truth (e.g., classification or regression errors), yet this paradigm often neglects the fidelity of intermediate feature representations. This limitation becomes especially pronounced for regression-heavy objectives such as bounding-box localization, where quantization errors can be amplified because accurate predictions rely on fine-grained spatial cues. Motivated by this, we argue that salient feature information should be explicitly incorporated to guide QAT in low-bit regimes.

Several feature-level supervision schemes have been proposed in the literature, such as [32][26][3]. However, they tend to be fragile under QAT: most of these methods directly match intermediate feature tensors, while quantization introduces non-stationary perturbations whose magnitude and pattern evolve during training (e.g., due to changing step sizes, clipping ranges, and rounding behavior). Consequently, the supervision target itself can drift, making strict tensor-level alignment unstable and preventing the student from reliably preserving fine-grained information.

To address this, we propose **Quantization-aware Attention Distribution Alignment (Q-ADA)**, a distribution-level distillation scheme that is explicitly designed for low-bit QAT. Rather than enforcing point-wise feature matching, Q-ADA aligns attention distributions using quantization-aware statistics that are more stable under evolving quantization noise: (i) *mean-centered responses* (deviation from the per-channel mean) to highlight saliency, (ii) *channel-wise variance* to normalize dynamic range and accommodate activation scaling, and (iii) a *local quantization distortion map* to emphasize regions/channels that are most vulnerable to quantization error.

Given a feature map $X \in \mathbb{R}^{C \times H \times W}$, let $\mu_c$ and $\sigma_c^2$ denote the mean and variance of channel $c$ over spatial positions. During QAT, we denote the quantized feature map as $\hat{X} = Q(X)$ and define the per-position quantization error

$$
\Delta_{c,ij} \triangleq |X_{c,ij} - \hat{X}_{c,ij}|. \qquad (13)
$$

Z. Wang and D. Wang

To ensure comparability across channels and bit-widths, we normalize the distortion by the per-channel quantization step size $s_c$ (or by $\sigma_c$ when $s_c$ is unavailable), and define two parameter-free statistics:

$$
z_{c,ij} \triangleq \frac{|X_{c,ij} - \mu_c|}{\sigma_c + \kappa}, \quad r_{c,ij} \triangleq \frac{\Delta_{c,ij}}{s_c + \kappa}, \qquad (14)
$$

where $\kappa > 0$ is a small constant for numerical stability.

Finally, the overall *saliency score* is calculated as:

$$
S_{c,ij}(X) \triangleq \log(1 + z_{c,ij}^2) + \gamma \log(1 + r_{c,ij}^2), \qquad (15)
$$

where $\gamma$ controls the strength of quantization-vulnerability emphasis. The first term highlights statistically salient locations relative to the channel distribution, while the second term explicitly increases the score for locations that incur larger quantization distortion.

The corresponding quantization-aware attention weight is computed as $\tilde{A}_{c,ij} = \text{Sigmoid}(S_{c,ij}(X))$. During distillation, we construct spatial probability distributions from the teacher (full-precision) and student (quantized) attention maps via normalization: $P_{c,ij} = \frac{\tilde{A}_{c,ij}^f}{\sum_{i',j'} \tilde{A}_{c,i'j'}^f}$, $R_{c,ij} = \frac{\tilde{A}_{c,ij}^q}{\sum_{i',j'} \tilde{A}_{c,i'j'}^q}$, and align these distributions using the Jensen-Shannon divergence:

$$
\mathcal{L}_{\text{ADA}}^{(c)} = \frac{1}{2} \sum_{i,j} P_{c,ij} \log \frac{P_{c,ij}}{M_{c,ij}} + \frac{1}{2} \sum_{i,j} R_{c,ij} \log \frac{R_{c,ij}}{M_{c,ij}}, \qquad (16)
$$

where $M_c = \frac{1}{2}(P_c+R_c)$. Notably, the divergence choice is not unique; Section 4.5 compares KL and its corresponding impact on performance. A visualization of the Q-ADA distillation process is provided in **Appendix 11**.

# 4 Experiments

## 4.1 Experimental Setup

We evaluate the proposed method on two representative visual tasks: object detection and image segmentation. For object detection, we adopt the latest CNN-based YOLOv11 and the widely cited YOLOv5 models, and further evaluate our method on the transformer-based RT-DETR[30]. Experimental results are reported on the PASCAL VOC [9] and COCO [17] datasets using mean Average Precision (mAP) as the primary evaluation metric. For image segmentation, we employ MK-UNet [20], a recent state-of-the-art architecture. Experiments are conducted on the BUSI medical imaging dataset [1], with performance measured by the mean Dice coefficient (mDICE). All experiments are performed on a server equipped with 8 × NVIDIA GeForce RTX 4090 GPUs. More detailed implementations and settings are provided in the **Appendix 8**.

Q² for Low-Bit Quantization

## 4.2 Evaluation of General Effectiveness

First, we integrate Q-GBFusion and Q-ADA with different QAT quantizers. Specifically, PACT, LSQ, and N2UQ are adopted as convolution-oriented quantizers, while Q-DETR[27], AQ-DETR[25], and GPLQ[16] are employed to quantize attention operators. We only report results on the VOC dataset in the main text, and provide COCO results in the **Appendix 10.1**.

**Table 1.** Quantization results of the object-detection models on the VOC dataset.

<table><thead><tr><th>Networks</th><th>Structure</th><th>BW</th><th>Method</th><th>Baseline</th><th>Ours</th><th>Gain</th></tr></thead><tbody><tr><td rowspan="6">YOLOv5s<br>(FP: 85.9%)</td><td rowspan="6">CNN</td><td rowspan="3">W4A4</td><td>N2UQ</td><td>82.1%</td><td><strong>84.2%</strong></td><td>+2.1</td></tr><tr><td>PACT</td><td>79.1%</td><td><strong>80.6%</strong></td><td>+1.5</td></tr><tr><td>LSQ</td><td>76.9%</td><td><strong>78.9%</strong></td><td>+2.0</td></tr><tr><td rowspan="3">W3A3</td><td>N2UQ</td><td>75.5%</td><td><strong>78.0%</strong></td><td>+2.5</td></tr><tr><td>PACT</td><td>62.9%</td><td><strong>66.6%</strong></td><td>+3.7</td></tr><tr><td>LSQ</td><td>59.9%</td><td><strong>66.8%</strong></td><td>+6.9</td></tr><tr><td rowspan="6">YOLOv11s<br>(FP: 89.4%)</td><td rowspan="6">CNN</td><td rowspan="3">W4A4</td><td>N2UQ</td><td>86.2%</td><td><strong>87.6%</strong></td><td>+1.4</td></tr><tr><td>PACT</td><td>83.8%</td><td><strong>84.8%</strong></td><td>+1.0</td></tr><tr><td>LSQ</td><td>82.9%</td><td><strong>84.2%</strong></td><td>+1.3</td></tr><tr><td rowspan="3">W3A3</td><td>N2UQ</td><td>83.0%</td><td><strong>84.3%</strong></td><td>+1.3</td></tr><tr><td>PACT</td><td>75.1%</td><td><strong>78.2%</strong></td><td>+3.1</td></tr><tr><td>LSQ</td><td>75.5%</td><td><strong>78.9%</strong></td><td>+3.4</td></tr><tr><td rowspan="6">RT-DETR<br>(FP: 90.1%)</td><td rowspan="6">Transformer</td><td rowspan="3">W4A4</td><td>Q-DETR</td><td>80.2%</td><td><strong>82.4%</strong></td><td>+2.2</td></tr><tr><td>AQ-DETR</td><td>81.4%</td><td><strong>83.3%</strong></td><td>+1.9</td></tr><tr><td>GPLQ</td><td>83.7%</td><td><strong>86.3%</strong></td><td>+2.6</td></tr><tr><td rowspan="3">W3A3</td><td>Q-DETR</td><td>77.1%</td><td><strong>79.8%</strong></td><td>+2.7</td></tr><tr><td>AQ-DETR</td><td>77.5%</td><td><strong>80.0%</strong></td><td>+2.5</td></tr><tr><td>GPLQ</td><td>78.3%</td><td><strong>81.8%</strong></td><td>+3.5</td></tr></tbody></table>

The object detection results are summarized in Table 1. Our method consistently improves accuracy across all baseline quantizers and network architectures. Specifically, under different bit-width settings, it achieves an average gain of up to +2.5% mAP, showing strong generalization regardless of the underlying quantization scheme. The gains become even more pronounced under the stricter W3A3 setting (up to +6.9% improvement), where quantization noise significantly degrades feature representation quality. Notably, when combined with N2UQ, it further narrows the accuracy gap to within 2% of the FP model, highlighting compatibility with advanced quantization designs.

We further validate our approach on an image segmentation model, MK-UNet, as reported in Table 8. Across all quantizers and bit-widths, our method yields an average +3.7% mDICE gain, with improvements reaching +4.9% under W3A3. These results confirm that the proposed strategy not only benefits detection tasks but also generalizes well to segmentation problems, where fine-grained spatial consistency is critical. Despite an 8.8% mDice gap relative to the FP baseline, our method (W4A4) notably outperforms the current 8-bit SOTA quantization scheme[29] by +4.4%.

Z. Wang and D. Wang

**Table 2.** Quantization results of the segmentation model on the BUSI dataset.

<table><thead><tr><th>Networks</th><th>BW</th><th>Method</th><th>Baseline</th><th>+</th><th>Ours</th><th>Gain</th></tr></thead><tbody><tr><td rowspan="5">MK-UNet<br>(FP: 69.5%)</td><td>W8A8</td><td>EQ[29]</td><td>55.9%</td><td>-</td><td>-</td><td></td></tr><tr><td rowspan="3">W4A4</td><td>N2UQ</td><td>55.4%</td><td><b>60.7%</b></td><td>+5.3</td><td></td></tr><tr><td>PACT</td><td>44.5%</td><td><b>46.3%</b></td><td>+1.8</td><td></td></tr><tr><td>LSQ</td><td>45.6%</td><td><b>49.7%</b></td><td>+4.1</td><td></td></tr><tr><td rowspan="3">W3A3</td><td>N2UQ</td><td>46.5%</td><td><b>53.9%</b></td><td>+7.4</td><td></td></tr><tr><td>PACT</td><td>39.3%</td><td><b>42.7%</b></td><td>+3.4</td><td></td></tr><tr><td>LSQ</td><td>40.9%</td><td><b>45.4%</b></td><td>+4.5</td><td></td></tr></tbody></table>

**Table 3.** Performance comparison of different optimization strategies on YOLOv5 (COCO) and MK-UNet (BUSI). All results are under W4A4 quantization. The base quantizer is N2UQ for YOLOv5 and LSQ for MK-UNet.

<table><thead><tr><th>Strategy</th><th>YOLOv5 (COCO), mAP</th><th>MK-UNet (BUSI), mDICE</th></tr></thead><tbody><tr><td>Baseline</td><td>31.1% (N2UQ)</td><td>45.6% (LSQ)</td></tr><tr><td>+ Compute-Optimal QAT [7] (ICLR 2026)</td><td>31.7%</td><td>46.3%</td></tr><tr><td>+ TR [15] (ICCV 2025)</td><td>31.4%</td><td>45.7%</td></tr><tr><td>+ HMQAT [13] (NN 2025)</td><td>31.3%</td><td>45.6%</td></tr><tr><td>+ QT-DoG [14] (ICML 2025)</td><td>31.7%</td><td>46.8%</td></tr><tr><td>+ EMA [10] (WACV 2024)</td><td>32.2%</td><td>45.9%</td></tr><tr><td>+ Ours</td><td><b>33.2%</b> (N2UQ + Ours)</td><td><b>49.7%</b> (LSQ + Ours)</td></tr><tr><td>+ Best combination</td><td><b>34.0%</b> (N2UQ + EMA [10] + Ours)</td><td><b>51.4%</b> (LSQ + QT-DoG [14] + Ours)</td></tr></tbody></table>

4.3 Comparison with SOTA Optimization Methods

We further evaluate the proposed framework against other optimization strategies across different quantization baselines and tasks. Specifically, Table 3 reports comparisons with state-of-the-art training-optimization-centric QAT methods. The reference results are rigorously re-implemented according to the original papers. More details on the reproduction procedure and configuration setup are provided in the **Appendix 9**.

Compared to the baseline, most reference optimization approaches yield only marginal improvements. Since EMA is specifically designed for YOLO, it exhibits obvious advantage over other reference methods. However, when transferred to MK-UNet, its accuracy drops drastically. This indicates that existing strategies are insufficient for addressing the inherent gradient imbalance problem in feature fusion. In contrast, the proposed method substantially alleviates this issue, achieving an average improvement of 3%–4% over competing approaches.

Finally, we augment both QT-DoG and EMA with our proposed approach. This yields consistent improvements of +4.6% and +1.8%, demonstrating that our method not only enhances the underlying quantizers but also provides complementary benefits to existing optimization strategies.

4.4 Visual Analysis

To better illustrate how Q-GBFusion facilitates quantization-aware training we provide, in this section, visualizations of intermediate representations during both forward and backward propagation in Fig. 3.

Q² for Low-Bit Quantization

![](images/task_4c3e0ecb40d1_page_12_pic_0.png)

**Fig. 3.** (a) Visualization of feature magnitudes across 32 channels at the concatenation layer in YOLOv5. (b) Gradient magnitudes from the two branches at the feature fusion node in YOLOv5 after applying Q-GBFusion.

**Table 4.** Component ablation on 4-bit quantized YOLOv5 (PASCAL VOC).

<table><thead><tr><th rowspan="2">Strategy</th><th colspan="2">N2UQ (W4A4)</th><th colspan="2">LSQ (W4A4)</th></tr><tr><th>mAP</th><th>Time(h)</th><th>mAP</th><th>Time(h)</th></tr></thead><tbody><tr><td>Baseline</td><td>82.1%</td><td>6.25</td><td>76.9%</td><td>3.80</td></tr><tr><td>+ Q-GBFusion</td><td>83.5%</td><td>6.47</td><td>78.4%</td><td>3.83</td></tr><tr><td>+ Q-GBFusion + Q-ADA (KL)</td><td>83.8%</td><td>3.31</td><td><strong>78.9%</strong></td><td>2.01</td></tr><tr><td>+ Q-GBFusion + Q-ADA (JS)</td><td><strong>84.2%</strong></td><td>2.98</td><td>78.5%</td><td>2.20</td></tr></tbody></table>

In Fig. 3(a), the original feature distributions exhibit significant disparities in dynamic range and scale. After applying Q-GBFusion, the fused outputs demonstrate a much more uniform and balanced distribution across all channels, indicating that our method effectively normalizes the cross-branch feature scales prior to fusion.

In the backward pass, Fig. 3(b) plots the mean gradient magnitudes flowing back through the two branches during quantization training. Compared to Fig. 1(b), it is clear that, after integrating Q-GBFusion, the gradient flows from both branches become significantly more balanced, with similar magnitude trajectories over training steps.

To demonstrate how Q-ADA improves localization in quantized object detection, we compare the IoU distributions of YOLOv11 predictions before and after applying Q-ADA under 4-bit N2UQ quantization (Fig. 4). It is clear that, without Q-ADA, quantization shifts IoU downward, sharply reducing high-IoU boxes (≥0.5/0.8) due to impaired spatial cues. Q-ADA aligns FP/quantized attention, preserving semantics and recovering accurate, high-confidence localization.

## 4.5 Ablation Study

**Component Ablation: Q-GBFusion vs. Q-ADA** We report the component-wise ablation results in Table 4. *Time* is measured under the same hardware/setup with validation-based early stopping, thus reporting time-to-convergence (not

Z. Wang and D. Wang

![](images/task_4c3e0ecb40d1_page_13_pic_0.png)

**Fig. 4.** Comparison of IoU distributions before and after integrating Q-ADA. (a) YOLOv11 quantized directly using N2UQ. (b) YOLOv11 quantized with Q-ADA integrated.

fixed epochs). The teacher is a frozen pretrained full-precision model, which does not participate in quantized parameter updates and is used only to generate supervision signals for distillation.

It can be observed from the table that Q-GBFusion consistently improves mAP by 1.4–1.5% across quantizers, indicating that balancing multi-scale fusion effectively stabilizes gradient propagation. Adding Q-ADA yields further gains (+0.3% with KL and +0.7% with JS) and substantially reduces training time due to faster convergence.

The choice of divergence interacts non-trivially with the quantization scheme. KL divergence, which places greater emphasis on discrepancies in low-probability regions, tends to work better with uniform quantizers (e.g., LSQ) that impose bounded activation ranges, thereby enabling more fine-grained modeling within a constrained domain. In contrast, JS divergence is symmetric and more tolerant to global distribution mismatch, making it a better match for non-uniform quantizers (e.g., N2UQ) whose activation statistics are inherently more flexible.

**Deployment Ablation: LayerNorm Removal Efficiency** Due to space limitations, we move additional YOLO results to the **Appendix 7.2**, and present the MK-UNet results in the main text. As summarized in Table 5, removing LayerNorm causes only a marginal average drop of 0.3%, which is negligible for most practical applications, while a short post-folding fine-tuning stage recovers performance within minutes, indicating that the proposed folding procedure is practically efficient.

5 Conclusion

In this work, we revisit low-bit quantization for complex vision models from a new perspective and identify a key factor behind the difficulty: biased gradient updates at feature fusion. To address this, we propose Q-GBFusion, a

Q² for Low-Bit Quantization 15

Table 5. LayerNorm removal on quantized MK-UNet (BUSI).

<table><thead><tr><th>Quantizer</th><th>With LN</th><th>Without LN</th><th>Time(min)</th></tr></thead><tbody><tr><td>N2UQ</td><td>60.7%</td><td>60.4%</td><td>5.6</td></tr><tr><td>PACT</td><td>46.3%</td><td>46.1%</td><td>3.4</td></tr><tr><td>LSQ</td><td>49.7%</td><td>49.3%</td><td>3.6</td></tr></tbody></table>

lightweight closed-loop fusion strategy that balances gradient flow across shallow and deep branches, and Q-ADA, a feature-aware supervision strategy that improves high-level semantic alignment during quantization-aware training. Extensive experiments demonstrate the effectiveness of both approaches.

References

1. Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A.: Dataset of breast ultrasound images. Data in Brief **28**, 104863 (2020). https://doi.org/10.1016/j.dib.2019.104863, https://www.sciencedirect.com/science/article/pii/S2352340919312181
2. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016)
3. Boo, Y., Shin, S., Choi, J., Sung, W.: Stochastic precision ensemble: self-knowledge distillation for quantized deep neural networks. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 6794–6802 (2021)
4. Choi, J., Wang, Z., Venkataramani, S., Chuang, P.I., Srinivasan, V., Gopalakrishnan, K.: PACT: parameterized clipping activation for quantized neural networks. CoRR **abs/1805.06085** (2018), http://arxiv.org/abs/1805.06085
5. Chu, X., Li, L., Zhang, B.: Make RepVGG greater again: A quantization-aware approach. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 11624–11632 (2024). https://doi.org/10.1609/aaai.v38i10.29045, https://doi.org/10.1609/aaai.v38i10.29045
6. Ding, Y., Feng, W., Chen, C., Guo, J., Liu, X.: Reg-ptq: Regression-specialized post-training quantization for fully quantized object detector. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16174–16184 (2024). https://doi.org/10.1109/CVPR52733.2024.01531
7. Dremov, A., Grangier, D., Katharopoulos, A., Hannun, A.: Compute-optimal quantization-aware training. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=QpbtT95S95
8. Esser, S.K., McKinstry, J.L., Bablani, D., Appuswamy, R., Modha, D.S.: Learned step size quantization. In: International Conference on Learning Representations (2020), https://openreview.net/forum?id=rkg066VKDS
9. Everingham, M., Gool, L.V., Williams, C.K.I., Winn, J., Zisserman, A.: The pascal visual object classes (voc) challenge. International Journal of Computer Vision **88**(2), 303–338 (2010)
10. Gupta, K., Asthana, A.: Reducing the side-effects of oscillations in training of quantized yolo networks. In: 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 2440–2449 (2024). https://doi.org/10.1109/WACV57701.2024.00244

Z. Wang and D. Wang

11. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: 2017 IEEE International Conference on Computer Vision (ICCV). pp. 2980–2988 (2017). https://doi.org/10.1109/ICCV.2017.322
12. Huang, L., Dong, Z., Chen, S.L., Zhang, R., Ti, S., Chen, F., Yin, X.C.: Hqod: Harmonious quantization for object detection. In: 2024 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6 (2024). https://doi.org/10.1109/ICME57554.2024.10687589
13. Huang, Z., Han, X., Yu, Z., Zhao, Y., Hou, M., Hu, S.: Hessian-based mixed-precision quantization with transition aware training for neural networks. Neural Networks 182, 106910 (2025)
14. Javed, S., Le, H., Salzmann, M.: QT-dog: Quantization-aware training for domain generalization. In: Forty-second International Conference on Machine Learning (2025), https://openreview.net/forum?id=OS2ZVeHI4U
15. Lee, J., Jeon, J., Kim, D., Ham, B.: Scheduling weight transitions for quantization-aware training. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23466–23475 (2025)
16. Liang, G., Liu, X., Wu, J.: GPLQ: A general, practical, and lightning QAT method for vision transformers. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum?id=58Vr1KOWG9
17. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (eds.) Computer Vision – ECCV 2014. pp. 740–755. Springer International Publishing, Cham (2014)
18. Liu, S., Qi, L., Qin, H., Shi, J., Jia, J.: Path aggregation network for instance segmentation. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8759–8768 (2018). https://doi.org/10.1109/CVPR.2018.00913
19. Liu, Z., Cheng, K.T., Huang, D., Xing, E., Shen, Z.: Nonuniform-to-uniform quantization: Towards accurate quantization via generalized straight-through estimation. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)
20. Rahman, M.M., Marculescu, R.: Mk-unet: Multi-kernel lightweight cnn for medical image segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 1042–1051 (October 2025)
21. Rokh, B., Azarpeyvand, A., Khanteymoori, A.: A comprehensive survey on model quantization for deep neural networks in image classification. ACM Trans. Intell. Syst. Technol. 14(6) (Nov 2023). https://doi.org/10.1145/3623402, https://doi.org/10.1145/3623402
22. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: 2017 IEEE International Conference on Computer Vision (ICCV). pp. 618–626 (2017). https://doi.org/10.1109/ICCV.2017.74
23. Tong, K., Wu, Y.: Small object detection using deep feature learning and feature fusion network. Engineering Applications of Artificial Intelligence 132, 107931 (2024). https://doi.org/10.1016/j.engappai.2024.107931, https://www.sciencedirect.com/science/article/pii/S0952197624000897
24. Wang, M., Sun, H., Shi, J., Liu, X., Cao, X., Zhang, L., Zhang, B.: Q-yolo: Efficient inference for real-time object detection. In: Lu, H., Blumenstein, M., Cho, S.B., Liu, C.L., Yagi, Y., Kamiya, T. (eds.) Pattern Recognition. pp. 307–321. Springer Nature Switzerland, Cham (2023)

Q² for Low-Bit Quantization

25. Wang, R., Sun, H., Yang, L., Lin, S., Liu, C., Gao, Y., Hu, Y., Zhang, B.: Aq-detr: Low-bit quantized detection transformer with auxiliary queries. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 15598–15606 (2024). https://doi.org/10.1609/aaai.v38i14.2948
26. Wei, Y., Pan, X., Qin, H., Ouyang, W., Yan, J.: Quantization mimic: Towards very tiny cnn for object detection. In: The European Conference on Computer Vision (ECCV) (September 2018)
27. Xu, S., Li, Y., Lin, M., Gao, P., Guo, G., Lü, J., Zhang, B.: Q-detr: An efficient low-bit quantized detection transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3842–3851 (2023)
28. Yamamoto, K.: Learnable companding quantization for accurate low-bit neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5029–5038 (2021)
29. Zhang, R., Chung, A.C.: Efficientq: An efficient and accurate post-training neural network quantization method for medical image segmentation. Medical Image Analysis 97, 103277 (2024). https://doi.org/10.1016/j.media.2024.103277, https://www.sciencedirect.com/science/article/pii/S1361841524002020
30. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J.: DETRs Beat YOLOs on Real-time Object Detection. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16965–16974. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2024). https://doi.org/10.1109/CVPR52733.2024.01605, https://doi.ieeecomputersociety.org/10.1109/CVPR52733.2024.01605
31. Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., Ren, D.: Distance-iou loss: Faster and better learning for bounding box regression. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 12993–13000 (2020). https://doi.org/10.1609/aaai.v34i07.6999, https://doi.org/10.1609/aaai.v34i07.6999
32. Zhu, K., He, Y.Y., Wu, J.: Quantized feature distillation for network quantization. Proceedings of the AAAI Conference on Artificial Intelligence 37(9), 11452–11460 (Jun 2023). https://doi.org/10.1609/aaai.v37i9.26354, https://ojs.aaai.org/index.php/AAAI/article/view/26354

Z. Wang and D. Wang

![](images/task_4c3e0ecb40d1_page_17_pic_0.png)

Fig. 5. (a) Gradient Measurement Results for YOLOv11. (b) Gradient Measurement Results for RT-DETR. (c) Gradient Measurement Results for MK-UNet. (d) Gradient Measurement Results for full-precision model(YOLOv11).

# Appendix

## 6 Visualization of Gradient Imbalance in Other Models

Using the same tracking protocol as in the main text, we further trace and visualize the gradient curves of YOLOv11, RT-DETR, and MK-UNet, as shown in Fig. 5. The results show that gradient imbalance is not an isolated phenomenon specific to a single model, but a common behavior observed across different architectures. Furthermore, as shown in Fig. 5(d), this phenomenon is not evident in the full-precision model (YOLOv11), suggesting that it is closely related to the low-bit quantization process. These observations provide cross-model evidence for our motivation that complex vision networks share a structural bottleneck at feature fusion stages.

## 7 How LayerNorm is Safely Removed

### 7.1 Theoretical Justification

In quantized neural networks, LayerNorm removal is substantially easier than in full-precision networks. Because the dynamic range of fixed-point activations

Q² for Low-Bit Quantization

(e.g., 4-bit) is strictly constrained, the input statistics of LayerNorm exhibit much smaller fluctuations. As a result, LayerNorm can be accurately approximated by a fixed affine transformation estimated from calibration data, enabling safe and efficient removal without extensive retraining, in contrast to full-precision settings.

The LayerNorm layers introduced by our framework can be removed via the following steps:

(1) After standard quantization-aware training (QAT) converges, we estimate fixed per-layer statistics $(\mu_0, \sigma_0)$, i.e., the approximate mean and standard deviation of each LayerNorm input, using a small calibration dataset;
(2) We then fold the resulting affine transformation into the adjacent convolutional layer as follows:

Assuming that the standard layer normalization is defined by

$$
\mathrm{LN}(x) = \frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \varepsilon}} \qquad (17)
$$

where $\mu(x)$ and $\sigma^2(x)$ denote the mean and variance of the input. Then, the LayerNorm operation can be approximated as the following channel-wise affine transformation:

$$
A = \frac{1}{\sqrt{\sigma_0^2 + \varepsilon}}, \quad B = -\frac{\mu_0}{\sqrt{\sigma_0^2 + \varepsilon}}. \qquad (18)
$$

where $A$ and $B$ serve as per-channel scaling and bias terms that replace the LayerNorm operation during inference.

Assuming the network computes:

$$
x \xrightarrow{\text{LN}} \tilde{x} \xrightarrow{\text{Conv}(W, b)} y, \qquad (19)
$$

where the LayerNorm output is an affine transform per channel:

$$
\tilde{x}_i = A_i x_i + B_i \qquad (20)
$$

Let $W_{o,i,k,l}$ denote the convolution weight for output channel $o$, input channel $i$, and spatial position $(k, l)$, and $b_o$ the corresponding bias. Then, the convolution is then:

$$
y_o = \sum_{i,k,l} W_{o,i,k,l} \tilde{x}_{i,k,l} + b_o = \sum_{i,k,l} W_{o,i,k,l} (A_i x_{i,k,l} + B_i) + b_o \qquad (21)
$$

This shows that the LayerNorm affine parameters can be absorbed into the subsequent convolution, yielding equivalent parameters:

$$
W'_{o,i,k,l} = W_{o,i,k,l} \cdot A_i, \quad b'_{o} = b_{o} + \sum_{i,k,l} W_{o,i,k,l} B_i \qquad (22)
$$

Replacing $(W, b)$ with $(W', b')$ leaves the forward pass unchanged, allowing the LayerNorm layer to be replaced by an identity mapping and safely removed during inference.

Z. Wang and D. Wang

**Table 6.** Accuracy of the quantized YOLOv5 model on the PASCAL VOC dataset before and after LayerNorm removal. Reported time denotes fine-tuning duration.

<table><thead><tr><th>Method</th><th>With LN</th><th>Without LN</th><th>Time(min)</th></tr></thead><tbody><tr><td>N2UQ</td><td>84.2%</td><td>84.1%</td><td>16.0</td></tr><tr><td>PACT</td><td>80.6%</td><td>80.6%</td><td>8.4</td></tr><tr><td>LSQ</td><td>78.9%</td><td>78.5%</td><td>9.0</td></tr></tbody></table>

(3) Finally, we perform post-LN-removal fine-tuning on a small subset of the quantization dataset for both tasks, using a unified strategy: no distillation is employed, and optimization relies solely on the native task losses. In both models, we use the Adam optimizer with a small learning rate of $2 \times 10^{-5}$ to stably readjust feature distributions after LayerNorm removal.

## 7.2 Experimental Verification

Tables 6 compare the accuracy of the quantized YOLO models before and after LayerNorm removal. The results show that removing LayerNorm incurs only a marginal average degradation of 0.17%, which is negligible for most practical applications. Moreover, the subsequent fine-tuning converges within minutes, demonstrating high efficiency.

# 8 Final Experimental Setup

All experiments were conducted on Ubuntu 20.04 with PyTorch 2.3.1 and CUDA 11.8. During training, we applied the Q-ADA strategy with a loss weight of 0.01, which may be slightly tuned depending on the quantization method.

## 8.1 Object Detection Experimental Configuration

For object detection on COCO and PASCAL VOC, we use SGD with models initialized from full-precision pretrained checkpoints. The initial learning rate is set to 0.00334, decayed via OneCycleLR with a final ratio of 0.15135. Momentum and weight decay are fixed at 0.74832 and 0.00025, respectively. All experiments use a batch size of 64, a fixed random seed of 0 for reproducibility, and 4 data loader workers to balance I/O efficiency and system overhead.

## 8.2 Image Segment Experimental Configuration

For medical image segmentation on the BUSI dataset, we adopt Adam with an initial learning rate of $10^{-4}$, initializing from a full-precision MK-UNet checkpoint [20]. UNet and its variants are canonical encoder-decoder architectures for segmentation, fusing shallow spatial details from the encoder with deep semantic features from the decoder. MK-UNet represents the latest advancement in this family and serves as our baseline.

Q² for Low-Bit Quantization

9 Reproduction Details

To demonstrate the effectiveness of our framework, we compare it against several recent quantization-aware training (QAT) optimization methods, including QT-DoG [14], TR [15], HMQAT [13], and EMA [10]. We also include EQ [29], a recent post-training quantization (PTQ) method designed for quantizing segmentation models. Since these methods were not all originally developed for YOLO or MK-UNet, we reimplement them using the authors' publicly released codebases and adapt them to our detection and segmentation benchmarks. Implementation details are as follows:

QT-DoG[14]. For QT-DoG, we extract its core QAT mechanism, namely, the stage-wise quantization scheduling, and integrate it into our framework. Following the original protocol, we begin with 10 epochs of full-precision (FP32) training to stabilize the shared backbone, then introduce 4-bit activation quantization for the next 10 epochs to allow gradual adaptation to quantization perturbations across tasks, and finally enable full W4A4 quantization for the remaining epochs to achieve low-bit convergence. No modifications are made to the architecture, task heads, loss functions, or training pipeline, ensuring a strictly fair comparison.

TR[15]. The TR scheduling mechanism, originally developed for classification, is adapted to our W4A4 setting by preserving its core principle: constraining the magnitude of latent weight updates to limit the transition frequency of quantized weights. We replace all learning rate based update logic in our framework with the TR driven update rule, effectively inserting a transition rate scheduler before the QAT optimizer. In this configuration, TR rather than the learning rate governs the update rhythm of quantized parameters, enforcing stable coarse to fine convergence. For fair comparison, the same TR schedule is applied to the shared backbone.

HMQAT[13]. We reformulate HMQAT's Hessian-driven mixed-precision strategy within our quantization framework as a second-order supervision mechanism. Rather than performing bit-width search, we leverage Hessian-based layer sensitivity estimates solely to guide gradient updates and quantization map adjustments. Following the original methodology, we compute the average Hessian trace for each layer using a full-precision teacher model, combine it with the layer's parameter scale to derive sensitivity scores, and interpret these scores as priors for quantization tolerance: layers with high sensitivity are more vulnerable to quantization noise, whereas less sensitive layers can tolerate greater distortion. Importantly, we discard the mixed-precision search component entirely and retain only two core ideas—Hessian- and parameter-based sensitivity modeling, and sensitivity-aware fine-tuning during quantization-aware training.

EMA[10]. For EMA-based quantization, we reproduce its core contribution—smoothing of latent weights and quantization scale factors—while enforcing uniform W4A4 precision across all layers. During training, we maintain exponential moving average (EMA) versions of both backbone and task-head latent weights as well as quantization scales. We also implement its quantization correction (QC) mechanism, but without per-channel extensions or architectural

Z. Wang and D. Wang

modifications. In our setup, QC serves as a lightweight post-quantization calibration step: after completing W4A4 quantization-aware training, we freeze all task parameters and optimize a minimal set of affine correction factors on a small calibration set.

**EQ[29].** To reproduce EfficientQ on MK-UNet, we begin with a pretrained full-precision model and apply layer-wise 8-bit post-training quantization. A single forward pass on one calibration sample is used to collect the full-precision output of each layer as the reconstruction target. For every convolutional layer, we first refine the activation quantization range via alternating minimization, then quantize the weights by solving a quadratic output-matching objective using ADMM, projecting the solution onto the discrete quantized set after each iteration. Following the original method, foreground regions are assigned higher reconstruction weights. The quantized output of each layer is propagated to the next, and no backpropagation or retraining is performed, resulting in an 8-bit post-training quantized version of MK-UNet for comparison.

**Compute-Optimal QAT[7].** To reproduce this optimization strategy in our visual tasks, we adopt the paper's cooldown-QAT fusion schedule while keeping the model, quantizer, losses, optimizer, data augmentation, and total training budget unchanged. Unlike the classic pipeline (full-precision training with cooldown, followed by QAT), we switch to QAT before the full-precision cooldown stage, enable fake quantization at the target bit-width, and perform the remaining learning-rate decay jointly with QAT (with a short QAT re-warmup after switching).

![](images/task_4c3e0ecb40d1_page_21_pic_0.png)

**Fig. 6.** Workflow of Q-ADA. DNN denotes the full-precision baseline model, and QNN represents its quantized counterpart. Highlighted regions in the heatmaps indicate areas of critical semantic importance, thereby guiding the reduction of quantization errors during training.

Q² for Low-Bit Quantization

**Table 7.** Supplementary experimental results of different quantizers on the COCO dataset.

<table><thead><tr><th>Network</th><th>BW</th><th>Method</th><th>mAP<sub>50</sub></th><th>mAP</th></tr></thead><tbody><tr><td rowspan="6">YOLOv5s W4A4</td><td rowspan="6"></td><td>PACT</td><td>45.3%</td><td>28.3%</td></tr><tr><td>PACT +Ours</td><td><b>47.0%</b></td><td><b>29.2%</b></td></tr><tr><td>LSQ</td><td>44.8%</td><td>26.9%</td></tr><tr><td>LSQ +Ours</td><td><b>45.8%</b></td><td><b>27.9%</b></td></tr><tr><td>N2UQ</td><td>50.2%</td><td>31.1%</td></tr><tr><td>N2UQ +Ours</td><td><b>51.4%</b></td><td><b>33.2%</b></td></tr><tr><td rowspan="6">YOLOv11s W4A4</td><td rowspan="6"></td><td>PACT</td><td>51.2%</td><td>35.5%</td></tr><tr><td>PACT +Ours</td><td><b>52.6%</b></td><td><b>36.7%</b></td></tr><tr><td>LSQ</td><td>49.4%</td><td>33.0%</td></tr><tr><td>LSQ +Ours</td><td><b>50.7%</b></td><td><b>34.8%</b></td></tr><tr><td>N2UQ</td><td>57.0%</td><td>40.2%</td></tr><tr><td>N2UQ +Ours</td><td><b>58.3%</b></td><td><b>41.2%</b></td></tr><tr><td rowspan="6">RT-DETR W4A4</td><td rowspan="6"></td><td>Q-DETR</td><td>58.3%</td><td>40.7%</td></tr><tr><td>Q-DETR +Ours</td><td><b>59.6%</b></td><td><b>41.9%</b></td></tr><tr><td>AQ-DETR</td><td>59.0%</td><td>41.6%</td></tr><tr><td>AQ-DETR +Ours</td><td><b>60.6%</b></td><td><b>43.0%</b></td></tr><tr><td>GPLQ</td><td>60.7%</td><td>43.8%</td></tr><tr><td>GPLQ +Ours</td><td><b>62.9%</b></td><td><b>45.4%</b></td></tr></tbody></table>

# 10 Supplementary Experimental Results

## 10.1 Experimental Quantization Results on COCO

Due to space constraints in the main paper, we reported only the results of combining N2UQ with our method on the COCO dataset, evaluated using the standard metric mAP$_{50-95}$. In this section, we present additional results for multiple quantizers on COCO, including both mAP$_{50}$ and mAP$_{50-95}$ (denoted as mAP). As shown in Table 7, experiments on the COCO benchmark also demonstrate that our strategy consistently improves performance by 1% to 2% in both mAP$_{50}$ and mAP across a range of mainstream quantization baselines, highlighting its effectiveness and broad applicability.

## 10.2 Additional Empirical Evidence

To further disentangle whether the low-bit degradation can be mitigated by simple optimization tuning, we run a controlled ablation that applies *fixed* branch-wise learning-coefficient scaling between the shallow and deep branches, and report the results in Table 8. Overall, varying a constant ratio yields *non-monotonic* and *inconsistent* changes in accuracy (e.g., a mild gain at $\times$4 but a clear drop at $\times$8), indicating that the observed degradation cannot be reliably resolved by a static re-weighting schedule. This supports our hypothesis that low-bit quantization induces a *structured* learning imbalance at feature-fusion

Z. Wang and D. Wang

modules, which calls for *adaptive* rebalancing rather than hand-tuned fixed coefficients.

In addition, when applied to the full-precision model, Q-GBFusion does not improve performance (and remains within normal training variance), suggesting that the module primarily acts as a corrective mechanism under quantization, rather than a generic architectural enhancement.

**Table 8.** Negative control: fixed branch-wise coefficient scaling is insufficient (×N: shallow/deep = N)

<table><thead><tr><th>Coeff. Ratio</th><th>BW</th><th>Network</th><th>Quantizer</th><th>mDICE</th></tr></thead><tbody><tr><td>×1</td><td>W4A4</td><td>MK-Unet</td><td>N2UQ</td><td>55.4%</td></tr><tr><td>×2</td><td>W4A4</td><td>MK-Unet</td><td>N2UQ</td><td>55.6%</td></tr><tr><td>×4</td><td>W4A4</td><td>MK-Unet</td><td>N2UQ</td><td>55.9%</td></tr><tr><td>×8</td><td>W4A4</td><td>MK-Unet</td><td>N2UQ</td><td>53.8%</td></tr><tr><td>×1</td><td>FP32</td><td>MK-Unet</td><td>–</td><td>69.5%</td></tr><tr><td>×1</td><td>FP32</td><td>MK-Unet + Q-GBFusion</td><td>–</td><td>69.3%</td></tr></tbody></table>

## 11 Visualizing the Mechanism of Q-ADA

As shown in Fig. 6, during the early stages of quantization training, the substantial quantization error introduced by naive quantization severely disrupts critical attention patterns. This results in significant degradation of semantic perception, particularly within target regions, and manifests as pronounced feature attenuation and information collapse. In effect, the model at this stage suffers from both severe semantic information loss and high levels of quantization noise.

As the distillation process progresses, soft supervision from the teacher model gradually guides the quantized model to realign its attention distribution. Responses associated with spatial focus, target shape, and boundary regions are progressively restored. Meanwhile, quantization error diminishes significantly over the course of training, enabling the model to locate and represent target regions more accurately and ultimately yielding stable and reliable quantized feature representations.