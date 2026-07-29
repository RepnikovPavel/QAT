# Paper arXiv:2503.11159

Source: arXiv:2503.11159 (pdftotext first-pass; ocrc parse supersedes when ready).

```
                                             Stabilizing Quantization-Aware Training by Implicit-Regularization on Hessian
                                                                                Matrix

                                                                                     Junbiao Pang1 , Tianyang Cai1

                                                                                    1
                                                                                   Beijing University Of Technology
                                                                          junbiao pang@bjut.edu.cn, tianyang cai@163.com
arXiv:2503.11159v1 [cs.CV] 14 Mar 2025




                                                                  Abstract                                                             Variation of Prediction Output
                                                                                                                        14
                                                                                                                                                                                 FP32
                                              Quantization-Aware Training (QAT) is one of the                                                                                    LSQ_W4A4
                                                                                                                        12                                                       LSQ_W2A4
                                              prevailing neural network compression solutions.                                                                                   FPQ_W4A4
                                              However, its stability has been challenged for yield-                     10                                                       FPQ_W2A4
                                              ing deteriorating performances as the quantization
                                                                                                                        8
                                              error is inevitable. We find that the sharp land-



                                                                                                             Variance
                                              scape of loss, which leads to a dramatic perfor-                          6
                                              mance drop, is an essential factor that causes in-
                                              stability. Theoretically, we have discovered that                         4

                                              the perturbations in the feature would bring a flat                       2
                                              local minima. However, simply adding perturba-
                                              tions into either weight or feature empirically dete-                     0
                                                                                                                             FP32   LSQ_W4A4       LSQ_W2A4         FPQ_W4A4   FPQ_W2A4
                                                                                                                                               Network Architecture
                                              riorates the performance of the Full Precision (FP)
                                              model. In this paper, we propose Feature-Perturbed
                                              Quantization (FPQ) to stochastically perturb the            Figure 1: The output variances of FP32, LSQ W4A4, LSQ W2A4,
                                                                                                          FPQ W4A4, FPQ W2A4 models of the ResNet18 on CIFAR-10
                                              feature and employ the feature distillation method
                                                                                                          dataset.
                                              to the quantized model. Our method generalizes
                                              well to different network architectures and various
                                              QAT methods. Furthermore, we mathematically                 to maintain its accuracy especially for an extremely low-bit
                                              show that FPQ implicitly regularizes the Hessian            width model.
                                              norm, which calibrates the smoothness of a loss
                                                                                                             Despite QAT tries to preserve the performance of the low
                                              landscape. Extensive experiments demonstrate that
                                                                                                          bit model by re-training (or fine-tuning) the weights, the sta-
                                              our approach significantly outperforms the current
                                                                                                          bility of a QAT model (i.e., the empirical performance is sen-
                                              State-Of-The-Art (SOTA) QAT methods and even
                                                                                                          sitive to the noise from the input for the quantizated model
                                              the FP counterparts.
                                                                                                          with the different bit widths) has been challenged. Impor-
                                                                                                          tantly, the stability of a QAT model is considered to be the
                                                                                                          empirical way to calibrate the Lipschitz constant of a net-
                                         1    Introduction                                                work [Lin et al., 2019]. That is, if the quantization error oc-
                                         Model compression has become an essential requirement for        curs, how stable is the output of a neural network?
                                         integrating deep models into edge computing devices. The            We have applied Gaussian-distribution perturbations to
                                         prevalent methods in the domain of model compression in-         the original inputs to investigate the stability of the quan-
                                         clude the search for optimal neural architectures [Zoph and      tized DNN. Fig. 1 illustrates that as the model parame-
                                         Le, 2016], network pruning [Han et al., 2015], and the Deep      ters are optimized, the stability of FP and the SOTA QAT
                                         Neural Network (DNN) quantization [Li et al., 2021a] [Esser      method (LSQ [Esser et al., 2019]) deteriorate. Such instabil-
                                         et al., 2019]. DNN quantization are categorized into two         ity makes QAT converge to the sharp loss landscape which
                                         sub-classes: Post-Training Quantization (PTQ) [Nagel et al.,     leads to the significant performance drops when the model
                                         2020], [Li et al., 2021a], [Wei et al., 2022], [Li et al.,       is quantized. This phenomenon discovers that the quantized
                                         2023b] and Quantization-Aware Training (QAT) [Esser et al.,      model is highly sensitive to small perturbations from the in-
                                         2019], [Nagel et al., 2022]. PTQ adjusts the quantized model     put [Lin et al., 2019] or the quantization error either from the
                                         with a limited calibration dataset, bypassing the need for re-   weights or the activations [Nagel et al., 2019a]. For instance,
                                         training. However, when dealing with low-bit widths, e.g., 2     BENN [Zhu et al., 2019] ensembled multiple binary models
                                         or 4 bits, PTQ may face a significant drop in performance.       to alleviate this problem. However, the inherent instability
                                         Conversely, QAT re-trains or fine-tunes the neural network       starts from the QAT training scheme and the ensambling ap-
proach is a compromise with the multiple quantized models                          minimizing perturbations in the feature of each layer im-
for a prediction.                                                                  plicitly minimizes this term. Theoretically, it is shown
   The important sources of such instability come from three                       that adding noise to features can explicitly optimize the
dimensions: 1) Straight-Through Estimator (STE)-based                              flatness of the model and improve the generalization
QAT has a bias gradient [Shin et al., 2023]; 2) the quan-                          ability of the model.
tization error would be propagated into the next layer of                        • We propose a approach to overcome the instability of
DNN [Pang et al., 2025]; and 3) the FP model itself is un-                         QAT by leveraging implicit Hessian regularization. In-
stable due to the improper Lipsciz constant [Qian and Weg-                         stead of simply adding perturbations to the weights, we
man, 2018]. When these factors appear together for a quan-                         formulate the smoothness of landscape as the minimiza-
tized DNN, there is often a huge performance drop. Given                           tion of stochastic perturbation in feature, defined as the
the above factors as θ1 , θ2 , θ3 receptively, the corresponding                   expected loss within the neighborhood of current neural
                                          (l)       (l)
loss is defined as L(θ1 , θ2 , θ3 ). Let fF P and fQ denote the                    network parameters. We theoretically discover the ad-
features at the l-th layer for the FP model and the quantized                      vantage of Stochastic Feature Perturbations (SFP), and
model, respectively. The quantization would introduce errors                       why SFP should be combined with a feature distillation
into the features as follows:                                                      technique which leads to the improved performance.
                    (l)      (l)                                                 • The proposed methods consistently improve QAT-based
                   fQ = fFP + ∆f (l) (θ1 , θ2 , θ3 ),                  (1)
                                                                                   methods and match or improve the SOTA results on vari-
where ∆f (l) (θ1 , θ2 , θ3 ) represents the perturbation in the fea-               ous network architectures on the CIFAR-10 and CIFAR-
ture caused by the factors θ1 , θ2 and θ3 at the l-th layer. The                   100. Besides, extensive experiments show that our
perturbation ∆f (l) (θ1 , θ2 , θ3 ) would propagate through all                    methods outperform other QAT approaches.
the layers of a neural network, leading to the variance in the
final output as follows:                                                     2     Related Work
             L(θ1 , θ2 , θ3 ) = Loriginal + ∆L(θ1 , θ2 , θ3 ),         (2)   2.1    Quantization-Aware Training
                                                                             As discussed in Section 1, the errors of the first two dimen-
where the Loriginal is from the unperturbed model, and
                                                                             sions are related to QAT, while the third one pertains to the
∆L(θ1 , θ2 , θ3 ) represents the loss caused by about three fac-
                                                                             process of pre-training FP models, which is not the focus of
tors. The perturbation in loss ∆L(θ1 , θ2 , θ3 ) is the errors be-
                                                                             this paper.
tween the quantized features and the FP features at each layer,
                                                                                In the first dimension, the instability is attributed to biased
which is controlled by the Lipsciz constants of different lay-
                                                                             backpropagated gradients due to the round operation. For
ers [Pang et al., 2025]. In summary, QAT lacks the mecha-
                                                                             instance, Straight-Through Estimator (STE) [Bengio et al.,
nism to “absorb” the perturbations (or errors1 ) in each layer
                                                                             2013] utilized the expected probability of stochastic quanti-
of the network caused by the above three aspects.
                                                                             zation as the gradient value for the backpropagation. The
   As a result, even minor perturbations would lead to a sub-
                                                                             EWGS [Lee et al., 2021] adaptively adjusted the quantized
stantial drop in performance, which is demonstrated as the
                                                                             gradients of the quantization error, thereby compensating for
poor generalization ability. As shown in Fig. 1, the instability
                                                                             the biased gradient. The PSG [Kim et al., 2020] scaled the
leads a quantizated model to converge to a local sharp min-
                                                                             gradients based on the position of the weight, effectively of-
ima [Deng et al., 2024]. Concretely, the sharp minima in the
                                                                             fering a form of gradient compensation. The instability of
landscape illustrates that the network weight is almost only
                                                                             gradients also leads to the oscillation problem during the QAT
applicable to the current samples.
                                                                             learning process. The DiffQ [Défossez et al., 2021] identified
   Based on the above analysis, we propose Feature-Perturbed                 that STE would lead to weight oscillations during training.
Quantization (FPQ) that leverages implicit-regularization on                 The Overcoming Oscillation Quantization (OOQ) [Nagel et
hessian matrix to smooth the loss landscape for overcoming                   al., 2022] addressed the oscillation issues by encouraging the
the instability problem. Intuitively, the perturbation makes                 latent weights to align closely with the center of the quantiza-
the optimization of the weights perform well on the nearby                   tion bin. The Resilient Binary Neural Network (ReBNN) [Xu
result rather than exactly the current one. Consequently,                    et al., 2023] introduced a weighted reconstruction loss to for-
we control perturbations of features in each layer during the                mulate an adaptive training objective.
training stage by encouraging the model to converge to flat
                                                                                In the second dimension, DQ [Lin et al., 2019] con-
minima. Besides, we theoretically discover the advantage of
                                                                             troled the error in feature transmission by managing the Lip-
Stochastic Feature Perturbations (SFP), and why SFP should
                                                                             schitz constant. NIPQ [Shin et al., 2023] employed pseudo-
be combined with a feature distillation technique. Our contri-
                                                                             quantization noise to simulate the quantization process, re-
butions are as follows:
                                                                             ducing quantization errors. Similarly, some PTQ efforts
   • Mathematically, we show that the performance drop                       aligned with the features of the FP model through layer-wise
     caused by quantization error is highly related to the                   [Nagel et al., 2020] and block-wise [Li et al., 2021a], [Wei
     norm of Hessian, which is also mentioned empirically                    et al., 2022] reconstruction, thereby preventing the further
     in [Deng et al., 2024]. Furthermore, we discover that                   propagation of errors within the network. To the best of our
                                                                             knowledge, we are the first to apply this idea to stabilize the
   1
       if it doesn’t cause confusion, we will abuse use these terms.         QAT.
   Specially, for the quantization of Stable Diffusion (SD), the
phenomenon of error accumulation is more pronounced due                                      gradient norm of adding noise to feature
to the multi-step inference of the denoising process. BitsFu-                        7       gradient norm of adding noise to weight
sion [Sui et al., 2024] employs a mixed-precision approach                           6
to mitigate the accumulation of errors during the denoising
process. Moreover, most of the work on SD quantization [Li                           5




                                                                             norms
et al., 2023a], [Huang et al., 2024], [He et al., 2024], [So et                      4
al., 2024] is based on BRECQ, as the block-wise reconstruc-
tion in BRECQ can minimize error propagation as much as                              3
possible. However, this work differs from their focus. Their
                                                                                     2
concern is addressing the accuracy issues caused by error ac-
cumulation, while we are concerned with the instability issues
                                                                                         0     25      50      75     100      125      150   175   200
arising from error propagation.                                                                                      Epoch


2.2       Adversarial Robustness                                   Figure 2: A comparison of the norm ∥∇w L(w)∥ trajectories of
                                                                   ResNet-18 on CIFIAR-10.
In this paper, we argue that QAT should exhibit robustness
against feature perturbations. This aim aligns with the topic
of adversarial robustness, which focused on mitigating the            We follow the practice in LSQ [Esser et al., 2019], where
vulnerability of neural networks to the perturbations from in-     s is a learnable parameter. Therefore, the loss function of a
put [Szegedy, 2013], e.g., random smoothing [Lecuyer et al.,       quantized model is given as follows:
2019], [Cohen et al., 2019], and adversarial training [Good-
fellow et al., 2014], [Madry et al., 2017]. Adversarial train-                               arg min Ex∼Dt [L (x; w, s)],                                 (6)
                                                                                                w,s
ing, in particular, optimized the worst-case training loss and
has been shown to not only improve robustness but also en-         where L (·; ·) is the predefined loss function.
hanced performance in tasks such as image classification [Xie         Notice that, the zero-points z are initialized by (5) with the
et al., 2020]. To the best of our knowledge, we are the first to   calibration set Dc , and are fixed throughout the entire QAT
apply this idea to stabilize QAT.                                  training process. During QAT training, the weights involved
                                                                   in the forward propagation are actually the quantized weights
3       Proposed Method                                            ŵ, rather than the floating-point weights w.

3.1       Notation and Background                                  3.2     Feature-Perturbation brings Implicit
Basic Notations. In this paper, x represents a matrix (or ten-             Regularization on Hessian Matrix
sor), a vector is denoted as x, f (x; w) represents a FP model     Motivation.
with the weight w and the input x, f (x; w, s, z) represents a     The convergence property of oscillation near the target value
quantized model with the parameter w, quantization param-          is greatly beneficial to enhance the robustness of the net-
eter s, z and the input x. We assume sample x is generated         work [Chen and Hsieh, 2020]. When some perturbation δ
from the training set Dt .                                         exists, the objective function can be approximated via Taylor
   Quantization. Step size s and zero point z serve as             expansion as follows:
a bridge between floating-point and fixed-point representa-
                                                                          E[L(w + δ)]
tions. Given the input tensor x2 , the quantization operation is
as follows:                                                                                       1
                                                                         ≈E[L(w) + δ · ∇w L(w) + δ T · ∇2w L(w) · δ]
                            x                                                                   2                                                       (7)
              xint = clip ⌊ ⌉ + z, 0, 2q − 1 ,                                                     E[δ 2 ]
                              s                             (3)          ≈L(w) + E[δ] · ∇w L(w) +          Tr(∇2w L(w))
                 x̂ = (xint − z) s,                                                                  2
                                                                   where w is the weight of a DNN f (w). The third term of
where ⌊·⌉ represents the rounding-to-nearest operator, q is
                                                                   Eq. (7) involves the Hessian matrix’s trace. [Keskar et al.,
the predefined quantization bit-width, s denotes the scale be-
                                                                   2017] and [Wen et al., 2020] verified that the smaller the trace
tween two subsequent quantization levels. z stands for the
                                                                   of the Hessian matrix is, the flatter the landscape of loss is. A
zero-points. Both s and z are initialized by a calibration set
                                                                   flat loss surface generally aids the model in finding good local
Dc from the training dataset Dt , i.e., Dc ∈ Dt .
                                                                   optima to enhance the model’s generalization capability.
                            xmax − xmin                               We should make the second term in Eq. (7) be zero as fol-
                        s=               ,                  (4)    lows:
                               2q − 1
                                   xmax                                                 E[δ] · ∇w L(w) = 0.                      (8)
                      z = ⌊qmax −       ⌉,                  (5)
                                     s                             There are two conditions make the Eq.(8) be as zero as pos-
                                                                   sible:
where qmax is the maximum value of the quantized integer.
                                                                       • Condition 1: The ∇w L(w) should be close enough to
    2
        It could either be feature x or weight w.                        zero.
   • Condition 2: The expectation of perturbation E[δ] of ev-        The first term in
                                                                                     Eq. (11) is a uniform distribution with a ex-
      ery layer should be zero.                                                        l
                                                                     pectation of E δ = 0. Consequently, we have:
Intuitively, when a well-trained model f (x, w) is used to
fine-tune the QAT model, the ∇w L(w) would be nearly                              E (δ total )
equal to zero. However, when a certain perturbation δ is in-                        h                           i                                                  (12)
jected into either the weight or the feature of each layer, a nat-               = E fl−1 xl−1 + δ l−1 − fl−1 xl−1     .
ural question is: which scheme would incur a smaller norm
                                                                       Assuming that the function fl−1 is smooth at the point
of the gradient ∇w L(w) than the other?
   Therefore, we investigated the norm of the gradient               xl−1 , and that the perturbation δ l−1 is NOT sufficiently
∇w L(w) when the perturbation δ is injected into either the          small, we apply the second-order Taylor approximation to
weight or the feature of resnet-18 model on the CIFAR-10             Eq. (12) as follows:
dataset. Results in Fig. 2 show that the norm of the gradient              E (δtotal )
norm ∥∇w L(w)∥ caused by the weight perturbation is larger                   
                                                                                                     1
                                                                                                                                    
than that of the feature perturbation. Therefore, in our work,            ≈ E ∇fl−1 (xl−1 )T δ l−1 + (δ l−1 )T ∇2 fl−1 (xl−1 )δ l−1
                                                                                                     2
we only apply perturbations to the features.                                                               
                                                                               1
Feature Perturbation Quantization                                         = E (δ l−1 )T ∇2 fl−1 (xl−1 )δ l−1 ̸= 0.
                                                                               2
Feature Perturbation Quantization (FPQ) introduces the per-                                                                                                             (13)
turbations into features, pushing the quantized model to live
at a flat local minima. Without loss of generality, taking DNN
as an example, we assume that a network has N convolutional             To reduce the bias, we propose a Stochastic Feature Pertur-
layers. Given the input sample x, the  inputs of each layer for     bations (SFP) with a predefined probability p(p > 0). Con-
a quantized model are represented as, x1s , x2s , . . . , xN
                                                           s . We
                                                                     cretely, the l-th features xl is stochastically injected noise as
apply perturbations to the inputs of each convolutional layer        follows:
of quantized model as follows:                                                       xls = xls + δ, with probability p.           (14)
                                                                        With SFP, the probability of injecting perturbations into all
                                           sl sl                     layers simultaneously is pN , which significantly reduces the
                xls = xls + δ, δ ∼ U [−      , ]              (9)
                                           2 2                       occurrence of the bias in Eq. (13), e.g., 0.110 = 1e−11 for
where xls is the input feature of the l-th layer of quantized        a DNN with 10 layers. Besides, we combined with a feature
model, δ is the injected perturbation which follows a uniform        distillation called Channel-wise Standardization Distillation
                    l    l                                           (CSD) to make the Condition 1 hold.
distribution U [− s2 , s2 ], and sl is the quantization parameter
scale corresponding to the input feature xl . However, simply        Channel-wise Standardization Distillation
injecting perturbations by FPQ could disrupt the features of
each layers of DNN, empirically resulting in inferior results.                                     Activation ranges of the ReLU layer output in ResNet-18.
                                                                               2.0
   Proposition 1 discovers that if wl represents the weights of
the l-th convolutional layer and xl is the corresponding input                 1.5
                                                                       Range




feature, the expected perturbation E[δ] is always biased.                      1.0

Proposition 1. (Inject noise into multiple layers would result                 0.5
in a biased expectation) Without loss of generality, given a                   0.0
                                                                                     1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
DNN with N (N ≥ 2) layers, if noises δ l (1 ≤ l ≤ N ) are                                                        Output channel index
simultaneously injected into different layers, the accumulated                               Quantized activation ranges of the ReLU layer output in ResNet-18.
perturbation from different layers is biased.                                  1.5

Proof. Without loss of generality, a DNN is represented as                     1.0
                                                                       Range




            fN (x) = (ϕN . . . ◦ ϕl ◦ . . . ◦ ϕ0 ) (x),      (10)              0.5

where ϕl could be any module of the neural network, e.g.,                      0.0
                                                                                     1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
conventional layer, attention layer.                                                                             Output channel index
   For the l-th layer, the perturbation comes from two sources:
the first one is the perturbation δ l that we directly injected by   Figure 3: Comparisons of the output feature of the ResNet-18 before
Eq. (9); and the second one is from the perturbation propa-          and after quantization. The orange line represents the mean value of
gated from the last layer ϕl−1 . Therefore, the expectation of       the feature (best viewed in color).
the perturbation of the l-th layer E (δ total ) is as follows:
                                                                        We have observed that the quantization operation can af-
    E (δ total )
      h                                 i                        fect the distribution of features. We visualized the output fea-
   = E δ l + fl−1 xl−1 + δ l−1 − fl−1 xl−1                   (11)    tures of a ReLU layer in the ResNet-18 model to compare
               h                         i                     the changes in features before and after quantization. The
   = E δ l + E fl−1 xl−1 + δ l−1 − fl−1 xl−1 .                       results are shown in Fig. 3. The findings indicate that the
quantization operation can cause a shift in the distribution.           fine tuning, if ŵ is the new optimal weight vector, we expect
Specifically, in Fig. 3, the feature with index 30 had a mean           Lval (ŵ) to be smaller than Lval (w∗ ) , provided that the train-
value of 0.12 before quantization, while after quantization,            ing and validation losses are highly correlated. Therefore, the
the mean value shifted to 0.11, resulting in a relative error of        performance of Lval (ŵ), which is the quantity that we care
nearly 10%. The drifted distribution is unfriendly for distilla-        for, will also be bounded by C. Note that the bound C could
tion [Nagel et al., 2019b] [He et al., 2024]. In this work, To          be quite loose since that the network weights changed when
mitigate drifted distribution caused by quantization, we stan-          transitioning from w∗ to ŵ. A more precise bound can be
dardize features before distillation for forcing Mean Squared           obtained by treating g(w) = Lval (w) as a function only pa-
Error (MSE) to focus the feature distribution rather than the           rameterized by w, and then calculate its derivative/Hessian
absolute values as follows:                                             by implicit function theory.
                                                                          The difference between our method and SAM. The con-
                l             l      zl − µ zl
              z̃ = N orm(z ) = p                     ,      (15)        cept of searching for minima characterized as “flat minima”
                                      σ 2 (z l ) + ϵ                    was introduced in [Hochreiter and Schmidhuber, 1994] and
                                               l
                                                                        extensive research has been conducted to explore its con-
where z l is the output of the l-th layer,l z̃ 2is the
                                                       normalized      nection with the generalization ability [Andriushchenko and
feature, µ z is the mean of feature x , σ z l is the vari-
              l
                                                                        Flammarion, 2022] [Zhang et al., 2021] . SAM [Foret et al.,
ance of the feature, and ϵ is the constant preventing the de-           2020] enhances the generalization ability of the model as fol-
nominator being 0.                                                      lows:
   The student model (quantized) fS (x; ws , ss , z s ) is initial-                   min LSAM (w) + λ∥w∥2 ,
ized with a calibration dataset, while the teacher model (FP)                           w
                                                                                                                                      (19)
fT (x; wt , st , z t ) serves as a reference. The standardized                      where LSAM (w) = max L(w + ϵ),
                                                                                                             ∥ϵ∥≤ρ
outputs of the l-th layer in the quantized model and that of
the FP model are as follows:                                            in which ϵ represents weight perturbations in an Euclidean
                                                                        ball within the radius ρ, LSAM is the perturbed loss, and
                      z̃ ls = N orm(ŵls xls ),                         λ∥w∥2 is the standard L2 norm.
                                                                (16)
                       z̃ lt = N orm(wlt xlt ),                            In contrast, the proposed FPQ is formally defined as fol-
                                                                        lows:
where z̃ ls and z̃ lt are the standardized features of the l-th layer                  min Lϵ∼U [− sl , sl ] (w; xl + ϵ),
                                                                                         w           2  2                      (20)
of the quantized model and the FP one, respectively.
   To make Condition 1 hold, one reasonable assumption is                                  s.t. E[δ] · ∇w L(w) = 0,
that if the output of student model fS (x; ws , ss , z s ) is equal     where xl is the input feature of the l-th layer in a DNN.
to that of the teacher one fT (x; wt , st , z t ), the gradient
∇ŵ L(ŵ) tends to be zero. The CSD is as follows:                      Table 1: The comparison of results for optimizing LSQ using SAM
                            X X                           2             and FPQ, and the accuracy (%) of the model with W4A4 quantiza-
             LCSD =                      z̃ ti;j − z̃ i;j
                                                      s     ,    (17)   tion on the CIFAR-10 dataset.
                                                     2
                       i∈[1,N ] j∈[1,c]
                                                                                         Methods         Res18      MBV2
where N represents the number of layers, c represents the
                                                             i;j
number of channels output by the i-th convolution, z̃ i;j
                                                      s , z̃ t
                                                                                         Full prec.      88.72       85.81
represent the standardized features of the j-th channel of the                          LSQ+SAM          89.75       84.72
i-th convolution of the quantized model and the FP model,                                  FPQ           90.16       85.53
respectively, ∥·∥2 denotes ℓ2 normalization.
                                                                           Rather than directly perturbing the weight in Eq. (19), we
3.3   Implicit Regularization on Hessian Matrix                         instead perturbing the features through Eq. (9). The compari-
In the following, we first explain why the spectral norm of             son of results between FPQ and SAM is shown in Tab. 1. As
Hessian is correlated with solution quality, and then formally          discussed in Eq. (7), FPQ imposes an implicit regularization
show that the difference of our algorithms with Sharpness-              on the Hessian matrix. The results indicate that the optimiza-
aware Minimization (SAM).                                               tion using SAM performs worse than that using FPQ. One
   Why is Hessian norm correlated with solution quality?                possible reason is that in high-dimensional spaces, there are
   Assume w∗ is the optimal weight of Eq. (6). Based on Tay-            many directions of perturbations δ and their size ρ to be cho-
lor expansion and assume ∇Lval (w∗ ) = 0 due to optimality              sen, and the random approach in FPQ may outperform the
condition, we have                                                      method of selecting the largest perturbation in SAM for QAT.
                          1
  Lval (ŵ) = Lval (w∗ ) + (ŵ − w∗ )T Ĥ(ŵ − w∗ ), (18)               3.4   Training Processing
                          2
              R ŵ 2                                                    Given a labeled sample (X, y) and the corresponding label y,
where Ĥ = w∗ ∇ Lval (w) dw is the average Hessian. If                  the output of the quantized model and the FP model are:
we assume that Hessian is stable in a local region, then the
quantity of C = ∥∇2 Lval (w∗ )∥∥ŵ − w∗ ∥ can approximately                                  z os = fS (x; ŵs , ss ) ,
bound the performance drop when projecting w∗ to ŵ. After                                                                           (21)
                                                                                             z ot = fT (x; ŵt , st ) .
                                                                                    feature distribution                           feature distribution                            feature distribution
Algorithm 1 QAT with FPQ                                             1.50                         naive quantization
                                                                                                  FP
                                                                                                                                                 naive quantization 1.50
                                                                                                                                                 FP
                                                                                                                                                                                                 naive quantization
                                                                                                                                                                                                 FP
                                                                     1.25                         Ours               1.5                         Ours               1.25                         Ours
 1: Input: labeled data X; FP model f (x; w)                         1.00
                                                                                                                     1.0
                                                                                                                                                                    1.00
                                                                                                                                                                    0.75
                                                                     0.75
 2: for i = 1 to · · · do                                            0.50
                                                                                                                     0.5
                                                                                                                                                                    0.50
                                                                     0.25                                                                                           0.25
 3:   Feed the X into models based on Eq. (21), while in-            0.00                                            0.0                                            0.00
      jecting perturbations into all layers of a DNN with                   1   0     1     2      3       4   5           1   0     1     2      3       4    5           1   0     1     2      3       4    5

      probability p by Eq. (14);
                                                                    Figure 4: Feature distributions of the FP ResNet-18 (blue lines),
 4:   Obtain and standarize the outputs of each layer of the        native baseline (LSQ, red lines), and ours method (green lines) for
      quantized model and the FP model based on Eq. (16);           the same feature of ResNet-18 model on CIFAR-10. We random
 5:   Update for student model based on Eq. (17) and                select three samples and plot the output feature of the same layer.
      Eq. (22);
 6: end for
 7: Output: fS (x; w s , ss , z s ).                                effects of the perturbation and LCSD . We used the W2A4
                                                                    model of ResNet-18 [He et al., 2016] with the LSQ method
                                                                    as the baseline, which achieved an accuracy of 88.36% on
   In the forward through Eq. (21), we apply with stochastic        CIFAR-10 dataset. Tab. 2 illustrated that FPQ has improved
perturbation δ in all the input of convolutional layer with a       the accuracy of W2A4 quantized models for ResNet archi-
predefined probability p. The perturbation drawn from a uni-        tectures. Adding only the perturbation in Eq. (14) improved
form distribution through Eq. (9).                                  the baseline by 0.94%, using only CSD improved the base-
   In this paper, we focus on the classification task. Therefore,   line by 1.3%, and when both were used together, the result
the whole loss consists of two components as follows:               was better, improving the baseline by 1.56%.
             Ltotal (ŵ, x) = CE (z os , y) + LCSD ,         (22)
                                                                    Table 3: The effectiveness of the probability p on CIFAR-10 with
where CE (·, ·) denotes the cross-entropy loss for classifica-      W2A4 quantization (Accuracy %).
tion tasks, zo presents the output from the quantized model
for the labeled data (X, y). Training procedure is shown in
                                                                                                p of FPQ in Eq. (14)                                          Val acc (%)
Algorithm 1.
                                                                                                                       0                                           89.66
4     Experiments                                                                                                     0.1                                          89.92
Experimental Protocols and Datasets. Our code is based on                                                             0.3                                          89.90
PyTorch [Paszke et al., 2019] and relies on the MQBench [Li                                                           0.5                                          89.83
et al., 2021b] package. We used asymmetric quantization by                                                            0.7                                          89.59
default. In this paper, we used the CIFAR-10 and CIFAR-                                                               0.9                                          89.68
100 [Krizhevsky et al., 2009] as dataset. We randomly se-                                                              1                                           89.48
lected 100 images for CIFAR-10 and CIFAR-100 as the cal-
ibration set. We also kept the first and last layers with 8-bit        Effectiveness of probability p. To investigate the impact
quantization, the same as QDrop [Wei et al., 2022]. Addi-           of probability p in Eq. (14), we used CIFAR-10 as the dataset.
tionally, we employed per-channel quantization for weight           Tab. 3 illustrated that the performances of the W2A4 model
quantization. We used WXAX to represent X-bit weight and            with the seven different sets of p. The experimental results
activation quantization.                                            indicated that when the random probability p is too high,
   Training Details. We used SGD as the optimizer, with a           excessive perturbations can disrupt the local minima of the
batch size of 256 and a base learning rate of 0.01. The default     quantized model. Conversely, when the perturbations are too
learning rate (LR) scheduler followed the cosine annealing          small, the insufficient noise failed to smooth the sharpness
method. The weight decay was 0.0005, and the SGD momen-             of the model’s loss landscape. An optimal probability p lies
tum was 0.9. We trained for 200 epochs otherwise specified.         within the range of 0.1 to 0.5.
Table 2: Ablation studies of perturbation and CSD (Accuracy %) on   4.2         Literature Comparison
CIFAR-10 with W2A4 quantization.                                    We selected ResNet-18 and ResNet-50 [He et al., 2016], Mo-
                                                                    bileNetV1 [Howard et al., 2017] and MobileNetV2 [Sandler
             Perturbations     CSD     ResNet-18                    et al., 2018] with depth-wise separable convolutions as the
                                          88.36                     representative network architectures.
                   ✓                      89.30                        CIFAR-10. In Tab. 4, we quantized the weights and acti-
                                ✓         89.66                     vations to 2-bit and 4-bit. We compared our approach with
                   ✓            ✓         89.92                     the effective baselines, including LSQ [Esser et al., 2019],
                                                                    LSQ+ [Bhalgat et al., 2020], PACT [Choi et al., 2018] and
                                                                    KD [Hinton et al., 2015]. Tab. 4 illustrated that when the en-
                                                                    tire training set of CIFAR-10 is used, FPQ significantly sur-
4.1    Ablation Study                                               passed the baselines. In W4A4 quantization, FPQ achieved
Effectiveness of probability FPQ. To investigate the impact         about 1∼2% accuracy improvements over LSQ. Furthermore,
of FPQ, we conducted ablation experiments to validate the           to explore the ability of FPQ, we conducted W2A4 and
Table 4: Comparison among different QAT strategies in terms of                                 9000
                                                                                                                            Comparison of Noise Trace and LSQ Trace                                              Comparison of Noise Trace and LSQ Trace
                                                                                                          Ours Trace                                                                           Ours Trace
accuracy on CIFAR-10.                                                                          8000
                                                                                                          LSQ Trace                                                                 3500       LSQ Trace


                                                                                                                                                                                    3000
                                                                                               7000

                                                                                                                                                                                    2500
                                                                                               6000

   Labeled




                                                                                       Trace




                                                                                                                                                                            Trace
                                                                                                                                                                                    2000
             Methods                       W/A     Res18   Res50   MBV1    MBV2                5000

    data                                                                                       4000                                                                                 1500

   50000     Full Prec.                    32/32   88.72   89.95   85.52   85.81               3000
                                                                                                                                                                                    1000
                                                                                               2000
             PACT [Choi et al., 2018]        4/4   88.15   85.27   80.77   79.88                      0                25       50      75      100     125     150   175
                                                                                                                                                                                    500
                                                                                                                                                                                           0                25       50      75      100     125     150   175
             LSQ [Esser et al., 2019]        4/4   86.69   90.01   82.39   84.45                                                             Epoch                                                                                Epoch
             LSQ+ [Bhalgat et al., 2020]     4/4   88.40   90.30   84.32   84.30
             KD [Hinton et al., 2015]        4/4   88.86   90.34   84.77   83.79                                            (a) ResNet-18                                                             (b) MobileNetV2
             FPQ (Ours)                      4/4   90.16   90.62   84.94   85.53
             PACT [Choi et al., 2018]        2/4   87.55   85.24   69.04   67.18
             LSQ [Esser et al., 2019]        2/4   88.36   90.01   78.15   78.15   Figure 5: Trajectory of the Hessian matrix trace for ResNet-18 and
   50000     LSQ+ [Bhalgat et al., 2020]     2/4   87.76   89.62   81.26   77.00   MobileNetV2 models on the CIFAR-10 dataset.
             KD [Hinton et al., 2015]        2/4   88.83   90.18   78.84   75.56
             FPQ (Ours)                      2/4   89.92   90.39   81.45   79.33
             PACT [Choi et al., 2018]        2/2   76.90   64.94   11.71   10.58
             LSQ [Esser et al., 2019]        2/2   87.60   87.79   75.29   70.32   and 0.6% respectively. Notably, FPQ even exceeds the full-
             LSQ+ [Bhalgat et al., 2020]     2/2   87.60   86.10   74.22   72.18
             KD [Hinton et al., 2015]        2/2   88.06   89.20   68.62   67.15
                                                                                   precision baseline by 0.44% on ResNet-18. In W2A4 set-
             FPQ (Ours)                      2/2   88.11   89.40   76.25   73.00   tings, FPQ maintains strong performance with ResNet-18
                                                                                   (75.77%) and ResNet-50 (78.11%), outperforming LSQ by
Table 5: Comparison among different QAT strategies regarding ac-
                                                                                   0.84% and 0.29%. For the challenging W2A2 configuration,
curacy on CIFAR-100.                                                               FPQ achieves the best results on ResNet-50 (65.23%), Mo-
                                                                                   bileNetV1 (58.46%), and MobileNetV2 (33.16%), demon-
   Labeled                                                                         strating robustness under extreme quantization.
             Methods                       W/A     Res18   Res50   MBV1    MBV2
    data
   50000     Full Prec.                    32/32   75.40   78.94   70.22   71.30   4.3                    Characteristics of FPQ
             PACT [Choi et al., 2018]        4/4   74.17   74.78   64.65   64.06
             LSQ [Esser et al., 2019]        4/4   75.30   78.20   68.63   69.01   Generalization of FPQ: It has been empirically pointed out
             LSQ+ [Bhalgat et al., 2020]     4/4   74.50   77.39   67.89   68.25
             KD [Hinton et al., 2015]        4/4   74.70   78.80   70.96   71.66   that the dominant eigenvalue of ∇2 Lval (w) (spectral norm of
             FPQ (Ours)                      4/4   75.84    78.8   67.55   68.24   Hessian) is highly correlated with the generalization quality
             PACT [Choi et al., 2018]        2/4   73.77   74.72   49.98   57.90   of QAT solutions [Keskar et al., 2017] [Wen et al., 2020].
             LSQ [Esser et al., 2019]        2/4   74.93   77.82   65.13   66.15
   50000     LSQ+ [Bhalgat et al., 2020]     2/4   73.90   76.61   65.28   66.24   In standard QAT training, the Hessian norm is usually great,
             KD [Hinton et al., 2015]        2/4   74.35   77.34   66.90   63.77
             FPQ (Ours)                      2/4   75.77   78.11   64.68   65.35   which leads to deteriorating (test) performance of the solu-
             PACT [Choi et al., 2018]        2/2   65.16   4.26    3.25     8.39   tions. In Fig. 5(a) and Fig. 5(b), we plot the Hessian trace
             LSQ [Esser et al., 2019]        2/2   71.80   62.79   55.53   31.08   during the training procedure and find that the Hessian trace
             LSQ+ [Bhalgat et al., 2020]     2/2   71.25   64.21   55.56   30.08
             KD [Hinton et al., 2015]        2/2   73.13   63.16   55.37   28.56   of the proposed methods is significantly lower than that of
             FPQ (Ours)                      2/2   71.73   65.23   58.46   33.16   the LSQ. The results demonstrate that FPQ would reduce the
                                                                                   trace of the Hessian matrix numerically, thereby enhancing
                                                                                   the model’s generalization ability.
W2A4 quantization experiments. In W2A4 quantization,                                  Distribution Visualization of FPQ: We visualized the
FPQ consistently achieved a 1∼2% accuracy improvement                              feature distribution before and after quantization and the FP
over LSQ in Tab. 4. In W2A2 setting, FPQ achieved about                            model to explore whether FPQ could align the distribution of
1∼3% accuracy improvements over LSQ. Moreover, there                               the quantized model with that of the FP model.
are two interesting observations as follows:                                          As shown in Fig. 4, there was a significant distribution drift
   • For W4A4, our method significantly surpassed the FP                           between the naive quantization and the FP model. This prob-
      counterparts for both ResNet-18 and ResNet-50. For                           lem is the key reason for the drop in performance of the quan-
      example, on the ResNet-18 model, FPQ surpassed the                           tized model. Our method effectively alleviated this problem.
      FP model by 1.44% in accuracy, and on the ResNet-50                          On the one hand, FPQ leveraged the feature distribution of the
      model, FPQ exceeded the FP model by 0.67%.                                   FP model as the ground truth, reducing the distribution drift
   • From W4A4 to W2A2, the performance drop of our                                caused by quantization. On the other hand, FPQ retained the
      method is significantly lower that the other SOTA meth-                      original task loss, making stochastic perturbation in Eq. (14)
      ods. For instance, on the ResNet-50 model, the LSQ                           to smooth the local minima.
      method decreased by 2.22% when reducing from W4A4
      to W2A2, while FPQ decreased by 1.22%. On the Mo-                            5                  Conclusion
      bileNetV1 model, the LSQ+ method saw a 10.1% drop                            In this paper, we propose the FPQ for QAT-based method. By
      when going from W4A4 to W2A2, while FPQ decreased                            applying perturbations to the features, FPQ introduces the im-
      by 8.69%                                                                     plicit regularization to the Hessian matrix, enhancing the sta-
   CIFAR-100. In Tab. 5, we evaluate quantization perfor-                          bility of the model. Specifically, the regularization is carried
mance across varying bit-widths (W4A4, W2A4, W2A2)                                 out with random smoothing. FPQ possesses a much smoother
on CIFAR-100. Same to CIFAR-10, FPQ demonstrates su-                               landscape and has the theoretical guarantee to regularize the
perior performance in most scenarios. For W4A4 quan-                               Hessian norm of the validation loss. Extensive experiments
tization, FPQ achieves the highest accuracy on ResNet-18                           have illustrate the effectiveness of FPQ and we outperform
(75.84%) and ResNet-50 (78.8%), surpassing LSQ by 0.54%                            various SOTA methods.
References                                                     [He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing
                                                                  Ren, and Jian Sun. Deep residual learning for image recog-
[Andriushchenko and Flammarion, 2022] Maksym         An-          nition. In Proceedings of the IEEE conference on computer
  driushchenko and Nicolas Flammarion.           Towards          vision and pattern recognition, pages 770–778, 2016.
  understanding sharpness-aware minimization. In Interna-
  tional Conference on Machine Learning, pages 639–668.        [He et al., 2024] Yefei He, Luping Liu, Jing Liu, Weijia Wu,
  PMLR, 2022.                                                     Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-
                                                                  training quantization for diffusion models. Advances in
[Bengio et al., 2013] Yoshua Bengio, Nicholas Léonard, and       Neural Information Processing Systems, 36, 2024.
  Aaron Courville. Estimating or propagating gradients
                                                               [Hinton et al., 2015] Geoffrey Hinton, Oriol Vinyals, and
  through stochastic neurons for conditional computation.
  arXiv preprint arXiv:1308.3432, 2013.                           Jeff Dean. Distilling the knowledge in a neural network.
                                                                  arXiv preprint arXiv:1503.02531, 2015.
[Bhalgat et al., 2020] Yash Bhalgat, Jinwon Lee, Markus        [Hochreiter and Schmidhuber, 1994] Sepp Hochreiter and
  Nagel, Tijmen Blankevoort, and Nojun Kwak. Lsq+:                Jürgen Schmidhuber. Simplifying neural nets by discov-
  Improving low-bit quantization through learnable offsets        ering flat minima. Advances in neural information pro-
  and better initialization. In Proceedings of the IEEE/CVF       cessing systems, 7, 1994.
  Conference on Computer Vision and Pattern Recognition
  Workshops, pages 696–697, 2020.                              [Howard et al., 2017] Andrew G Howard, Menglong Zhu,
                                                                  Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias
[Chen and Hsieh, 2020] Xiangning Chen and Cho-Jui                 Weyand, Marco Andreetto, and Hartwig Adam. Mo-
  Hsieh. Stabilizing differentiable architecture search via       bilenets: Efficient convolutional neural networks for mo-
  perturbation-based regularization. In International con-        bile vision applications. arXiv preprint arXiv:1704.04861,
  ference on machine learning, pages 1554–1565. PMLR,             2017.
  2020.
                                                               [Huang et al., 2024] Yushi Huang, Ruihao Gong, Jing Liu,
[Choi et al., 2018] Jungwook Choi, Zhuo Wang, Swagath             Tianlong Chen, and Xianglong Liu. Tfmq-dm: Temporal
  Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srini-        feature maintenance quantization for diffusion models. In
  vasan, and Kailash Gopalakrishnan. Pact: Parameterized          Proceedings of the IEEE/CVF Conference on Computer
  clipping activation for quantized neural networks. arXiv        Vision and Pattern Recognition, pages 7362–7371, 2024.
  preprint arXiv:1805.06085, 2018.                             [Keskar et al., 2017] Nitish Shirish Keskar, Dheevatsa
[Cohen et al., 2019] Jeremy Cohen, Elan Rosenfeld, and            Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping
  Zico Kolter. Certified adversarial robustness via random-       Tak Peter Tang. On large-batch training for deep learning:
  ized smoothing. In international conference on machine          Generalization gap and sharp minima. In International
  learning, pages 1310–1320. PMLR, 2019.                          Conference on Learning Representations, 2017.
[Défossez et al., 2021] Alexandre Défossez, Yossi Adi, and   [Kim et al., 2020] Jangho Kim, KiYoon Yoo, and Nojun
  Gabriel Synnaeve.         Differentiable model compres-         Kwak. Position-based scaled gradient for model quanti-
  sion via pseudo quantization noise.         arXiv preprint      zation and pruning. Advances in neural information pro-
  arXiv:2104.09987, 2021.                                         cessing systems, 33:20415–20426, 2020.
[Deng et al., 2024] Jiaxin Deng, Junbiao Pang, and             [Krizhevsky et al., 2009] Alex Krizhevsky, Geoffrey Hinton,
  Baochang Zhang. Asymptotic unbiased sample sam-                 et al. Learning multiple layers of features from tiny im-
  pling to speed up sharpness-aware minimization, 2024.           ages. 2009.
                                                               [Lecuyer et al., 2019] Mathias Lecuyer, Vaggelis Atlidakis,
[Esser et al., 2019] Steven K Esser, Jeffrey L McKinstry,
                                                                  Roxana Geambasu, Daniel Hsu, and Suman Jana. Cer-
   Deepika Bablani, Rathinakumar Appuswamy, and Dhar-
                                                                  tified robustness to adversarial examples with differential
   mendra S Modha. Learned step size quantization. arXiv
                                                                  privacy. In 2019 IEEE symposium on security and privacy
   preprint arXiv:1902.08153, 2019.
                                                                  (SP), pages 656–672. IEEE, 2019.
[Foret et al., 2020] Pierre Foret, Ariel Kleiner, Hossein      [Lee et al., 2021] Junghyup Lee, Dohyung Kim, and Bum-
   Mobahi, and Behnam Neyshabur. Sharpness-aware min-             sub Ham. Network quantization with element-wise gradi-
   imization for efficiently improving generalization. arXiv      ent scaling. In Proceedings of the IEEE/CVF conference
   preprint arXiv:2010.01412, 2020.                               on computer vision and pattern recognition, pages 6448–
[Goodfellow et al., 2014] Ian J Goodfellow, Jonathon              6457, 2021.
  Shlens, and Christian Szegedy. Explaining and harnessing     [Li et al., 2021a] Yuhang Li, Ruihao Gong, Xu Tan, Yang
  adversarial examples. arXiv preprint arXiv:1412.6572,           Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang,
  2014.                                                           and Shi Gu. Brecq: Pushing the limit of post-training
[Han et al., 2015] Song Han, Huizi Mao, and William J             quantization by block reconstruction. arXiv preprint
  Dally. Deep compression: Compressing deep neural net-           arXiv:2102.05426, 2021.
  works with pruning, trained quantization and huffman cod-    [Li et al., 2021b] Yuhang Li, Mingzhu Shen, Jian Ma, Yan
  ing. arXiv preprint arXiv:1510.00149, 2015.                     Ren, Mingxin Zhao, Qi Zhang, Ruihao Gong, Fengwei Yu,
   and Junjie Yan. Mqbench: Towards reproducible and de-          Mobilenetv2: Inverted residuals and linear bottlenecks. In
   ployable model quantization benchmark. arXiv preprint          Proceedings of the IEEE conference on computer vision
   arXiv:2111.03759, 2021.                                        and pattern recognition, pages 4510–4520, 2018.
[Li et al., 2023a] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui   [Shin et al., 2023] Juncheol Shin, Junhyuk So, Sein Park,
   Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and             Seungyeop Kang, Sungjoo Yoo, and Eunhyeok Park.
   Kurt Keutzer. Q-diffusion: Quantizing diffusion models.        Nipq: Noise proxy-based integrated pseudo-quantization.
   In Proceedings of the IEEE/CVF International Conference        In Proceedings of the IEEE/CVF Conference on Computer
   on Computer Vision, pages 17535–17545, 2023.                   Vision and Pattern Recognition, pages 3852–3861, 2023.
[Li et al., 2023b] Zhikai Li, Mengjuan Chen, Junrui Xiao,      [So et al., 2024] Junhyuk So, Jungwon Lee, Daehyun Ahn,
   and Qingyi Gu. Psaq-vit v2: Toward accurate and gen-           Hyungjun Kim, and Eunhyeok Park. Temporal dynamic
   eral data-free quantization for vision transformers. IEEE      quantization for diffusion models. Advances in Neural In-
   Transactions on Neural Networks and Learning Systems,          formation Processing Systems, 36, 2024.
   pages 1–12, 2023.                                           [Sui et al., 2024] Yang Sui, Yanyu Li, Anil Kag, Yerlan Idel-
[Lin et al., 2019] Ji Lin, Chuang Gan, and Song Han. De-          bayev, Junli Cao, Ju Hu, Dhritiman Sagar, Bo Yuan,
   fensive quantization: When efficiency meets robustness.        Sergey Tulyakov, and Jian Ren. Bitsfusion: 1.99 bits
   arXiv preprint arXiv:1904.08444, 2019.                         weight quantization of diffusion model. arXiv preprint
[Madry et al., 2017] Aleksander        Madry,     Aleksandar      arXiv:2406.04333, 2024.
   Makelov, Ludwig Schmidt, Dimitris Tsipras, and              [Szegedy, 2013] C Szegedy. Intriguing properties of neural
   Adrian Vladu. Towards deep learning models resistant to        networks. arXiv preprint arXiv:1312.6199, 2013.
   adversarial attacks. stat, 1050(9), 2017.                   [Wei et al., 2022] Xiuying Wei, Ruihao Gong, Yuhang Li,
[Nagel et al., 2019a] Markus Nagel, Mart van Baalen, Tij-         Xianglong Liu, and Fengwei Yu. Qdrop: Randomly
   men Blankevoort, and Max Welling. Data-free quantiza-          dropping quantization for extremely low-bit post-training
   tion through weight equalization and bias correction. In       quantization. arXiv preprint arXiv:2203.05740, 2022.
   Proceedings of the IEEE/CVF International Conference        [Wen et al., 2020] Yeming Wen, Kevin Luk, Maxime
   on Computer Vision, pages 1325–1334, 2019.                     Gazeau, Guodong Zhang, Harris Chan, and Jimmy Ba. An
[Nagel et al., 2019b] Markus Nagel, Mart van Baalen, Tij-         empirical study of stochastic gradient descent with struc-
   men Blankevoort, and Max Welling. Data-free quantiza-          tured covariance noise. In International Conference on
   tion through weight equalization and bias correction. In       Artificial Intelligence and Statistics, 2020.
   Proceedings of the IEEE/CVF International Conference        [Xie et al., 2020] Cihang Xie, Mingxing Tan, Boqing Gong,
   on Computer Vision, pages 1325–1334, 2019.                     Jiang Wang, Alan L Yuille, and Quoc V Le. Adversarial
[Nagel et al., 2020] Markus Nagel, Rana Ali Amjad, Mart           examples improve image recognition. In Proceedings of
   Van Baalen, Christos Louizos, and Tijmen Blankevoort.          the IEEE/CVF conference on computer vision and pattern
   Up or down? adaptive rounding for post-training quanti-        recognition, pages 819–828, 2020.
   zation. In International Conference on Machine Learning,    [Xu et al., 2023] Sheng Xu, Yanjing Li, Teli Ma, Mingbao
   pages 7197–7206. PMLR, 2020.                                   Lin, Hao Dong, Baochang Zhang, Peng Gao, and Jinhu
[Nagel et al., 2022] Markus Nagel, Marios Fournarakis, Yel-       Lu. Resilient binary neural network. In Proceedings of
   ysei Bondarenko, and Tijmen Blankevoort. Overcoming            the AAAI Conference on Artificial Intelligence, volume 37,
   oscillations in quantization-aware training. In Interna-       pages 10620–10628, 2023.
   tional Conference on Machine Learning, pages 16318–         [Zhang et al., 2021] Chiyuan Zhang, Samy Bengio, Moritz
   16330. PMLR, 2022.
                                                                  Hardt, Benjamin Recht, and Oriol Vinyals. Understand-
[Pang et al., 2025] Junbiao Pang, Tianyang Cai, Baochang          ing deep learning (still) requires rethinking generalization.
   Zhang, and Jiaqi Wu. In-distribution consistency regular-      Communications of the ACM, 64(3):107–115, 2021.
   ization improves the generalization of quantization-aware   [Zhu et al., 2019] Shilin Zhu, Xin Dong, and Hao Su. Binary
   training, 2025.
                                                                  ensemble neural network: More bits per network or more
[Paszke et al., 2019] Adam Paszke, Sam Gross, Francisco           networks per bit? In Proceedings of the IEEE/CVF con-
   Massa, Adam Lerer, James Bradbury, Gregory Chanan,             ference on computer vision and pattern recognition, pages
   Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca           4923–4932, 2019.
   Antiga, et al. Pytorch: An imperative style, high-          [Zoph and Le, 2016] Barret Zoph and Quoc V Le. Neural
   performance deep learning library. Advances in neural in-
                                                                  architecture search with reinforcement learning. arXiv
   formation processing systems, 32, 2019.
                                                                  preprint arXiv:1611.01578, 2016.
[Qian and Wegman, 2018] Haifeng Qian and Mark N Weg-
   man. L2-nonexpansive neural networks. arXiv preprint
   arXiv:1802.07896, 2018.
[Sandler et al., 2018] Mark Sandler, Andrew Howard, Men-
   glong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen.
```
