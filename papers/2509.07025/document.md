1 BIT IS ALL WE NEED: BINARY NORMALIZED NEURAL NETWORKS

A PREPRINT

Eduardo L. L. Cabral
Mauá Institute of Technology
São Caetano do Sul, São Paulo, SP, Brazil
Nuclear and Energy Research Institute
São Paulo, SP, Brazil
elcabral@maua.br

Paulo Pirozelli
Mauá Institute of Technology
São Caetano do Sul, São Paulo, SP, Brazil
paulopirozelli@gmail.com

Larissa Driemeier
Department of Mechatronics and Mechanical Systems Engineering
Polytechnic School – University of São Paulo
São Paulo, SP, Brazil
driemeie@usp.br

September 10, 2025

ABSTRACT

The increasing size of large neural network models, specifically language models and foundational image models, poses deployment challenges, prompting efforts to reduce memory requirements and enhance computational efficiency. These efforts are critical to ensure practical deployment and effective utilization of these models across various applications. In this work, a novel type of neural network layers and models is developed that uses only single-bit parameters. In this novel type of models all parameters of all layers, including kernel weights and biases, only have values equal to zero or one. This novel type of models uses layers named as binary normalized layer. These binary normalized layers can be of any type, such as fully connected, convolutional, attention, etc., and they consist of slight variations of the corresponding conventional layers. To show the effectiveness of the binary normalized layers, two different models are configured to solve a multiclass image classification problem and a language decoder to predict the next token of a sequence. The model to solve the image classification has convolutional and fully connected layers, and the language model is composed of transformer blocks with multi-head attention. The results show that models with binary normalized layers present almost the same results obtained by equivalent models with real 32-bit parameters. The binary normalized layers allow to develop models that use 32 times less memory than current models and have equivalent performance. Besides, the binary normalized layers can be easily implemented on current computers using 1-bit arrays, and do not require the development of dedicated electronic hardware. This novel type of layers opens a new era for large neural network models with reduced memory requirements that can be deployed using simple and cheap hardware, such as mobile devices or only cpus.

Keywords Neural networks · binary parameters · binary normalized layers

1 Introduction

Recent advances in machine learning techniques and hardware have enabled extraordinary performance in a wide range of applications — from traditional tasks such as pattern recognition and natural language processing to more complex domains, including autonomous control systems and the discovery of new materials. These developments have been

arXiv:2509.07025v1 [cs.LG] 7 Sep 2025

A PREPRINT

1 BIT IS ALL WE NEED

made possible thanks to improved neural network architectures, increased computational power, and the availability of large datasets Goodfellow et al. [2016].

However, despite such impressive progress, current artificial intelligence models still face serious limitations when applied to embedded systems. Most state-of-the-art AI solutions rely heavily on cloud computing infrastructure and high-performance specialized hardware. Deep neural networks, such as those used for image classification, often require billions of floating point operations to process a single input sample Henzinger et al. [2021]. The increasing size of large-scale AI models, especially language models and foundational image models, introduces significant deployment challenges. These models require substantial computational resources, energy, and memory requirements that are difficult to satisfy outside data centers.

This dependency makes it impractical to implement these models in systems with limited computational resources, especially in contexts where connectivity is restricted or nonexistent. Applications in isolated environments, such as underwater, underground, aerospace, or agricultural systems, face severe communication challenges, including physical interference, high latency, and limited bandwidth. As discussed in Plastiras et al. [2018], local data processing becomes an evolutionary necessity to ensure the autonomy and responsiveness of such systems, particularly where rapid decision making is required under resource constraints.

To address these constraints, quantization has emerged as a key technique for optimizing neural networks in resource-limited environments Henzinger et al. [2021]. Instead of relying on high-precision floating-point arithmetic (e.g., 32-bit), quantization uses lower-bit integer formats. It typically operates in the 2 to 8 bit range, which is widely adopted in industry with minimal impact on accuracy Henzinger et al. [2021]. This approach enables significantly reduced memory usage and bandwidth requirements, with compression ratios ranging from 35× to 49×, as shown by Han et al. [2016]. It also provides substantial speed-ups, achieving up to 3 times faster execution on standard CPUs and up to 10 times on specialized fixed point hardware such as Qualcomm DSPs with HVX support Jacob et al. [2018]. Furthermore, energy efficiency is greatly improved, which is especially critical for mobile and edge devices Krishnamoorthi [2018], Banner et al. [2019].

However, quantization presents challenges, particularly in preserving accuracy due to the non-differentiable nature of quantization functions and the sensitivity to ranges of varying values in weights and activations Hubara et al. [2017]. To address these issues, several techniques have been developed.

Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) are two main strategies to reduce the precision of parameters of neural network models. PTQ applies quantization after a model has been fully trained in high precision (typically 32-bit floating point), aiming to reduce model size and inference latency without retraining. In contrast, QAT simulates quantization effects during training, allowing the model to learn and adapt to the information loss introduced by low-precision representations. As a result, QAT generally achieves significantly higher accuracy than PTQ, particularly at lower bit-widths.

Several key studies have shaped the development of both PTQ and QAT. Jacob et al. [2018] presented a foundational quantization scheme that has been widely adopted in TensorFlow Lite, offering practical guidelines for deploying low-precision models in real-world applications. Krishnamoorthi [2018] provided a comprehensive white paper that systematically covers the principles and implementation details of both PTQ and QAT, serving as a central reference for researchers. For low-bit PTQ, Banner et al. [2019] proposed ACIQ (Analytical Clipping for Integer Quantization), a method for minimizing quantization error without retraining, particularly effective in the 4-bit setting.

Regarding QAT specifically, Hubara et al. [2017] introduced a seminal approach to training Quantized Neural Networks (QNNs) with low-precision weights and activations using the Straight-Through Estimator (STE) to handle non-differentiability. As proposed in Jacob et al. [2018], weights and biases are stored in 32-bit floating-point format to allow precise updates, but are quantized during the forward pass to simulate low-precision inference. Backpropagation still occurs using full-precision gradients, and the STE is employed to handle the non-differentiability of quantization. This dual approach ensures that small gradient updates are not lost, which would happen if parameters were permanently quantized during training. Once training is complete, only the low-precision parameters are retained for efficient inference. In van den Oord et al. [2018] the authors discussed the Vector Quantised-Variational AutoEncoder (VQ-VAE), a generative model that learns discrete latent representations. In Choi et al. [2019] the authors proposed PACT (Parameterized Clipping Activation), a method that improves activation quantization by learning optimal clipping thresholds during training. The field has also advanced with techniques such as SAWB (Statistical Aware Weight Binning) for weight quantization, which selects quantization ranges based on the weight distribution to reduce accuracy loss in low-precision networks. Zhuang et al. [2018] proposed training strategies for quantized networks, including two-stage optimization, progressive quantization, and the use of a full-precision teacher model to guide learning and improve final performance.

1 BIT IS ALL WE NEED

A PREPRINT

Recently, Cabral and Driemeier [2025] explored the trade-offs involved in using low-bit representations for network weights. Their study demonstrates that 2.32-bit weights—corresponding to five discrete levels—can provide a favorable balance between memory efficiency and model accuracy. They also observe that low-resolution models with fewer parameters may need more training epochs to reach the accuracy of 32-bit models, while larger models can achieve similar performance within typical training configurations.

Recent progress includes BitNet, a transformer architecture tailored for large language models using binary weights and low-precision activations while preserving full-precision states for optimizers and gradients. Its variant, BitNet 1.58, introduces ternary weights (−1, 0, +1), offering comparable performance to 16-bit models with reduced memory footprint, lower latency, and improved energy efficiency.

Particularly, 1-bit quantization refers to the technique of reducing the numerical precision of neural network weights and/or activations to just one bit, typically representing values as +1 or -1 Hubara et al. [2017]. This extreme form of quantization, known as Binary Neural Networks (BNNs), enables the replacement of costly floating-point operations with highly efficient bitwise operations such as XNOR and bit-counting. The primary goal is to drastically reduce memory usage, energy consumption, and computational complexity, making deep learning models suitable for deployment on low-power, resource-constrained devices. In theory, 1-bit quantization can achieve up to 32× compression over 32-bit floating-point parameters and similarly large energy savings.

Early works by Hwang and Sung [2014], Courbariaux et al. [2015] demonstrated the feasibility of training deep models with binary weights. Hubara et al. [2016] extended this to both weights and activations, training BNNs on datasets like MNIST, CIFAR-10, SVHN, and even ImageNet. Later, Rastegari et al. [2016] introduced XNOR-Net, which used a scaling factor to improve the performance of binarized layers, achieving competitive top-1 accuracy on ImageNet. Despite these advances, 1-bit models still suffer from accuracy degradation on complex tasks. For instance, BNN proposed by Hubara et al. [2016] achieved 41.8% top-1 accuracy on AlexNet with ImageNet, while XNOR-Net reached 44.2%.

In this work, we propose a novel class of neural network models built entirely with single-bit parameters, using binary normalized layers. Unlike traditional models that rely on 32-bit floating-point precision, our approach constrains all layer parameters—including kernel weights and biases—to a single bit of resolution. The binary normalized layer concept is versatile and can be applied across various architectures such as fully connected, convolutional, and attention layers. To demonstrate their effectiveness, we apply these layers to two distinct problems: multiclass image classification using a convolutional binary model, and a language decoder for next-token prediction in language sequences using a binary transformer model. Our results show that these binary models achieve performance comparable to their full-precision 32-bit counterparts, without exhibiting common training instabilities associated with low-resolution parameter networks. This significant memory reduction, of up to 32 times less than conventional models, combined with their straightforward implementation on standard hardware using 1-bit arrays, opens the door to deploying large-scale neural networks on resource-limited platforms such as mobile devices and CPUs. Moreover, the reduced memory footprint allows scaling to larger models, making advanced AI feasible on embedded systems.

Section 2 of this paper outlines the binary normalized layers, Sections 3 details the convolutional model and the image dataset used to train this model for a multiclass classification problem, Section 4 shows the language models and the dataset used to train this model to predict the next token. The results of the both models are compared with results obtained with the corresponding conventional models with 32-bit parameters. Finally, Section 5 summarizes the conclusions.

## 2 Binary normalized layers

In our binary normalized layers, including the kernel and bias, each parameter exists in two forms simultaneously during training: a full-precision 32-bit floating-point value ($p$) used for gradient updates, and its binarized counterpart ($p_b$) used for forward computations. The quantization process is straightforward and is performed as follows,

$$
p_b = \begin{cases} 1, & \text{if } p > p_{mean} \\ 0, & \text{if } p \le p_{mean} \end{cases} \qquad (1)
$$

where $p_{mean}$ is the mean value of the parameters of the layer.

The 32-bit parameters ($p$) are essential because gradient updates during backpropagation are typically very small ($10^{-4}$ to $10^{-2}$), and would be completely lost if parameters were permanently binarized during training. During the forward pass, we use the binary parameters ($p_b$) to compute activations, but during backpropagation the full-precision parameters ($p$) are used to calculate the gradients.

1 BIT IS ALL WE NEED

A PREPRINT

This dual representation approach, inspired by VQ-VAE van den Oord et al. [2018] and related to QAT principles Jacob et al. [2018], allows effective training while ultimately delivering the benefits of 1-bit inference. After training completes, we discard the 32-bit parameters and retain only the 1-bit $p_b$ parameters for deployment.

Obviously, training the model with the help of 32-bit parameters requires a large amount of memory, the same as that required in conventional models. But the final trained model only has 1-bit parameters, thus demanding a much smaller amount of memory. Note that to avoid using 32-bit parameters during training, current neural network training methods based on gradient descent could not be used, and a new training method for neural networks with 1-bit parameters would have to be developed.

The fundamental operation in any neural network layer consists of multiplying input data by the layer's weights (kernel), adding biases, and applying an activation function. This linear transformation followed by nonlinear activation enables the network to learn complex patterns. However, when weights are constrained to only zeros and ones, this transformation exhibits two critical limitations. First, it disproportionately amplifies large positive and negative input values while suppressing small ones, making it inadequate for extracting complex features from the input data. Second, the binary nature significantly intensifies both vanishing and exploding gradient problems during backpropagation.

One effective way to address the challenges posed by using 1-bit parameters is to normalize the output of the linear transformation before applying the activation function. While this strategy is commonly employed in modern architectures through normalization layers, it becomes particularly crucial in the context of binary-weighted networks. In these models, normalization not only stabilizes training but also compensates for the severe limitations introduced by extreme quantization, enabling effective learning despite the low resolution of the parameters.

The motivation for this normalization shares similarities with its role in conventional networks but takes on greater importance in binary models due to their restricted representational capacity and sensitivity to input scale. Specifically:

* **Equalizing feature influence:** when input features vary in scale, the limited expressiveness of low-precision weights prevents the model from compensating for dominant features. Normalization ensures that all inputs contribute more equally to learning.
* **Improving convergence stability:** scale discrepancies in the input can lead to unstable or inefficient optimization. Normalization mitigates this by aligning feature scales, facilitating smoother convergence.
* **Controlling gradient magnitudes:** quantized parameters make gradient updates more sensitive to input scale. Normalizing inputs helps keep gradients within a stable range, avoiding saturation or stagnation during training.
* **Avoiding biased learning:** when features have unequal numerical ranges, the model may overemphasize those with larger absolute values. Normalization enforces fairer treatment of all features, improving generalization.
* **Mitigating vanishing/exploding gradients:** limited-precision models are more prone to unstable gradient propagation. Normalization helps maintain consistent signal flow across layers, especially in deeper networks.

Four types of normalized binary layers are implemented and used in different models: fully connected, convolutional, attention and embedding layers. These layers are described in the sections that follow.

## 2.1 Binary normalized fully connected layer (BNFCL)

Algorithm 1 illustrates the forward propagation process in a binary normalized fully connected layer (BNFCL). In Algorithm 1, $Quant$ denotes the function that performs weight quantization (defined by equation 1); $NoGradient$ is a placeholder function that prevents gradient calculation for its argument during model training; $Normalize$ is the normalization function; $Activation$ represents the chosen activation function for the layer; and $trainable$ is a flag to indicate if the layer is in training or predicting. Note that the $Normalize$ function normalizes the features of each example so that it has zero mean and unit standard deviation.

In Algorithm 1, $W$ and $W_q$ represent respectively the 32-bit float and binary kernel weights, while $b$ and $b_q$ are the corresponding 32-bit float and binary bias vectors. The arrays $W$ and $W_q$ have shape $(n_x, n_{units})$, where $n_x$ is the number of input features and $n_{units}$ is the number of neurons in the layer. The bias vectors $b$ and $b_q$ are one-dimensional, each containing $n_{units}$ elements.

During both training and inference, the binary weights are used to compute the layer activations. However, during training, when $trainable == True$, the gradients are computed and applied to the 32-bit floating-point weight matrix $W$ and bias vector $b$. These high-precision values are retained and updated throughout training, ensuring that no information is lost during the optimization process. This approach enables parameter updates using full-precision values while still performing forward passes with quantized weights. This scheme is adapted from the method proposed in Alcorn [2023], and is similar in spirit to Quantization Aware Training (QAT) Jacob et al. [2018].

1 BIT IS ALL WE NEED

A PREPRINT

**Algorithm 1** Forward propagation calculation process in a binary normalized fully connected layer (BNFCL)

**Require:** Input $x$, weights $W$, bias $b$, flag *trainable*, activation function
**Ensure:** Activations $a$
1: **if** *trainable* **then**
2: &nbsp;&nbsp;&nbsp;&nbsp;Quantize kernel for training: $W_q = W + \text{NoGradient}(\text{Quant}(W) - W)$
3: &nbsp;&nbsp;&nbsp;&nbsp;Quantize bias for training: $b_q = b + \text{NoGradient}(\text{Quant}(b) - b)$
4: **else**
5: &nbsp;&nbsp;&nbsp;&nbsp;Quantize kernel for inference: $W_q = \text{Quant}(W)$
6: &nbsp;&nbsp;&nbsp;&nbsp;Quantize bias for inference: $b_q = \text{Quant}(b)$
7: **end if**
8: Apply linear transformation: $z = W_q x + b_q$
9: Normalize features of each example: $z = \text{Normalize}(z)$
10: Calculate activations: $a = \text{Activation}(z)$
11: **return** $a$

It is important to observe that after training only the quantized 1-bit parameters ($W_q$ and $b_q$) are needed for inference and the calculations performed in the BNFC layer are modified according to Algorithm 2.

**Algorithm 2** Forward propagation calculation process in a binary normalized fully connected layer (BNFCL) after training

**Require:** Input $x$, binary eights $W_q$, binary bias $b_q$, activation function
**Ensure:** Activations $a$
1: Apply linear transformation: $z = W_q x + b_q$
2: Normalize features of each example: $z = \text{Normalize}(z)$
3: Calculate activations: $a = \text{Activation}(z)$
4: **return** $a$

## 2.2 Binary normalized convolutional layer (BNCVL)

The only difference between the binary normalized convolutional layer (BNCVL) and the binary normalized fully connected layer (BNFCL) is that a convolution operation is used between the filters (kernel with binary weights) and the input tensor of the layer, rather than a simple matrix multiplication. Equation (2) performs a convolution operation in calculating activations in a BNCV layer.

$$
z = \text{Conv}(W_q x) + b_q \qquad (2)
$$

where $\text{Conv}(W_q x)$ performs the convolution of $x$ by $W_q$. In this case $W_q$ is the binary kernel parameters of the layer, which is a four-dimensional array with dimensions ($n_H, n_W, n_C, n_F$), where $n_H, n_W$ and $n_C$ are respectively the height, the width and the number of channels of the input data, and $n_F$ is the number of filters used in the convolution layer, and $b_q$ is the binary bias vector with $n_F$ elements. The forward propagation process in a binary normalized convolutional layer (BNCVL) is defined in Algorithm 3. It should be noted that in the BNCVL, equation 2 replaces the linear transformation in Algorithm 1.

It is important to observe that after training only the quantized 1-bit parameters are needed for inference and the calculations performed in the BNCV layer are modified similarly to what it is done for the BNFC layer in Algorithm 2.

## 2.3 Binary embedding layer (BEMBL)

Algorithm 4 presents the calculations performed in a binary token and position embedding layer (BEMBL). In Algorithm 4, *seq* is the input of the layer which consists of a sequence of tokens with maximum length equal to *max_len*, *emb_dim* is the dimension of the embedding vectors for each token, and *vocab_size* is the number of tokens in the dictionary.

In Algorithm 4 $\text{LinearSeq}(\text{min\_value}, \text{max\_value})$ denotes a function that creates a vector varying linearly from the minimum value (min_value) to the maximum value minus one (max_value-1); $\text{BNFCL}(\text{units})$ is a binary normalized fully connected layer with number of units equal to units, as described in Algorithm 1; and $\text{ToCategorical}(s, n\_cats)$ is a function that performs one-hot codification of a sequence $s$ with $n\_cats$ categories. Note that the activation functions of both BNFCL layers are linear.

1 BIT IS ALL WE NEED

A PREPRINT

**Algorithm 3** Forward propagation calculation process in a binary normalized convolutional layer (BNCVL)

**Require:** Input $x$, weights $W$, bias $b$, flag *trainable*, activation function
**Ensure:** Activations $a$
1: **if** *trainable* **then**
2: &nbsp;&nbsp;&nbsp;&nbsp;Quantize kernel for training: $W_q = W + \text{NoGradient}(\text{Quant}(W) - W)$
3: &nbsp;&nbsp;&nbsp;&nbsp;Quantize bias for training: $b_q = b + \text{NoGradient}(\text{Quant}(b) - b)$
4: **else**
5: &nbsp;&nbsp;&nbsp;&nbsp;Quantize kernel for inference: $W_q = \text{Quant}(W)$
6: &nbsp;&nbsp;&nbsp;&nbsp;Quantize bias for inference: $b_q = \text{Quant}(b)$
7: **end if** $\text{alg:03}$
8: Calculate convolution and add the bias: $z = \text{Conv}(W_q x) + b_q$
9: Normalize features of each example: $z = \text{Normalize}(z)$
10: Calculate activations: $a = \text{Activation}(z)$
11: **return** $a$

**Algorithm 4** Forward propagation calculation process in a binary normalized embedding layer (BEMB)

**Require:** Input $seq$, sequence maximum length $max\_len$, embedding dimension $emb\_dim$, vocabulary size $vocab\_size$
**Ensure:** Token and position embeddings $tk\_pos\_emb$
1: Create linear position vector varying from 0 to $max\_len$:
2: &nbsp;&nbsp;&nbsp;&nbsp;$pos = \text{LinearSeq}(\text{min\_value} = 0, \text{max\_value} = max\_len)$
3: One-hot codification of position vector: $one\_hot\_pos = \text{ToCategorical}(pos, max\_len)$
4: Embedding codification of positions: $pos\_emb = \text{BNFCL}(units = emb\_dim)(one\_hot\_pos)$
5: One-hot codification of token sequence: $one\_hot\_tk = \text{ToCategorical}(seq, vocab\_size)$
6: Embedding codification of token sequence:
7: &nbsp;&nbsp;&nbsp;&nbsp;$tk\_emb = \text{BNFCL}(units = emb\_dim)(one\_hot\_tk)$
8: Add token and position embeddings: $tk\_pos\_emb = tk\_emb + pos\_emb$
9: **return** $tk\_pos\_emb$

## 2.4 Binary transformer block with binary attention layer (BTFB)

A transformer block with its attention mechanism uses only embedding layers, fully connected layers, and normalization layers. The fully connected layers of the binary transformer block (BTFB) are the binary normalized layers (BNFCL) presented in Algorithm 1. The embedding layer is the binary embedding layer (BEMBL) presented in Algorithm 4 and the normalization layer performs a simple normalization, i.e., it returns data with zero mean and unit standard deviation.

Algorithm 5 presents the forward propagation process in a binary transformer block (BTFB). The required inputs of Algorithm 5, are the input token sequence ($seq$), the embeddings dimension ($emb\_dim$), the number of heads in the attention layer ($num\_heads$), and the number of units of the first BNFC layer ($ff\_dim$).

**Algorithm 5** Forward propagation calculation process in a binary transformer block (BTFB)

**Require:** Input $seq$, embedding dimension $emb\_dim$, number of heads in the attention layer $num\_heads$, number of units of the first FCL layer $ff\_dim$
**Ensure:** Transformer output $output$
1: Apply binary normalized attention mechanism:
2: &nbsp;&nbsp;&nbsp;&nbsp;$attention\_output = \text{BNFCL}(emb\_dim = emb\_dim, num\_heads = num\_heads)(seq, seq, seq)$
3: Add and normalize: $add\_norm = \text{Normalize}(seq + attention\_output)$
4: Apply binary normalized fully connected layers (BNFCL):
5: &nbsp;&nbsp;&nbsp;&nbsp;$ffn\_output = \text{BNFCL}(units = ff\_dim, activation = 'gelu')(add\_norm)$
6: &nbsp;&nbsp;&nbsp;&nbsp;$ffn\_output = \text{BNFCL}(units = emb\_dim)(ffn\_output)$
7: Add and normalize again: $output = \text{Normalize}(add\_norm + ffn\_output)$
8: **return** $output$

In Algorithm 5, BNFCL($units$, $activation$) represents the binary normalized fully connected layer, whose calculation process is shown in Algorithm 1; note that the $gelu$ activation function is used in the first BNFC layer; $\text{Normalize}()$ is the function that normalizes the features of each example so that it has zero mean and unit standard deviation.

1 BIT IS ALL WE NEED

A PREPRINT

deviation; and `BATL(emb_dim, num heads)` represents the binary multi-head attention layer. The calculation process of this attention layer is presented in Algorithm 6.

Algorithm 6 presents the forward propagation process in a binary multi-head attention layer (BATL). The required inputs are the input token sequences `query`, `key` and `value`, the causal mask (`mask`), the embeddings dimension (`emb_dim`), and the number of heads in the attention layer (`num heads`).

The functions used in Algorithm 6 are: `LengthOfSequence()` is a function that retrieves the length of a sequence; `Reshape()` is a function that reallocates the elements of a tensor according to the provided shape list; `Permute()` denotes a function that swaps the axes of a tensor based on the provided order list; `Matmul()` is a function that performs tensor multiplication according to the linear algebra rules; `Where()` is the standard where function that operates conditions along all elements of a tensors; and `Softmax()` is the standard softmax function. All the other functions and terms used in Algorithm 6 have been defined previously.

**Algorithm 6** Forward propagation calculation process in a binary multi-head attention layer (BATL)
***
**Require:** Input `query`, `key` and `value`, causal mask `mask`, embedding dimension `emb_dim`, number of heads `num heads`
**Ensure:** Final linear projection `projection`
1: Calculate number of keys: `num_key = emb_dim // num heads`
2: Apply linear projections to get Q, K, V
3: $Q = \text{BNFLC}(\text{units} = \text{emb\_dim})(\text{query})$
4: $K = \text{BNFLC}(\text{units} = \text{emb\_dim})(\text{key})$
5: $V = \text{BNFLC}(\text{units} = \text{emb\_dim})(\text{value})$
6: Get sequence length from query: $\text{seq\_len} = \text{LengthOfSequence}(\text{query})$
7: Split each tensor into `num heads` to support multi-head attention:
8: $Q = \text{Reshape}(Q, \text{shape} = [-1, \text{seq\_len}, \text{num\_heads}, \text{num\_key}])$
9: $K = \text{Reshape}(K, \text{shape} = [-1, \text{seq\_len}, \text{num\_heads}, \text{num\_key}])$
10: $V = \text{Reshape}(V, \text{shape} = [-1, \text{seq\_len}, \text{num\_heads}, \text{num\_key}])$
11: Permute axis of Q, K, V to support multi-head attention
12: $Q = \text{Permute}(Q, \text{order} = [0, 2, 1, 3])$
13: $K = \text{Permute}(K, \text{order} = [0, 2, 1, 3])$
14: $V = \text{Permute}(V, \text{order} = [0, 2, 1, 3])$
15: Compute scaled dot-product attention scores:
16: $\text{attention\_scores} = \text{Matmul}(Q, \text{Permute}(K, \text{order} = [0, 1, 3, 2])) / \text{Sqrt}(\text{num\_key})$
17: Apply causal mask: $\text{scale\_dot} = \text{Where}(\text{mask} == 0, -1.0e-10, \text{scale\_dot})$
18: Apply softmax to get attention probabilities: $\text{attn\_prob} = \text{Softmax}(\text{scale\_dot}, \text{axis} = -1)$
19: Calculate attention: $A = \text{Matmul}(\text{attn\_prob}, V)$
20: Reshape attention back to the original dimension:
21: $A = \text{Permute}(A, \text{order} = [0, 2, 1, 3])$
22: $A = \text{Reshape}(A, \text{shape} = [-1, \text{seq\_len}, \text{num\_heads} * \text{num\_key}])$
23: Apply final linear projection: $\text{projection} = \text{BNFLC}(\text{units} = \text{emb\_dim})(A)$
24: **return** `projection`

## 3 Image classification problem

For the image multiclass classification problem, a binary convolutional model is configured and the Food-101 dataset Bossard et al. [2014] is used. The Food-101 dataset has a total of 101,000 images in varying resolutions with 101 categories of foods. This dataset is used for identification of types of food in a dish. The data is divided in two sets: the training data with 75,750 images and the validation data with 25,250 images.

### 3.1 Configuration of the models with convolutional layers

Algorithm 7 outlines the binary convolutional model (BCVNN) for the image classification task. The inputs for the model are an image (`image`) and the filter dimension used in the convolutional layers `f`. The output of the model are the 101 class probabilities calculated for the input image (`probs`). In Algorithm 7 the following functions are used: `BNCVL()` represents the binary normalized convolutional layer presented in 3; `MAXPOOL2D()` represents a max-pooling layer; `BNFCL()` represents the binary normalized fully connected layer presented in Algorithm 1; and `GLOBALAVG()` is a standard global average pooling layer that averages a three-axis tensor across the first two dimensions resulting a

1 BIT IS ALL WE NEED

A PREPRINT

tensor with only one axis. All convolutional layers use *relu* activation function, a stride of 1, and padding to maintain the width and height of the tensors. All max-pooling layers use $2 \times 2$ windows and stride equal to 2. The first and second binary normalized fully connected layer use *relu* activation, and the output layer uses softmax activation.

**Algorithm 7** Binary convolutional model used for the image classification problem (BCVNN)
___
**Require:** Input *image*, filter dimension *f*
**Ensure:** Classes probabilities *prob*
1: First block of convolutional layers
2:     $a1$ = BNCVL(units = 32, ($f, f$), activation = 'relu', padding = 'same')(*image*)
3:     $a1$ = BNCVL(units = 32, ($f, f$), activation = 'relu', padding = 'same')(*a1*)
4:     $a1$ = MAXPOOL2D(window = (2, 2), stride = (2, 2))(*a1*)
5: Second block of convolutional layers
6:     $a2$ = BNCVL(units = 64, ($f, f$), activation = 'relu', padding = 'same')(*a1*)
7:     $a2$ = BNCVL(units = 64, ($f, f$), activation = 'relu', padding = 'same')(*a2*)
8:     $a2$ = MAXPOOL2D(window = (2, 2), stride = (2, 2))(*a2*)
9: Third block of convolutional layers
10:    $a3$ = BNCVL(units = 64, ($f, f$), activation = 'relu', padding = 'same')(*a2*)
11:    $a3$ = BNCVL(units = 64, ($f, f$), activation = 'relu', padding = 'same')(*a3*)
12:    $a3$ = MAXPOOL2D(window = (2, 2), stride = (2, 2))(*a3*)
13: Fourth block of convolutional layers
14:    $a4$ = BNCVL(units = 128, ($f, f$), activation = 'relu', padding = 'same')(*a3*)
15:    $a4$ = BNCVL(units = 128, ($f, f$), activation = 'relu', padding = 'same')(*a4*)
16:    $a4$ = MAXPOOL2D(window = (2, 2), stride = (2, 2))(*a4*)
17: Fifth block of convolutional layers
18:    $a5$ = BNCVL(units = 256, ($f, f$), activation = 'relu', padding = 'same')(*a4*)
19:    $a5$ = BNCVL(units = 256, ($f, f$), activation = 'relu', padding = 'same')(*a5*)
20:    $a6$ = GLOBALAVG()(*a5*)
21: Classification layers
22:    $a7$ = BNFCL(units = 256, activation = 'relu')(*a6*)
23:    $a8$ = BNFCL(units = 256, activation = 'relu')(*a7*)
24:    $prob$ = BNFCL(units = 101, activation = 'softmax')(*a8*)
25: **return** *prob*

## 3.2 Convolutional models training

To verify whether the number of parameters of the models influences training stability and performance of the binary models, two models with different filter dimensions are configured and trained: $3 \times 3$ and $5 \times 5$. The model with $3 \times 3$ filters has 5,132,165 parameters and the model with $5 \times 5$ filters has 13,505,925 parameters.

To verify if the binary models are effective, two models with float 32-bit parameters ("standard" models) with the same configurations of the binary models are also configured and trained. In these "standard" models dropout layers are introduced after the first and second fully connected layers with dropout rates of 0.4 and 0.3 respectively. Dropout is necessary in the standard models to prevent excess overfitting.

In all models, connection weights and biases are initialized using the standard methods: Glorot Uniform for weights and zeros for biases. No regularization methods or parameter constraints are applied in the binary models and only dropout are used in the standard models. Table 1 presents the hyperparameters used for training the convolutional models.

It is observed that a considerable number of training epochs are needed for the cost function to completely converge during training.

## 3.3 Results obtained with the models with convolutional layers

In Figure 1, the training results are displayed for the convolutional models. The results from the standard models with 32-bit parameters are included as a benchmark for the desired performance. It is important to note that multiple training tests were conducted for all models, and all results are very similar. These results are summarized in Table 2 that presents the best results obtained during training for each model.

Analyzing the training results of the convolutional models shown in Figures 1 and Table 2, the following observations can be made:

1 BIT IS ALL WE NEED

A PREPRINT

Table 1: Hyperparameters used for training the convolutional models.

<table><thead><tr><td>Hyperparameter</td><td>Value</td></tr></thead><tbody><tr><td>Image resolution</td><td>256 × 256 × 3</td></tr><tr><td>Cost function</td><td>Categorical cross entropy</td></tr><tr><td>Metrics</td><td>Accuracy</td></tr><tr><td>Optimization method</td><td>Adam</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Learning rate schedule</td><td>Warmup and decay</td></tr><tr><td>Maximum learning rate</td><td>0.0001</td></tr><tr><td>Warmup steps</td><td>20</td></tr><tr><td>Decay steps</td><td>1100</td></tr><tr><td>Number of epochs</td><td>1000</td></tr></tbody></table>

![](images/task_14a83adeb726_page_8_pic_0.png)

Figure 1: Training results of image classification problem with the convolutional models.

* The binary models are capable to train without any kind of instability and their performance is almost equal to the standard models;
* The standard models learn more rapidly than the binary models, i.e., they need fewer epochs for training;
* The standard models present strong overfitting while the binary models do show overfitting;
* The accuracies for the validation data of the standard models are slightly better than the ones of the binary models;
* The binary model with 5x5 filters presents better performance than the 3x3 filters binary model;
* The results of the binary models are very good considering that they have only 1-bit parameters.

1 BIT IS ALL WE NEED

A PREPRINT

Table 2: Summary of the results of the image classification problem with the convolutional models.

<table><thead><tr><th>Model</th><th>Training loss</th><th>Validation loss</th><th>Training accuracy</th><th>Validation accuracy</th></tr></thead><tbody><tr><td>Standard model with 3 × 3 filter</td><td>0.0369</td><td>1.37</td><td>0.989</td><td>0.703</td></tr><tr><td>Binary model with 3 × 3 filter</td><td>1.46</td><td>1.55</td><td>0.670</td><td>0.637</td></tr><tr><td>Standard model with 5 × 5 filter</td><td>0.0495</td><td>1.48</td><td>0.986</td><td>0.679</td></tr><tr><td>Binary model with 5 × 5 filter</td><td>0.836</td><td>1.35</td><td>0.834</td><td>0.686</td></tr></tbody></table>

It is important to observe that the binary normalization layers are effective to solve the problems of training instabilities and low accuracy of models with binary parameters. According to the study performed by Cabral and Driemeier [2025], that analyzed the impact of low-resolution parameters on the performance of neural networks, models with binary parameters are not able to train effectively.

# 4 Language decoder problem

For the language decoder problem, a binary transformer model is configured and the WikiText-103-raw dataset is used. This dataset was created by Salesforce Research Merity et al. [2016]. The WikiText-103-raw dataset is primarily sourced from English Wikipedia. Specifically, it was created from high-quality Wikipedia articles to provide clean and representative data for training language models. It includes 25,000 carefully selected Wikipedia articles, containing around 103 million words. The "raw" version preserves the original punctuation and basic formatting, unlike the tokenized version. The dataset was pre-processed in the form of sentences performing 782,208 examples. The data was divided into training dataset with 95% of the examples and validation dataset with the rest 5%.

The text is tokenized using the WordPiece Tokenizer Song et al. [2021] which uses a sub-word strategy. Its vocabulary size is 30,522, and any token not appearing in the vocabulary is replaced by [UNK] ("unknown").

## 4.1 Configuration and training the language decoder

Algorithm 8 outlines the binary language decoder model (BLM). The inputs for the model are the sequence of tokens (*seq*), the maximum sequence length *max_len*, the embedding dimension *emb_dim*, the number of attention heads *num heads*, the vocabulary size *vocab_size*, the numbers of units in the MLP head *mlp_units_0* and *mlp_units_1*. The output of the model are the *vocab_size* probabilities calculated for the next token (*probs*). In Algorithm 8 the following functions are used: BEMB() represents the binary embedding layer presented in Algorithm 3; Normalize() is the function that normalizes the features of each example so that it has zero mean and unit standard deviation; BTFB() is the transformer block presented in Algorithm 5; BNFCL() represents the binary normalized fully connected layer presented in Algorithm 1. The activation functions of the MLP head layers are *gelu* and for the last layer is *softmax*.

**Algorithm 8** Binary language decoder model (BLM)
***
**Require:** Input token sequences *seq*, maximum sentence length *max_len*, embedding dimension *emb_dim*, number of attention heads *num heads*, vocabulary size *vocab_size*, number of units in the mlp head layers *mlp_units_0* and *mlp_units_1*
**Ensure:** Classes probabilities *probs*
1: Embedded coding of the token sequences: $embs = \text{BEMB}(max\_len, emb\_dim, vocab\_size)(seq)$
2: Embedding normalizations: $x = \text{Normalize}(embs)$
3: Pass through a sequence of transformer blocks:
4: **for** *i* from 1 to *num_blocks* **do**
5: &nbsp;&nbsp;&nbsp;&nbsp;$x = \text{BTFB}(emb\_dim, num\_heads, ff\_dim = 2 * emb\_dim)(x)$
6: &nbsp;&nbsp;&nbsp;&nbsp;**end for**
7: Process transformer output with fully connected layers (MLP head)
8: &nbsp;&nbsp;&nbsp;&nbsp;$features = \text{BNFCL}(units = mlp\_units\_0, activation = 'gelu')(x)$
9: &nbsp;&nbsp;&nbsp;&nbsp;$features = \text{BNFCL}(units = mlp\_units\_1, activation = 'gelu')(features)$
10: Final fully connected layer to calculate the probabilities
11: &nbsp;&nbsp;&nbsp;&nbsp;$probs = \text{BNFCL}(units = vocab\_size, activation = 'softmax')(features)$
12: **return** *probs*

To verify the influence of the number of parameters on training stability and performance, two binary transformer neural networks are configured with different number of parameters. Table 3 presents the hyperparameters used to configure

1 BIT IS ALL WE NEED

A PREPRINT

the binary language models. The small model has about 154.4 million binary parameters and the large model has about 332.8 million.

Table 3: Hyperparameters used to configure the language models.

<table><thead><tr><th>Hyperparameter</th><th>Small model</th><th>Large model</th></tr></thead><tbody><tr><td>Maximum sentence length</td><td>256</td><td>256</td></tr><tr><td>Number of transformer blocks</td><td>12</td><td>16</td></tr><tr><td>Embedding dimension</td><td>768</td><td>1024</td></tr><tr><td>Number of attention heads</td><td>16</td><td>16</td></tr><tr><td>Number of units in the MLP head</td><td>4096 - 2048</td><td>8192 - 4096</td></tr><tr><td>Vocabulary size</td><td>30522</td><td>30522</td></tr><tr><td>Total number of parameters</td><td>154.4 million</td><td>332.8 million</td></tr></tbody></table>

To verify if the binary models are effective, a model with float 32-bit parameters ("standard" model) with the same configurations of the small binary model is also configured and trained. In the "standard" model the normalize layers has trainable parameters to adjust the best mean and standard deviation of the data, therefore, the standard model has a slightly larger total number of parameters than the equivalent binary model.

In all models, connection weights and biases are initialized using the standard methods: Glorot Uniform for weights and zeros for biases. No dropout is used in any model and no regularization methods or parameter constraints are applied in all models. Table 4 presents the hyperparameters used for training the language models.

Table 4: Hyperparameters used for training the language decoder models.

<table><thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead><tbody><tr><td>Cost function</td><td>Categorical cross entropy</td></tr><tr><td>Metrics</td><td>Accuracy and Perplexity</td></tr><tr><td>Optimization method</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1.0 × 10<sup>-5</sup></td></tr><tr><td>Number of epochs</td><td>100</td></tr></tbody></table>

## 4.2 Results obtained with the language decoders

In Figure 2 the training results are displayed for the language decoder models and in Table 5 these results are summarized. The values given in Table 5 are the best results obtained during training. The results from the standard language decoder with 32-bit parameters are included as a benchmark for the desired performance. It is important to note that multiple training tests were conducted for all models, and all results are very similar.

Table 5: Summary of the language decoder models results

<table><thead><tr><th>Model</th><th>Training loss</th><th>Validation loss</th><th>Training accuracy</th><th>Validation accuracy</th><th>Training perplexity</th><th>Validation perplexity</th></tr></thead><tbody><tr><td>Standard model</td><td>1.35</td><td>1.94</td><td>0.726</td><td>0.664</td><td>3.98</td><td>7.47</td></tr><tr><td>Small binary model</td><td>2.03</td><td>1.99</td><td>0.653</td><td>0.659</td><td>8.08</td><td>7.92</td></tr><tr><td>Large binary model</td><td>1.87</td><td>1.91</td><td>0.666</td><td>0.666</td><td>6.95</td><td>7.47</td></tr></tbody></table>

Analyzing the training results of the decoders models shown in Figure 2 and Table 5, the following observations can be made:

* The binary models are capable to train without any kind of instability and their performance are similar to the standard model;
* The standard model shows a large overfitting while none of the binary models present overfitting;
* The results for the validation data of the binary models are very similar of the results of the standard model;
* The results of the large binary model are slightly better than the results of the small binary model and are equivalent to the best results of the standard model. Note that the large binary model has roughly the double number of parameters of the small binary model and the standard model;
* The results of the binary models are very good considering that they have only 1-bit parameters.

1 BIT IS ALL WE NEED

A PREPRINT

![](images/task_14a83adeb726_page_11_pic_0.png)

Figure 2: Training results of the language decoders.

1 BIT IS ALL WE NEED

A PREPRINT

Again, it is important to observe that the binary normalization layers are effective to solve the problems of training instabilities and low accuracy of models with binary parameters.

## 5 Conclusions

In this work, a novel type of neural network model is developed, in which all parameters have only a single bit. In this new class of models, all layer parameters — including kernel weights and biases — are restricted to 1-bit resolution. These models are built using a type of layer referred to as a binary normalized layer. Binary normalized layers can be implemented in various architectures, such as fully connected, convolutional, or attention-based layers.

To show the effectiveness of the binary normalized layers, two different types of problems are solved: a multiclass image classification problem, and a language decoder to predict the next token of a sequence. A convolutional binary model is configured to solve the image classification problem and a binary transformer model is configured to solve the language problem. Standard 32-bit float parameters models with the same configuration of the binary models are used to provide benchmarks for the results.

The model used for image classification includes convolutional and fully connected layers. Two versions of this model are configured and tested, differing only in the number of parameters. Similarly, the model used for the language decoding task consists of transformer blocks with multi-head attention layers, and again, two versions are tested, differing in the number of transformer blocks and total parameters.

Results show that models using binary normalized layers achieve almost the same performance to that of the corresponding models with 32-bit floating-point parameters. Moreover, the binary models do not exhibit instability during training—an important result, since training instability is a major concern in low-resolution models. As observed by Cabral and Driemeier [2025], traditional models with binary parameters typically fail to train effectively due to such instability.

As expected, increasing the number of parameters in the binary models improves their performance without causing overfitting. This is a key advantage of the proposed model type, allowing it to reach performance levels similar to those of 32-bit models.

It is important to emphasize that no effort was made to fine-tune complex architectures for maximal performance or to eliminate overfitting entirely. The primary goal of this study is to introduce the binary normalized model and demonstrate its ability to generalize and perform comparably to standard 32-bit models. For this purpose, a direct comparison with equivalent 32-bit models is sufficient.

Binary normalized layers make it possible to build models that use 32 times less memory than conventional networks while delivering equivalent performance. Furthermore, these layers can be efficiently implemented on standard hardware using 1-bit arrays and do not require dedicated electronic components. This new type of layer opens the door to large-scale neural networks with drastically reduced memory requirements, making them deployable on simple and inexpensive hardware such as mobile devices or CPUs alone.

In addition, 1-bit parameter models offer the significant advantage of enabling the development of more complex architectures with more processing units while still using far less memory than traditional 32-bit models or even those using 8-bit quantization. These low-resolution networks have the potential to support the deployment of large language models on embedded devices.

Future work will focus on the implementation of binary normalization layers using single-bit arrays operations, as well as on quantizing layer activations to 8 or 16-bit precision. These improvements are expected to further enhance the efficiency and performance of the binary neural network models.

## References

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT Press, Cambridge, MA, 2016.

Thomas A. Henzinger, Mathias Lechner, and Djórđe Žikelić. Scalable verification of quantized neural networks. In *Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)*, pages 3787–3795, 2021.

George Plastiras, Maria Terzi, Christos Kyrkou, and Theocharis Theocharides. Edge intelligence: Challenges and opportunities of near-sensor machine learning applications. In *2018 IEEE 29th International Conference on Application-specific Systems, Architectures and Processors (ASAP)*, pages 1–7, 2018. doi:10.1109/ASAP.2018.8445118.

Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *International Conference on Learning Representations (ICLR)*, 2016.

A PREPRINT

1 BIT IS ALL WE NEED

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. arXiv preprint arXiv:1712.05877, 2018. doi:10.48550/arXiv.1712.05877.

Raghuraman Krishnamoorthi. Quantizing deep convolutional networks for efficient inference: A whitepaper, 2018. arXiv:1806.08342.

Ron Banner, Yaniv Nahshan, and Daniel Soudry. Post-training 4-bit quantization of convolutional networks for rapid-deployment. Advances in Neural Information Processing Systems (NeurIPS), 2019.

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Quantized neural networks: Training neural networks with low precision weights and activations. Journal of Machine Learning Research, 18 (187):1–30, 2017.

Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, Long Beach, CA, USA, 2018. URL https://arxiv.org/abs/1711.00937. Apresentado na NIPS 2017.

Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Accurate and efficient 2-bit quantized neural networks. In International Conference on Machine Learning (ICML), 2019.

B. Zhuang, C. Shen, M. Tan, L. Liu, and I. Reid. Towards effective low-bitwidth convolutional neural networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7920–7928, 2018.

Eduardo Lobo Lustosa Cabral and Larissa Driemeier. Neural networks with low-resolution parameters. Neural Networks, 2025.

Kyuyeon Hwang and Wonyong Sung. Fixed-point feedforward deep neural network design using weights +1, 0, and -1. In Signal Processing Systems (SiPS), 2014 IEEE Workshop on, pages 1–6. IEEE, 2014.

Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural networks with binary weights during propagations. In Advances in Neural Information Processing Systems (NIPS), pages 3123–3131, 2015.

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks. In Advances in Neural Information Processing Systems (NIPS), pages 4107–4115, 2016.

Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. In European Conference on Computer Vision (ECCV), volume 9908 of Lecture Notes in Computer Science, pages 525–542. Springer, 2016. doi:10.1007/978-3-319-46493-0_32.

Michael A. Alcorn. Aquamam: An autoregressive, quaternion manifold model for rapidly estimating complex so(3) distributions, 2023. URL https://arxiv.org/abs/2301.08838.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), volume 8694 of Lecture Notes in Computer Science, pages 446–461. Springer, 2014.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. URL https://arxiv.org/abs/1609.07843.

Xinying Song, Alexandru Salcianu, Yang Song, Dave Dopson, and Denny Zhou. Fast wordpiece tokenization. arXiv preprint arXiv:2012.15524, 2021. URL https://arxiv.org/abs/2012.15524.