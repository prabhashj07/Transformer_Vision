# Literature Review: AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

## Introduction:
When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

Transformers, introduced by Vaswani et al. (2017), have become dominant in natural language processing (NLP) due to their efficiency and scalability. These models are typically pre-trained on large text corpora and fine-tuned for specific tasks, achieving remarkable results with up to 100 billion parameters (Brown et al., 2020).

In contrast, computer vision has largely relied on convolutional neural networks (CNNs) (LeCun et al., 1989; Krizhevsky et al., 2012). Recent attempts to integrate self-attention with CNNs or replace convolutions entirely have faced challenges in scaling and have not yet surpassed traditional architectures (Wang et al., 2018; Ramachandran et al., 2019).

This paper explores applying Transformers directly to images by treating image patches as tokens, similar to NLP. Initial results on datasets like ImageNet show modest performance compared to CNNs but improve significantly with larger datasets. Vision Transformers (ViT), when pre-trained on large datasets like ImageNet-21k or JFT-300M, achieve or surpass state-of-the-art results on several benchmarks.

### Related Work

#### Transformers in NLP 🗣️
- **Vaswani et al. (2017)**: Introduced Transformers for machine translation, which have since become the state-of-the-art for various NLP tasks.
- **BERT (Devlin et al., 2019)**: Utilizes denoising self-supervised pre-training.
- **GPT Series (Radford et al., 2018; 2019; Brown et al., 2020)**: Uses language modeling as a pre-training task.

#### Transformers for Images 🖼️
- **Parmar et al. (2018)**: Applied self-attention locally to reduce computational cost.
- **Hu et al. (2019); Ramachandran et al. (2019); Zhao et al. (2020)**: Replaced convolutions with local multi-head self-attention.
- **Child et al. (2019)**: Introduced Sparse Transformers with scalable approximations for global self-attention.
- **Weissenborn et al. (2019); Ho et al. (2019); Wang et al. (2020a)**: Explored attention in blocks or along individual axes to improve scalability.

#### Cordonnier et al. (2020) 🔍
- **Model Overview**: Extracts 2 × 2 image patches and applies full self-attention. This approach is similar to ViT but limited to small-resolution images.

#### CNNs and Self-Attention 🤖
- **Bello et al. (2019)**: Augmented CNN feature maps with self-attention for image classification.
- **Hu et al. (2018); Carion et al. (2020)**: Used self-attention to further process CNN outputs for object detection and other tasks.
- **Wu et al. (2020)**: Integrated self-attention with CNNs for image classification.
- **Chen et al. (2020c); Lu et al. (2019); Li et al. (2019)**: Combined CNNs with self-attention for unified text-vision tasks.

#### Image GPT (Chen et al., 2020a) 🖼️🤖
- **Approach**: Applies Transformers to image pixels after reducing resolution and color space. Achieved a maximum accuracy of 72% on ImageNet through unsupervised training.

#### Scaling and Dataset Size 📈
- **Mahajan et al. (2018); Touvron et al. (2019); Xie et al. (2020)**: Demonstrated state-of-the-art results with larger datasets.
- **Sun et al. (2017)**: Studied CNN performance scaling with dataset size.
- **Kolesnikov et al. (2020); Djolonga et al. (2020)**: Explored CNN transfer learning with large datasets like ImageNet-21k and JFT-300M.

#### Paper Contribution 🚀
- Focuses on applying Transformers to large-scale image recognition, specifically using large datasets such as ImageNet-21k and JFT-300M, and achieving state-of-the-art results compared to traditional ResNet-based models.

### Outline
- Vision Transformer(ViT)
- Hybrid Architecture
- Some Training Details 
- Experimental Results


#### Vision Transformer (ViT)
![Vision Transformer](../assets/vision_transformer.png)
*Vision Transformer Netwokr Architecture*

- To handle 2D images, the image x is reshaped from H×W×C into a sequence of flattened 2D patches xp, with the shape of N×(P²×C), where (H, W) is the resolution of the original image, C is the number of channels, (P, P) is the resolution of each image patch, and N=HW/P² is the resulting number of patches.

![ViT Maths](../assets/ViT_equations.png)

- Eq. 1: The Transformer uses constant latent vector size D through all of its layers, so the patches are flattened and map to D dimensions with a trainable linear projection. The output of this projection as the **patch embeddings**.
- Similar to BERT’s [class] token, a learnable embedding is prepended to the sequence of embedded patches (z00=xclass)
- Eq. 4: The state at the output of the Transformer encoder (z0L) serves as the **image representation y**.
- Both during pre-training and fine-tuning, a classification head is attached to z0L. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.
- Position embeddings are added to the patch embeddings to retain positional information. Standard learnable 1D position embeddings is used.
- Eq. 2, 3: The Transformer encoder consists of alternating layers of multiheaded self-attention (MSA) and MLP blocks.
- Layernorm (LN) is applied before every block, and residual connections after every block. The MLP contains two layers with a GELU non-linearity.
- The “Base” and “Large” models are directly adopted from BERT and the larger “Huge” model is added.
- ViT-L/16 means the “Large” variant with 16×16 input patch size. Note that the Transformer’s sequence length is inversely proportional to the square of the patch size, and models with smaller patch size are computationally more expensive.
