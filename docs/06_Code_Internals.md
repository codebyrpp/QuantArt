# Code Internals: Stage 1 & Stage 2

This document provides a comprehensive deep dive into the internal implementation of the two primary model classes in QuantArt: `BaseVQGAN` (Stage 1) and `StyleTransfer` (Stage 2).

## Stage 1: BaseVQGAN (`quantart/models/stage_1.py`)

The `BaseVQGAN` class implements a Vector Quantized Generative Adversarial Network (VQGAN). Its primary goal is to learn a discrete latent representation of images that preserves high-quality reconstruction.

### 1. Initialization (`__init__`)

The constructor sets up the three main components of the VQGAN:

1.  **Encoder (`self.encoder`)**: A convolutional neural network that compresses the input image into a spatial latent representation. It reduces the spatial resolution by a factor determined by `ch_mult` and `num_res_blocks`.
2.  **Vector Quantizer (`self.quantize`)**: A layer that maps continuous latent vectors to the nearest neighbor in a learnable codebook.
3.  **Decoder (`self.decoder`)**: A convolutional neural network that reconstructs the image from the quantized latent representation.
4.  **Loss Module (`self.loss`)**: Instantiated from config (typically `VQLPIPS`), handling both reconstruction (L1/L2/LPIPS) and adversarial losses.

Key parameters:

- `embed_dim`: Dimension of the latent vectors (before quantization).
- `n_embed`: Size of the codebook (number of discrete tokens).
- `ckpt_quantize`: Optional path to load pre-trained quantizer weights separately.

### 2. Forward Pass Mechanics

The data flow is split into modular methods:

- **`encode(x)`**:

  1.  Passes `x` through `self.encoder` -> `h`.
  2.  Projects `h` to `embed_dim` using `self.quant_conv`.
  3.  If `use_quantize` is True, passes the result through `self.quantize`.
  4.  **Returns**: `quant` (quantized latents), `emb_loss` (commitment loss), and `info` (indices).

- **`decode(quant)`**:

  1.  Projects `quant` back to the decoder's channel dimension using `self.post_quant_conv`.
  2.  Passes the result through `self.decoder` to get the reconstructed image.

- **`forward(input)`**:
  - Orchestrates `encode` and `decode` in sequence.
  - Returns the reconstruction `dec` and the quantization loss `diff`.

### 3. Training Loop (`training_step`)

PyTorch Lightning handles the training loop. VQGAN training involves two competing objectives, managed via `optimizer_idx`:

- **Optimizer 0 (Generator)**:

  - Calculates `aeloss` (Autoencoder Loss).
  - **Components**: Reconstruction Loss (L1/L2 + LPIPS) + Codebook Commitment Loss + Adversarial Generator Loss (trying to fool the discriminator).
  - **Goal**: Make reconstructed images look real and match the input.

- **Optimizer 1 (Discriminator)**:
  - Calculates `discloss`.
  - **Components**: Adversarial Discriminator Loss (Real vs Fake classification).
  - **Goal**: Distinguish between real input images and the model's reconstructions.

---

## Stage 2: StyleTransfer (`quantart/models/stage_2.py`)

The `StyleTransfer` class (formerly `ExperimentStage2`) implements the novel style transfer mechanism. It operates entirely within the discrete latent space learned by Stage 1.

### 1. Initialization (`__init__`)

This class is more complex as it loads pre-trained weights from Stage 1 but freezes them to focus on learning the transfer module.

1.  **Frozen Backbone**:

    - `self.encoder` (Content) and `self.encoder_real` (Style/Reference) are instantiated and their weights loaded from `checkpoint_encoder` (Stage 1 checkpoint).
    - `self.decoder` is similarly loaded from `checkpoint_decoder`.
    - These components are typically **frozen** (gradients disabled) during Stage 2 training.

2.  **Style Transfer Module (`self.model_x2y`)**:

    - Instantiated as `SGAModule` (Style-Guided Attention Module).
    - This is the **only** major component being trained (along with the quantizers in some configs).

3.  **Quantizers**:
    - `self.quantize_enc`: Quantizes the input content/style features.
    - `self.quantize_dec`: Quantizes the output of the style transfer module.

### 2. The Transfer Mechanism (`transfer`)

This method defines the core logic of QuantArt:

1.  **Content Encoding**:

    - Input `x` (Content) is passed through `self.encoder` -> `quant_x`.
    - `quant_x` is detached from the graph to prevent gradients flowing back into the encoder.

2.  **Style Encoding**:

    - Input `ref` (Style) is passed through `self.encoder_real` -> `quant_ref`.
    - `quant_ref` is also detached.

3.  **Style-Guided Attention**:

    - `h_x = self.model_x2y(quant_x, quant_ref)`
    - The `SGAModule` uses cross-attention to inject style information from `quant_ref` into `quant_x` while preserving the structure of `quant_x`.

4.  **Result Quantization**:
    - The output `h_x` is effectively a continuous latent map.
    - It is passed through `self.quantize_dec` to map it back to the discrete codebook -> `quant_y`.
    - **Crucial Step**: This ensures the stylized result uses valid tokens from the pre-trained codebook, guaranteeing that the frozen decoder can interpret it correctly.

### 3. Training Loop (`training_step`)

Similar to Stage 1, this uses a GAN setup, but the objectives differ:

- **Optimizer 0 (Generator/Transfer)**:

  - Calculates `aeloss` via `Stage2Loss`.
  - **Components**:
    - **Content Loss**: MSE between `quant_y` and `quant_x` (ensures structure is preserved).
    - **Style Loss**: Mean/Std statistics matching between `quant_y` and `quant_ref` (ensures style is transferred).
    - **Adversarial Loss**: To ensure the output looks like a realistic image.
  - **Note**: Gradients primarily update `self.model_x2y` (SGAModule).

- **Optimizer 1 (Discriminator)**:
  - Updates the discriminator to distinguish between real images and the stylized outputs.
