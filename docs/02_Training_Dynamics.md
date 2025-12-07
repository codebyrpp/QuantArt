# Training Dynamics

This document explains the loss functions, optimization strategies, and training loop details implemented in the codebase.

## Loss Landscape

The model optimization is driven by a composite loss function defined in `taming.modules.losses.vqperceptual_ref.VQLPIPS_Ref`.

### Total Loss Composition

The generator (Autoencoder/Transfer) loss is calculated as:

$$ \mathcal{L}_{total} = \mathcal{L}_{AE} + \lambda*{adv} \cdot \mathcal{L}*{GAN} $$

Where $\mathcal{L}_{AE}$ (Autoencoder Loss) is composed of:

$$ \mathcal{L}_{AE} = \lambda_{rev} \mathcal{L}_{rev} + \lambda_{code} \mathcal{L}_{code} + \lambda_{style} \mathcal{L}_{style} + (\lambda_{id} \mathcal{L}\_{id}) $$

#### 1. Reverse/Content Loss (`reverse_loss`)

- **Goal**: Ensure the transferred **latent representation** stays faithful to the content structure.
- **Code**: `calc_content_loss` (MSE).
- **Input**: `quant_y` (Transferred Latent) vs `quant_x` (Content Latent).
- **Weight**: `reverse_weight` (Default: 1.0).

#### 2. Style Loss (`style_loss`)

- **Goal**: Ensure the **latent statistics** of the output match the reference style.
- **Code**: `calc_style_loss` (Mean/Std matching).
- **Formula**:
  $$ \mathcal{L}_{style} = \| \mu(z_y) - \mu(z_{ref}) \|_2^2 + \| \sigma(z_y) - \sigma(z_{ref}) \|\_2^2 $$
- **Weight**: `style_weight` (Default: 10.0).

#### 3. Codebook Loss (`codebook1_loss`)

- **Goal**: Move the embedding vectors towards the encoder outputs.
- **Formula**: $\beta \|z - sg[e]\|_2^2$ (Commitment loss).
- **Weight**: `codebook1_weight` (Default: 1.0).

#### 4. Adversarial Loss (`g_loss`)

- **Goal**: Force realistic outputs.
- **Implementation**: Hinge Loss or Vanilla GAN loss via `NLayerDiscriminator` (PatchGAN).
- **Adaptive Weighting**: The code implements `calculate_adaptive_weight` to balance gradients between reconstruction and adversarial loss, though it seems `discriminator_weight` is often fixed in config.

## Hyperparameters

Common configurations found in `configs/*.yaml`:

- **Learning Rate**: `4.5e-6` (Base LR).
- **Optimizer**: Adam (`betas=(0.5, 0.9)`).
- **Batch Size**: 8 (per GPU/node).
- **Weights**:
  - `style_weight`: 10.0 (Emphasizes style transfer).
  - `disc_start`: 0 (Discriminator starts immediately or after N steps).

## Training Step (`VQModel_Ref.training_step`)

The training process uses PyTorch Lightning's optimization loop with two optimizers:

1.  **Optimizer 0 (Generator/Transfer)**:

    - Calculates `transfer(x1, x2)`.
    - Computes `total_loss`.
    - Backpropagates through `model_x2y` (and potentially encoder/decoder if not frozen, though usually frozen).

2.  **Optimizer 1 (Discriminator)**:
    - Computes `logits_real` (on reference `x1` or `x2`) and `logits_fake` (on transferred output).
    - Computes `d_loss`.

**Note**: In many VQGAN-based experiments, the Encoder and Decoder are loaded from a checkpoint and `frozen` (via `disable_grad`). Only the `StyleTransferModule` and `VectorQuantizer` (sometimes) are trained.
