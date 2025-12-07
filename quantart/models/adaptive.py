import torch
import torch.nn as nn
import pytorch_lightning as pl
from main import instantiate_from_config
from quantart.models.stage_2 import StyleTransfer
from quantart.components.adaptive import AdaptiveFusionNet


class AdaptiveQuantArtModel(pl.LightningModule):
    """
    Adaptive QuantArt Model (PL Module).
    """

    def __init__(self,
                 base_model_config,
                 lossconfig,
                 stage1_encoder_path=None,
                 stage1_decoder_path=None,
                 stage2_ckpt_path=None,
                 embed_dim=256,
                 learning_rate=4.5e-6,
                 image_key1="image1",  # Content
                 image_key2="image2",  # Style
                 monitor=None,
                 lambda_content=1.0,   # Added config for weights
                 lambda_style=10.0,
                 lambda_sparsity=0.1
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_key1 = image_key1
        self.image_key2 = image_key2

        # Loss Weights
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_sparsity = lambda_sparsity

        # 1. Initialize Base Model (StyleTransfer)
        self.style_transfer = StyleTransfer(**base_model_config)

        # Load checkpoints
        if stage2_ckpt_path:
            print(f"Loading Stage 2 weights from {stage2_ckpt_path}")
            self.style_transfer.init_from_ckpt(stage2_ckpt_path)

        # CRITICAL: FREEZE THE BASE MODEL
        # If you don't do this, you waste VRAM computing gradients for the whole VQGAN
        self.style_transfer.eval()
        for param in self.style_transfer.parameters():
            param.requires_grad = False

        # 2. Initialize Adaptive Component (The ONLY trainable part)
        self.net = AdaptiveFusionNet(self.style_transfer, embed_dim=embed_dim)
        # Ensure mask predictor is trainable
        for param in self.net.mask_predictor.parameters():
            param.requires_grad = True

        # 3. Loss Helper
        self.loss_helper = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor

    def forward(self, content, style):
        # AdaptiveFusionNet handles feature extraction -> mask -> fusion -> decode
        return self.net(content, style)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def shared_step(self, batch):
        """Shared logic for train/val"""
        content = self.get_input(batch, self.image_key1)
        style = self.get_input(batch, self.image_key2)

        output_image, mask = self(content, style)

        # Calculate Losses
        l_content = self.loss_helper.calc_content_loss(output_image, content)
        l_style = self.loss_helper.calc_style_loss(output_image, style)

        # Sparsity Loss: Encourages mask to be binary (0 or 1)
        # Using L1 distance from 0.5 pushes values to edges
        l_sparsity = torch.mean(torch.abs(mask - 0.5))

        total_loss = (self.lambda_content * l_content) + \
                     (self.lambda_style * l_style) + \
                     (self.lambda_sparsity * l_sparsity)

        return total_loss, l_content, l_style, l_sparsity, output_image, mask

    def training_step(self, batch, batch_idx):
        loss, l_c, l_s, l_sp, _, _ = self.shared_step(batch)

        self.log("train/loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/content_loss", l_c, prog_bar=False,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/style_loss", l_s, prog_bar=False,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/sparsity_loss", l_sp, prog_bar=False,
                 logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        VALIDATION STEP IS CRUCIAL FOR STYLE TRANSFER
        We need to know if the model is ignoring the style (mask=0) 
        or destroying content (mask=1).
        """
        loss, l_c, l_s, l_sp, _, _ = self.shared_step(batch)

        self.log("val/loss", loss, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/content_loss", l_c, logger=True,
                 on_epoch=True, sync_dist=True)
        self.log("val/style_loss", l_s, logger=True,
                 on_epoch=True, sync_dist=True)
        self.log("val/sparsity_loss", l_sp, logger=True,
                 on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # We only train the mask predictor
        optimizer = torch.optim.Adam(
            self.net.mask_predictor.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        """
        This is called by ImageLogger callback in PL
        """
        log = dict()
        content = self.get_input(batch, self.image_key1).to(self.device)
        style = self.get_input(batch, self.image_key2).to(self.device)

        output_image, mask = self(content, style)

        # Visualize Mask:
        # Mask is likely Bx1x16x16 (latent size).
        # We must upsample it to image size for visualization
        if mask.shape[2] != content.shape[2]:
            mask_vis = torch.nn.functional.interpolate(
                mask, size=content.shape[2:], mode='nearest'
            )
        else:
            mask_vis = mask

        mask_vis = mask_vis.repeat(1, 3, 1, 1)  # Make RGB for logging

        log["inputs_content"] = content
        log["inputs_style"] = style
        log["output"] = output_image
        log["mask_heatmap"] = mask_vis

        return log
