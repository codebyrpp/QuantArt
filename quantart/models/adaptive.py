import torch
import torch.nn as nn
import pytorch_lightning as pl
from main import instantiate_from_config
from quantart.models.stage_2 import StyleTransfer
from quantart.components.adaptive import AdaptiveFusionNet


class AdaptiveQuantArtModel(pl.LightningModule):
    """
    Adaptive QuantArt Model (PL Module).

    Wraps the AdaptiveFusionNet (component) and StyleTransfer (backbone) 
    into a LightningModule for training.
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
                 monitor=None
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_key1 = image_key1
        self.image_key2 = image_key2

        # 1. Initialize Base Model (StyleTransfer)
        # We need to ensure the config passed creates a StyleTransfer model
        # The base_model_config should be the 'params' for StyleTransfer
        self.style_transfer = StyleTransfer(**base_model_config)

        # Load checkpoints if provided
        if stage2_ckpt_path:
            self.style_transfer.init_from_ckpt(stage2_ckpt_path)

        # If specific stage 1 weights are provided and not handled by stage2_ckpt_path
        # (Though StyleTransfer init handles them if in config, here we allow override if needed,
        # but usually StyleTransfer loads them in __init__ via its own args.
        # For simplicity, we assume base_model_config contains checkpoint_encoder/decoder paths
        # or they are loaded via stage2_ckpt_path)

        # 2. Initialize Adaptive Component
        self.net = AdaptiveFusionNet(self.style_transfer, embed_dim=embed_dim)

        # 3. Loss Helper
        self.loss_helper = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor

    def forward(self, content, style):
        return self.net(content, style)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        # Only one optimizer (for mask predictor)

        content = self.get_input(batch, self.image_key1)
        style = self.get_input(batch, self.image_key2)

        output_image, mask = self(content, style)

        # Losses
        # L_content: MSE between output and content
        l_content = self.loss_helper.calc_content_loss(output_image, content)

        # L_style: Mean/Std matching between output and style
        l_style = self.loss_helper.calc_style_loss(output_image, style)

        # L_sparsity: Mean(|mask - 0.5|)
        l_sparsity = torch.mean(torch.abs(mask - 0.5))

        # Total Loss
        total_loss = l_content + 10 * l_style + 0.1 * l_sparsity

        self.log("train/loss", total_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/content_loss", l_content, prog_bar=False,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/style_loss", l_style, prog_bar=False,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/sparsity_loss", l_sparsity, prog_bar=False,
                 logger=True, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        # We only train the mask predictor
        optimizer = torch.optim.Adam(
            self.net.mask_predictor.parameters(), lr=self.learning_rate)
        return optimizer

    def log_images(self, batch, **kwargs):
        log = dict()
        content = self.get_input(batch, self.image_key1)
        style = self.get_input(batch, self.image_key2)

        content = content.to(self.device)
        style = style.to(self.device)

        output_image, mask = self(content, style)

        # Mask is 1-channel, repeat to 3 for viz
        mask_vis = mask.repeat(1, 3, 1, 1)

        log["inputs_content"] = content
        log["inputs_style"] = style
        log["output"] = output_image
        log["mask"] = mask_vis

        return log
