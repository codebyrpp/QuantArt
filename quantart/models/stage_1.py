import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from quantart.components.encoder import Encoder
from quantart.components.decoder import Decoder
from quantart.components.quantizer import VectorQuantizer
from quantart.util import load_model


class BaseVQGAN(pl.LightningModule):
    """
    VQGAN Model (Stage 1) for learning a discrete latent representation of images.

    This model consists of an Encoder, a VectorQuantizer, and a Decoder. It is trained
    to reconstruct images while learning a compact discrete codebook. This learned
    latent space serves as the foundation for the Stage 2 style transfer model.

    References:
        "Taming Transformers for High-Resolution Image Synthesis" (Esser et al., 2021)
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_quantize=True,
                 freeze_decoder=False,
                 ckpt_quantize=None,
                 ):
        """
        Initialize the VQGAN model.

        Args:
            ddconfig (dict): Configuration for the Encoder and Decoder (e.g., ch, out_ch, ch_mult).
            lossconfig (dict): Configuration for the VQLPIPS loss (discriminator, weights).
            n_embed (int): Number of embeddings in the codebook (codebook size).
            embed_dim (int): Dimension of the embedding vectors.
            ckpt_path (str, optional): Path to a checkpoint to load weights from.
            ignore_keys (list, optional): List of keys to ignore when loading from checkpoint.
            image_key (str, optional): Key to retrieve images from the batch (default: "image").
            colorize_nlabels (int, optional): If set, enables colorization for segmentation masks.
            monitor (str, optional): Metric to monitor.
            remap (str, optional): Path to a file for remapping indices.
            sane_index_shape (bool, optional): If True, returns indices as (B, H, W).
            use_quantize (bool, optional): Whether to use the quantization layer.
            freeze_decoder (bool, optional): Whether to freeze the decoder weights.
            ckpt_quantize (str, optional): Checkpoint path for loading specific quantizer weights.
        """
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.use_quantize = use_quantize
        self.freeze_decoder = freeze_decoder
        if self.use_quantize:
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                            remap=remap, sane_index_shape=sane_index_shape)
        if self.freeze_decoder:
            checkpoint_quantize = torch.load(ckpt_quantize)['state_dict']
            load_model(self.quantize, checkpoint_quantize, 'quantize')

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer(
                "colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        Encode input images into quantized representations.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            tuple:
                - quant (torch.Tensor): Quantized latent feature map (B, embed_dim, h, w).
                - emb_loss (torch.Tensor): Codebook commitment loss.
                - info (tuple): Additional info (perplexity, min_encodings, indices).
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.use_quantize:
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info
        else:
            return h, None, None

    def decode(self, quant):
        """
        Decode quantized latent features back to image space.

        Args:
            quant (torch.Tensor): Quantized features (B, embed_dim, h, w).

        Returns:
            torch.Tensor: Reconstructed images (B, C, H, W).
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        """
        Forward pass of the VQGAN.

        1. Encode input to latent codes.
        2. Quantize latent codes.
        3. Decode quantized codes to reconstruct input.

        Args:
            input (torch.Tensor): Input batch.

        Returns:
            tuple:
                - dec (torch.Tensor): Reconstructed images.
                - diff (torch.Tensor): Quantization loss.
        """
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Single training step for PyTorch Lightning.

        Handles two alternating optimization steps:
        - Optimizer 0: Generator/Autoencoder (Reconstruction + Codebook + Adversarial Generator Loss).
        - Optimizer 1: Discriminator (Adversarial Discriminator Loss).

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of batch.
            optimizer_idx (int): Index of optimizer (0 or 1).

        Returns:
            torch.Tensor: The calculated loss value.
        """
        self.decoder.conv_out.weight.requires_grad = True

        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        self.decoder.conv_out.weight.requires_grad = True

        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        parameter_list = list(self.encoder.parameters()) + \
            list(self.quant_conv.parameters())
#                           list(self.decoder.parameters())+ \
#                           list(self.post_quant_conv.parameters())
        if not self.freeze_decoder:
            parameter_list = parameter_list + \
                list(self.decoder.parameters()) + \
                list(self.post_quant_conv.parameters())
        if self.use_quantize:
            parameter_list = parameter_list + \
                list(self.quantize.parameters())
        opt_ae = torch.optim.Adam(parameter_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer(
                "colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
