import torch
import torch.nn as nn


class MaskPredictor(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class AdaptiveFusionNet(nn.Module):
    """
    Component that performs the adaptive fusion logic using a provided backbone (StyleTransfer)
    and an internal MaskPredictor.
    """

    def __init__(self, base_model, embed_dim=256):
        super().__init__()
        self.base_model = base_model
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.mask_predictor = MaskPredictor(in_channels=embed_dim)

    def forward(self, content, style):
        # 1. Encode Content and Style using frozen encoders
        # quant_x: quantized content latents (z_content)
        with torch.no_grad():
            quant_x, _, _ = self.base_model.encode(content, quantize=True)
            quant_x = quant_x.detach()

            quant_ref, _, info_ref = self.base_model.encode_real(
                style, quantize=True)
            quant_ref = quant_ref.detach()

            # 2. Get Continuous Stylized Features (z_continuous)
            # Pass through SGAModule
            z_continuous = self.base_model.model_x2y(quant_x, quant_ref)

            # 3. Get Quantized Stylized Features (z_quantized)
            # Pass z_continuous through quantizer
            z_quantized, _, _ = self.base_model.quantize_dec(z_continuous)

        # 4. Predict Mask
        # Pass z_content (quant_x) through MaskPredictor
        mask = self.mask_predictor(quant_x)

        # 5. Perform Fusion
        # z_fused = mask * z_quantized + (1-mask) * z_continuous
        z_fused = mask * z_quantized + (1 - mask) * z_continuous

        # 6. Decode
        output_image = self.base_model.decode(z_fused)

        return output_image, mask
