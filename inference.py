import argparse
import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF
from main import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True,
                        help="Path to content image")
    parser.add_argument("--style", type=str, required=True,
                        help="Path to style image")
    parser.add_argument("--config_discrete", type=str, required=True,
                        help="Path to discrete model config")
    parser.add_argument("--config_continuous", type=str, required=True,
                        help="Path to continuous model config")
    parser.add_argument("--ckpt_discrete", type=str,
                        required=True, help="Path to discrete model checkpoint")
    parser.add_argument("--ckpt_continuous", type=str,
                        required=True, help="Path to continuous model checkpoint")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Visual fidelity control (0.0 to 1.0)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Style fidelity control (0.0 to 1.0)")
    parser.add_argument("--output", type=str,
                        default="output.png", help="Path to save output image")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    # Load checkpoint
    if ckpt_path.endswith('.ckpt'):
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    else:
        # Assume it's a raw state dict or other format if needed
        sd = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, size, device):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size), Image.LANCZOS)
    image = TF.to_tensor(image).to(device)
    image = image * 2.0 - 1.0
    return image.unsqueeze(0)


def weighted_sum(a, b, p):
    return p * a + (1 - p) * b


def fuse_models(model_a, model_b, alpha):
    """
    Fuse the decoders of two models based on alpha.
    Returns a new model (or modifies one) with fused weights for decoder.
    Here we create a copy of model_a and update its decoder weights.
    """
    fused_model = model_a  # We'll just use model_a structure and update weights

    # Fuse Decoder
    for (name_a, param_a), (name_b, param_b) in zip(model_a.decoder.named_parameters(), model_b.decoder.named_parameters()):
        assert name_a == name_b
        fused_param = weighted_sum(param_a.data, param_b.data, alpha)
        param_a.data.copy_(fused_param)

    # Fuse Post Quant Conv
    for (name_a, param_a), (name_b, param_b) in zip(model_a.post_quant_conv.named_parameters(), model_b.post_quant_conv.named_parameters()):
        assert name_a == name_b
        fused_param = weighted_sum(param_a.data, param_b.data, alpha)
        param_a.data.copy_(fused_param)

    return fused_model


def save_image(tensor, path):
    grid = tensor.detach().cpu()
    grid = (grid + 1.0) / 2.0
    grid = grid.clamp(0, 1)
    grid = grid.squeeze(0).permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)
    image = Image.fromarray(grid)
    image.save(path)
    try:
        image.show()
    except Exception as e:
        print(f"Could not display image: {e}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"Loading models on {args.device}...")
    model_d = load_model(args.config_discrete, args.ckpt_discrete, args.device)
    model_c = load_model(args.config_continuous,
                         args.ckpt_continuous, args.device)

    print("Loading images...")
    content_img = preprocess_image(args.content, args.size, args.device)
    style_img = preprocess_image(args.style, args.size, args.device)

    with torch.no_grad():
        # 1. Extract Features
        # Discrete Features
        # model.encode returns: quant, emb_loss, info
        z_c_hat, _, _ = model_d.encode(content_img, quantize=True)
        z_s_hat, _, _ = model_d.encode_real(style_img, quantize=True)

        # Continuous Features & Transform
        z_c, z_y, _ = model_c.transfer_without_quantization(
            content_img, style_img)

        # 2. Transform Features (SGA)
        # Discrete Path
        h_x_d = model_d.model_x2y(z_c_hat, z_s_hat)
        z_y_hat, _, _ = model_d.quantize_dec(
            h_x_d)  # Quantized stylized feature

        # 3. Combine Features
        # ztest = ⊕α(⊕β (ˆzy , ˆzc), ⊕β (zy , zc))
        # ⊕p(a, b) = pa + (1 − p)b

        # Term 1: Discrete mix
        term1 = weighted_sum(z_y_hat, z_c_hat, args.beta)

        # Term 2: Continuous mix
        term2 = weighted_sum(z_y, z_c, args.beta)

        # Final z_test
        z_test = weighted_sum(term1, term2, args.alpha)

        # 4. Decode
        # Fuse decoders: ¯DS = ⊕α( ˆDS , DS )
        # ˆDS is discrete model decoder (model_d), DS is continuous model decoder (model_c)
        # We fuse into model_d's decoder instance
        alpha = args.alpha
        if alpha == 0:
            fused_model = model_c
        elif alpha == 1:
            fused_model = model_d
        else:
            fused_model = fuse_models(model_d, model_c, alpha)

        # Decode z_test
        # decode() calls post_quant_conv then decoder
        stylized_image = fused_model.decode(z_test)

        print(f"Saving output to {args.output}...")
        save_image(stylized_image, args.output)


if __name__ == "__main__":
    main()
