import argparse
import os
import torch
import numpy as np
import datetime
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF
from main import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_discrete", type=str, required=True,
                        help="Path to discrete model config")
    parser.add_argument("--config_continuous", type=str, required=True,
                        help="Path to continuous model config")
    parser.add_argument("--ckpt_discrete", type=str,
                        required=True, help="Path to discrete model checkpoint")
    parser.add_argument("--ckpt_continuous", type=str,
                        required=True, help="Path to continuous model checkpoint")
    parser.add_argument("--output", type=str,
                        default=None, help="Path to save output image")
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

    # Save original state dict of model_d's decoder parts to avoid accumulation of fusion
    # We only need to save the parts modified by fuse_models
    model_d_decoder_sd = {k: v.clone()
                          for k, v in model_d.decoder.state_dict().items()}
    model_d_post_quant_sd = {
        k: v.clone() for k, v in model_d.post_quant_conv.state_dict().items()}

    while True:
        # Get user input for parameters
        content_path = input(
            "Enter path to content image (or 'q' to quit): ").strip('"').strip("'")
        if content_path.lower() == 'q':
            break

        style_path = input(
            "Enter path to style image: ").strip('"').strip("'")

        alpha_input = input(
            "Enter alpha (visual fidelity, 0.0-1.0) [0.5]: ")
        alpha_val = float(alpha_input) if alpha_input.strip() else 0.5

        beta_input = input("Enter beta (style fidelity, 0.0-1.0) [0.5]: ")
        beta_val = float(beta_input) if beta_input.strip() else 0.5

        print("Loading images...")
        try:
            content_img = preprocess_image(
                content_path, args.size, args.device)
            style_img = preprocess_image(style_path, args.size, args.device)

            # Determine output path
            if args.output is None:
                os.makedirs("results", exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    "results", f"stylized_alpha_{alpha_val}_beta_{beta_val}_{timestamp}.png")
            else:
                output_path = args.output

            with torch.no_grad():
                # Restore model_d decoder weights before each run
                model_d.decoder.load_state_dict(model_d_decoder_sd)
                model_d.post_quant_conv.load_state_dict(model_d_post_quant_sd)

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
                term1 = weighted_sum(z_y_hat, z_c_hat, beta_val)

                # Term 2: Continuous mix
                term2 = weighted_sum(z_y, z_c, beta_val)

                # Final z_test
                z_test = weighted_sum(term1, term2, alpha_val)

                # 4. Decode
                # Fuse decoders: ¯DS = ⊕α( ˆDS , DS )
                # ˆDS is discrete model decoder (model_d), DS is continuous model decoder (model_c)
                # We fuse into model_d's decoder instance
                if alpha_val == 0:
                    fused_model = model_c
                elif alpha_val == 1:
                    fused_model = model_d
                else:
                    fused_model = fuse_models(model_d, model_c, alpha_val)

                # Decode z_test
                # decode() calls post_quant_conv then decoder
                stylized_image = fused_model.decode(z_test)

                print(f"Saving output to {output_path}...")
                save_image(stylized_image, output_path)

        except Exception as e:
            print(f"Error processing image: {e}")

        # Reset output to None if it was auto-generated?
        # args.output is from argparse, so it stays the same throughout the loop.
        # If args.output was provided by user, it will be reused (overwritten).
        # If args.output was None, output_path is recalculated each time.
        # So we don't need to reset anything here.


if __name__ == "__main__":
    main()
