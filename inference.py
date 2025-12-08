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

        content_path = "imgs/content/" + content_path

        style_path = input(
            "Enter path to style image: ").strip('"').strip("'")

        style_path = "imgs/style/" + style_path

        alpha_input = input(
            "Enter alpha (visual fidelity, 0.0-1.0) [Leave empty for combinations 0, 0.5, 1]: ")
        if alpha_input.strip():
            alphas = [float(alpha_input)]
        else:
            alphas = [0.0, 0.5, 1.0]

        beta_input = input(
            "Enter beta (style fidelity, 0.0-1.0) [Leave empty for combinations 0, 0.5, 1]: ")
        if beta_input.strip():
            betas = [float(beta_input)]
        else:
            betas = [0.0, 0.5, 1.0]

        decoder_input = input(
            "Enter decoder to use (continuous(c), discrete(d), fused_model(f)) [d]: ")
        decoder_choice = decoder_input.strip().lower()
        if not decoder_choice:
            decoder_choice = 'd'

        print("Loading images...")
        try:
            content_img = preprocess_image(
                content_path, args.size, args.device)
            style_img = preprocess_image(style_path, args.size, args.device)

            # Determine if we are in combo mode
            is_combo = (len(alphas) > 1 or len(betas) > 1)

            result_dir = None
            if is_combo:
                content_name = os.path.splitext(
                    os.path.basename(content_path))[0]
                style_name = os.path.splitext(os.path.basename(style_path))[0]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                result_dir = os.path.join(
                    "results", f"{content_name}_{style_name}_{decoder_choice}_{timestamp}")
                os.makedirs(result_dir, exist_ok=True)

            with torch.no_grad():
                # Ensure model_d is clean before feature extraction
                model_d.decoder.load_state_dict(model_d_decoder_sd)
                model_d.post_quant_conv.load_state_dict(model_d_post_quant_sd)

                # 1. Extract Features
                # Discrete Features
                z_c_hat, _, _ = model_d.encode(content_img, quantize=True)
                z_s_hat, _, _ = model_d.encode_real(style_img, quantize=True)

                # Continuous Features & Transform
                z_c, z_y, _ = model_c.transfer_without_quantization(
                    content_img, style_img)

                # 2. Transform Features (SGA)
                h_x_d = model_d.model_x2y(z_c_hat, z_s_hat)
                z_y_hat, _, _ = model_d.quantize_dec(h_x_d)

                for alpha_val in alphas:
                    for beta_val in betas:
                        # Restore model_d decoder weights before each fusion
                        model_d.decoder.load_state_dict(model_d_decoder_sd)
                        model_d.post_quant_conv.load_state_dict(
                            model_d_post_quant_sd)

                        # Determine output path
                        if is_combo:
                            output_path = os.path.join(
                                result_dir, f"alpha_{alpha_val}_beta_{beta_val}.png")
                        elif args.output is None:
                            os.makedirs("results", exist_ok=True)
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = os.path.join(
                                "results", f"stylized_alpha_{alpha_val}_beta_{beta_val}_{timestamp}.png")
                        else:
                            output_path = args.output

                        # 3. Combine Features
                        term1 = weighted_sum(z_y_hat, z_c_hat, beta_val)
                        term2 = weighted_sum(z_y, z_c, beta_val)
                        z_test = weighted_sum(term1, term2, alpha_val)

                        # 4. Decode
                        if decoder_choice == 'f':
                            if alpha_val == 0:
                                decoder_model = model_c
                            elif alpha_val == 1:
                                decoder_model = model_d
                            else:
                                decoder_model = fuse_models(
                                    model_d, model_c, alpha_val)
                        elif decoder_choice == 'c':
                            decoder_model = model_c
                        elif decoder_choice == 'd':
                            decoder_model = model_d
                        else:
                            # Fallback to d
                            decoder_model = model_d

                        stylized_image = decoder_model.decode(z_test)

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
