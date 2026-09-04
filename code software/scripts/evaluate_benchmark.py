import os
import time
import json
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from model_architecture import Generator
from quantize_model import fuse_generator_bn, replace_conv2d_with_quant, quantize_residual_blocks_only


def compute_epi(hr_img, sr_img):
    """
    Edge Preservation Index (EPI) based on Laplacian operator.
    """
    hr_gray = np.array(Image.fromarray(hr_img).convert('L'), dtype=np.float64)
    sr_gray = np.array(Image.fromarray(sr_img).convert('L'), dtype=np.float64)

    # 3x3 Laplacian kernel
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    
    from scipy.signal import convolve2d
    delta_hr = convolve2d(hr_gray, lap, mode='same', boundary='symm')
    delta_sr = convolve2d(sr_gray, lap, mode='same', boundary='symm')

    mu_hr = np.mean(delta_hr)
    mu_sr = np.mean(delta_sr)

    d_hr = delta_hr - mu_hr
    d_sr = delta_sr - mu_sr

    numerator = np.sum(d_hr * d_sr)
    denominator = np.sqrt(np.sum(d_hr ** 2) * np.sum(d_sr ** 2)) + 1e-10
    return float(numerator / denominator)


def load_model(weights_path, upscale_factor=4, device=torch.device('cpu')):
    """
    Loads FP32 or Q7/INT8 Generator weights.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Check if Q7 txt file
    if weights_path.endswith('.txt'):
        print(f"[INFO] Loading Q7 text weights from {weights_path}...")
        with open(weights_path, 'r') as f:
            q7_values = [float(line.strip()) for line in f if line.strip()]
        
        base_model = Generator(upscale_factor=upscale_factor).to(device)
        model = fuse_generator_bn(base_model)
        
        state_dict = model.state_dict()
        ptr = 0
        new_state_dict = {}
        for name, param in model.named_parameters():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                continue
            if 'weight' in name or 'bias' in name:
                numel = param.numel()
                q7_slice = q7_values[ptr : ptr + numel]
                ptr += numel
                q7_tensor = torch.tensor(q7_slice, dtype=torch.float32, device=device)
                dequant_tensor = q7_tensor / 128.0
                new_state_dict[name] = dequant_tensor.view_as(param)
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Q7_Quantized"

    # PyTorch Checkpoint (.pth / .pth.tar)
    checkpoint = torch.load(weights_path, map_location=device)
    is_quantized = False
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        if checkpoint.get("quantized") is True:
            is_quantized = True
            state_dict = checkpoint["model"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

    base_model = Generator(upscale_factor=upscale_factor).to(device)
    if is_quantized or any("weight_int8" in k for k in state_dict.keys()):
        fused_model = fuse_generator_bn(base_model)
        quant_model = quantize_residual_blocks_only(fused_model, per_channel=True)
        quant_model.to(device)
        for m in quant_model.modules():
            if hasattr(m, "calibrating"):
                m.calibrating = False
        quant_model.load_state_dict(state_dict)
        model = quant_model
        model_type = "INT8_Quantized"
    else:
        # Strip DataParallel module. prefix if present
        clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(clean_state)
        model = base_model
        model_type = "FP32_Baseline"

    model.eval()
    return model, model_type


def run_benchmark(data_dir, weights_path, output_json, output_csv=None, save_images_dir=None, upscale_factor=4, device_str='auto', max_images=None):
    device = torch.device('cuda' if (device_str == 'cuda' or (device_str == 'auto' and torch.cuda.is_available())) else 'cpu')
    print(f"[INFO] Using Device: {device}")

    model, model_type = load_model(weights_path, upscale_factor=upscale_factor, device=device)
    print(f"[INFO] Loaded Model: Swift-SRGAN Generator ({model_type})")

    # Find image paths
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))
    
    image_paths.sort()
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"[INFO] Found {len(image_paths)} images in '{data_dir}'")
    if not image_paths:
        print("[ERROR] No valid images found.")
        return

    lr_resize = transforms.Resize((256, 256), interpolation=Image.BICUBIC)
    bicubic_upscale = transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)

    if save_images_dir:
        os.makedirs(save_images_dir, exist_ok=True)

    results = []
    start_wall_time = time.perf_counter()

    # Warmup GPU
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for img_path in tqdm(image_paths, desc="Benchmarking"):
        dataset_name = os.path.basename(os.path.dirname(img_path)) or "unknown"
        filename = os.path.basename(img_path)

        try:
            hr_pil = Image.open(img_path).convert('RGB')
            # Ensure 1024x1024 target size
            if hr_pil.size != (1024, 1024):
                hr_pil = hr_pil.resize((1024, 1024), Image.BICUBIC)

            hr_np = np.array(hr_pil)

            # Generate LR (256x256)
            lr_pil = lr_resize(hr_pil)
            
            # Baseline Bicubic (1024x1024)
            bic_pil = bicubic_upscale(lr_pil)
            bic_np = np.array(bic_pil)

            # Metrics for Bicubic
            psnr_bic = float(psnr_fn(hr_np, bic_np, data_range=255))
            ssim_bic = float(ssim_fn(hr_np, bic_np, data_range=255, channel_axis=2))

            # Model Inference
            lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                sr_tensor = model(lr_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Convert to numpy uint8
            sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

            # Metrics for Model
            psnr_model = float(psnr_fn(hr_np, sr_np, data_range=255))
            ssim_model = float(ssim_fn(hr_np, sr_np, data_range=255, channel_axis=2))
            psnr_gain = float(psnr_model - psnr_bic)
            mse_val = float(np.mean((hr_np.astype(np.float64) / 255.0 - sr_np.astype(np.float64) / 255.0) ** 2))

            # Optional save image
            if save_images_dir:
                out_path = os.path.join(save_images_dir, f"sr_{filename}")
                Image.fromarray(sr_np).save(out_path)

            results.append({
                "source_path": img_path,
                "dataset": dataset_name,
                "filename": filename,
                "status": "ok",
                "resolution": f"{hr_np.shape[1]}x{hr_np.shape[0]}",
                "scale_factor": upscale_factor,
                "latency_ms": round(latency_ms, 3),
                "psnr_bicubic_db": round(psnr_bic, 3),
                "ssim_bicubic": round(ssim_bic, 4),
                "psnr_model_db": round(psnr_model, 3),
                "ssim_model": round(ssim_model, 4),
                "psnr_gain_db": round(psnr_gain, 3),
                "mse": round(mse_val, 6),
                "mean_intensity": round(float(np.mean(sr_np)), 2),
                "std_contrast": round(float(np.std(sr_np)), 2)
            })

        except Exception as e:
            results.append({
                "source_path": img_path,
                "dataset": dataset_name,
                "filename": filename,
                "status": f"error: {str(e)}"
            })

    total_wall_time_sec = time.perf_counter() - start_wall_time
    ok_results = [r for r in results if r.get("status") == "ok"]

    if ok_results:
        psnr_bics = [r["psnr_bicubic_db"] for r in ok_results]
        ssim_bics = [r["ssim_bicubic"] for r in ok_results]
        psnr_mods = [r["psnr_model_db"] for r in ok_results]
        ssim_mods = [r["ssim_model"] for r in ok_results]
        gains     = [r["psnr_gain_db"] for r in ok_results]
        latencies = [r["latency_ms"] for r in ok_results]

        summary = {
            "model": f"Swift-SRGAN Generator 4x ({model_type})",
            "weights_source": os.path.basename(weights_path),
            "device": str(device),
            "images_evaluated": len(ok_results),
            "images_error": len(results) - len(ok_results),
            "scale_factor": upscale_factor,
            "avg_psnr_bicubic_db": round(float(np.mean(psnr_bics)), 3),
            "std_psnr_bicubic_db": round(float(np.std(psnr_bics)), 3),
            "avg_ssim_bicubic": round(float(np.mean(ssim_bics)), 4),
            "avg_psnr_model_db": round(float(np.mean(psnr_mods)), 3),
            "std_psnr_model_db": round(float(np.std(psnr_mods)), 3),
            "avg_ssim_model": round(float(np.mean(ssim_mods)), 4),
            "avg_psnr_gain_db": round(float(np.mean(gains)), 3),
            "avg_latency_ms": round(float(np.mean(latencies)), 3),
            "elapsed_sec_total": round(total_wall_time_sec, 3),
            "wall_time_min": round(total_wall_time_sec / 60.0, 2)
        }
    else:
        summary = {"error": "No valid results"}

    # Export JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "elapsed_sec_accumulated": round(total_wall_time_sec, 3),
            "summary": summary,
            "per_image_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Benchmark JSON saved to: {output_json}")

    # Export CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df = pd.DataFrame(ok_results)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Benchmark CSV saved to: {output_csv}")

    # Print summary
    print("\n" + "=" * 65)
    print("                 BENCHMARK SUMMARY RESULTS                ")
    print("=" * 65)
    for k, v in summary.items():
        print(f" {k:<25s}: {v}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Medical Super-Resolution Benchmark (Bicubic vs Swift-SRGAN)")
    parser.add_argument('--data_dir', type=str, default='./eval_images', help='Path to images directory')
    parser.add_argument('--weights', type=str, default='./models/netG_4x_epoch5.pth.tar', help='Path to weights (.pth.tar or .txt)')
    parser.add_argument('--output_json', type=str, default='./logs/srgan_benchmark_results.json', help='Output JSON path')
    parser.add_argument('--output_csv', type=str, default='./logs/srgan_benchmark_results.csv', help='Output CSV path')
    parser.add_argument('--save_images_dir', type=str, default=None, help='Directory to save generated SR images')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor (default: 4)')
    parser.add_argument('--device', type=str, default='auto', help="Device ('cuda', 'cpu', 'auto')")
    parser.add_argument('--max_images', type=int, default=None, help='Limit number of images')

    args = parser.parse_args()
    run_benchmark(
        data_dir=args.data_dir,
        weights_path=args.weights,
        output_json=args.output_json,
        output_csv=args.output_csv,
        save_images_dir=args.save_images_dir,
        upscale_factor=args.upscale_factor,
        device_str=args.device,
        max_images=args.max_images
    )
