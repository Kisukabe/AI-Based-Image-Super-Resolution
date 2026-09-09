#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_bicubic.py — Đánh giá Mốc Cơ Sở Bicubic Baseline Đa Tỉ Lệ (2×, 3×, 4×)
Thư mục: code hardware/

Hỗ trợ chạy CLI:
    python "code hardware/benchmark_bicubic.py" --scales 2 3 4 --max_images 2200
"""

import os
import sys
import time
import json
import copy
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve2d

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CẤU TRÚC MẠNG PHẦN CỨNG (SWIFT-SRGAN GENERATOR)
# ─────────────────────────────────────────────────────────────────────────────

class SeperableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SeperableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, groups=in_channels, bias=bias, padding=padding
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=True, discriminator=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.cnn = SeperableConv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        res = self.bn(self.cnn(x))
        return self.act(res) if self.use_act else res

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = SeperableConv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)

    def forward(self, x):
        return self.block2(self.block1(x)) + x

class SwiftSRGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_blocks: int = 16, upscale_factor: int = 4):
        super(SwiftSRGANGenerator, self).__init__()
        self.upscale_factor = upscale_factor
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residual = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)

        if upscale_factor == 2:
            self.upsampler = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2))
        elif upscale_factor == 3:
            self.upsampler = nn.Sequential(UpsampleBlock(num_channels, scale_factor=3))
        elif upscale_factor == 4:
            self.upsampler = nn.Sequential(
                UpsampleBlock(num_channels, scale_factor=2),
                UpsampleBlock(num_channels, scale_factor=2)
            )
        else:
            self.upsampler = nn.Sequential(*[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor // 2)])
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        init = self.initial(x)
        x = self.convblock(self.residual(init)) + init
        x = self.upsampler(x)
        return (torch.tanh(self.final_conv(x)) + 1.0) / 2.0

# ─────────────────────────────────────────────────────────────────────────────
# 2. CÁC HÀM TÍNH TOÁN 7 CHỈ SỐ KHOA HỌC
# ─────────────────────────────────────────────────────────────────────────────

def compute_epi(hr_np, sr_np):
    hr_gray = np.array(Image.fromarray(hr_np).convert('L'), dtype=np.float64)
    sr_gray = np.array(Image.fromarray(sr_np).convert('L'), dtype=np.float64)
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    d_hr = convolve2d(hr_gray, lap, mode='same', boundary='symm'); d_hr -= np.mean(d_hr)
    d_sr = convolve2d(sr_gray, lap, mode='same', boundary='symm'); d_sr -= np.mean(d_sr)
    denom = np.sqrt(np.sum(d_hr**2) * np.sum(d_sr**2)) + 1e-10
    return float(np.sum(d_hr * d_sr) / denom)

def compute_metrics_record(hr_np, sr_np, bic_np, hr_tensor, sr_tensor, bic_tensor,
                           latency_ms, img_path, dataset_name, filename, scale_val,
                           ms_ssim_fn, lpips_fn, niqe_fn, saved_path=None):
    # 1. PSNR, MSE, RMSE
    psnr_bic = float(psnr_fn(hr_np, bic_np, data_range=255))
    mse_bic  = float(np.mean((hr_np.astype(np.float64) - bic_np.astype(np.float64))**2))
    rmse_bic = float(np.sqrt(mse_bic))

    psnr_sr  = float(psnr_fn(hr_np, sr_np, data_range=255)) if sr_np is not None else psnr_bic
    mse_sr   = float(np.mean((hr_np.astype(np.float64) - sr_np.astype(np.float64))**2)) if sr_np is not None else mse_bic
    rmse_sr  = float(np.sqrt(mse_sr))

    # 2. SSIM
    ssim_bic = float(ssim_fn(hr_np, bic_np, data_range=255, channel_axis=2))
    ssim_sr  = float(ssim_fn(hr_np, sr_np, data_range=255, channel_axis=2)) if sr_np is not None else ssim_bic

    # 3. MS-SSIM
    msssim_bic, msssim_sr = None, None
    if ms_ssim_fn is not None:
        try:
            with torch.no_grad():
                msssim_bic = float(ms_ssim_fn(hr_tensor, bic_tensor, data_range=1.0).item())
                msssim_sr  = float(ms_ssim_fn(hr_tensor, sr_tensor, data_range=1.0).item()) if sr_tensor is not None else msssim_bic
        except Exception:
            msssim_bic, msssim_sr = ssim_bic, ssim_sr

    # 4. LPIPS
    lpips_bic, lpips_sr = None, None
    if lpips_fn is not None:
        try:
            with torch.no_grad():
                hr_n  = (hr_tensor * 2.0) - 1.0
                bic_n = (bic_tensor * 2.0) - 1.0
                lpips_bic = float(lpips_fn(hr_n, bic_n).item())
                lpips_sr  = float(lpips_fn(hr_n, (sr_tensor * 2.0) - 1.0).item()) if sr_tensor is not None else lpips_bic
        except Exception:
            pass

    # 5. NIQE
    niqe_bic, niqe_sr = None, None
    if niqe_fn is not None:
        try:
            with torch.no_grad():
                niqe_bic = float(niqe_fn(bic_tensor).item())
                niqe_sr  = float(niqe_fn(sr_tensor).item()) if sr_tensor is not None else niqe_bic
        except Exception:
            pass

    # 6. EPI & Mean/STD
    eval_target = sr_np if sr_np is not None else bic_np
    epi_val  = float(compute_epi(hr_np, eval_target))
    mean_val = float(np.mean(eval_target))
    std_val  = float(np.std(eval_target))

    # 7. Gains
    psnr_gain   = round(psnr_sr - psnr_bic, 3)
    msssim_gain = round(msssim_sr - msssim_bic, 4) if msssim_sr and msssim_bic else 0.0
    lpips_gain  = round(lpips_bic - lpips_sr, 4) if lpips_bic and lpips_sr else 0.0
    niqe_gain   = round(niqe_bic - niqe_sr, 4) if niqe_bic and niqe_sr else 0.0

    return {
        "source_path":      img_path,
        "saved_path":       saved_path,
        "dataset":          dataset_name,
        "filename":         filename,
        "status":           "ok",
        "resolution":       f"{hr_np.shape[1]}x{hr_np.shape[0]}",
        "scale_factor":     scale_val,
        "patches_count":    1,
        "latency_ms":       round(latency_ms, 2),

        # Bicubic
        "psnr_bicubic_db":  round(psnr_bic, 3),
        "mse_bicubic":      round(mse_bic, 4),
        "rmse_bicubic":     round(rmse_bic, 4),
        "ssim_bicubic":     round(ssim_bic, 4),
        "msssim_bicubic":   round(msssim_bic, 4) if msssim_bic is not None else None,
        "lpips_bicubic":    round(lpips_bic, 4) if lpips_bic is not None else None,
        "niqe_bicubic":     round(niqe_bic, 4) if niqe_bic is not None else None,

        # FPGA / Model Metrics
        "psnr_fpga_db":     round(psnr_sr, 3),
        "mse_fpga":         round(mse_sr, 4),
        "rmse_fpga":        round(rmse_sr, 4),
        "ssim_fpga":        round(ssim_sr, 4),
        "msssim_fpga":      round(msssim_sr, 4) if msssim_sr is not None else None,
        "lpips_fpga":       round(lpips_sr, 4) if lpips_sr is not None else None,
        "lpips":            round(lpips_sr, 4) if lpips_sr is not None else None,
        "lpips_srgan":      round(lpips_sr, 4) if lpips_sr is not None else None,
        "niqe_fpga":        round(niqe_sr, 4) if niqe_sr is not None else None,
        "mean_fpga":        round(mean_val, 2),
        "std_fpga":         round(std_val, 2),
        "mean":             round(mean_val, 2),
        "std":              round(std_val, 2),
        "epi":              round(epi_val, 4),

        # Aliases
        "psnr_model_db":    round(psnr_sr, 3),
        "ssim_model":       round(ssim_sr, 4),
        "mse_model":        round(mse_sr, 4),
        "rmse_model":       round(rmse_sr, 4),

        # Gains
        "psnr_gain_db":     psnr_gain,
        "msssim_gain":      msssim_gain,
        "lpips_gain":       lpips_gain,
        "niqe_gain":        niqe_gain
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. QUÉT DATASET CHUẨN
# ─────────────────────────────────────────────────────────────────────────────

def find_sub_xray_images(data_dir=None):
    valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    search_dirs = [data_dir] if data_dir else [
        '/kaggle/input/datasets/duc24kdl/sub-x-ray/sub_X-Ray',
        '/kaggle/input/sub-x-ray/sub_X-Ray',
        'code software/eval_images',
        'eval_images',
        'data'
    ]

    nih_imgs, chest_imgs = [], []
    for cand in search_dirs:
        if cand and os.path.exists(cand):
            nih_p = os.path.join(cand, 'sub_NIH')
            chest_p = os.path.join(cand, 'sub_chest')
            if os.path.exists(nih_p):
                nih_imgs = [os.path.join(nih_p, f) for f in sorted(os.listdir(nih_p)) if f.endswith(valid_exts)]
            if os.path.exists(chest_p):
                chest_imgs = [os.path.join(chest_p, f) for f in sorted(os.listdir(chest_p)) if f.endswith(valid_exts)]
            if nih_imgs or chest_imgs:
                print(f"✓ Đã phát hiện dataset tại: {cand}")
                break

    if not nih_imgs and not chest_imgs:
        for search_base in ['/kaggle/input', '.']:
            if os.path.exists(search_base):
                for root, dirs, files in os.walk(search_base):
                    bname = os.path.basename(root)
                    if bname == 'sub_NIH' and not nih_imgs:
                        nih_imgs = [os.path.join(root, f) for f in sorted(files) if f.endswith(valid_exts)]
                    elif bname == 'sub_chest' and not chest_imgs:
                        chest_imgs = [os.path.join(root, f) for f in sorted(files) if f.endswith(valid_exts)]

    return nih_imgs, chest_imgs

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Bicubic & SwiftSRGAN Đa Tỉ Lệ (2x, 3x, 4x)")
    parser.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4], help="Danh sách các scale cần chạy (vd: 2 3 4)")
    parser.add_argument('--data_dir', type=str, default=None, help="Đường dẫn thư mục sub_X-Ray")
    parser.add_argument('--max_images', type=int, default=2200, help="Số lượng ảnh tối đa")
    parser.add_argument('--device', type=str, default=None, help="Device: cuda hoặc cpu")
    parser.add_argument('--out_dir', type=str, default='./output_bicubic_benchmark', help="Thư mục xuất kết quả")
    args = parser.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    os.makedirs(args.out_dir, exist_ok=True)

    print("═" * 70)
    print(f"🚀 BẮT ĐẦU BENCHMARK BICUBIC BASELINE ĐA TỈ LỆ: {args.scales}")
    print(f"  • Thiết bị         : {device_str.upper()}")
    print(f"  • Thư mục xuất file: {args.out_dir}")
    print("═" * 70)

    # Khởi tạo LPIPS, MS-SSIM, NIQE
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    except Exception:
        lpips_fn = None

    try:
        from pytorch_msssim import ms_ssim
        ms_ssim_fn = ms_ssim
    except Exception:
        ms_ssim_fn = None

    try:
        import pyiqa
        niqe_fn = pyiqa.create_metric('niqe', device=device)
    except Exception:
        niqe_fn = None

    nih_imgs, chest_imgs = find_sub_xray_images(args.data_dir)
    all_imgs = nih_imgs + chest_imgs
    if not all_imgs:
        print("❌ Không tìm thấy ảnh nào! Vui lòng chỉ định --data_dir.")
        return

    imgs_to_run = all_imgs[:args.max_images] if args.max_images else all_imgs
    print(f"✓ Quét được {len(all_imgs):,} ảnh (sub_NIH: {len(nih_imgs):,}, sub_chest: {len(chest_imgs):,}). Sẽ chạy {len(imgs_to_run):,} ảnh.")

    HR_SIZE = (1024, 1024)
    all_summaries = []

    for sc in args.scales:
        lr_dim = HR_SIZE[0] // sc
        lr_trans = transforms.Resize((lr_dim, lr_dim), interpolation=Image.BICUBIC)
        bic_trans = transforms.Resize(HR_SIZE, interpolation=Image.BICUBIC)

        print(f"\n--- BẮT ĐẦU SCALE {sc}× (LR {lr_dim}x{lr_dim} -> HR 1024x1024) ---")
        results = []
        t_start = time.perf_counter()

        for idx, img_path in enumerate(tqdm(imgs_to_run, desc=f"Scale {sc}x")):
            ds_name = os.path.basename(os.path.dirname(img_path)) or "unknown"
            fname = os.path.basename(img_path)
            try:
                hr_pil = Image.open(img_path).convert('RGB')
                if hr_pil.size != HR_SIZE:
                    hr_pil = hr_pil.resize(HR_SIZE, Image.BICUBIC)
                hr_np = np.array(hr_pil)
                hr_tensor = to_tensor(hr_pil).unsqueeze(0).to(device)

                lr_pil = lr_trans(hr_pil)
                t0 = time.perf_counter()
                bic_pil = bic_trans(lr_pil)
                if device_str == 'cuda': torch.cuda.synchronize()
                lat_ms = (time.perf_counter() - t0) * 1000.0

                bic_np = np.array(bic_pil)
                bic_tensor = to_tensor(bic_pil).unsqueeze(0).to(device)

                rec = compute_metrics_record(
                    hr_np, bic_np, bic_np, hr_tensor, bic_tensor, bic_tensor,
                    lat_ms, img_path, ds_name, fname, sc,
                    ms_ssim_fn, lpips_fn, niqe_fn
                )
                results.append(rec)
            except Exception as e:
                results.append({"source_path": img_path, "dataset": ds_name, "filename": fname, "scale_factor": sc, "status": f"error: {e}"})

        wall_time = time.perf_counter() - t_start
        ok_res = [r for r in results if r.get('status') == 'ok']

        # Lưu JSON và CSV
        json_file = os.path.join(args.out_dir, f'bicubic_{sc}x_benchmark.json')
        csv_file = os.path.join(args.out_dir, f'bicubic_{sc}x_benchmark.csv')

        avg_p = float(np.mean([r['psnr_bicubic_db'] for r in ok_res])) if ok_res else 0.0
        avg_s = float(np.mean([r['ssim_bicubic'] for r in ok_res])) if ok_res else 0.0
        avg_lp = float(np.mean([r['lpips_bicubic'] for r in ok_res if r.get('lpips_bicubic') is not None])) if any(r.get('lpips_bicubic') is not None for r in ok_res) else None
        avg_nq = float(np.mean([r['niqe_bicubic'] for r in ok_res if r.get('niqe_bicubic') is not None])) if any(r.get('niqe_bicubic') is not None for r in ok_res) else None
        avg_lat = float(np.mean([r['latency_ms'] for r in ok_res])) if ok_res else 0.0

        summary = {
            "scale_factor": sc,
            "resolution_in_out": f"{lr_dim}x{lr_dim} -> 1024x1024",
            "images_evaluated": len(ok_res),
            "avg_psnr_db": round(avg_p, 3),
            "avg_ssim": round(avg_s, 4),
            "avg_lpips": round(avg_lp, 4) if avg_lp else None,
            "avg_niqe": round(avg_nq, 2) if avg_nq else None,
            "avg_latency_ms": round(avg_lat, 2),
            "fps": round(1000.0 / avg_lat, 1) if avg_lat > 0 else 0.0,
            "elapsed_sec": round(wall_time, 2)
        }
        all_summaries.append(summary)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary, "per_image_results": results}, f, indent=2, ensure_ascii=False)
        pd.DataFrame(ok_res).to_csv(csv_file, index=False)

        print(f"✓ Hoàn tất scale {sc}x: PSNR = {avg_p:.2f}dB, SSIM = {avg_s:.4f} -> Đã lưu: {json_file}")

    # Bảng tổng hợp
    sum_df = pd.DataFrame(all_summaries)
    sum_df.to_csv(os.path.join(args.out_dir, 'bicubic_multiscale_summary.csv'), index=False)
    with open(os.path.join(args.out_dir, 'bicubic_multiscale_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 70)
    print("📊 BẢNG TỔNG HỢP SO SÁNH CÁC SCALE:")
    print(sum_df[['scale_factor', 'resolution_in_out', 'avg_psnr_db', 'avg_ssim', 'avg_lpips', 'avg_niqe', 'avg_latency_ms', 'fps']].to_string(index=False))
    print("═" * 70)

if __name__ == '__main__':
    main()
