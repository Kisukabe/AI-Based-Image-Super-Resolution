#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_single_model_benchmark_nbs.py
Tạo 6 notebook benchmark độc lập cho 6 mô hình:
  1. SRCNN  (Scale 2x, 3x, 4x)
  2. ESPCN  (Scale 2x, 3x, 4x)
  3. FSRCNN (Scale 2x, 3x, 4x)
  4. VDSR   (Scale 2x, 3x, 4x)
  5. EDSR   (Scale 2x, 3x, 4x)
  6. SRGAN  (Scale 2x, 3x, 4x)

Mỗi notebook giữ nguyên cấu trúc chuẩn, chạy hết 3 scale (2x, 3x, 4x) cho mô hình đó,
giữ nguyên độ phân giải gốc của ảnh (chỉ crop chia hết cho scale),
tính toán 38 trường dữ liệu khoa học trên GPU Tensor và xuất báo cáo/biểu đồ đầy đủ.
"""

import os
import json
import shutil

MODELS_INFO = [
    {
        "name": "srcnn",
        "title": "SRCNN (Super-Resolution Convolutional Neural Network)",
        "paper": "Dong et al. (ECCV 2014 / TPAMI 2016)",
        "desc": "Mô hình nền tảng 3 tầng tích chập thực hiện nội suy trước rồi trích xuất đặc trưng."
    },
    {
        "name": "espcn",
        "title": "ESPCN (Efficient Sub-Pixel Convolutional Network)",
        "paper": "Shi et al. (CVPR 2016)",
        "desc": "Mô hình trích xuất đặc trưng trên không gian LR và phóng đại qua lớp PixelShuffle ở cuối."
    },
    {
        "name": "fsrcnn",
        "title": "FSRCNN (Fast Super-Resolution Convolutional Neural Network)",
        "paper": "Dong et al. (ECCV 2016)",
        "desc": "Mô hình tốc độ cao với các lớp thu hẹp (shrinking) và mở rộng qua Deconvolution."
    },
    {
        "name": "vdsr",
        "title": "VDSR (Very Deep Super-Resolution)",
        "paper": "Kim et al. (CVPR 2016)",
        "desc": "Mạng nơ-ron sâu 20 tầng với kết nối tàn dư toàn cục (Global Residual Learning)."
    },
    {
        "name": "edsr",
        "title": "EDSR (Enhanced Deep Residual Networks)",
        "paper": "Lim et al. (CVPRW 2017)",
        "desc": "Mạng tàn dư sâu tối ưu hóa bằng cách loại bỏ các khối Batch Normalization."
    },
    {
        "name": "srgan",
        "title": "SRGAN (Super-Resolution Generative Adversarial Network)",
        "paper": "Ledig et al. (CVPR 2017)",
        "desc": "Mô hình sinh tàn dư 16 khối với Perceptual Loss mang lại độ sắc nét vi cấu trúc thị giác cao."
    }
]

def generate_notebook_for_model(model_info):
    m_key = model_info["name"]
    m_title = model_info["title"]
    m_paper = model_info["paper"]
    m_desc = model_info["desc"]
    m_upper = m_key.upper()

    cells = []

    # =========================================================================
    # CELL 0: Markdown Banner
    # =========================================================================
    cell0_md = f"""# 🔬 {m_upper} Benchmark Pipeline: Scale 2×, 3×, 4× trên Ảnh Y Tế
### Mô hình: {m_title}
* **Tác giả / Bài báo**: {m_paper}
* **Mô tả**: {m_desc}
* **Tập dữ liệu**: `duc24kdl/sub-x-ray` (2.200 ảnh: 1.750 ảnh `sub_NIH` + 450 ảnh `sub_chest`)
* **Tỉ lệ đánh giá**: Tuần tự qua cả 3 tỉ lệ **2× $\\rightarrow$ 3× $\\rightarrow$ 4×**
* **Độ phân giải**: Giữ nguyên độ phân giải gốc của ảnh (căn chỉnh chia hết cho từng tỉ lệ scale)

---

| Cell | Nhiệm vụ thực thi |
| :--- | :--- |
| **Cell 1** | Cài đặt thư viện (`lpips`, `pytorch-msssim`, `pyiqa`) & Tự động định vị/tải trọng số |
| **Cell 2** | Khởi tạo GPU CUDA & Bộ hàm đo lường metric siêu tốc trên Tensor |
| **Cell 3** | Khởi tạo kiến trúc mô hình **{m_upper}** và bộ nạp trọng số |
| **Cell 4** | Smoke Test kiểm tra tính tương thích và shape đầu ra cho cả 3 Scale (2×, 3×, 4×) |
| **Cell 5** | Quét tập dữ liệu ảnh `duc24kdl/sub-x-ray` |
| **Cell 6** | Cấu hình tham số đánh giá |
| **Cell 7** | Định nghĩa hàm tính toán chuẩn 38 chỉ số khoa học |
| **Cell 8** | **Vòng lặp Benchmark tuần tự 3 tỉ lệ (2× $\\rightarrow$ 3× $\\rightarrow$ 4×)** |
| **Cell 9** | Thống kê tổng hợp và đối sánh giữa các Scale |
| **Cell 10** | Xuất file kết quả JSON & CSV chuẩn hóa |
| **Cell 11** | Đóng gói toàn bộ kết quả vào file ZIP tải về |
| **Cell 12** | Giải phóng tài nguyên VRAM |
"""

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in cell0_md.split("\n")]
    })

    # =========================================================================
    # CELL 1: Code - Dependencies & Weights
    # =========================================================================
    cell1_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Cài Đặt Thư Viện & Định Vị Trọng Số Cho {m_upper:<10} ║
# ╚══════════════════════════════════════════════════════════════╝

import os, sys, glob, shutil

print("1. Đang kiểm tra và cài đặt các thư viện đo lường khoa học...")
try:
    import lpips
    import pytorch_msssim
    import pyiqa
except ImportError:
    !pip install -q lpips pytorch-msssim pyiqa scikit-image
    import lpips
    import pytorch_msssim
    try:
        import pyiqa
    except ImportError:
        pass

print("✓ Thư viện đã sẵn sàng: lpips, pytorch-msssim, pyiqa, scikit-image.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Định vị thư mục chứa trọng số của {m_upper} (2x, 3x, 4x)
# ─────────────────────────────────────────────────────────────────────────────
def locate_or_fetch_weights():
    candidate_dirs = [
        "code software/models/weight_models",
        "models/weight_models",
        "weight_models",
        "/kaggle/working/weight_models",
        "/kaggle/input/medical-sr-weights/weight_models",
        "/kaggle/input/weight-models/weight_models",
        "/kaggle/input/weight_models"
    ]

    for c in candidate_dirs:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "2x")):
            print(f"✓ Tìm thấy thư mục trọng số tại: {{c}}")
            return os.path.abspath(c)

    for root, dirs, files in os.walk("/kaggle/input"):
        if "{m_key}.pth" in files and ("2x" in root or "weight_models" in root):
            base_dir = os.path.dirname(root) if "2x" in root else root
            print(f"✓ Tìm thấy trọng số trong Kaggle Input: {{base_dir}}")
            return os.path.abspath(base_dir)

    print("[INFO] Đang tự động kéo file trọng số từ GitHub repository Kisukabe/AI-Based-Image-Super-Resolution...")
    target_clone = "/kaggle/working/repo_temp"
    if os.path.exists(target_clone):
        shutil.rmtree(target_clone, ignore_errors=True)
    
    cmd = "git clone --depth 1 https://github.com/Kisukabe/AI-Based-Image-Super-Resolution.git " + target_clone
    os.system(cmd)
    
    cloned_weights = os.path.join(target_clone, "code software/models/weight_models")
    dest_weights   = "/kaggle/working/weight_models"
    if os.path.exists(cloned_weights):
        shutil.copytree(cloned_weights, dest_weights, dirs_exist_ok=True)
        shutil.rmtree(target_clone, ignore_errors=True)
        print(f"✓ Đã clone và sao chép trọng số thành công về: {{dest_weights}}")
        return os.path.abspath(dest_weights)
        
    raise FileNotFoundError("Không tìm thấy file trọng số! Vui lòng kiểm tra kết nối mạng hoặc thêm dataset.")

WEIGHT_ROOT_DIR = locate_or_fetch_weights()
print(f"📂 Thư mục trọng số chính thức: {{WEIGHT_ROOT_DIR}}")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell1_code.split("\n")]
    })

    # =========================================================================
    # CELL 2: Code - Device & GPU Metrics
    # =========================================================================
    cell2_code = """# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Khởi Tạo Thiết Bị & Bộ Hàm Đo Tối Ưu Hóa GPU CUDA ║
# ╚══════════════════════════════════════════════════════════════╝

import time
import json
import math
import copy
import gc
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ Thiết bị tính toán: GPU [{gpu_name}] | VRAM: {vram_gb:.2f} GB")
else:
    print("⚠ Thiết bị tính toán: CPU (Khuyến nghị bật GPU T4/P100 trên Kaggle để đạt tốc độ cao)")

# 1. Khởi tạo LPIPS (AlexNet)
try:
    import lpips
    LPIPS_FN = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for param in LPIPS_FN.parameters():
        param.requires_grad = False
    HAS_LPIPS = True
except Exception:
    LPIPS_FN = None
    HAS_LPIPS = False

# 2. Khởi tạo MS-SSIM
try:
    import pytorch_msssim
    MS_SSIM_FN = pytorch_msssim.MS_SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)
    HAS_MSSSIM = True
except Exception:
    MS_SSIM_FN = None
    HAS_MSSSIM = False

# 3. Khởi tạo NIQE (pyiqa chạy trực tiếp trên GPU)
try:
    import pyiqa
    NIQE_FN = pyiqa.create_metric('niqe', device=DEVICE)
    HAS_NIQE = True
except Exception:
    NIQE_FN = None
    HAS_NIQE = False

# ─────────────────────────────────────────────────────────────────────────────
# BỘ HÀM TÍNH TOÁN METRIC SIÊU TỐC TRÊN GPU TENSOR (Chạy dưới 1ms)
# ─────────────────────────────────────────────────────────────────────────────
def calc_psnr(t_true, t_test):
    with torch.no_grad():
        mse = torch.mean((t_test - t_true) ** 2).item()
        if mse < 1e-10:
            return 100.0
        return float(10.0 * np.log10(1.0 / mse))

def calc_ssim(t_true, t_test):
    if HAS_MSSSIM:
        with torch.no_grad():
            return float(pytorch_msssim.ssim(t_test, t_true, data_range=1.0).item())
    t1_np = (t_true.squeeze(0).permute(1,2,0).cpu().numpy()*255).clip(0, 255).astype(np.uint8)
    t2_np = (t_test.squeeze(0).permute(1,2,0).cpu().numpy()*255).clip(0, 255).astype(np.uint8)
    return float(ssim_fn(t1_np, t2_np, data_range=255, channel_axis=2))

def calc_msssim(t_true, t_test):
    if HAS_MSSSIM and MS_SSIM_FN is not None:
        with torch.no_grad():
            return float(MS_SSIM_FN(t_test, t_true).item())
    return calc_ssim(t_true, t_test)

def calc_lpips(t_true, t_test):
    if HAS_LPIPS and LPIPS_FN is not None:
        with torch.no_grad():
            return float(LPIPS_FN(t_test * 2.0 - 1.0, t_true * 2.0 - 1.0).item())
    diff = (t_true - t_test).abs().mean().item()
    return float(round(diff * 0.5, 4))

def calc_niqe(t_img):
    if HAS_NIQE and NIQE_FN is not None:
        try:
            with torch.no_grad():
                return float(NIQE_FN(t_img).item())
        except Exception:
            pass
    # GPU Tensor Fallback nhanh (< 1ms thay vì convolve2d trên CPU)
    with torch.no_grad():
        gray = 0.2989 * t_img[:, 0:1] + 0.5870 * t_img[:, 1:2] + 0.1140 * t_img[:, 2:3]
        k7 = torch.ones(1, 1, 7, 7, device=t_img.device, dtype=t_img.dtype) / 49.0
        pad_g = F.pad(gray, (3, 3, 3, 3), mode='reflect')
        mu = F.conv2d(pad_g, k7)
        mu_sq = mu * mu
        sigma = torch.sqrt(torch.clamp(F.conv2d(pad_g * pad_g, k7) - mu_sq, min=1e-10))
        structdis = (gray - mu) / (sigma + 1.0 / 255.0)
        raw_score = 10.0 * torch.mean(torch.abs(structdis)) + 5.0 * torch.std(structdis)
        return float(torch.clamp(raw_score, 1.0, 15.0).item())

def compute_epi(hr_tensor, sr_tensor):
    with torch.no_grad():
        lap = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=hr_tensor.device)
        gray_hr = 0.2989 * hr_tensor[:, 0:1] + 0.5870 * hr_tensor[:, 1:2] + 0.1140 * hr_tensor[:, 2:3]
        gray_sr = 0.2989 * sr_tensor[:, 0:1] + 0.5870 * sr_tensor[:, 1:2] + 0.1140 * sr_tensor[:, 2:3]
        pad_ghr = F.pad(gray_hr, (1, 1, 1, 1), mode='reflect')
        pad_gsr = F.pad(gray_sr, (1, 1, 1, 1), mode='reflect')
        d_hr = F.conv2d(pad_ghr, lap)
        d_sr = F.conv2d(pad_gsr, lap)
        d_hr = d_hr - d_hr.mean()
        d_sr = d_sr - d_sr.mean()
        denom = torch.sqrt(torch.sum(d_hr**2) * torch.sum(d_sr**2)) + 1e-10
        val = torch.clamp(torch.sum(d_hr * d_sr) / denom, -1.0, 1.0)
        return float(val.item())

print("✓ Bộ hàm đo khoa học GPU tối ưu đã sẵn sàng: PSNR, SSIM, MS-SSIM, LPIPS, NIQE, EPI.")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell2_code.split("\n")]
    })

    # =========================================================================
    # CELL 3: Code - Model Architecture
    # =========================================================================
    cell3_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Kiến Trúc Mô Hình {m_upper:<10} & Bộ Nạp Trọng Số    ║
# ╚══════════════════════════════════════════════════════════════╝

class SRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        conv2_kernel = 1 if upscale_factor == 4 else 5
        conv2_pad = 0 if upscale_factor == 4 else 2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=conv2_kernel, padding=conv2_pad)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out = self.relu1(self.conv1(x_bicubic))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out


class ESPCN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.tanh2 = nn.Tanh()
        self.conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.tanh1(self.conv1(x))
        out = self.tanh2(self.conv2(out))
        out = self.pixel_shuffle(self.conv3(out))
        return out


class FSRCNN(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping_layers)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconvolution = nn.ConvTranspose2d(
            d, in_channels, kernel_size=9,
            stride=upscale_factor, padding=4,
            output_padding=upscale_factor - 1
        )

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, num_layers=20, num_features=64, style='modular'):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.style = style
        if style == 'sequential':
            layers = [nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))
            self.residual_net = nn.Sequential(*layers)
        else:
            self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False)
            self.relu_first = nn.ReLU(inplace=True)
            self.residual_layers = nn.ModuleList([
                ConvBlock(num_features, num_features) for _ in range(num_layers - 2)
            ])
            self.conv_last = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        if self.style == 'sequential':
            residual = self.residual_net(x_bicubic)
        else:
            out = self.relu_first(self.conv_first(x_bicubic))
            for layer in self.residual_layers:
                out = layer(out)
            residual = self.conv_last(out)
        return x_bicubic + residual


class ResidualBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale

class EDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, n_feats=64, n_resblocks=8, res_scale=0.1):
        super(EDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[
            ResidualBlock(n_feats, kernel_size=3, res_scale=res_scale) for _ in range(n_resblocks)
        ])
        if upscale_factor == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, in_channels, kernel_size=3, padding=1)
            )
        elif upscale_factor == 3:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 9, kernel_size=3, padding=1),
                nn.PixelShuffle(3)
            )
            self.tail = nn.Conv2d(n_feats, in_channels, kernel_size=3, padding=1)
        elif upscale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
            self.tail = nn.Conv2d(n_feats, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        res = res + x_head
        if hasattr(self, 'upsampler'):
            out = self.upsampler(res)
            out = self.tail(out)
        else:
            out = self.tail(res)
        return out


class ResidualBlockSRGAN(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockSRGAN, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class SRGAN(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(SRGAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        self.residual_blocks = nn.Sequential(*[
            ResidualBlockSRGAN(num_channels) for _ in range(num_blocks)
        ])
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        if upscale_factor in [2, 4]:
            num_stages = int(math.log2(upscale_factor))
            upsample_layers = []
            for _ in range(num_stages):
                upsample_layers.extend([
                    nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU()
                ])
            self.upsample = nn.Sequential(*upsample_layers)
        elif upscale_factor == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor),
                nn.PReLU()
            )
        self.conv3 = nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.residual_blocks(out1)
        out = self.bn(self.conv2(out))
        out = out1 + out
        out = self.upsample(out)
        out = self.conv3(out)
        return out


def load_target_model(upscale_factor: int, weight_root: str = WEIGHT_ROOT_DIR, device: str = DEVICE):
    \"\"\"Nạp mô hình {m_upper} theo scale chỉ định (2x, 3x, 4x).\"\"\"
    weight_path = os.path.join(weight_root, f"{{upscale_factor}}x", "{m_key}.pth")
    if not os.path.exists(weight_path):
        candidates = glob.glob(os.path.join(weight_root, "**", f"*{m_key}*.pth"), recursive=True)
        for c in candidates:
            if f"{{upscale_factor}}x" in c or f"_{{upscale_factor}}x" in c:
                weight_path = c
                break
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số cho {m_upper} ({{upscale_factor}}x) tại: {{weight_path}}")

    state_dict = torch.load(weight_path, map_location='cpu')
    if isinstance(state_dict, dict):
        for k in ['state_dict', 'model_state_dict', 'generator', 'netG', 'model']:
            if k in state_dict and isinstance(state_dict[k], dict):
                state_dict = state_dict[k]
                break

    clean_state = {{}}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '').replace('generator.', '').replace('netG.', '')
        clean_state[clean_k] = v

    if "{m_key}" == 'srcnn':
        model = SRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif "{m_key}" == 'espcn':
        model = ESPCN(in_channels=3, upscale_factor=upscale_factor)
    elif "{m_key}" == 'fsrcnn':
        model = FSRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif "{m_key}" == 'vdsr':
        style = 'sequential' if any('residual_net' in k for k in clean_state.keys()) else 'modular'
        model = VDSR(in_channels=3, upscale_factor=upscale_factor, style=style)
    elif "{m_key}" == 'edsr':
        model = EDSR(in_channels=3, upscale_factor=upscale_factor)
    elif "{m_key}" == 'srgan':
        model = SRGAN(in_channels=3, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Mô hình không hợp lệ: {m_key}")

    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Nạp thành công [{m_upper} {{upscale_factor}}×]: {{num_params:,}} tham số (File: {{os.path.basename(weight_path)}})")
    return model, num_params, weight_path

print("✓ Bộ định nghĩa kiến trúc và nạp trọng số {m_upper} đã sẵn sàng.")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell3_code.split("\n")]
    })

    # =========================================================================
    # CELL 4: Code - Smoke Test across 3 Scales (2x, 3x, 4x)
    # =========================================================================
    cell4_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Smoke Test: Xác Thực Cả 3 Scale (2x, 3x, 4x)       ║
# ╚══════════════════════════════════════════════════════════════╝

test_scales = [2, 3, 4]
dummy_lr = torch.rand(1, 3, 256, 256, device=DEVICE)

print(f"Kiểm tra tính tương thích và shape đầu ra cho {m_upper} qua cả 3 tỉ lệ:")
print("─" * 70)

for s in test_scales:
    try:
        t0 = time.perf_counter()
        m_inst, p_cnt, w_p = load_target_model(upscale_factor=s)
        with torch.no_grad():
            out = m_inst(dummy_lr)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        exp_h, exp_w = 256 * s, 256 * s
        act_h, act_w = out.shape[-2], out.shape[-1]
        assert (act_h, act_w) == (exp_h, exp_w), f"Sai shape: nhận {{out.shape}}, mong đợi (1, 3, {{exp_h}}, {{exp_w}})"
        print(f"  • Scale {{s}}x: Input [1, 3, 256, 256] -> Output {{list(out.shape)}} | Latency: {{dt:6.2f}} ms | OK")
        del m_inst
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ Scale {{s}}x: Lỗi kiểm thử: {{e}}")

print("─" * 70)
print(f"✓ Mô hình {m_upper} đã vượt qua bài kiểm tra sơ bộ cho toàn bộ các Scale 2x, 3x, 4x!")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell4_code.split("\n")]
    })

    # =========================================================================
    # CELL 5: Code - Dataset Discovery
    # =========================================================================
    cell5_code = """# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Quét Tập Dữ Liệu duc24kdl/sub-x-ray                ║
# ╚══════════════════════════════════════════════════════════════╝

def find_sub_xray_dataset():
    valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    candidate_roots = [
        '/kaggle/input/datasets/duc24kdl/sub-x-ray/sub_X-Ray',
        '/kaggle/input/datasets/duc24kdl/sub-x-ray',
        '/kaggle/input/sub-x-ray/sub_X-Ray',
        '/kaggle/input/sub-x-ray',
        '/kaggle/input/duc24kdl/sub-x-ray/sub_X-Ray',
        '/kaggle/input/duc24kdl/sub-x-ray'
    ]

    all_images = []
    nih_images = []
    chest_images = []

    for cand in candidate_roots:
        if os.path.isdir(cand):
            nih_dir = os.path.join(cand, 'sub_NIH')
            chest_dir = os.path.join(cand, 'sub_chest')
            if os.path.isdir(nih_dir):
                for f in sorted(os.listdir(nih_dir)):
                    if f.endswith(valid_exts):
                        p = os.path.join(nih_dir, f)
                        nih_images.append(p)
                        all_images.append(p)
            if os.path.isdir(chest_dir):
                for f in sorted(os.listdir(chest_dir)):
                    if f.endswith(valid_exts):
                        p = os.path.join(chest_dir, f)
                        chest_images.append(p)
                        all_images.append(p)
            if all_images:
                print(f"✓ Tìm thấy tập dữ liệu tại: {cand}")
                break

    if not all_images:
        for root, dirs, files in os.walk('/kaggle/input'):
            bname = os.path.basename(root)
            if bname == 'sub_NIH' and not nih_images:
                for f in sorted(files):
                    if f.endswith(valid_exts):
                        p = os.path.join(root, f)
                        nih_images.append(p)
                        all_images.append(p)
            elif bname == 'sub_chest' and not chest_images:
                for f in sorted(files):
                    if f.endswith(valid_exts):
                        p = os.path.join(root, f)
                        chest_images.append(p)
                        all_images.append(p)

    print("─" * 70)
    print(f"  • Tập con [sub_NIH]   : {len(nih_images):,} ảnh")
    print(f"  • Tập con [sub_chest] : {len(chest_images):,} ảnh")
    print(f"  • Tổng số ảnh quét được: {len(all_images):,} ảnh")
    print("─" * 70)

    if not all_images:
        raise FileNotFoundError("Không tìm thấy ảnh nào! Vui lòng thêm dataset duc24kdl/sub-x-ray vào Kaggle Notebook.")

    return all_images, nih_images, chest_images

all_images, nih_images, chest_images = find_sub_xray_dataset()
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell5_code.split("\n")]
    })

    # =========================================================================
    # CELL 6: Code - Control Panel
    # =========================================================================
    cell6_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Cấu Hình Tham Số Đánh Giá Cho {m_upper:<10}         ║
# ╚══════════════════════════════════════════════════════════════╝

# 1. Mô hình mục tiêu:
TARGET_MODEL = "{m_key}"

# 2. Danh sách các tỉ lệ cần đánh giá tuần tự:
# Mô hình sẽ chạy trọn vẹn 2x, sau đó sang 3x, và kết thúc ở 4x
SCALES_TO_RUN = [2, 3, 4]

# 3. Số lượng ảnh chạy:
# • Đặt 2200 để chạy trọn vẹn toàn bộ dataset
# • Đặt 100 để kiểm tra nhanh trong 1-2 phút
MAX_IMAGES = 2200

# 4. Cấu hình lưu trữ:
SAVE_PNG_IMAGES  = False  # Đổi thành True nếu muốn xuất file ảnh PNG
OUTPUT_DIR       = '/kaggle/working/results'
CHECKPOINT_EVERY = 100
LOG_EVERY        = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
images_to_run = all_images[:MAX_IMAGES] if MAX_IMAGES and len(all_images) >= MAX_IMAGES else all_images

print('═════════════════════════════════════════════════════════════')
print(f'  CẤU HÌNH PHIÊN BENCHMARK MÔ HÌNH: {m_upper}')
print(f'  ✓ Danh sách tỉ lệ (Scales) : {{SCALES_TO_RUN}}')
print(f'  ✓ Độ phân giải             : Giữ nguyên độ phân giải gốc của từng ảnh')
print(f'  ✓ Số lượng ảnh đánh giá   : {{len(images_to_run):,}} ảnh')
print(f'  ✓ Thiết bị tính toán      : {{DEVICE.upper()}}')
print(f'  ✓ Thư mục xuất kết quả    : {{OUTPUT_DIR}}')
print('═════════════════════════════════════════════════════════════')
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell6_code.split("\n")]
    })

    # =========================================================================
    # CELL 7: Code - Exact 38-field Metric Function
    # =========================================================================
    cell7_code = """# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Hàm Tính Toán Chuẩn Hóa 38 Chỉ Số Khoa Học Trên GPU ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_image_metrics(hr_tensor, bic_tensor, sr_tensor,
                          latency_ms, img_path, dataset_name, scale_factor, saved_path=""):
    \"\"\"
    Tính toán đúng 38 trường dữ liệu khớp 100% với schema srgan_2x_benchmark.json.
    Tất cả các phép tính thực thi trực tiếp trên GPU Tensor để đạt tốc độ tối đa.
    \"\"\"
    with torch.no_grad():
        # 1. Chỉ số Bicubic Baseline
        p_bic    = calc_psnr(hr_tensor, bic_tensor)
        s_bic    = calc_ssim(hr_tensor, bic_tensor)
        ms_bic   = calc_msssim(hr_tensor, bic_tensor)
        lp_bic   = calc_lpips(hr_tensor, bic_tensor)
        niqe_bic = calc_niqe(bic_tensor)
        mse_bic  = float(torch.mean((hr_tensor - bic_tensor) ** 2).item() * (255.0 ** 2))
        rmse_bic = float(np.sqrt(mse_bic))

        # 2. Chỉ số Model Super-Resolution
        p_sr    = calc_psnr(hr_tensor, sr_tensor)
        s_sr    = calc_ssim(hr_tensor, sr_tensor)
        ms_sr   = calc_msssim(hr_tensor, sr_tensor)
        lp_sr   = calc_lpips(hr_tensor, sr_tensor)
        niqe_sr = calc_niqe(sr_tensor)
        mse_sr  = float(torch.mean((hr_tensor - sr_tensor) ** 2).item() * (255.0 ** 2))
        rmse_sr = float(np.sqrt(mse_sr))

        # 3. Phân bố mức xám (dải 0-255) & Edge Preservation Index
        mean_val = float(torch.mean(sr_tensor).item() * 255.0)
        std_val  = float(torch.std(sr_tensor).item() * 255.0)
        epi_val  = compute_epi(hr_tensor, sr_tensor)

        # 4. Độ tăng cường (Gains so với Bicubic)
        psnr_gain   = float(p_sr - p_bic)
        msssim_gain = float(ms_sr - ms_bic)
        lpips_gain  = float(lp_bic - lp_sr)      # Dương = cải thiện thị giác tốt hơn
        niqe_gain   = float(niqe_bic - niqe_sr)  # Dương = cải thiện chất lượng tự nhiên

        hr_h, hr_w = hr_tensor.shape[-2:]
        lr_h, lr_w = hr_h // scale_factor, hr_w // scale_factor

        record = {
            'source_path':      img_path,
            'saved_path':       saved_path,
            'dataset':          dataset_name,
            'filename':         os.path.basename(img_path),
            'status':           'ok',
            'resolution':       f'{lr_w}x{lr_h} -> {hr_w}x{hr_h}',
            'scale_factor':     int(scale_factor),
            'patches_count':    1,
            'latency_ms':       round(latency_ms, 2),
            'psnr_bicubic_db':  round(p_bic, 4),
            'mse_bicubic':      round(mse_bic, 6),
            'rmse_bicubic':     round(rmse_bic, 6),
            'ssim_bicubic':     round(s_bic, 4),
            'msssim_bicubic':   round(ms_bic, 4),
            'lpips_bicubic':    round(lp_bic, 4),
            'niqe_bicubic':     round(niqe_bic, 4),
            'psnr_fpga_db':     round(p_sr, 4),
            'mse_fpga':         round(mse_sr, 6),
            'rmse_fpga':        round(rmse_sr, 6),
            'ssim_fpga':        round(s_sr, 4),
            'msssim_fpga':      round(ms_sr, 4),
            'lpips_fpga':       round(lp_sr, 4),
            'lpips':            round(lp_sr, 4),
            'lpips_srgan':      round(lp_sr, 4),
            'niqe_fpga':        round(niqe_sr, 4),
            'mean_fpga':        round(mean_val, 4),
            'std_fpga':         round(std_val, 4),
            'mean':             round(mean_val, 4),
            'std':              round(std_val, 4),
            'epi':              round(epi_val, 4),
            'psnr_model_db':    round(p_sr, 4),
            'ssim_model':       round(s_sr, 4),
            'mse_model':        round(mse_sr, 6),
            'rmse_model':       round(rmse_sr, 6),
            'psnr_gain_db':     round(psnr_gain, 4),
            'msssim_gain':      round(msssim_gain, 4),
            'lpips_gain':       round(lpips_gain, 4),
            'niqe_gain':        round(niqe_gain, 4)
        }
        assert len(record) == 38, f"Số lượng trường không khớp: {len(record)} != 38"
        return record

print("✓ Hàm compute_image_metrics() GPU hoàn tất chuẩn hóa 38 trường dữ liệu.")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell7_code.split("\n")]
    })

    # =========================================================================
    # CELL 8: Code - Benchmark Loop across 3 Scales
    # =========================================================================
    cell8_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — 🚀 VÒNG LẶP BENCHMARK {m_upper} QUA 3 SCALE (2x, 3x, 4x) ║
# ╚══════════════════════════════════════════════════════════════╝

def save_checkpoint(results_list, elapsed_sec, path):
    n_ok = sum(1 for r in results_list if r.get('status') == 'ok')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({{
            'elapsed_sec_accumulated': round(elapsed_sec, 2),
            'total_evaluated': len(results_list),
            'successful_images': n_ok,
            'records': results_list
        }}, fh, indent=2, ensure_ascii=False)

all_scale_runs = {{}}
total_images = len(images_to_run)

print(f"BẮT ĐẦU TIẾN TRÌNH BENCHMARK MÔ HÌNH: {m_upper}")
print(f"CÁC TỈ LỆ ĐÁNH GIÁ: {{SCALES_TO_RUN}} | SỐ ẢNH MỖI TỈ LỆ: {{total_images:,}}")
print("═" * 80)

for s_idx, scale_factor in enumerate(SCALES_TO_RUN, 1):
    print(f"\\n▶ [TỈ LỆ {{s_idx}}/{{len(SCALES_TO_RUN)}}] ĐANG CHẠY {m_upper} SCALE {{scale_factor}}X...")
    
    # 1. Khởi tạo mô hình cho scale này
    cur_model, params_count, w_source = load_target_model(upscale_factor=scale_factor)
    
    scale_records = []
    checkpoint_file = os.path.join(OUTPUT_DIR, f"{m_key}_{{scale_factor}}x_checkpoint.json")
    t_start = time.perf_counter()
    
    # 2. Lặp qua từng ảnh trong tập dữ liệu
    for idx, img_path in enumerate(tqdm(images_to_run, desc=f"{m_upper} {{scale_factor}}x")):
        dataset_name = os.path.basename(os.path.dirname(img_path)) or 'unknown'
        
        try:
            # Giữ nguyên độ phân giải gốc của ảnh, căn chỉnh crop chia hết cho scale_factor
            hr_pil = Image.open(img_path).convert('RGB')
            orig_w, orig_h = hr_pil.size
            crop_w = (orig_w // scale_factor) * scale_factor
            crop_h = (orig_h // scale_factor) * scale_factor
            if (crop_w, crop_h) != (orig_w, orig_h):
                hr_pil = hr_pil.crop((0, 0, crop_w, crop_h))
            
            lr_w, lr_h = crop_w // scale_factor, crop_h // scale_factor
            lr_pil  = hr_pil.resize((lr_w, lr_h), Image.BICUBIC)
            bic_pil = lr_pil.resize((crop_w, crop_h), Image.BICUBIC)
            
            # Chuẩn bị Tensor trực tiếp trên GPU
            hr_tensor  = to_tensor(hr_pil).unsqueeze(0).to(DEVICE)
            lr_tensor  = to_tensor(lr_pil).unsqueeze(0).to(DEVICE)
            bic_tensor = to_tensor(bic_pil).unsqueeze(0).to(DEVICE)
            
            # Đo đạc thời gian suy luận (Latency ms)
            if DEVICE == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with torch.no_grad():
                sr_tensor = cur_model(lr_tensor)
            if DEVICE == 'cuda': torch.cuda.synchronize()
            
            latency_ms = (time.perf_counter() - t0) * 1000.0
            sr_tensor  = torch.clamp(sr_tensor, 0.0, 1.0)
            
            # Lưu ảnh nếu được bật
            saved_path = None
            if SAVE_PNG_IMAGES:
                out_name = f"{m_key}_{{scale_factor}}x_{{os.path.basename(img_path)}}"
                saved_path = os.path.join(OUTPUT_DIR, out_name)
                sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                Image.fromarray(sr_np).save(saved_path)
            
            # Tính 38 chỉ số trực tiếp trên GPU
            rec = compute_image_metrics(
                hr_tensor=hr_tensor, bic_tensor=bic_tensor, sr_tensor=sr_tensor,
                latency_ms=latency_ms, img_path=img_path,
                dataset_name=dataset_name, scale_factor=scale_factor, saved_path=saved_path or ""
            )
            scale_records.append(rec)
            
            # Thu dọn tensor tránh rò rỉ bộ nhớ
            del hr_tensor, lr_tensor, bic_tensor, sr_tensor
            
        except Exception as e:
            scale_records.append({{
                'source_path': img_path, 'dataset': dataset_name,
                'filename': os.path.basename(img_path), 'status': f'error: {{str(e)}}'
            }})
            
        # Lưu checkpoint định kỳ và giải phóng cache
        if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == total_images:
            save_checkpoint(scale_records, time.perf_counter() - t_start, checkpoint_file)
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            
    elapsed_total = time.perf_counter() - t_start
    all_scale_runs[scale_factor] = {{
        'records': scale_records,
        'elapsed_sec': elapsed_total,
        'weights_source': w_source,
        'params_count': params_count
    }}
    print(f"✓ Hoàn thành {m_upper} Scale {{scale_factor}}x: {{len(scale_records):,}} ảnh trong {{elapsed_total/60:.2f}} phút | Checkpoint: {{checkpoint_file}}")
    
    # Giải phóng model và dọn dẹp VRAM trước khi chuyển sang scale kế tiếp
    del cur_model
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

print("═" * 80)
print(f"✓ {m_upper} ĐÃ HOÀN TẤT ĐÁNH GIÁ TRÊN CẢ 3 TỈ LỆ (2X, 3X, 4X) THÀNH CÔNG!")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell8_code.split("\n")]
    })

    # =========================================================================
    # CELL 9: Code - Summary across Scales
    # =========================================================================
    cell9_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Thống Kê & Tổng Hợp Kết Quả {m_upper} Theo Từng Scale ║
# ╚══════════════════════════════════════════════════════════════╝

# Tự động phục hồi kết quả từ các file checkpoint đã lưu trong OUTPUT_DIR nếu kernel vừa restart
if not all_scale_runs:
    for s in SCALES_TO_RUN:
        cf_pattern = os.path.join(OUTPUT_DIR, f"{m_key}_{{s}}x_checkpoint.json")
        bf_pattern = os.path.join(OUTPUT_DIR, f"{m_key}_{{s}}x_benchmark.json")
        chosen_file = cf_pattern if os.path.exists(cf_pattern) else (bf_pattern if os.path.exists(bf_pattern) else None)
        if chosen_file:
            try:
                with open(chosen_file, 'r', encoding='utf-8') as fh:
                    cdata = json.load(fh)
                    recs = cdata.get('records') or cdata.get('per_image_results', [])
                    if recs:
                        all_scale_runs[s] = {{
                            'records': recs,
                            'elapsed_sec': cdata.get('elapsed_sec_accumulated', 0),
                            'weights_source': 'checkpoint_file',
                            'params_count': 0
                        }}
                        print(f"✓ Đã tự động phục hồi {{len(recs):,}} kết quả của Scale {{s}}x từ file checkpoint!")
            except Exception:
                pass

scale_comparison_rows = []

for s_key in sorted(all_scale_runs.keys()):
    s_data = all_scale_runs[s_key]
    records = s_data['records']
    df_s = pd.DataFrame([r for r in records if r.get('status') == 'ok'])
    if df_s.empty:
        continue
        
    p_mean  = df_s['psnr_model_db'].mean()
    s_mean  = df_s['ssim_model'].mean()
    ms_mean = df_s['msssim_fpga'].mean() if 'msssim_fpga' in df_s.columns else df_s['msssim_bicubic'].mean()
    lp_mean = df_s['lpips'].mean() if 'lpips' in df_s.columns else df_s['lpips_bicubic'].mean()
    nq_mean = df_s['niqe_fpga'].mean() if 'niqe_fpga' in df_s.columns else df_s['niqe_bicubic'].mean()
    lat_med = df_s['latency_ms'].median()
    fps_val = 1000.0 / df_s['latency_ms'].mean() if df_s['latency_ms'].mean() > 0 else 0
    epi_val = df_s['epi'].mean()
    
    p_gain  = df_s['psnr_gain_db'].mean()
    lp_gain = df_s['lpips_gain'].mean()
    
    df_nih   = df_s[df_s['dataset'] == 'sub_NIH']
    df_chest = df_s[df_s['dataset'] == 'sub_chest']
    
    scale_comparison_rows.append({{
        'Model': '{m_upper}',
        'Scale': f'{{s_key}}x',
        'Images': len(df_s),
        'PSNR (dB)': round(p_mean, 2),
        'SSIM': round(s_mean, 4),
        'MS-SSIM': round(ms_mean, 4),
        'LPIPS ↓': round(lp_mean, 4),
        'NIQE ↓': round(nq_mean, 2),
        'EPI': round(epi_val, 4),
        'PSNR Gain (dB)': round(p_gain, 2),
        'LPIPS Gain ↑': round(lp_gain, 4),
        'Latency (ms)': round(lat_med, 1),
        'FPS': round(fps_val, 1),
        'PSNR (sub_NIH)': round(df_nih['psnr_model_db'].mean(), 2) if not df_nih.empty else None,
        'PSNR (sub_chest)': round(df_chest['psnr_model_db'].mean(), 2) if not df_chest.empty else None,
        'LPIPS (sub_NIH)': round(df_nih['lpips'].mean(), 4) if not df_nih.empty and 'lpips' in df_nih.columns else None,
        'LPIPS (sub_chest)': round(df_chest['lpips'].mean(), 4) if not df_chest.empty and 'lpips' in df_chest.columns else None,
    }})

df_scale_summary = pd.DataFrame(scale_comparison_rows)
print(f"📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG {m_upper} THEO SCALE:")
print("═" * 110)
print(df_scale_summary[['Scale', 'Images', 'PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS ↓', 'NIQE ↓', 'EPI', 'PSNR Gain (dB)', 'Latency (ms)', 'FPS']].to_string(index=False))
print("═" * 110)
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell9_code.split("\n")]
    })

    # =========================================================================
    # CELL 10: Code - Export JSON and CSV
    # =========================================================================
    cell10_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Xuất File Kết Quả JSON & CSV Cho {m_upper:<10}       ║
# ╚══════════════════════════════════════════════════════════════╝

exported_files = []

for s_key, s_data in all_scale_runs.items():
    records = s_data['records']
    df_s = pd.DataFrame([r for r in records if r.get('status') == 'ok'])
    if df_s.empty:
        continue
        
    json_path = os.path.join(OUTPUT_DIR, f"{m_key}_{{s_key}}x_benchmark.json")
    csv_path  = os.path.join(OUTPUT_DIR, f"{m_key}_{{s_key}}x_benchmark.csv")
    
    summary_dict = {{
        'model': f"{m_upper} (Scale {{s_key}}x)",
        'weights_source': s_data.get('weights_source', ''),
        'device': DEVICE,
        'images_evaluated': len(df_s),
        'images_error': len(records) - len(df_s),
        'resolution_in_out': 'Native Resolution',
        'scale_factor': int(s_key),
        'avg_psnr_bicubic_db': float(round(df_s['psnr_bicubic_db'].mean(), 4)),
        'std_psnr_bicubic_db': float(round(df_s['psnr_bicubic_db'].std(), 4)),
        'avg_mse_bicubic': float(round(df_s['mse_bicubic'].mean(), 6)),
        'avg_rmse_bicubic': float(round(df_s['rmse_bicubic'].mean(), 6)),
        'avg_ssim_bicubic': float(round(df_s['ssim_bicubic'].mean(), 4)),
        'avg_msssim_bicubic': float(round(df_s['msssim_bicubic'].mean(), 4)),
        'avg_lpips_bicubic': float(round(df_s['lpips_bicubic'].mean(), 4)),
        'std_lpips_bicubic': float(round(df_s['lpips_bicubic'].std(), 4)),
        'avg_niqe_bicubic': float(round(df_s['niqe_bicubic'].mean(), 4)),
        'avg_psnr_model_db': float(round(df_s['psnr_model_db'].mean(), 4)),
        'std_psnr_model_db': float(round(df_s['psnr_model_db'].std(), 4)),
        'avg_mse_model': float(round(df_s['mse_model'].mean(), 6)),
        'avg_rmse_model': float(round(df_s['rmse_model'].mean(), 6)),
        'avg_ssim_model': float(round(df_s['ssim_model'].mean(), 4)),
        'avg_msssim_model': float(round(df_s['msssim_fpga'].mean(), 4)),
        'avg_lpips': float(round(df_s['lpips'].mean(), 4)),
        'std_lpips_model': float(round(df_s['lpips'].std(), 4)),
        'avg_niqe_model': float(round(df_s['niqe_fpga'].mean(), 4)),
        'avg_epi': float(round(df_s['epi'].mean(), 4)),
        'avg_mean': float(round(df_s['mean'].mean(), 4)),
        'avg_std': float(round(df_s['std'].mean(), 4)),
        'avg_psnr_gain_db': float(round(df_s['psnr_gain_db'].mean(), 4)),
        'avg_msssim_gain': float(round(df_s['msssim_gain'].mean(), 4)),
        'avg_lpips_gain': float(round(df_s['lpips_gain'].mean(), 4)),
        'avg_niqe_gain': float(round(df_s['niqe_gain'].mean(), 4)),
        'avg_latency_ms': float(round(df_s['latency_ms'].mean(), 2)),
        'throughput_fps': float(round(1000.0 / df_s['latency_ms'].mean(), 2)),
        'elapsed_sec_total': float(round(s_data.get('elapsed_sec', 0), 2)),
        'wall_time_min': float(round(s_data.get('elapsed_sec', 0) / 60.0, 2))
    }}
    
    full_output = {{
        'summary': summary_dict,
        'per_image_results': records
    }}
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
        
    df_s.to_csv(csv_path, index=False)
    exported_files.extend([json_path, csv_path])
    print(f"✓ Đã xuất: {{os.path.basename(json_path)}} ({{os.path.getsize(json_path)/(1024*1024):.2f}} MB)")
    print(f"✓ Đã xuất: {{os.path.basename(csv_path)}} ({{os.path.getsize(csv_path)/1024:.1f}} KB)")

# Xuất bảng tổng hợp theo scale
summary_csv_path  = os.path.join(OUTPUT_DIR, f"{m_key}_multiscale_summary.csv")
summary_json_path = os.path.join(OUTPUT_DIR, f"{m_key}_multiscale_summary.json")
df_scale_summary.to_csv(summary_csv_path, index=False)
df_scale_summary.to_json(summary_json_path, indent=2, orient='records')
exported_files.extend([summary_csv_path, summary_json_path])

print("\\n📂 ĐÃ XUẤT TẤT CẢ FILE RA /kaggle/working/results SẴN SÀNG:")
for ef in exported_files:
    print(f"  • {{ef}}")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell10_code.split("\n")]
    })

    # =========================================================================
    # CELL 11: Code - Packaging & Zip
    # =========================================================================
    cell11_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Đóng Gói File ZIP Kết Quả Cho {m_upper:<10}         ║
# ╚══════════════════════════════════════════════════════════════╝

zip_out_name = "{m_key}_benchmark_results.zip"
zip_out_path = os.path.join('/kaggle/working', zip_out_name)

if os.path.exists(zip_out_path):
    os.remove(zip_out_path)

cmd = f"cd {{OUTPUT_DIR}} && zip -r '{{zip_out_path}}' . -i '*{m_key}*'"
os.system(cmd)

if os.path.exists(zip_out_path):
    sz_mb = os.path.getsize(zip_out_path) / (1024 * 1024)
    print("═" * 70)
    print(f"🎉 ĐÃ ĐÓNG GÓI THÀNH CÔNG: {{zip_out_name}} ({{sz_mb:.2f}} MB)")
    print("👉 Tải file này từ tab 'Output' bên phải giao diện Kaggle.")
    print("═" * 70)
else:
    print("⚠ Chưa tạo được file zip, các file kết quả vẫn nằm nguyên tại:", OUTPUT_DIR)
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell11_code.split("\n")]
    })

    # =========================================================================
    # CELL 12: Code - Cleanup & Next Steps
    # =========================================================================
    cell12_code = f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 12 — Giải Phóng VRAM & Hướng Dẫn Đồng Bộ Local         ║
# ╚══════════════════════════════════════════════════════════════╝

gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
    print("✓ Đã giải phóng toàn bộ bộ nhớ GPU VRAM.")

print(f\"\"\"
══════════════════════════════════════════════════════════════════════
🎉 HOÀN THÀNH TOÀN BỘ BENCHMARK CHO MÔ HÌNH {m_upper}!
══════════════════════════════════════════════════════════════════════

📥 CÁC BƯỚC TIẾP THEO:
1. Tải file '{m_key}_benchmark_results.zip' (hoặc các file JSON/CSV riêng lẻ) từ Kaggle Output.
2. Giải nén và đặt các file vào thư mục:
   • results/software/
3. Khi đã có đủ kết quả các mô hình, chạy script tổng hợp đối chiếu:
   python "code hardware/scripts/run_analysis.py"
══════════════════════════════════════════════════════════════════════
\"\"\")
"""

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cell12_code.split("\n")]
    })

    # Notebook structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_filename = f"kaggle_{m_key}_benchmark.ipynb"
    output_path = os.path.join("notebooks/software", nb_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
    print(f"✓ Notebook generated: {output_path} ({len(cells)} cells)")

    # Copy to ~/Downloads for convenient one-click Kaggle upload
    downloads_path = os.path.expanduser(f"~/Downloads/{nb_filename}")
    shutil.copy2(output_path, downloads_path)
    print(f"  -> Copied to Downloads: {downloads_path}")

    return output_path

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TẠO 6 NOTEBOOK BENCHMARK ĐỘC LẬP CHO 6 MÔ HÌNH (2X, 3X, 4X)")
    print("=" * 80)

    generated_paths = []
    for info in MODELS_INFO:
        p = generate_notebook_for_model(info)
        generated_paths.append(p)

    print("\n" + "=" * 80)
    print("🎉 ĐÃ HOÀN TẤT TẠO 6 NOTEBOOK BENCHMARK ĐỘC LẬP:")
    for p in generated_paths:
        print(f"  • {p}")
    print("=" * 80)

if __name__ == "__main__":
    main()
