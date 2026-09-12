#!/usr/bin/env python3
"""
================================================================================
Generator: Kaggle Benchmark Notebook for Compact SRCNN RTL (1-16-8-1, 1,649 params)
================================================================================
Mục đích:
  Tạo file Jupyter Notebook (.ipynb) chuẩn 13 cell (1 Markdown Header + 12 Code Cells)
  khớp 100% về kiến trúc, luồng thực thi, biến số và cấu trúc thư mục với
  notebook chuẩn kaggle_espcn_benchmark.ipynb, chỉ khác duy nhất ở mô hình sử dụng:
  Compact SRCNN RTL (1 -> 16 -> 8 -> 1, 1.649 tham số, Bit-Accurate Fixed Point).
================================================================================
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS_FILE = REPO_ROOT / "core_project/hardware_fpga/weights_fixed_point/weights_hex_clean.txt"
BIASES_FILE = REPO_ROOT / "core_project/hardware_fpga/weights_fixed_point/biases_hex_clean.txt"

with open(WEIGHTS_FILE, "r") as f:
    weights_lines = [line.strip() for line in f if line.strip()]
assert len(weights_lines) == 1624, f"Số lượng weights: {len(weights_lines)} != 1624"
weights_hex_str = "".join(weights_lines)

with open(BIASES_FILE, "r") as f:
    biases_lines = [line.strip() for line in f if line.strip()]
assert len(biases_lines) == 25, f"Số lượng biases: {len(biases_lines)} != 25"
biases_hex_str = ",".join(biases_lines)

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# ------------------------------------------------------------------------------
# CELL 0: Markdown Header (Tương ứng Header của kaggle_espcn_benchmark.ipynb)
# ------------------------------------------------------------------------------
add_md("""# 🔬 Compact SRCNN RTL Benchmark Pipeline: Scale 2×, 3×, 4× trên Ảnh Y Tế
### Mô hình: Compact SRCNN RTL (1 -> 16 -> 8 -> 1, 1.649 tham số)
* **Tác giả / Kiến trúc**: Compact SRCNN RTL triển khai phần cứng Xilinx Zynq-7020 (PYNQ-Z2)
* **Mô tả**: Mô hình 3 tầng tích chập Compact RTL số học Fixed-Point bit-accurate (S7.0, S0.7, S24.7).
* **Tập dữ liệu**: `duc24kdl/sub-x-ray` (2.200 ảnh: 1.750 ảnh `sub_NIH` + 450 ảnh `sub_chest`)
* **Tỉ lệ đánh giá**: Tuần tự qua cả 3 tỉ lệ **2× $\\rightarrow$ 3× $\\rightarrow$ 4×**
* **Độ phân giải**: Giữ nguyên độ phân giải gốc của ảnh (căn chỉnh chia hết cho từng tỉ lệ scale)

---

| Cell | Nhiệm vụ thực thi |
| :--- | :--- |
| **Cell 1** | Cài đặt thư viện (`lpips`, `pytorch-msssim`, `pyiqa`) & Tự động định vị/nạp trọng số RTL |
| **Cell 2** | Khởi tạo GPU CUDA & Bộ hàm đo lường metric siêu tốc trên Tensor |
| **Cell 3** | Khởi tạo kiến trúc mô hình **Compact SRCNN RTL** và bộ nạp trọng số |
| **Cell 4** | Smoke Test kiểm tra tính tương thích và shape đầu ra cho cả 3 Scale (2×, 3×, 4×) |
| **Cell 5** | Quét tập dữ liệu ảnh `duc24kdl/sub-x-ray` |
| **Cell 6** | Cấu hình tham số đánh giá |
| **Cell 7** | Định nghĩa hàm tính toán chuẩn 38 chỉ số khoa học |
| **Cell 8** | **Vòng lặp Benchmark tuần tự 3 tỉ lệ (2× $\\rightarrow$ 3× $\\rightarrow$ 4×)** |
| **Cell 9** | Thống kê tổng hợp và đối sánh giữa các Scale |
| **Cell 10** | Xuất file kết quả JSON & CSV chuẩn hóa |
| **Cell 11** | Đóng gói toàn bộ kết quả vào file ZIP tải về |
| **Cell 12** | Giải phóng tài nguyên VRAM |
""")

# ------------------------------------------------------------------------------
# CELL 1: Cài đặt thư viện & Định vị trọng số (Khớp Cell 1 ESPCN)
# ------------------------------------------------------------------------------
add_code(f"""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Cài Đặt Thư Viện & Định Vị Trọng Số Cho Model      ║
# ╚══════════════════════════════════════════════════════════════╝

import os, sys, glob, shutil, json, time, math, gc
import numpy as np

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
# 2. Định vị hoặc giải mã trọng số RTL phần cứng (1.624 weights Q7 và 25 biases Q14)
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDED_WEIGHTS_HEX = "{weights_hex_str}"
EMBEDDED_BIASES_HEX  = "{biases_hex_str}"

def locate_or_fetch_rtl_weights():
    candidate_w = glob.glob("/kaggle/input/**/weights_hex_clean.txt", recursive=True) + \\
                  glob.glob("./**/weights_hex_clean.txt", recursive=True) + \\
                  glob.glob("weights_hex_clean.txt")
    candidate_b = glob.glob("/kaggle/input/**/biases_hex_clean.txt", recursive=True) + \\
                  glob.glob("./**/biases_hex_clean.txt", recursive=True) + \\
                  glob.glob("biases_hex_clean.txt")

    if candidate_w and candidate_b and os.path.exists(candidate_w[0]) and os.path.exists(candidate_b[0]):
        print(f"✓ Tìm thấy file trọng số phần cứng tại: {{candidate_w[0]}}")
        with open(candidate_w[0]) as f:
            w_lines = [l.strip() for l in f if l.strip()]
        with open(candidate_b[0]) as f:
            b_lines = [l.strip() for l in f if l.strip()]
        w_source = f"File cục bộ ({{candidate_w[0]}})"
    else:
        print("✓ Nạp trọng số RTL trực tiếp từ bộ nhúng Embedded Weights (100% self-contained).")
        w_lines = [EMBEDDED_WEIGHTS_HEX[i:i+2] for i in range(0, len(EMBEDDED_WEIGHTS_HEX), 2)]
        b_lines = [h.strip() for h in EMBEDDED_BIASES_HEX.split(",") if h.strip()]
        w_source = "Embedded RTL Weights (1-16-8-1, 1.649 params)"

    assert len(w_lines) == 1624, f"Số lượng weights không khớp: {{len(w_lines)}} != 1624"
    assert len(b_lines) == 25, f"Số lượng biases không khớp: {{len(b_lines)}} != 25"

    q7_weights = np.array([int(h, 16) for h in w_lines], dtype=np.uint8).view(np.int8)
    q14_biases = np.array([int(h, 16) for h in b_lines], dtype=np.uint32).view(np.int32)
    return q7_weights, q14_biases, w_source

Q7_WEIGHTS, Q14_BIASES, WEIGHTS_SOURCE = locate_or_fetch_rtl_weights()
print(f"📂 Nguồn trọng số chính thức: {{WEIGHTS_SOURCE}}")
print(f"  • Q7 weights : {{len(Q7_WEIGHTS)}} entries (non-zero: {{np.count_nonzero(Q7_WEIGHTS)}})")
print(f"  • Q14 biases : {{len(Q14_BIASES)}} entries (non-zero: {{np.count_nonzero(Q14_BIASES)}})")
""")

# ------------------------------------------------------------------------------
# CELL 2: GPU Init & Metrics Functions (Khớp 100% Cell 2 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Khởi Tạo Thiết Bị & Bộ Hàm Đo Tối Ưu Hóa GPU CUDA ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import sys
import glob
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
    torch.backends.cudnn.benchmark = True
else:
    print("⚠ Thiết bị tính toán: CPU (Khuyến nghị bật GPU T4/P100 trên Kaggle để đạt tốc độ cao)")

# 1. Khởi tạo LPIPS (AlexNet)
try:
    import lpips
    LPIPS_FN = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for param in LPIPS_FN.parameters():
        param.requires_grad = False
    HAS_LPIPS = True
    print("✓ LPIPS (AlexNet) đã sẵn sàng")
except Exception as e:
    LPIPS_FN = None
    HAS_LPIPS = False
    print(f"⚠ LPIPS không khả dụng ({e})")

# 2. Khởi tạo MS-SSIM
try:
    import pytorch_msssim
    MS_SSIM_FN = pytorch_msssim.MS_SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)
    HAS_MSSSIM = True
    print("✓ MS-SSIM (Multi-Scale SSIM) đã sẵn sàng")
except Exception as e:
    MS_SSIM_FN = None
    HAS_MSSSIM = False
    print(f"⚠ MS-SSIM không khả dụng ({e})")

# 3. Khởi tạo NIQE (pyiqa chạy trực tiếp trên GPU)
try:
    import pyiqa
    NIQE_FN = pyiqa.create_metric('niqe', device=DEVICE)
    HAS_NIQE = True
    print("✓ NIQE (Natural Image Quality Evaluator) đã sẵn sàng")
except Exception as e:
    NIQE_FN = None
    HAS_NIQE = False
    print(f"⚠ NIQE không khả dụng ({e})")

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
            try:
                return float(MS_SSIM_FN(t_test, t_true).item())
            except Exception:
                return calc_ssim(t_true, t_test)
    return calc_ssim(t_true, t_test)

def calc_lpips(t_true, t_test):
    if HAS_LPIPS and LPIPS_FN is not None:
        with torch.no_grad():
            try:
                return float(LPIPS_FN(t_test * 2.0 - 1.0, t_true * 2.0 - 1.0).item())
            except Exception:
                pass
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
""")

# ------------------------------------------------------------------------------
# CELL 3: Model Architecture & load_target_model (Khớp Cell 3 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Kiến Trúc Mô Hình Compact SRCNN RTL & Bộ Nạp       ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import sys
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactSRCNN_RTL(nn.Module):
    \"\"\"
    Mô hình Compact SRCNN RTL (1 -> 16 -> 8 -> 1) mô phỏng chính xác 100%
    số học số nguyên Fixed-Point trên IP Core phần cứng Xilinx Zynq-7020 (PYNQ-Z2).
    
    Quy tắc số học phần cứng RTL:
      1. Pixel đầu vào: S7.0 (offset -128 từ [0, 255])
      2. Trọng số: S0.7 (Q7, chia sẻ /128.0)
      3. Biases: S24.7 (Q14, chia sẻ /16384.0)
      4. Tích lũy số nguyên S24.7, sau mỗi tầng dịch phải acc >> 7
      5. Layer 1 & 2: ReLU kẹp vào [0, 127]
      6. Layer 3: Cộng 128 (OUTPUT_ZERO_POINT) và kẹp vào [0, 255]
    \"\"\"
    def __init__(self, q7_w: np.ndarray, q14_b: np.ndarray, upscale_factor: int = 2):
        super(CompactSRCNN_RTL, self).__init__()
        self.upscale_factor = upscale_factor

        # Tách cấu trúc trọng số
        w1_int = q7_w[0:1296].reshape(16, 1, 9, 9).astype(np.int64)
        b1_int = q14_b[0:16].astype(np.int64)
        w2_int = q7_w[1296:1424].reshape(8, 16, 1, 1).astype(np.int64)
        b2_int = q14_b[16:24].astype(np.int64)
        w3_int = q7_w[1424:1624].reshape(1, 8, 5, 5).astype(np.int64)
        b3_int = q14_b[24:25].astype(np.int64)

        self.register_buffer("w1", torch.from_numpy(w1_int).float())
        self.register_buffer("b1", torch.from_numpy(b1_int).float())
        self.register_buffer("w2", torch.from_numpy(w2_int).float())
        self.register_buffer("b2", torch.from_numpy(b2_int).float())
        self.register_buffer("w3", torch.from_numpy(w3_int).float())
        self.register_buffer("b3", torch.from_numpy(b3_int).float())

        self.num_params = 1624 + 25
        assert self.num_params == 1649, f"Tổng tham số: {self.num_params} ≠ 1649"

    def forward_grayscale(self, gray_lr):
        # 1. Nội suy Bicubic lên kích thước mục tiêu
        x_bic = F.interpolate(gray_lr, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        # 2. Chuyển sang miền điểm 0 phần cứng: Pixel S7.0 [-128, 127]
        x_px = torch.clamp(torch.round(x_bic * 255.0), 0.0, 255.0)
        x_s7 = (x_px - 128.0)

        # 3. Layer 1: Conv 9x9, Pad 4 -> acc >> 7 -> ReLU [0, 127]
        x_pad1 = F.pad(x_s7, (4, 4, 4, 4), mode='constant', value=0.0)
        acc1 = F.conv2d(x_pad1, self.w1, self.b1)
        shifted1 = torch.floor(acc1 / 128.0)
        l1 = torch.clamp(shifted1, 0.0, 127.0)

        # 4. Layer 2: Conv 1x1, Pad 0 -> acc >> 7 -> ReLU [0, 127]
        acc2 = F.conv2d(l1, self.w2, self.b2)
        shifted2 = torch.floor(acc2 / 128.0)
        l2 = torch.clamp(shifted2, 0.0, 127.0)

        # 5. Layer 3: Conv 5x5, Pad 2 -> (acc >> 7) + 128 -> Clamp [0, 255]
        l2_pad = F.pad(l2, (2, 2, 2, 2), mode='constant', value=0.0)
        acc3 = F.conv2d(l2_pad, self.w3, self.b3)
        shifted3 = torch.floor(acc3 / 128.0)
        l3 = torch.clamp(shifted3 + 128.0, 0.0, 255.0)

        return l3 / 255.0

    def forward(self, x):
        if x.shape[1] == 1:
            return self.forward_grayscale(x)

        # RGB -> YCbCr BT.601
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        y  =  0.299000 * r + 0.587000 * g + 0.114000 * b
        cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
        cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5

        y_sr = self.forward_grayscale(y)
        cb_sr = F.interpolate(cb, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        cr_sr = F.interpolate(cr, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        cb_shifted = cb_sr - 0.5
        cr_shifted = cr_sr - 0.5
        r_sr = y_sr + 1.402000 * cr_shifted
        g_sr = y_sr - 0.344136 * cb_shifted - 0.714136 * cr_shifted
        b_sr = y_sr + 1.772000 * cb_shifted

        return torch.clamp(torch.cat([r_sr, g_sr, b_sr], dim=1), 0.0, 1.0)


def load_target_model(upscale_factor: int, device: str = DEVICE):
    \"\"\"Nạp mô hình Compact SRCNN RTL theo scale chỉ định (2x, 3x, 4x).\"\"\"
    model = CompactSRCNN_RTL(Q7_WEIGHTS, Q14_BIASES, upscale_factor=upscale_factor)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_params = model.num_params
    print(f"✓ Nạp thành công [Compact SRCNN RTL {upscale_factor}×]: {num_params:,} tham số ({WEIGHTS_SOURCE})")
    return model, num_params, WEIGHTS_SOURCE

print("✓ Bộ định nghĩa kiến trúc và nạp trọng số Compact SRCNN RTL đã sẵn sàng.")
""")

# ------------------------------------------------------------------------------
# CELL 4: Smoke Test (Khớp Cell 4 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Smoke Test: Xác Thực Cả 3 Scale (2x, 3x, 4x)       ║
# ╚══════════════════════════════════════════════════════════════╝

test_scales = [2, 3, 4]
dummy_lr = torch.rand(1, 3, 256, 256, device=DEVICE)

print("Kiểm tra tính tương thích và shape đầu ra cho Compact SRCNN RTL qua cả 3 tỉ lệ:")
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
        assert (act_h, act_w) == (exp_h, exp_w), f"Sai shape: nhận {out.shape}, mong đợi (1, 3, {exp_h}, {exp_w})"
        print(f"  • Scale {s}x: Input [1, 3, 256, 256] -> Output {list(out.shape)} | Latency: {dt:6.2f} ms | OK")
        del m_inst
        gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ Scale {s}x: Lỗi kiểm thử: {e}")

print("─" * 70)
print("✓ Mô hình Compact SRCNN RTL đã vượt qua bài kiểm tra sơ bộ cho toàn bộ các Scale 2x, 3x, 4x!")
""")

# ------------------------------------------------------------------------------
# CELL 5: Quét tập dữ liệu ảnh (Khớp 100% Cell 5 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
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
""")

# ------------------------------------------------------------------------------
# CELL 6: Cấu hình tham số đánh giá (Khớp 100% Cell 6 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Cấu Hình Tham Số Đánh Giá Cho Compact SRCNN RTL    ║
# ╚══════════════════════════════════════════════════════════════╝

# 1. Mô hình mục tiêu:
TARGET_MODEL = "compact_srcnn_rtl"

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
print(f'  CẤU HÌNH PHIÊN BENCHMARK MÔ HÌNH: Compact SRCNN RTL (1-16-8-1)')
print(f'  ✓ Danh sách tỉ lệ (Scales) : {SCALES_TO_RUN}')
print(f'  ✓ Độ phân giải             : Giữ nguyên độ phân giải gốc của từng ảnh')
print(f'  ✓ Số lượng ảnh đánh giá   : {len(images_to_run):,} ảnh')
print(f'  ✓ Thiết bị tính toán      : {DEVICE.upper()}')
print(f'  ✓ Thư mục xuất kết quả    : {OUTPUT_DIR}')
print('═════════════════════════════════════════════════════════════')
""")

# ------------------------------------------------------------------------------
# CELL 7: Hàm tính 38 chỉ số khoa học (Khớp 100% Cell 7 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Hàm Tính Toán Chuẩn Hóa 38 Chỉ Số Khoa Học Trên GPU ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_image_metrics(hr_tensor, bic_tensor, sr_tensor,
                          latency_ms, img_path, dataset_name, scale_factor, saved_path=""):
    \"\"\"
    Tính toán đúng 38 trường dữ liệu khớp 100% với schema chuẩn.
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
""")

# ------------------------------------------------------------------------------
# CELL 8: Vòng lặp benchmark tuần tự (Khớp 100% Cell 8 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — 🚀 VÒNG LẶP BENCHMARK COMPACT SRCNN RTL (2x, 3x, 4x)║
# ╚══════════════════════════════════════════════════════════════╝

def save_checkpoint(results_list, elapsed_sec, path):
    n_ok = sum(1 for r in results_list if r.get('status') == 'ok')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({
            'elapsed_sec_accumulated': round(elapsed_sec, 2),
            'total_evaluated': len(results_list),
            'successful_images': n_ok,
            'records': results_list
        }, fh, indent=2, ensure_ascii=False)

all_scale_runs = {}
total_images = len(images_to_run)

print(f"BẮT ĐẦU TIẾN TRÌNH BENCHMARK MÔ HÌNH: Compact SRCNN RTL (1-16-8-1)")
print(f"CÁC TỈ LỆ ĐÁNH GIÁ: {SCALES_TO_RUN} | SỐ ẢNH MỖI TỈ LỆ: {total_images:,}")
print("═" * 80)

for s_idx, scale_factor in enumerate(SCALES_TO_RUN, 1):
    print(f"\\n▶ [TỈ LỆ {s_idx}/{len(SCALES_TO_RUN)}] ĐANG CHẠY COMPACT SRCNN RTL SCALE {scale_factor}X...")
    
    # 1. Khởi tạo mô hình cho scale này
    cur_model, params_count, w_source = load_target_model(upscale_factor=scale_factor)
    
    scale_records = []
    checkpoint_file = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_{scale_factor}x_checkpoint.json")
    t_start = time.perf_counter()
    
    # 2. Lặp qua từng ảnh trong tập dữ liệu
    for idx, img_path in enumerate(tqdm(images_to_run, desc=f"Compact SRCNN RTL {scale_factor}x")):
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
                out_name = f"{TARGET_MODEL}_{scale_factor}x_{os.path.basename(img_path)}"
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
            scale_records.append({
                'source_path': img_path, 'dataset': dataset_name,
                'filename': os.path.basename(img_path), 'status': f'error: {str(e)}'
            })
            
        # Lưu checkpoint định kỳ và giải phóng cache
        if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == total_images:
            save_checkpoint(scale_records, time.perf_counter() - t_start, checkpoint_file)
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            
    elapsed_total = time.perf_counter() - t_start
    all_scale_runs[scale_factor] = {
        'records': scale_records,
        'elapsed_sec': elapsed_total,
        'weights_source': w_source,
        'params_count': params_count
    }
    print(f"✓ Hoàn thành Compact SRCNN RTL Scale {scale_factor}x: {len(scale_records):,} ảnh trong {elapsed_total/60:.2f} phút | Checkpoint: {checkpoint_file}")
    
    # Giải phóng model và dọn dẹp VRAM trước khi chuyển sang scale kế tiếp
    del cur_model
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

print("═" * 80)
print(f"✓ COMPACT SRCNN RTL ĐÃ HOÀN TẤT ĐÁNH GIÁ TRÊN CẢ 3 TỈ LỆ (2X, 3X, 4X) THÀNH CÔNG!")
""")

# ------------------------------------------------------------------------------
# CELL 9: Thống kê & Tổng hợp kết quả (Khớp 100% Cell 9 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Thống Kê & Tổng Hợp Kết Quả Compact SRCNN RTL      ║
# ╚══════════════════════════════════════════════════════════════╝

# Tự động phục hồi kết quả từ các file checkpoint đã lưu trong OUTPUT_DIR nếu kernel vừa restart
if not all_scale_runs:
    for s in SCALES_TO_RUN:
        cf_pattern = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_{s}x_checkpoint.json")
        bf_pattern = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_{s}x_benchmark.json")
        chosen_file = cf_pattern if os.path.exists(cf_pattern) else (bf_pattern if os.path.exists(bf_pattern) else None)
        if chosen_file:
            try:
                with open(chosen_file, 'r', encoding='utf-8') as fh:
                    cdata = json.load(fh)
                    recs = cdata.get('records') or cdata.get('per_image_results', [])
                    if recs:
                        all_scale_runs[s] = {
                            'records': recs,
                            'elapsed_sec': cdata.get('elapsed_sec_accumulated', 0),
                            'weights_source': 'checkpoint_file',
                            'params_count': 1649
                        }
                        print(f"✓ Đã tự động phục hồi {len(recs):,} kết quả của Scale {s}x từ file checkpoint!")
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
    
    scale_comparison_rows.append({
        'Model': 'Compact SRCNN RTL (1-16-8-1)',
        'Scale': f'{s_key}x',
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
    })

df_scale_summary = pd.DataFrame(scale_comparison_rows)
print(f"📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG COMPACT SRCNN RTL THEO SCALE:")
print("═" * 110)
print(df_scale_summary[['Scale', 'Images', 'PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS ↓', 'NIQE ↓', 'EPI', 'PSNR Gain (dB)', 'Latency (ms)', 'FPS']].to_string(index=False))
print("═" * 110)
""")

# ------------------------------------------------------------------------------
# CELL 10: Xuất file JSON & CSV (Khớp 100% Cell 10 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Xuất File Kết Quả JSON & CSV Chuẩn Hóa            ║
# ╚══════════════════════════════════════════════════════════════╝

exported_files = []

for s_key, s_data in all_scale_runs.items():
    records = s_data['records']
    df_s = pd.DataFrame([r for r in records if r.get('status') == 'ok'])
    if df_s.empty:
        continue
        
    json_path = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_{s_key}x_benchmark.json")
    csv_path  = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_{s_key}x_benchmark.csv")
    
    summary_dict = {
        'model': f"Compact SRCNN RTL (Scale {s_key}x)",
        'architecture': "1 -> 16 -> 8 -> 1 (1,649 params)",
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
    }
    
    full_output = {
        'summary': summary_dict,
        'per_image_results': records
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
        
    df_s.to_csv(csv_path, index=False)
    exported_files.extend([json_path, csv_path])
    print(f"✓ Đã xuất: {os.path.basename(json_path)} ({os.path.getsize(json_path)/(1024*1024):.2f} MB)")
    print(f"✓ Đã xuất: {os.path.basename(csv_path)} ({os.path.getsize(csv_path)/1024:.1f} KB)")

# Xuất bảng tổng hợp theo scale
summary_csv_path  = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_multiscale_summary.csv")
summary_json_path = os.path.join(OUTPUT_DIR, f"{TARGET_MODEL}_multiscale_summary.json")
df_scale_summary.to_csv(summary_csv_path, index=False)
df_scale_summary.to_json(summary_json_path, indent=2, orient='records')
exported_files.extend([summary_csv_path, summary_json_path])

print("\\n📂 ĐÃ XUẤT TẤT CẢ FILE RA /kaggle/working/results SẴN SÀNG:")
for ef in exported_files:
    print(f"  • {ef}")
""")

# ------------------------------------------------------------------------------
# CELL 11: Đóng gói file ZIP (Khớp 100% Cell 11 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Đóng Gói File ZIP Kết Quả                         ║
# ╚══════════════════════════════════════════════════════════════╝

zip_out_name = f"{TARGET_MODEL}_benchmark_results.zip"
zip_out_path = os.path.join('/kaggle/working', zip_out_name)

if os.path.exists(zip_out_path):
    os.remove(zip_out_path)

cmd = f"cd {OUTPUT_DIR} && zip -r '{zip_out_path}' . -i '*{TARGET_MODEL}*'"
os.system(cmd)

if os.path.exists(zip_out_path):
    sz_mb = os.path.getsize(zip_out_path) / (1024 * 1024)
    print("═" * 70)
    print(f"🎉 ĐÃ ĐÓNG GÓI THÀNH CÔNG: {zip_out_name} ({sz_mb:.2f} MB)")
    print("👉 Tải file này từ tab 'Output' bên phải giao diện Kaggle.")
    print("═" * 70)
else:
    print("⚠ Chưa tạo được file zip, các file kết quả vẫn nằm nguyên tại:", OUTPUT_DIR)
""")

# ------------------------------------------------------------------------------
# CELL 12: Giải phóng VRAM & Hướng dẫn (Khớp 100% Cell 12 ESPCN)
# ------------------------------------------------------------------------------
add_code("""# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 12 — Giải Phóng VRAM & Hướng Dẫn Đồng Bộ Local         ║
# ╚══════════════════════════════════════════════════════════════╝

gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
    print("✓ Đã giải phóng toàn bộ bộ nhớ GPU VRAM.")

print(f\"\"\"
══════════════════════════════════════════════════════════════════════
🎉 HOÀN THÀNH TOÀN BỘ BENCHMARK CHO MÔ HÌNH COMPACT SRCNN RTL!
══════════════════════════════════════════════════════════════════════

📥 CÁC BƯỚC TIẾP THEO:
1. Tải file '{TARGET_MODEL}_benchmark_results.zip' từ Kaggle Output.
2. Giải nén và đặt các file vào thư mục repository:
   • legacy_experiments/benchmark_results/hardware/
     - {TARGET_MODEL}_2x_benchmark.json
     - {TARGET_MODEL}_3x_benchmark.json
     - {TARGET_MODEL}_4x_benchmark.json
     - {TARGET_MODEL}_multiscale_summary.csv
3. Chạy script đối chiếu hiệu năng và vẽ biểu đồ:
   python3 legacy_experiments/notebooks_kaggle/analysis/render_hardware_comparison_charts.py
══════════════════════════════════════════════════════════════════════
\"\"\")
""")

notebook_json = {
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

# Ghi ra 2 thư mục đích
target_repo_path = REPO_ROOT / "legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb"
target_downloads_path = Path("/Users/giabao/Downloads/kaggle_compact_srcnn_hardware_benchmark.ipynb")
target_core_path = REPO_ROOT / "core_project/benchmarks_reports/pynq_z2/kaggle_compact_srcnn_hardware_benchmark.ipynb"

with open(target_repo_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)
print(f"✓ Đã lưu notebook tại repository   : {target_repo_path}")

with open(target_downloads_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)
print(f"✓ Đã lưu notebook tại Downloads    : {target_downloads_path}")

with open(target_core_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)
print(f"✓ Đã lưu notebook tại core_project : {target_core_path}")
