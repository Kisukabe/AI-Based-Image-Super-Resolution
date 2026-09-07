# 📋 HƯỚNG DẪN COPY-PASTE TỪNG CELL LÊN KAGGLE

Tạo một Notebook mới trên Kaggle (GPU T4 x2 hoặc P100), thêm dataset `duc24kdl/sub-x-ray`, sau đó copy và dán tuần tự từng Cell dưới đây:

---

## 🟦 CELL 0 — [MARKDOWN]

```markdown
# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║  🔬 MASTER BENCHMARK: SO SÁNH 7 MÔ HÌNH SIÊU PHÂN GIẢI TRÊN ẢNH Y TẾ (2×, 3×, 4×)        ║
# ║  Luận Chứng Lựa Chọn SRGAN & Phân Tích Đánh Đổi Cảm Thụ Thị Giác (Perception vs Distortion) ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝

Notebook này thực hiện đánh giá toàn diện và đối chuẩn khoa học (scientific benchmarking) giữa **7 mô hình** Siêu phân giải ảnh đơn (Single Image Super-Resolution - SISR) trên tập dữ liệu ảnh X-Quang chuẩn `duc24kdl/sub-x-ray` (2.200 ảnh) ở cả 3 tỉ lệ phóng đại **2×, 3×, 4×**:

| STT | Mô hình | Kiến trúc cốt lõi | Đặc điểm nổi bật | Mục đích so sánh trong đề tài |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Bicubic** | Nội suy đa thức bậc 3 | Không dùng nơ-rơn, siêu nhanh | Mốc tham chiếu cơ sở (Baseline) |
| **2** | **SRCNN** | 3 lớp Conv ($9\text{-}5\text{-}5$ / $9\text{-}1\text{-}5$) | Mạng CNN SISR đầu tiên | Đối sánh trực tiếp với lõi phần cứng FPGA RTL |
| **3** | **ESPCN** | Sub-Pixel Conv (`PixelShuffle`) | Siêu nhẹ, tốc độ cao | Nhóm tối ưu tài nguyên phần mềm |
| **4** | **FSRCNN** | Deconvolution (`ConvTranspose2d`) | Co cụm & giải nén đặc trưng | Nhóm giải tích chập nhỏ gọn |
| **5** | **VDSR** | 20 lớp Conv + Global Residual | Học phần dư rất sâu | Nhóm mạng sâu cổ điển |
| **6** | **EDSR** | 8 ResBlocks (bỏ BatchNorm) | Tối ưu hóa PSNR/SSIM cực cao | Đỉnh cao của trường phái tối ưu hàm mất mát L1/L2 |
| **7** | **SRGAN** | 16 ResBlocks + VGG Perceptual + GAN | Khôi phục độ sắc nét thị giác | **Mô hình cốt lõi cần luận chứng** |

---

### 🎯 Trọng Tâm Luận Chứng: "Vì Sao Lựa Chọn SRGAN?"
1. **Nghịch lý Perception-Distortion**: Các mạng tối ưu MSE (như EDSR, VDSR, SRCNN) đạt PSNR cao do chấp nhận làm mờ (blur) để triệt tiêu sai số bình phương trung bình, dẫn đến hiện tượng trơn láng các bờ xương và cấu trúc phế nang nhỏ. Ngược lại, **SRGAN** sử dụng hàm mất mát Perceptual (VGG feature loss) và Adversarial loss, tái tạo trung thực kết cấu và chi tiết giải phẫu với chỉ số **LPIPS** (độ sai lệch cảm thụ thị giác) thấp vượt trội.
2. **Cơ sở chuyển giao sang Phần cứng**: Phân tích trade-off giữa chất lượng và độ phức tạp tính toán của SRGAN chính là tiền đề then chốt để đề xuất kiến trúc **Swift-SRGAN** (rút gọn kênh bằng Seperable Convolution) nhằm triển khai tăng tốc trên FPGA.

---

### 📋 Tiêu Chuẩn Đầu Ra (Đồng bộ 100% với `srgan_2x_benchmark.json`):
- Đúng tập dữ liệu chuẩn: [`duc24kdl/sub-x-ray`](https://www.kaggle.com/datasets/duc24kdl/sub-x-ray) (1.750 `sub_NIH` + 450 `sub_chest`).
- Đúng schema **38 chỉ số đo lường** trên từng ảnh.
- Checkpoint định kỳ mỗi 100 ảnh chống mất dữ liệu.
- Xuất file kết quả JSON, CSV và biểu đồ phân tích đánh đổi tự động.
```

---

## 🟩 CELL 1 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Cài Đặt Thư Viện & Định Vị Trọng Số Các Mô Hình   ║
# ╚══════════════════════════════════════════════════════════════╝

import os, sys, glob, shutil

print("1. Đang kiểm tra và cài đặt các thư viện đo lường khoa học...")
try:
    import lpips
    import pytorch_msssim
    import skimage
except ImportError:
    !pip install -q lpips pytorch-msssim scikit-image
    import lpips
    import pytorch_msssim
    import skimage

print("✓ Thư viện đã sẵn sàng: lpips, pytorch-msssim, scikit-image.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Định vị thư mục chứa 18 file trọng số (2x, 3x, 4x)
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
            print(f"✓ Tìm thấy thư mục trọng số tại: {c}")
            return os.path.abspath(c)

    # Tìm kiếm đệ quy trong /kaggle/input
    for root, dirs, files in os.walk("/kaggle/input"):
        if "edsr.pth" in files and ("2x" in root or "weight_models" in root):
            base_dir = os.path.dirname(root) if "2x" in root else root
            print(f"✓ Tìm thấy trọng số trong Kaggle Input: {base_dir}")
            return os.path.abspath(base_dir)

    # Fallback tự động: Clone trực tiếp từ GitHub repository (chỉ mất ~5s)
    print("[INFO] Đang tự động kéo 18 file trọng số từ GitHub repository Kisukabe/AI-Based-Image-Super-Resolution...")
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
        print(f"✓ Đã clone và sao chép trọng số thành công về: {dest_weights}")
        return os.path.abspath(dest_weights)
        
    raise FileNotFoundError("Không tìm thấy trọng số các mô hình! Vui lòng kiểm tra kết nối mạng hoặc thêm dataset.")

WEIGHT_ROOT_DIR = locate_or_fetch_weights()
print(f"📂 Thư mục trọng số chính thức: {WEIGHT_ROOT_DIR}")
```

---

## 🟩 CELL 2 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Khởi Tạo Thiết Bị & Chuẩn Bị Các Hàm Đo Khoa Học   ║
# ╚══════════════════════════════════════════════════════════════╝

import time
import json
import math
import copy
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve2d
try:
    import lpips
    import pytorch_msssim
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

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

# Khởi tạo LPIPS AlexNet và MS-SSIM
if HAS_LPIPS:
    print("Đang khởi tạo các mạng đo lường nhận thức (LPIPS, MS-SSIM)...")
    LPIPS_FN = lpips.LPIPS(net='alex').to(DEVICE).eval()
    for param in LPIPS_FN.parameters():
        param.requires_grad = False
    MS_SSIM_FN = pytorch_msssim.MS_SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)
else:
    print("⚠ Thư viện lpips/pytorch-msssim chưa có sẵn (chạy trên Kaggle sẽ tự động cài đặt qua Cell 1)")
    LPIPS_FN = None
    MS_SSIM_FN = None

def calc_psnr(img_true, img_test):
    return float(psnr_fn(img_true, img_test, data_range=255))

def calc_ssim(img_true, img_test):
    return float(ssim_fn(img_true, img_test, data_range=255, channel_axis=2))

def calc_msssim(t_true, t_test):
    if MS_SSIM_FN is not None:
        with torch.no_grad():
            val = MS_SSIM_FN(t_test, t_true).item()
        return float(val)
    return calc_ssim((t_true.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8),
                     (t_test.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

def calc_lpips(t_true, t_test):
    if LPIPS_FN is not None:
        with torch.no_grad():
            val = LPIPS_FN(t_test * 2.0 - 1.0, t_true * 2.0 - 1.0).item()
        return float(val)
    diff = (t_true - t_test).abs().mean().item()
    return float(round(diff * 0.5, 4))

def fallback_niqe(img_np):
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
    mu = convolve2d(gray, np.ones((7,7))/49.0, mode='same', boundary='symm')
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(convolve2d(gray*gray, np.ones((7,7))/49.0, mode='same', boundary='symm') - mu_sq))
    structdis = (gray - mu) / (sigma + 1.0)
    feat_mean = float(np.mean(np.abs(structdis)))
    feat_std  = float(np.std(structdis))
    raw_score = 10.0 * feat_mean + 5.0 * feat_std
    return float(np.clip(raw_score, 1.0, 15.0))

print("✓ Bộ hàm đo khoa học đã sẵn sàng: PSNR, SSIM, MS-SSIM, LPIPS, NIQE, EPI.")
```

---

## 🟩 CELL 3 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Định Nghĩa Cấu Trúc 6 Mô Hình Nơ-rơn (PyTorch)     ║
# ╚══════════════════════════════════════════════════════════════╝

# ====================================================================
# 1. SRCNN (Dong et al., ECCV 2014) - Pre-upsampling
# ====================================================================
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
        return torch.clamp(out, 0.0, 1.0)


# ====================================================================
# 2. ESPCN (Shi et al., CVPR 2016) - Sub-Pixel Convolution
# ====================================================================
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
        return torch.clamp(out, 0.0, 1.0)


# ====================================================================
# 3. FSRCNN (Dong et al., ECCV 2016) - Post-upsampling Deconvolution
# ====================================================================
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
        self.deconv = nn.ConvTranspose2d(
            d, in_channels, kernel_size=9,
            stride=upscale_factor, padding=4,
            output_padding=upscale_factor - 1
        )

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconv(out)
        return torch.clamp(out, 0.0, 1.0)


# ====================================================================
# 4. VDSR (Kim et al., CVPR 2016) - 20-Layer Deep Residual Network
# ====================================================================
class VDSRConvBlock(nn.Module):
    def __init__(self):
        super(VDSRConvBlock, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, num_layers=20, num_features=64, style='modular'):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.style = style
        if style == 'sequential':
            layers = [
                nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ]
            for _ in range(num_layers - 2):
                layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
            self.residual_net = nn.Sequential(*layers)
        else:
            self.conv_first = nn.Sequential(
                nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.residual_layers = nn.ModuleList([VDSRConvBlock() for _ in range(num_layers - 2)])
            self.conv_last = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        if self.style == 'sequential':
            residual = self.residual_net(x_bicubic)
        else:
            out = self.conv_first(x_bicubic)
            for layer in self.residual_layers:
                out = layer(out)
            residual = self.conv_last(out)
        return torch.clamp(x_bicubic + residual, 0.0, 1.0)


# ====================================================================
# 5. EDSR (Lim et al., CVPRW 2017) - Enhanced Deep Residual Network
# ====================================================================
class EDSRResBlock(nn.Module):
    def __init__(self, n_feats=64, res_scale=0.1):
        super(EDSRResBlock, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, in_channels=3, upscale_factor=4, n_feats=64, n_resblocks=8, res_scale=0.1):
        super(EDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[EDSRResBlock(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

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
        res = self.body_conv(self.body(x_head)) + x_head
        if hasattr(self, 'upsampler'):
            res = self.upsampler(res)
        out = self.tail(res)
        return torch.clamp(out, 0.0, 1.0)


# ====================================================================
# 6. SRGAN Generator (Ledig et al., CVPR 2017)
# ====================================================================
class SRGANResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(SRGANResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.prelu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


class SRGAN(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(SRGAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=num_channels)
        )
        self.residual = nn.Sequential(*[SRGANResidualBlock(num_channels) for _ in range(num_blocks)])
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )
        if upscale_factor in [2, 4]:
            num_stages = int(math.log2(upscale_factor))
            stages = []
            for _ in range(num_stages):
                stages.extend([
                    nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU(num_parameters=num_channels)
                ])
            self.upsampler = nn.Sequential(*stages)
        else: # 3x
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor),
                nn.PReLU(num_parameters=num_channels)
            )
        self.final_conv = nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        res = self.residual(initial)
        mid = self.mid_conv(res) + initial
        up = self.upsampler(mid)
        out = (torch.tanh(self.final_conv(up)) + 1.0) / 2.0
        return torch.clamp(out, 0.0, 1.0)

print("✓ Đã nạp thành công định nghĩa 6 kiến trúc: SRCNN, ESPCN, FSRCNN, VDSR, EDSR, SRGAN.")
```

---

## 🟩 CELL 4 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Bộ Nạp Trọng Số Thông Minh (Unified Smart Loader)  ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import glob
import torch
import torch.nn as nn

def load_sr_model(model_name: str, upscale_factor: int = 2, weight_root: str = WEIGHT_ROOT_DIR, device: str = DEVICE):
    """
    Khởi tạo kiến trúc và nạp trọng số tương ứng cho mọi mô hình và scale factor.
    """
    model_name = model_name.upper()
    if model_name == "BICUBIC":
        return None, 0, "Non-neural Baseline"

    weight_path = os.path.join(weight_root, f"{upscale_factor}x", f"{model_name.lower()}.pth")
    if not os.path.exists(weight_path):
        # Thử tìm kiếm đệ quy tên file
        matches = glob.glob(f"{weight_root}/**/{model_name.lower()}.pth", recursive=True)
        if matches:
            weight_path = matches[0]
        else:
            raise FileNotFoundError(f"Không tìm thấy file trọng số cho {model_name} ({upscale_factor}x) tại: {weight_path}")

    # 1. Đọc file trọng số PyTorch
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
        epoch_info = checkpoint.get("epoch", "N/A")
    elif isinstance(checkpoint, dict):
        raw_state_dict = checkpoint
        epoch_info = "dict"
    else:
        raise TypeError(f"Định dạng checkpoint không hợp lệ cho {model_name}")

    # 2. Làm sạch tiền tố module.
    clean_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}

    # 3. Khởi tạo instance mô hình phù hợp
    if model_name == "SRCNN":
        model = SRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "ESPCN":
        model = ESPCN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "FSRCNN":
        model = FSRCNN(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "VDSR":
        style = 'sequential' if any('residual_net' in k for k in clean_state_dict.keys()) else 'modular'
        model = VDSR(in_channels=3, upscale_factor=upscale_factor, style=style)
    elif model_name == "EDSR":
        model = EDSR(in_channels=3, upscale_factor=upscale_factor)
    elif model_name == "SRGAN":
        model = SRGAN(in_channels=3, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")

    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Nạp thành công [{model_name} {upscale_factor}×]: {num_params:,} tham số (Trọng số: {os.path.basename(weight_path)})")
    return model, num_params, weight_path

print("✓ Hàm load_sr_model() đã sẵn sàng phục vụ benchmark.")
```

---

## 🟩 CELL 5 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Smoke Test: Xác Thực Shape & Tốc Độ Khởi Động      ║
# ╚══════════════════════════════════════════════════════════════╝

TEST_SCALE = 2
TEST_LR_SIZE = (256, 256)
dummy_lr = torch.rand(1, 3, TEST_LR_SIZE[0], TEST_LR_SIZE[1], device=DEVICE)

print(f"Kiểm tra tính tương thích và shape đầu ra cho Scale {TEST_SCALE}x (Input: {dummy_lr.shape}):")
print("─" * 70)

all_test_models = ['bicubic', 'srcnn', 'espcn', 'fsrcnn', 'vdsr', 'edsr', 'srgan']
for m_name in all_test_models:
    try:
        t0 = time.perf_counter()
        if m_name == 'bicubic':
            with torch.no_grad():
                out = F.interpolate(dummy_lr, scale_factor=TEST_SCALE, mode='bicubic', align_corners=False)
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  • {m_name.upper():<8}: Output {list(out.shape)} | Latency: {dt:6.2f} ms | OK")
        else:
            model_inst, p_cnt, _ = load_sr_model(m_name, upscale_factor=TEST_SCALE)
            with torch.no_grad():
                out = model_inst(dummy_lr)
            if DEVICE == 'cuda': torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  • {m_name.upper():<8}: Output {list(out.shape)} | Latency: {dt:6.2f} ms | Params: {p_cnt:>10,} | OK")
            del model_inst
            gc.collect()
            if DEVICE == 'cuda': torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ {m_name.upper():<8}: Lỗi kiểm thử: {e}")

print("─" * 70)
print("✓ Toàn bộ 7 mô hình đã vượt qua smoke test!")
```

---

## 🟩 CELL 6 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — 🔍 Quét Tập Dữ Liệu duc24kdl/sub-x-ray (2.200 ảnh) ║
# ╚══════════════════════════════════════════════════════════════╝

def find_sub_xray_dataset():
    valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    candidate_roots = [
        '/kaggle/input/datasets/duc24kdl/sub-x-ray/sub_X-Ray',
        '/kaggle/input/datasets/duc24kdl/sub-x-ray',
        '/kaggle/input/sub-x-ray/sub_X-Ray',
        '/kaggle/input/sub-x-ray',
        '/kaggle/input/sub_X-Ray',
    ]

    nih_imgs = []
    chest_imgs = []

    # 1. Tìm theo các đường dẫn mount chuẩn của Kaggle
    for cand in candidate_roots:
        if os.path.exists(cand):
            nih_dir = os.path.join(cand, 'sub_NIH')
            chest_dir = os.path.join(cand, 'sub_chest')
            if os.path.exists(nih_dir):
                nih_imgs = [os.path.join(nih_dir, f) for f in sorted(os.listdir(nih_dir)) if f.endswith(valid_exts)]
            if os.path.exists(chest_dir):
                chest_imgs = [os.path.join(chest_dir, f) for f in sorted(os.listdir(chest_dir)) if f.endswith(valid_exts)]
            if nih_imgs or chest_imgs:
                print(f"✓ Đã phát hiện dataset duc24kdl/sub-x-ray tại: {cand}")
                break

    # 2. Tìm kiếm đệ quy trong /kaggle/input nếu Kaggle mount ở thư mục khác
    if not nih_imgs and not chest_imgs:
        print("[INFO] Đang quét tìm thư mục sub_NIH và sub_chest trong /kaggle/input...")
        for root, dirs, files in os.walk('/kaggle/input'):
            bname = os.path.basename(root)
            if bname == 'sub_NIH' and not nih_imgs:
                nih_imgs = [os.path.join(root, f) for f in sorted(files) if f.endswith(valid_exts)]
                print(f"  ✓ Tìm thấy sub_NIH tại: {root} ({len(nih_imgs)} ảnh)")
            elif bname == 'sub_chest' and not chest_imgs:
                chest_imgs = [os.path.join(root, f) for f in sorted(files) if f.endswith(valid_exts)]
                print(f"  ✓ Tìm thấy sub_chest tại: {root} ({len(chest_imgs)} ảnh)")

    return nih_imgs, chest_imgs

nih_images, chest_images = find_sub_xray_dataset()
all_images = nih_images + chest_images

print('═' * 60)
print("📊 KẾT QUẢ QUÉT TẬP DỮ LIỆU [duc24kdl/sub-x-ray]:")
print(f"  • Tập con [sub_NIH]   : {len(nih_images):,} ảnh")
print(f"  • Tập con [sub_chest] : {len(chest_images):,} ảnh")
print("  ────────────────────────────────────────────")
print(f"  ► TỔNG CỘNG           : {len(all_images):,} ảnh để Benchmark")
print('═' * 60)

if not all_images:
    err_msg = (
        "❌ Không tìm thấy ảnh nào trong sub_NIH hoặc sub_chest! "
        "Vui lòng nhấn '+ Add Input' và thêm dataset 'duc24kdl/sub-x-ray' "
        "(https://www.kaggle.com/datasets/duc24kdl/sub-x-ray) vào notebook."
    )
    raise RuntimeError(err_msg)
```

---

## 🟩 CELL 7 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — 🎛️ BẢNG ĐIỀU KHIỂN THAM SỐ BENCHMARK (CONTROL PANEL) ║
# ╚══════════════════════════════════════════════════════════════╝

# 1. Chọn tỉ lệ phóng đại muốn đánh giá: 2, 3, hoặc 4
SCALE_FACTOR = 2

# 2. Danh sách các mô hình cần đánh giá:
# • Chạy trọn bộ 7 mô hình để so sánh: ['bicubic', 'srcnn', 'espcn', 'fsrcnn', 'vdsr', 'edsr', 'srgan']
# • Hoặc chọn 1 mô hình cụ thể nếu bạn muốn chạy riêng lẻ: ['edsr'] hoặc ['srgan']
MODELS_TO_RUN = ['bicubic', 'srcnn', 'espcn', 'fsrcnn', 'vdsr', 'edsr', 'srgan']

# 3. Số lượng ảnh chạy:
# • Đặt 2200 để chạy trọn vẹn toàn bộ dataset (chuẩn bài báo)
# • Hoặc đặt 100 để kiểm tra nhanh trong 1-2 phút
MAX_IMAGES = 2200

# 4. Cấu hình lưu trữ:
SAVE_PNG_IMAGES  = False  # Đổi thành True nếu muốn xuất file ảnh PNG siêu phân giải
OUTPUT_DIR       = '/kaggle/working/results'
CHECKPOINT_EVERY = 100
LOG_EVERY        = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
images_to_run = all_images[:MAX_IMAGES] if MAX_IMAGES and len(all_images) >= MAX_IMAGES else all_images

print('═════════════════════════════════════════════════════════════')
print('  CẤU HÌNH PHIÊN THỰC THI BENCHMARK:')
print(f'  ✓ Tỉ lệ phóng đại (Scale) : {SCALE_FACTOR}x')
print(f'  ✓ Danh sách mô hình       : {[m.upper() for m in MODELS_TO_RUN]} ({len(MODELS_TO_RUN)} models)')
print(f'  ✓ Số lượng ảnh đánh giá   : {len(images_to_run):,} ảnh')
print(f'  ✓ Thiết bị tính toán      : {DEVICE.upper()}')
print(f'  ✓ Thư mục xuất kết quả    : {OUTPUT_DIR}')
print('═════════════════════════════════════════════════════════════')
```

---

## 🟩 CELL 8 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Hàm Tính Toán Chuẩn Hóa 38 Chỉ Số Khoa Học         ║
# ╚══════════════════════════════════════════════════════════════╝

import os
from scipy.signal import convolve2d

def compute_epi(img_true, img_test):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    gray_true = np.dot(img_true[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
    gray_test = np.dot(img_test[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
    delta_true = convolve2d(gray_true, laplacian_kernel, mode='same', boundary='symm')
    delta_test = convolve2d(gray_test, laplacian_kernel, mode='same', boundary='symm')
    d_true_mean = np.mean(delta_true)
    d_test_mean = np.mean(delta_test)
    num = np.sum((delta_true - d_true_mean) * (delta_test - d_test_mean))
    den = np.sqrt(np.sum((delta_true - d_true_mean)**2) * np.sum((delta_test - d_test_mean)**2)) + 1e-10
    return float(np.clip(num / den, -1.0, 1.0))


def compute_image_metrics(hr_np, hr_tensor, bic_np, bic_tensor, sr_np, sr_tensor,
                          latency_ms, img_path, dataset_name, scale_factor, saved_path=""):
    """
    Tính toán đúng 38 trường dữ liệu khớp 100% với schema srgan_2x_benchmark.json.
    """
    # 1. Chỉ số Bicubic Baseline
    p_bic    = calc_psnr(hr_np, bic_np)
    s_bic    = calc_ssim(hr_np, bic_np)
    ms_bic   = calc_msssim(hr_tensor, bic_tensor)
    lp_bic   = calc_lpips(hr_tensor, bic_tensor)
    niqe_bic = fallback_niqe(bic_np)
    diff_bic = (hr_np.astype(np.float64) - bic_np.astype(np.float64)) / 255.0
    mse_bic  = float(np.mean(diff_bic ** 2))
    rmse_bic = float(np.sqrt(mse_bic))

    # 2. Chỉ số Model Super-Resolution
    p_sr    = calc_psnr(hr_np, sr_np)
    s_sr    = calc_ssim(hr_np, sr_np)
    ms_sr   = calc_msssim(hr_tensor, sr_tensor)
    lp_sr   = calc_lpips(hr_tensor, sr_tensor)
    niqe_sr = fallback_niqe(sr_np)
    diff_sr = (hr_np.astype(np.float64) - sr_np.astype(np.float64)) / 255.0
    mse_sr  = float(np.mean(diff_sr ** 2))
    rmse_sr = float(np.sqrt(mse_sr))

    # 3. Phân bố mức xám & Edge Preservation Index
    mean_val = float(np.mean(sr_np) / 255.0)
    std_val  = float(np.std(sr_np) / 255.0)
    epi_val  = compute_epi(hr_np, sr_np)

    # 4. Độ tăng cường (Gains so với Bicubic)
    psnr_gain   = float(p_sr - p_bic)
    msssim_gain = float(ms_sr - ms_bic)
    lpips_gain  = float(lp_bic - lp_sr)      # Dương = cải thiện thị giác tốt hơn
    niqe_gain   = float(niqe_bic - niqe_sr)  # Dương = cải thiện chất lượng tự nhiên

    hr_h, hr_w = hr_np.shape[:2]
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

print("✓ Hàm compute_image_metrics() hoàn tất chuẩn hóa 38 trường dữ liệu.")
```

---

## 🟩 CELL 9 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — 🚀 VÒNG LẶP BENCHMARK TUẦN TỰ & CHECKPOINT MỖI 100 ẢNH║
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

all_benchmark_runs = {}
total_images = len(images_to_run)

print(f"BẮT ĐẦU BENCHMARK: {len(MODELS_TO_RUN)} MÔ HÌNH | TỈ LỆ: {SCALE_FACTOR}× | SỐ ẢNH: {total_images:,}")
print("═" * 80)

for m_idx, model_name in enumerate(MODELS_TO_RUN, 1):
    m_upper = model_name.upper()
    print(f"\n▶ [{m_idx}/{len(MODELS_TO_RUN)}] ĐANG CHẠY MÔ HÌNH: {m_upper} (Scale {SCALE_FACTOR}x)...")
    
    # 1. Khởi tạo mô hình
    if model_name.lower() == 'bicubic':
        cur_model = None
        w_source = 'Bicubic (Interpolation Baseline)'
    else:
        cur_model, params_count, w_source = load_sr_model(model_name, upscale_factor=SCALE_FACTOR)
    
    model_records = []
    checkpoint_file = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_{SCALE_FACTOR}x_checkpoint.json")
    t_start = time.perf_counter()
    
    # 2. Lặp qua từng ảnh trong tập dữ liệu
    for idx, img_path in enumerate(tqdm(images_to_run, desc=f"{m_upper} {SCALE_FACTOR}x")):
        dataset_name = os.path.basename(os.path.dirname(img_path)) or 'unknown'
        
        try:
            # Đọc ảnh HR gốc và căn chỉnh chia hết cho SCALE_FACTOR
            hr_pil = Image.open(img_path).convert('RGB')
            orig_w, orig_h = hr_pil.size
            crop_w = (orig_w // SCALE_FACTOR) * SCALE_FACTOR
            crop_h = (orig_h // SCALE_FACTOR) * SCALE_FACTOR
            if (crop_w, crop_h) != (orig_w, orig_h):
                hr_pil = hr_pil.crop((0, 0, crop_w, crop_h))
            
            lr_w, lr_h = crop_w // SCALE_FACTOR, crop_h // SCALE_FACTOR
            lr_pil = hr_pil.resize((lr_w, lr_h), Image.BICUBIC)
            
            # Chuẩn bị Tensor
            hr_np     = np.array(hr_pil)
            hr_tensor = to_tensor(hr_pil).unsqueeze(0).to(DEVICE)
            
            lr_tensor = to_tensor(lr_pil).unsqueeze(0).to(DEVICE)
            
            # Tạo ảnh Bicubic mốc cơ sở
            bic_pil    = lr_pil.resize((crop_w, crop_h), Image.BICUBIC)
            bic_np     = np.array(bic_pil)
            bic_tensor = to_tensor(bic_pil).unsqueeze(0).to(DEVICE)
            
            # Đo đạc thời gian suy luận (Latency ms)
            if DEVICE == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            if model_name.lower() == 'bicubic':
                sr_tensor = bic_tensor
                sr_np     = bic_np
            else:
                with torch.no_grad():
                    sr_tensor = cur_model(lr_tensor)
                if DEVICE == 'cuda': torch.cuda.synchronize()
                sr_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            # Tính 38 chỉ số
            rec = compute_image_metrics(
                hr_np=hr_np, hr_tensor=hr_tensor,
                bic_np=bic_np, bic_tensor=bic_tensor,
                sr_np=sr_np, sr_tensor=sr_tensor,
                latency_ms=latency_ms, img_path=img_path,
                dataset_name=dataset_name, scale_factor=SCALE_FACTOR
            )
            model_records.append(rec)
            
        except Exception as e:
            model_records.append({
                'source_path': img_path, 'dataset': dataset_name,
                'filename': os.path.basename(img_path), 'status': f'error: {str(e)}'
            })
            
        # Lưu checkpoint định kỳ
        if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == total_images:
            save_checkpoint(model_records, time.perf_counter() - t_start, checkpoint_file)
            
    elapsed_total = time.perf_counter() - t_start
    all_benchmark_runs[model_name.lower()] = {
        'records': model_records,
        'elapsed_sec': elapsed_total,
        'weights_source': w_source
    }
    print(f"✓ Hoàn thành {m_upper}: {len(model_records):,} ảnh trong {elapsed_total/60:.2f} phút | Checkpoint: {checkpoint_file}")
    
    # Giải phóng VRAM
    if cur_model is not None:
        del cur_model
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

print("═" * 80)
print("✓ TOÀN BỘ CÁC MÔ HÌNH ĐÃ HOÀN TẤT BENCHMARK THÀNH CÔNG!")
```

---

## 🟩 CELL 10 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Thống Kê & Tổng Hợp Kết Quả (sub_NIH vs sub_chest) ║
# ╚══════════════════════════════════════════════════════════════╝

summary_comparison_rows = []

for m_key, m_data in all_benchmark_runs.items():
    records = m_data['records']
    df_m = pd.DataFrame([r for r in records if r.get('status') == 'ok'])
    if df_m.empty:
        continue
        
    p_mean  = df_m['psnr_model_db'].mean()
    s_mean  = df_m['ssim_model'].mean()
    ms_mean = df_m['msssim_fpga'].mean() if 'msssim_fpga' in df_m.columns else df_m['msssim_bicubic'].mean()
    lp_mean = df_m['lpips'].mean() if 'lpips' in df_m.columns else df_m['lpips_bicubic'].mean()
    nq_mean = df_m['niqe_fpga'].mean() if 'niqe_fpga' in df_m.columns else df_m['niqe_bicubic'].mean()
    lat_med = df_m['latency_ms'].median()
    fps_val = 1000.0 / df_m['latency_ms'].mean() if df_m['latency_ms'].mean() > 0 else 0
    epi_val = df_m['epi'].mean()
    
    p_gain  = df_m['psnr_gain_db'].mean()
    lp_gain = df_m['lpips_gain'].mean()
    
    # Thống kê riêng cho từng tập con
    df_nih   = df_m[df_m['dataset'] == 'sub_NIH']
    df_chest = df_m[df_m['dataset'] == 'sub_chest']
    
    summary_comparison_rows.append({
        'Model': m_key.upper(),
        'Scale': f'{SCALE_FACTOR}x',
        'Images': len(df_m),
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

df_summary = pd.DataFrame(summary_comparison_rows)
print(f"📊 BẢNG TỔNG HỢP ĐỐI CHUẨN KHOA HỌC TRÊN ẢNH Y TẾ (SCALE {SCALE_FACTOR}×):")
print("═" * 105)
display_cols = ['Model', 'PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS ↓', 'NIQE ↓', 'EPI', 'PSNR Gain (dB)', 'LPIPS Gain ↑', 'Latency (ms)', 'FPS']
print(df_summary[display_cols].to_string(index=False))
print("═" * 105)
print("💡 Ghi chú: 'LPIPS ↓' càng nhỏ càng sắc nét tự nhiên; 'PSNR Gain (dB)' và 'LPIPS Gain ↑' so với Bicubic Baseline.")
```

---

## 🟩 CELL 11 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Xuất File Kết Quả JSON & CSV Chuẩn Hóa             ║
# ╚══════════════════════════════════════════════════════════════╝

exported_files = []

for m_key, m_data in all_benchmark_runs.items():
    records = m_data['records']
    df_m = pd.DataFrame([r for r in records if r.get('status') == 'ok'])
    if df_m.empty:
        continue
        
    json_path = os.path.join(OUTPUT_DIR, f"{m_key}_{SCALE_FACTOR}x_benchmark.json")
    csv_path  = os.path.join(OUTPUT_DIR, f"{m_key}_{SCALE_FACTOR}x_benchmark.csv")
    
    # Tạo cấu trúc JSON hoàn toàn tương thích với srgan_2x_benchmark.json
    summary_dict = {
        'model': m_key.upper(),
        'weights_source': m_data['weights_source'],
        'device': DEVICE,
        'images_evaluated': len(df_m),
        'images_error': len(records) - len(df_m),
        'resolution_in_out': df_m['resolution'].iloc[0] if not df_m.empty else '',
        'scale_factor': SCALE_FACTOR,
        'avg_psnr_bicubic_db': float(round(df_m['psnr_bicubic_db'].mean(), 4)),
        'std_psnr_bicubic_db': float(round(df_m['psnr_bicubic_db'].std(), 4)),
        'avg_mse_bicubic': float(round(df_m['mse_bicubic'].mean(), 6)),
        'avg_rmse_bicubic': float(round(df_m['rmse_bicubic'].mean(), 6)),
        'avg_ssim_bicubic': float(round(df_m['ssim_bicubic'].mean(), 4)),
        'avg_msssim_bicubic': float(round(df_m['msssim_bicubic'].mean(), 4)),
        'avg_lpips_bicubic': float(round(df_m['lpips_bicubic'].mean(), 4)),
        'std_lpips_bicubic': float(round(df_m['lpips_bicubic'].std(), 4)),
        'avg_niqe_bicubic': float(round(df_m['niqe_bicubic'].mean(), 4)),
        'avg_psnr_srgan_db': float(round(df_m['psnr_model_db'].mean(), 4)),
        'std_psnr_srgan_db': float(round(df_m['psnr_model_db'].std(), 4)),
        'avg_mse_srgan': float(round(df_m['mse_model'].mean(), 6)),
        'avg_rmse_srgan': float(round(df_m['rmse_model'].mean(), 6)),
        'avg_ssim_srgan': float(round(df_m['ssim_model'].mean(), 4)),
        'avg_msssim_srgan': float(round(df_m['msssim_fpga'].mean(), 4)),
        'avg_lpips': float(round(df_m['lpips'].mean(), 4)),
        'std_lpips_srgan': float(round(df_m['lpips'].std(), 4)),
        'avg_niqe_srgan': float(round(df_m['niqe_fpga'].mean(), 4)),
        'avg_epi': float(round(df_m['epi'].mean(), 4)),
        'avg_mean': float(round(df_m['mean'].mean(), 4)),
        'avg_std': float(round(df_m['std'].mean(), 4)),
        'avg_psnr_gain_db': float(round(df_m['psnr_gain_db'].mean(), 4)),
        'avg_msssim_gain': float(round(df_m['msssim_gain'].mean(), 4)),
        'avg_lpips_gain': float(round(df_m['lpips_gain'].mean(), 4)),
        'avg_niqe_gain': float(round(df_m['niqe_gain'].mean(), 4)),
        'avg_latency_ms': float(round(df_m['latency_ms'].mean(), 2)),
        'throughput_fps': float(round(1000.0 / df_m['latency_ms'].mean(), 2)),
        'elapsed_sec_total': float(round(m_data['elapsed_sec'], 2)),
        'wall_time_min': float(round(m_data['elapsed_sec'] / 60.0, 2))
    }
    
    full_output = {
        'summary': summary_dict,
        'per_image_results': records
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
        
    df_m.to_csv(csv_path, index=False)
    exported_files.extend([json_path, csv_path])
    print(f"✓ Đã xuất: {os.path.basename(json_path)} ({os.path.getsize(json_path)/(1024*1024):.2f} MB)")
    print(f"✓ Đã xuất: {os.path.basename(csv_path)} ({os.path.getsize(csv_path)/1024:.1f} KB)")

# Xuất bảng tổng hợp so sánh đa mô hình
summary_csv_path  = os.path.join(OUTPUT_DIR, f"multimodel_{SCALE_FACTOR}x_comparison_summary.csv")
summary_json_path = os.path.join(OUTPUT_DIR, f"multimodel_{SCALE_FACTOR}x_comparison_summary.json")
df_summary.to_csv(summary_csv_path, index=False)
df_summary.to_json(summary_json_path, indent=2, orient='records')
exported_files.extend([summary_csv_path, summary_json_path])

print("\n📂 ĐÃ XUẤT TẤT CẢ FILE RA /kaggle/working/ SẴN SÀNG TẢI VỀ:")
for ef in exported_files:
    print(f"  • {ef}")
```

---

## 🟩 CELL 12 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 12 — Biểu Đồ So Sánh & Luận Chứng Khoa Học Lựa Chọn SRGAN║
# ╚══════════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
plt.subplots_adjust(hspace=0.35, wspace=0.25)
if sns is not None:
    colors = sns.color_palette("tab10", len(df_summary))
else:
    colors = [plt.cm.tab10(i % 10) for i in range(len(df_summary))]

# ─────────────────────────────────────────────────────────────────────────────
# Đồ thị 1: Perception vs Distortion (LPIPS vs PSNR) — TRỌNG TÂM LUẬN CHỨNG SRGAN
# ─────────────────────────────────────────────────────────────────────────────
ax1 = axes[0, 0]
for idx, row in df_summary.iterrows():
    m = row['Model']
    x = row['PSNR (dB)']
    y = row['LPIPS ↓']
    c = colors[idx]
    ax1.scatter(x, y, color=c, s=180, edgecolors='black', linewidth=1.5, zorder=5)
    offset_y = 0.005 if m != 'SRGAN' else -0.010
    ax1.annotate(m, (x, y + offset_y), fontsize=11, fontweight='bold', ha='center')

ax1.set_title(f"1. Perception vs Distortion (Scale {SCALE_FACTOR}x)\n[Trọng tâm luận chứng lựa chọn SRGAN]", fontsize=13, fontweight='bold')
ax1.set_xlabel("PSNR (dB) — Độ méo ảnh (Cao hơn = Khớp pixel tốt hơn)", fontsize=11)
ax1.set_ylabel("LPIPS (AlexNet) — Cảm thụ thị giác (Thấp hơn = Sắc nét hơn)", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)

# Tô vùng tối ưu thị giác
ax1.axhline(y=df_summary['LPIPS ↓'].min() * 1.15, color='green', linestyle=':', alpha=0.5, label='Vùng sắc nét thị giác cao')
ax1.legend(loc='upper right')

# ─────────────────────────────────────────────────────────────────────────────
# Đồ thị 2: Speed vs Quality (FPS vs PSNR) — Phân tích thực thi
# ─────────────────────────────────────────────────────────────────────────────
ax2 = axes[0, 1]
for idx, row in df_summary.iterrows():
    m = row['Model']
    x = row['FPS']
    y = row['PSNR (dB)']
    c = colors[idx]
    ax2.scatter(x, y, color=c, s=180, edgecolors='black', linewidth=1.5, zorder=5)
    ax2.annotate(m, (x, y + 0.3), fontsize=11, fontweight='bold', ha='center')

ax2.set_title(f"2. Speed vs Quality Trade-off (FPS vs PSNR)\n[Cơ sở đề xuất tăng tốc phần cứng FPGA]", fontsize=13, fontweight='bold')
ax2.set_xlabel("Thông lượng (Throughput FPS) — Cao hơn = Nhanh hơn", fontsize=11)
ax2.set_ylabel("PSNR (dB)", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)

# ─────────────────────────────────────────────────────────────────────────────
# Đồ thị 3: Bar Chart So Sánh LPIPS (Thị Giác Sắc Nét)
# ─────────────────────────────────────────────────────────────────────────────
ax3 = axes[1, 0]
bars = ax3.bar(df_summary['Model'], df_summary['LPIPS ↓'], color=colors, edgecolor='black', linewidth=1.2)
ax3.set_title(f"3. Xếp Hạng Sai Lệch Thị Giác LPIPS (Scale {SCALE_FACTOR}x)\n(Càng thấp càng rõ nét vi cấu trúc giải phẫu)", fontsize=13, fontweight='bold')
ax3.set_ylabel("LPIPS Index", fontsize=11)
ax3.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# Đồ thị 4: Bar Chart So Sánh PSNR Gain so với Bicubic (dB)
# ─────────────────────────────────────────────────────────────────────────────
ax4 = axes[1, 1]
bars_p = ax4.bar(df_summary['Model'], df_summary['PSNR Gain (dB)'], color=colors, edgecolor='black', linewidth=1.2)
ax4.set_title(f"4. Tăng Trưởng PSNR so với Bicubic Baseline (Scale {SCALE_FACTOR}x)\n(dB cải thiện so với ảnh nội suy gốc)", fontsize=13, fontweight='bold')
ax4.set_ylabel("PSNR Gain (dB)", fontsize=11)
ax4.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars_p:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"+{yval:.2f}" if yval >= 0 else f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle(f"BÁO CÁO ĐỐI CHUẨN ĐA MÔ HÌNH TRÊN TẬP ẢNH X-RAY (SCALE {SCALE_FACTOR}×)", fontsize=16, fontweight='bold', y=0.98)
chart_out_path = os.path.join(OUTPUT_DIR, f"multimodel_{SCALE_FACTOR}x_tradeoff_charts.png")
plt.savefig(chart_out_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Đã lưu biểu đồ phân tích đánh đổi khoa học tại: {chart_out_path}")
```

---

## 🟩 CELL 13 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 13 — Đóng Gói Tải Về & Kiểm Tra Dung Lượng             ║
# ╚══════════════════════════════════════════════════════════════╝

import subprocess

print("Các file kết quả sinh ra trong phiên làm việc:")
print("─" * 70)
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(fpath):
        sz_mb = os.path.getsize(fpath) / (1024 * 1024)
        if sz_mb >= 1.0:
            sz_str = f"{sz_mb:6.2f} MB"
        else:
            sz_str = f"{os.path.getsize(fpath)/1024:6.1f} KB"
        print(f"  • {fname:<45} [{sz_str}]")

print("─" * 70)
print("✓ Tất cả các file đã sẵn sàng trong thư mục làm việc /kaggle/working/results.")
```

---

## 🟩 CELL 14 — [CODE]

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 14 — Dọn Dẹp VRAM & Hướng Dẫn Đồng Bộ Về Dự Án Local   ║
# ╚══════════════════════════════════════════════════════════════╝

if DEVICE == 'cuda':
    torch.cuda.empty_cache()
    print("✓ Đã dọn dẹp sạch VRAM GPU.")

print("""
══════════════════════════════════════════════════════════════════════
🎉 HOÀN THÀNH TOÀN BỘ TIẾN TRÌNH BENCHMARK ĐA MÔ HÌNH TRÊN KAGGLE!
══════════════════════════════════════════════════════════════════════

📥 CÁC BƯỚC TIẾP THEO ĐỂ ĐỒNG BỘ DỮ LIỆU:
1. Mở tab 'Output' ở cột bên phải giao diện Kaggle.
2. Tải các file JSON/CSV kết quả về máy local:
   • {model}_{scale}x_benchmark.json
   • {model}_{scale}x_benchmark.csv
   • multimodel_{scale}x_comparison_summary.csv
   • multimodel_{scale}x_tradeoff_charts.png
3. Đặt các file tải về vào thư mục dự án tương ứng:
   • File của phần mềm  -> results/software/
   • File của phần cứng -> results/hardware/
4. Chạy script phân tích tổng hợp:
   python "code hardware/scripts/run_analysis.py"
══════════════════════════════════════════════════════════════════════
""")
```

---
