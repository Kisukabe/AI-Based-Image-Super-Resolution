#!/usr/bin/env python3
"""
================================================================================
Train PyTorch Float32 Baseline Models for Medical Image Super-Resolution
Dự án: Siêu phân giải ảnh y tế trên FPGA Xilinx Zynq-7020 (PYNQ-Z2)
Căn cứ: Request1.txt (Mục 1) & TASK_ROADMAP.md (Bước 5 / Task 1.5)
================================================================================
Mục tiêu:
  1. Huấn luyện 2 kiến trúc mạng chuẩn 1 kênh xám (Grayscale):
     - Compact SRCNN (1 -> 16 -> 8 -> 1): Đúng 1.649 tham số (khớp RTL phần cứng).
     - SRCNN Original (1 -> 64 -> 32 -> 1): Đúng 8.129 tham số (Mô hình đối chứng baseline).
  2. Sử dụng hàm mất mát hỗn hợp (Composite Loss):
     - Loss = MSE + 0.1 * Sobel_Edge_Loss.
  3. Áp dụng cơ chế chống rò rỉ dữ liệu (Strict Anti-Data Leakage):
     - Tự động quét và loại trừ 338 ảnh trong tập testset đã niêm phong.
  4. Xuất đồ thị hội tụ Train/Val Loss chuẩn xuất bản 300 DPI và checkpoint PyTorch.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

# Import mô hình và hàm loss chuẩn từ core_project
try:
    from core_project.ai_software.models.models_srcnn import Compact_SRCNN, SRCNN_Original
    from core_project.ai_software.losses.custom_loss import SRCNN_CompositeLoss, SobelEdgeLoss
except ImportError:
    # Hỗ trợ đường dẫn tương đối khi chạy trực tiếp trong thư mục training
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from core_project.ai_software.models.models_srcnn import Compact_SRCNN, SRCNN_Original
    from core_project.ai_software.losses.custom_loss import SRCNN_CompositeLoss, SobelEdgeLoss

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Rule-03: Prefix standard)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainBaseline")

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_CROP_SIZE = 128
DEFAULT_SCALE = 2


# ------------------------------------------------------------------------------
# REPRODUCIBILITY SEEDING (Rule-02)
# ------------------------------------------------------------------------------
def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Cố định seed ngẫu nhiên đảm bảo tính tái lập 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"[SEED] Thiết lập seed ngẫu nhiên cố định: {seed}")


# ------------------------------------------------------------------------------
# ANTI-DATA LEAKAGE TESTSET SCANNER
# ------------------------------------------------------------------------------
def get_sealed_testset_filenames(testset_dir: Optional[Path] = None) -> Set[str]:
    """
    Quét danh sách các file ảnh trong tập testset đã niêm phong để loại trừ,
    đảm bảo không xảy ra rò rỉ dữ liệu (Anti-Data Leakage).
    """
    if testset_dir is None:
        testset_dir = Path(__file__).resolve().parent.parent.parent / "data" / "degraded_testset" / "HR"

    sealed_names: Set[str] = set()
    if testset_dir.exists():
        for p in testset_dir.glob("*.png"):
            sealed_names.add(p.name)
        for p in testset_dir.glob("*.jpg"):
            sealed_names.add(p.name)
        logger.info(f"[AUDIT] Đã niêm phong {len(sealed_names)} ảnh testset từ: {testset_dir}")
    else:
        logger.warning(f"[AUDIT] Không tìm thấy thư mục testset tại {testset_dir}. Tiếp tục mà không lọc.")
    return sealed_names


# ------------------------------------------------------------------------------
# DATASET & AUGMENTATION PIPELINE (Rule-01 & Rule-02)
# ------------------------------------------------------------------------------
class MedicalXRaySRDataset(Dataset):
    """
    Dataset nạp ảnh X-quang y tế 1 kênh Grayscale:
      - Tự động cắt ngẫu nhiên patch HR (crop_size x crop_size).
      - Ngẫu nhiên lật ngang/dọc (Horizontal/Vertical Flip).
      - Tạo cặp ảnh LR bằng cách hạ mẫu Bicubic theo hệ số scale.
      - Phóng đại Bicubic trở lại (Pre-upsampling) làm đầu vào cho mạng SRCNN.
    """

    def __init__(
        self,
        image_paths: List[Path],
        crop_size: int = DEFAULT_CROP_SIZE,
        scale: int = DEFAULT_SCALE,
        is_train: bool = True,
    ) -> None:
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.scale = scale
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            # Bắt buộc chuyển sang ảnh xám 1 kênh chuẩn y tế (Grayscale)
            gray_img = img.convert("L")

        w, h = gray_img.size

        # Cắt patch kích thước crop_size x crop_size
        if w >= self.crop_size and h >= self.crop_size:
            if self.is_train:
                left = random.randint(0, w - self.crop_size)
                top = random.randint(0, h - self.crop_size)
            else:
                left = (w - self.crop_size) // 2
                top = (h - self.crop_size) // 2
            patch_hr = gray_img.crop((left, top, left + self.crop_size, top + self.crop_size))
        else:
            patch_hr = gray_img.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)

        # Data augmentation cho tập train
        if self.is_train:
            if random.random() > 0.5:
                patch_hr = patch_hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                patch_hr = patch_hr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Sinh ảnh LR bằng cách hạ mẫu Bicubic
        lr_w = self.crop_size // self.scale
        lr_h = self.crop_size // self.scale
        patch_lr = patch_hr.resize((lr_w, lr_h), Image.Resampling.BICUBIC)

        # Pre-upsampling: Phóng đại Bicubic về kích thước HR để đưa vào mạng SRCNN
        patch_lr_bicubic = patch_lr.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)

        # Chuyển đổi sang Float32 Tensor [1, H, W] trong khoảng [0.0, 1.0]
        tensor_in = to_tensor(patch_lr_bicubic)
        tensor_target = to_tensor(patch_hr)

        return tensor_in, tensor_target


# ------------------------------------------------------------------------------
# BALANCED SAMPLING DATASET STRATEGY (Mục 1.1 trong models/description.md)
# ------------------------------------------------------------------------------
def scan_and_split_dataset(
    dataset_dirs: List[Path],
    sealed_names: Set[str],
    images_per_batch: int = 1000,
    target_batches: int = 12,
    train_ratio: float = 0.85,
    max_images: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Path], List[Path]]:
    """
    Chiến lược lấy mẫu cân bằng (Balanced Sampling Strategy) bám sát Mục 1.1 trong models/description.md:
      1. Quét tìm 12 lô dữ liệu NIH ChestX-ray14 (images_001 -> images_012 hoặc các thư mục con tương ứng).
      2. Với mỗi lô dữ liệu, trích xuất đúng 1.000 ảnh đầu tiên (loại trừ tuyệt đối ảnh thuộc testset niêm phong).
      3. Đạt quy mô chuẩn 12.000 ảnh X-quang lồng ngực (12 lô x 1.000 ảnh).
      4. Phân chia tập dữ liệu: đúng 85% Training (10.200 ảnh) và 15% Validation (1.800 ảnh) với seed cố định 42.
      5. Nếu tập dữ liệu trên Kaggle là thư mục phẳng hoặc không chia 12 lô (ví dụ sub_NIH/sub_chest), tự động
         phân bổ cân đối giữa các thư mục con hoặc trích xuất tất định 12.000 ảnh và phân chia 85/15.
    """
    extensions = {".png", ".jpg", ".jpeg"}
    dir_to_files: Dict[Path, List[Path]] = {}

    for d in dataset_dirs:
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            root_path = Path(root)
            valid_files = []
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in extensions and f not in sealed_names:
                    valid_files.append(root_path / f)
            if valid_files:
                dir_to_files[root_path] = sorted(valid_files)

    if not dir_to_files:
        logger.error("[SAMPLE] Không tìm thấy bất kỳ file ảnh hợp lệ nào trong các thư mục chỉ định.")
        return [], []

    sampled_images: List[Path] = []
    sorted_dirs = sorted(list(dir_to_files.keys()), key=lambda p: p.name)

    # Nhận diện các thư mục dạng lô dữ liệu (images_001 .. images_012 hoặc các thư mục con)
    batch_dirs = [d for d in sorted_dirs if any(kw in d.name.lower() for kw in ["images_", "batch_", "sub_"])]
    if not batch_dirs:
        # Nếu không có tên chuẩn, lấy các thư mục chứa ảnh
        batch_dirs = sorted_dirs

    logger.info(f"[SAMPLE] Phát hiện {len(batch_dirs)} thư mục/lô dữ liệu ảnh.")

    # Áp dụng Balanced Sampling: lấy đúng 1.000 ảnh từ mỗi lô
    for bdir in batch_dirs[:target_batches]:
        b_files = dir_to_files[bdir]
        take_count = min(images_per_batch, len(b_files))
        selected = b_files[:take_count]
        sampled_images.extend(selected)
        logger.info(f"  [BATCH] {bdir.name}: Lấy {len(selected):,}/{len(b_files):,} ảnh (Mục tiêu: {images_per_batch})")

    # Nếu tổng số ảnh thu được từ các lô chưa đủ 12.000 và còn ảnh ở các thư mục khác
    if len(sampled_images) < target_batches * images_per_batch:
        remaining_pool = []
        sampled_set = set(sampled_images)
        for b_files in dir_to_files.values():
            for f in b_files:
                if f not in sampled_set:
                    remaining_pool.append(f)
        needed = (target_batches * images_per_batch) - len(sampled_images)
        if needed > 0 and remaining_pool:
            remaining_pool = sorted(remaining_pool)
            additional = remaining_pool[:needed]
            sampled_images.extend(additional)
            logger.info(f"[SAMPLE] Bổ sung thêm {len(additional):,} ảnh để đạt mục tiêu.")

    # Loại bỏ trùng lặp và sắp xếp lại
    sampled_images = sorted(list(set(sampled_images)))
    logger.info(f"[SAMPLE] Tổng số ảnh sau Balanced Sampling: {len(sampled_images):,} ảnh.")

    # Hỗ trợ giới hạn max_images nếu có cờ chạy thử (--dry-run)
    if max_images is not None and max_images < len(sampled_images):
        random.seed(seed)
        sampled_images = random.sample(sampled_images, max_images)
        logger.info(f"[SAMPLE] Giới hạn lấy mẫu theo cấu hình: {len(sampled_images):,} ảnh.")

    # Trộn ngẫu nhiên với seed cố định 42 để phân chia Train / Val
    random.seed(seed)
    random.shuffle(sampled_images)

    split_idx = int(len(sampled_images) * train_ratio)
    train_files = sampled_images[:split_idx]
    val_files = sampled_images[split_idx:]

    logger.info(
        f"[SPLIT] Phân chia dữ liệu chuẩn 85% - 15% (Seed={seed}):\\n"
        f"  - Training (85%)   : {len(train_files):,} ảnh (Kỳ vọng: 10.200 khi đủ 12.000)\\n"
        f"  - Validation (15%) : {len(val_files):,} ảnh (Kỳ vọng: 1.800 khi đủ 12.000)"
    )
    return train_files, val_files


# ------------------------------------------------------------------------------
# METRIC EVALUATION (PSNR & SSIM)
# ------------------------------------------------------------------------------
def calculate_batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Tính PSNR trung bình của một batch tensor [B, 1, H, W] trong miền [0.0, 1.0]."""
    diff = pred - target
    mse = torch.mean(diff ** 2, dim=(1, 2, 3))
    # Tránh chia cho 0
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(torch.mean(psnr).item())


# ------------------------------------------------------------------------------
# TRAINING LOG DATA STRUCTURE
# ------------------------------------------------------------------------------
@dataclass
class TrainingHistory:
    """Lưu trữ lịch sử hội tụ qua các Epoch để vẽ đồ thị."""

    epochs: List[int]
    train_loss: List[float]
    train_mse: List[float]
    train_sobel: List[float]
    val_loss: List[float]
    val_mse: List[float]
    val_psnr: List[float]
    learning_rates: List[float]


# ------------------------------------------------------------------------------
# CORE MODEL TRAINER
# ------------------------------------------------------------------------------
def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: SRCNN_CompositeLoss,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("checkpoints"),
) -> Tuple[Path, TrainingHistory]:
    """
    Huấn luyện một mô hình SRCNN trong đúng số epochs chỉ định.
    
    Args:
        model: Kiến trúc mạng PyTorch.
        model_name: Tên định danh ("Compact_SRCNN" hoặc "SRCNN_Original").
        train_loader: DataLoader tập huấn luyện.
        val_loader: DataLoader tập kiểm định.
        criterion: Hàm mất mát SRCNN_CompositeLoss.
        epochs: Số epoch (mặc định 10).
        lr: Tốc độ học ban đầu.
        device: CPU hoặc CUDA device.
        output_dir: Thư mục lưu checkpoint.
        
    Returns:
        Đường dẫn file checkpoint tốt nhất và đối tượng TrainingHistory.
    """
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"================================================================================")
    logger.info(f"[TRAIN] BẮT ĐẦU HUẤN LUYỆN: {model_name} (Số tham số: {total_params:,}) TRÊN {device}")
    logger.info(f"================================================================================")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / f"{model_name.lower()}_float32.pth"

    history = TrainingHistory(
        epochs=[],
        train_loss=[],
        train_mse=[],
        train_sobel=[],
        val_loss=[],
        val_mse=[],
        val_psnr=[],
        learning_rates=[],
    )

    best_val_psnr = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # ----------------- TRAINING PHASE -----------------
        model.train()
        running_train_loss = 0.0
        running_train_mse = 0.0
        running_train_sobel = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss, loss_dict = criterion(pred, batch_y)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_mse += loss_dict["loss_mse"]
            running_train_sobel += loss_dict["loss_sobel"]
            train_batches += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        avg_train_loss = running_train_loss / max(1, train_batches)
        avg_train_mse = running_train_mse / max(1, train_batches)
        avg_train_sobel = running_train_sobel / max(1, train_batches)

        # ----------------- VALIDATION PHASE -----------------
        model.eval()
        running_val_loss = 0.0
        running_val_mse = 0.0
        running_val_psnr = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                pred = model(batch_x)
                loss, loss_dict = criterion(pred, batch_y)
                psnr_val = calculate_batch_psnr(pred, batch_y)

                running_val_loss += loss.item()
                running_val_mse += loss_dict["loss_mse"]
                running_val_psnr += psnr_val
                val_batches += 1

        avg_val_loss = running_val_loss / max(1, val_batches)
        avg_val_mse = running_val_mse / max(1, val_batches)
        avg_val_psnr = running_val_psnr / max(1, val_batches)

        epoch_time = time.perf_counter() - t0

        # Ghi nhận lịch sử
        history.epochs.append(epoch)
        history.train_loss.append(avg_train_loss)
        history.train_mse.append(avg_train_mse)
        history.train_sobel.append(avg_train_sobel)
        history.val_loss.append(avg_val_loss)
        history.val_mse.append(avg_val_mse)
        history.val_psnr.append(avg_val_psnr)
        history.learning_rates.append(current_lr)

        logger.info(
            f"[EPOCH {epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, Sobel: {avg_train_sobel:.6f}) | "
            f"Val Loss: {avg_val_loss:.6f} | Val PSNR: {avg_val_psnr:.2f} dB | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Lưu checkpoint tốt nhất theo PSNR kiểm định
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_epoch = epoch

            checkpoint_dict = {
                "model_name": model_name,
                "upscale_factor": DEFAULT_SCALE,
                "in_channels": 1,
                "total_params": total_params,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_psnr": best_val_psnr,
                "state_dict": model.state_dict(),
                "history": asdict(history),
            }
            torch.save(checkpoint_dict, best_checkpoint_path)
            logger.info(f"  >>> Lưu checkpoint xuất sắc nhất: {best_checkpoint_path.name} (Val PSNR: {best_val_psnr:.2f} dB)")

    logger.info(f"[DONE] Hoàn thành huấn luyện {model_name}. Kỷ lục: Epoch {best_epoch} đạt {best_val_psnr:.2f} dB.")
    return best_checkpoint_path, history


# ------------------------------------------------------------------------------
# CONVERGENCE PLOTTER (300 DPI - Deliverable D1.4)
# ------------------------------------------------------------------------------
def plot_convergence_curves(
    compact_history: TrainingHistory,
    original_history: Optional[TrainingHistory],
    output_path: Path,
) -> None:
    """
    Vẽ đồ thị hội tụ hàm mất mát (Train vs Val Loss) và PSNR kiểm định chuẩn 300 DPI.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    epochs = compact_history.epochs

    # Đồ thị 1: Hàm mất mát hỗn hợp (Composite Loss)
    ax1.plot(epochs, compact_history.train_loss, "b-o", label="Compact SRCNN (Train)", linewidth=2)
    ax1.plot(epochs, compact_history.val_loss, "b--s", label="Compact SRCNN (Val)", linewidth=1.8)

    if original_history is not None:
        ax1.plot(epochs, original_history.train_loss, "r-o", label="SRCNN Original (Train)", linewidth=2)
        ax1.plot(epochs, original_history.val_loss, "r--s", label="SRCNN Original (Val)", linewidth=1.8)

    ax1.set_title("Hội Tụ Hàm Mất Mát Hỗn Hợp (MSE + 0.1 × Sobel)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Composite Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Đồ thị 2: PSNR trên tập kiểm định
    ax2.plot(epochs, compact_history.val_psnr, "b-^", label="Compact SRCNN (1.649 params)", linewidth=2)
    if original_history is not None:
        ax2.plot(epochs, original_history.val_psnr, "r-^", label="SRCNN Original (8.129 params)", linewidth=2)

    ax2.set_title("Động Học Cải Thiện Chỉ Số PSNR Kiểm Định", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation PSNR (dB)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"[FIGURE] Đã xuất đồ thị hội tụ độ phân giải cao tại: {output_path} (300 DPI)")


# ------------------------------------------------------------------------------
# CLI INTERFACE
# ------------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình Float32 Baseline (Compact SRCNN & SRCNN Original) - Request 1."
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        type=str,
        default=None,
        help="Danh sách thư mục chứa ảnh X-quang huấn luyện",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Số lượng Epochs huấn luyện (mặc định 10)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Kích thước batch (mặc định 32)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Tốc độ học Learning Rate ban đầu (mặc định 1e-4)")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE, help="Kích thước patch cắt ngẫu nhiên (128)")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Hệ số phóng đại (mặc định 2)")
    parser.add_argument("--max-images", type=int, default=None, help="Giới hạn tối đa số lượng ảnh nạp vào")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Thư mục lưu checkpoints và đồ thị")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: cuda hoặc cpu")
    parser.add_argument("--dry-run", action="store_true", help="Chế độ chạy thử nhanh 1 epoch trên 10 ảnh")
    parser.add_argument("--allow-testset-images", action="store_true", help="Cho phép dùng ảnh testset (chỉ dùng cho kiểm thử debug/dry-run)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    seed_everything(DEFAULT_SEED)

    # Lựa chọn thiết bị tính toán
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[DEVICE] Sử dụng thiết bị tính toán: {device}")

    # Nhận diện các thư mục dữ liệu
    if args.data_dirs:
        data_dirs = [Path(d) for d in args.data_dirs]
    else:
        # Đường dẫn mặc định (Kaggle hoặc local)
        candidate_dirs = [
            Path("/kaggle/input"),
            Path(__file__).resolve().parent.parent.parent / "data" / "test_images",
            Path(__file__).resolve().parent.parent.parent / "data" / "degraded_testset" / "HR",
        ]
        data_dirs = [d for d in candidate_dirs if d.exists()]

    # Quét danh sách ảnh testset đã niêm phong để chống rò rỉ dữ liệu
    if args.allow_testset_images:
        logger.warning("[AUDIT] Cảnh báo: Kích hoạt cờ --allow-testset-images (chỉ phục vụ dry-run/debug).")
        sealed_names: Set[str] = set()
    else:
        sealed_names = get_sealed_testset_filenames()

    # Phân chia dữ liệu
    max_imgs = 20 if args.dry_run else args.max_images
    epochs = 1 if args.dry_run else args.epochs
    batch_size = 4 if args.dry_run else args.batch_size

    train_files, val_files = scan_and_split_dataset(
        dataset_dirs=data_dirs,
        sealed_names=sealed_names,
        train_ratio=0.85,
        max_images=max_imgs,
        seed=DEFAULT_SEED,
    )

    if len(train_files) == 0 or len(val_files) == 0:
        logger.error("[ERROR] Không tìm thấy dữ liệu huấn luyện hợp lệ. Vui lòng kiểm tra lại cờ --data-dirs.")
        sys.exit(1)

    # Khởi tạo DataLoader
    train_dataset = MedicalXRaySRDataset(train_files, crop_size=args.crop_size, scale=args.scale, is_train=True)
    val_dataset = MedicalXRaySRDataset(val_files, crop_size=args.crop_size, scale=args.scale, is_train=False)

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hàm mất mát hỗn hợp (Composite Loss)
    criterion = SRCNN_CompositeLoss(sobel_weight=0.1).to(device)

    # 1. Huấn luyện Compact SRCNN (1.649 params)
    model_compact = Compact_SRCNN(pre_upsample=False, upscale_factor=args.scale)
    compact_ckpt, compact_history = train_single_model(
        model=model_compact,
        model_name="Compact_SRCNN",
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=epochs,
        lr=args.lr,
        device=device,
        output_dir=output_dir,
    )

    # 2. Huấn luyện SRCNN Original (8.129 params)
    model_orig = SRCNN_Original(pre_upsample=False, upscale_factor=args.scale)
    orig_ckpt, orig_history = train_single_model(
        model=model_orig,
        model_name="SRCNN_Original",
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=epochs,
        lr=args.lr,
        device=device,
        output_dir=output_dir,
    )

    # 3. Xuất đồ thị hội tụ 300 DPI
    plot_path = output_dir / "loss_convergence_dpi300.png"
    plot_convergence_curves(compact_history, orig_history, plot_path)

    logger.info("================================================================================")
    logger.info("✅ HOÀN TẤT HUẤN LUYỆN TOÀN BỘ BASELINE FLOAT32 (BƯỚC 5)")
    logger.info(f"  - Checkpoint Compact SRCNN (D1.1): {compact_ckpt}")
    logger.info(f"  - Checkpoint SRCNN Original:       {orig_ckpt}")
    logger.info(f"  - Đồ thị hội tụ (D1.4):           {plot_path}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
