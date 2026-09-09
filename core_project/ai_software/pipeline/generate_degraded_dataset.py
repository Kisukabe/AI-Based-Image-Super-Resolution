#!/usr/bin/env python3
"""
generate_degraded_dataset.py
================================================================================
Mục đích:
    Xây dựng pipeline mô phỏng suy thoái vật lý quang học và cảm biến y tế thực tế
    theo yêu cầu mục 2 trong Request1.txt (Sản phẩm bàn giao Deliverable 2):
        I_lr = (I_hr * k) ↓s + n

    Trong đó:
    - I_hr: Ảnh gốc High-Resolution Ground Truth (1 kênh xám uint8 [0, 255]).
    - k   : Hàm làm mờ quang học Gaussian blur (độ lệch chuẩn sigma ∈ [0.5, 1.5]).
    - ↓s  : Hạ mẫu không gian tỉ lệ 2x (1024x1024 -> 512x512).
    - n   : Nhiễu cảm biến y tế Poisson-Gaussian (Shot noise tia X + Readout noise).

Kiến trúc tuân thủ rule-01:
    [EXTRACT] -> [VALIDATE] -> [TRANSFORM] -> [LOAD] -> [AUDIT]

Cách chạy:
    python3 generate_degraded_dataset.py --seed 42
    python3 generate_degraded_dataset.py --num-images 20 --dry-run
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DEGRADE_PIPELINE")


# ------------------------------------------------------------------------------
# CONFIGURATION CONTRACT
# ------------------------------------------------------------------------------
@dataclass
class DegradationConfig:
    """Cấu hình tham số cho toàn bộ pipeline suy thoái vật lý."""

    input_dir: Path = Path("code software/test_images")
    output_dir: Path = Path("data/degraded_testset")
    scale_factor: int = 2
    sigma_min: float = 0.5
    sigma_max: float = 1.5
    photon_scale: float = 2500.0        # Peak photon count cho Poisson shot noise
    readout_noise_sigma: float = 0.008   # Readout noise Gaussian (tương đương ~2 mức xám / 255)
    seed: int = 42
    target_hr_size: int = 1024          # Kích thước chuẩn HR 1024x1024 (LR sẽ là 512x512)
    dry_run: bool = False
    num_images: Optional[int] = None


@dataclass
class DegradationMetadata:
    """Metadata kết quả suy thoái cho 1 ảnh."""

    filename: str
    hr_shape: list[int]
    lr_shape: list[int]
    blur_sigma: float
    kernel_size: int
    seed_used: int
    bicubic_psnr_db: float
    bicubic_ssim: float


# ------------------------------------------------------------------------------
# LAYER 1: EXTRACT
# ------------------------------------------------------------------------------
def extract_raw_image_paths(config: DegradationConfig) -> list[Path]:
    """
    Trích xuất danh sách đường dẫn ảnh thô từ thư mục đầu vào.
    Chỉ đọc đường dẫn, không biến đổi dữ liệu.
    """
    logger.info(f"[EXTRACT] Starting discovery in {config.input_dir}")
    if not config.input_dir.exists():
        raise FileNotFoundError(f"[EXTRACT] Input directory does not exist: {config.input_dir}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [
            p
            for p in config.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions and not p.name.startswith(".")
        ]
    )

    if not image_paths:
        raise ValueError(f"[EXTRACT] No valid image files found in {config.input_dir}")

    if config.num_images is not None and config.num_images > 0:
        image_paths = image_paths[: config.num_images]

    logger.info(f"[EXTRACT] Done. Discovered {len(image_paths)} valid images.")
    return image_paths


# ------------------------------------------------------------------------------
# LAYER 2: VALIDATE
# ------------------------------------------------------------------------------
def validate_and_normalize_raw_image(
    image_path: Path, target_size: int
) -> tuple[bool, Optional[np.ndarray], str]:
    """
    Kiểm tra tính toàn vẹn của file ảnh và chuyển đổi về dạng 1 kênh xám (Grayscale) uint8.
    Nếu kích thước khác target_size, đưa về target_size với phép nội suy bảo toàn chi tiết.
    """
    try:
        with Image.open(image_path) as pil_img:
            # Đảm bảo định dạng 1 kênh xám (Grayscale 'L')
            if pil_img.mode != "L":
                pil_img = pil_img.convert("L")

            # Đảm bảo kích thước chuẩn 1024x1024 nếu có sai lệch
            if pil_img.size != (target_size, target_size):
                pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

            arr = np.array(pil_img, dtype=np.uint8)

            if arr.ndim != 2:
                return False, None, f"Image must be 2D grayscale, got ndim={arr.ndim}"

            if arr.shape != (target_size, target_size):
                return False, None, f"Invalid shape: {arr.shape} != ({target_size}, {target_size})"

            return True, arr, "OK"
    except Exception as e:
        return False, None, f"Failed reading {image_path.name}: {str(e)}"


# ------------------------------------------------------------------------------
# LAYER 3: TRANSFORM (PHYSICAL DEGRADATION MODEL)
# ------------------------------------------------------------------------------
def compute_kernel_size(sigma: float) -> int:
    """Tính kích thước kernel lẻ cho Gaussian blur: k_size = 2 * ceil(3 * sigma) + 1."""
    k = 2 * int(math.ceil(3.0 * sigma)) + 1
    return max(k, 3)


def apply_optical_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Thực hiện làm mờ quang học (Point Spread Function) bằng Gaussian blur.
    Input: uint8 [0, 255], Output: float32 [0.0, 255.0].
    """
    k_size = compute_kernel_size(sigma)
    img_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_spatial_downsample(image: np.ndarray, scale: int) -> np.ndarray:
    """
    Hạ mẫu không gian tỉ lệ scale (mặc định 2x).
    1024x1024 -> 512x512 qua phép nội suy Bicubic tiêu chuẩn.
    """
    h, w = image.shape[:2]
    new_h, new_w = h // scale, w // scale
    downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return downsampled


def apply_medical_poisson_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    photon_scale: float,
    readout_sigma: float,
) -> np.ndarray:
    """
    Mô phỏng vật lý nhiễu cảm biến y tế (Poisson-Gaussian noise trên đầu dò FPD).

    Công thức:
        I_norm = image / 255.0
        Poisson component: N_photons ~ Poisson(I_norm * photon_scale) / photon_scale
        Gaussian component: N_readout ~ Gaussian(0, readout_sigma^2)
        I_noisy = clip((Poisson + Gaussian) * 255.0, 0, 255)
    """
    img_norm = np.clip(image / 255.0, 0.0, 1.0)

    # 1. Thành phần Poisson (Photon Shot Noise)
    scaled_photons = img_norm * photon_scale
    poisson_photons = rng.poisson(lam=scaled_photons)
    poisson_component = poisson_photons.astype(np.float32) / photon_scale

    # 2. Thành phần Gaussian (Readout Electronic Noise)
    readout_component = rng.normal(loc=0.0, scale=readout_sigma, size=image.shape).astype(np.float32)

    # 3. Tổng hợp và kẹp biên
    noisy_float = (poisson_component + readout_component) * 255.0
    noisy_uint8 = np.clip(np.round(noisy_float), 0, 255).astype(np.uint8)
    return noisy_uint8


def transform_single_image(
    image_hr: np.ndarray,
    sigma: float,
    image_seed: int,
    config: DegradationConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Thực hiện quy trình suy thoái vật lý hoàn chỉnh trên 1 ảnh:
        I_lr = (I_hr * k) ↓s + n
    Đồng thời tính PSNR và SSIM của ảnh Bicubic nội suy ngược để kiểm tra mức độ suy thoái.
    """
    rng = np.random.default_rng(image_seed)

    # Bước 1: Làm mờ quang học Gaussian blur (k)
    blurred = apply_optical_gaussian_blur(image_hr, sigma)

    # Bước 2: Hạ mẫu không gian 2x (↓s)
    downsampled = apply_spatial_downsample(blurred, config.scale_factor)

    # Bước 3: Thêm nhiễu cảm biến y tế Poisson-Gaussian (n)
    image_lr = apply_medical_poisson_gaussian_noise(
        downsampled, rng, config.photon_scale, config.readout_noise_sigma
    )

    # Đo đạc chỉ số suy thoái bằng Bicubic upscale ngược lại để đánh giá
    lr_upscaled = cv2.resize(
        image_lr, (image_hr.shape[1], image_hr.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    mse = np.mean((image_hr.astype(np.float64) - lr_upscaled.astype(np.float64)) ** 2)
    psnr_db = 10.0 * np.log10((255.0**2) / (mse + 1e-10))

    # Tính SSIM đơn giản
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(image_hr.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(lr_upscaled.astype(np.float64), (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(image_hr.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(lr_upscaled.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(image_hr.astype(np.float64) * lr_upscaled.astype(np.float64), (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim_val = float(np.mean(ssim_map))

    return image_hr, image_lr, float(psnr_db), ssim_val


# ------------------------------------------------------------------------------
# LAYER 4: LOAD
# ------------------------------------------------------------------------------
def load_degraded_dataset(
    processed_records: list[tuple[Path, np.ndarray, np.ndarray, DegradationMetadata]],
    config: DegradationConfig,
) -> None:
    """
    Lưu trữ các cặp ảnh và metadata vào thư mục đích.
    """
    hr_dir = config.output_dir / "HR"
    lr_dir = config.output_dir / f"LR_{config.scale_factor}x"

    if not config.dry_run:
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []
    logger.info(f"[LOAD] Starting saving {len(processed_records)} image pairs to {config.output_dir}")

    for raw_path, hr_img, lr_img, meta in processed_records:
        meta_dict = asdict(meta)
        metadata_list.append(meta_dict)

        if not config.dry_run:
            hr_path = hr_dir / raw_path.name
            lr_path = lr_dir / raw_path.name

            # Lưu ảnh dạng PNG không mất mát (Lossless compression)
            cv2.imwrite(str(hr_path), hr_img)
            cv2.imwrite(str(lr_path), lr_img)

    if not config.dry_run:
        meta_path = config.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "scale_factor": config.scale_factor,
                        "sigma_min": config.sigma_min,
                        "sigma_max": config.sigma_max,
                        "photon_scale": config.photon_scale,
                        "readout_noise_sigma": config.readout_noise_sigma,
                        "seed": config.seed,
                        "total_images": len(metadata_list),
                    },
                    "images": metadata_list,
                },
                f,
                indent=2,
            )
        logger.info(f"[LOAD] Successfully wrote metadata to {meta_path}")


# ------------------------------------------------------------------------------
# LAYER 5: AUDIT
# ------------------------------------------------------------------------------
def audit_pipeline_run(
    processed_records: list[tuple[Path, np.ndarray, np.ndarray, DegradationMetadata]],
    elapsed_time: float,
    config: DegradationConfig,
) -> None:
    """
    Kiểm tra tính toàn vẹn và báo cáo thống kê kết quả pipeline.
    """
    total_imgs = len(processed_records)
    if total_imgs == 0:
        logger.warning("[AUDIT] No records to audit.")
        return

    psnr_values = [meta.bicubic_psnr_db for _, _, _, meta in processed_records]
    ssim_values = [meta.bicubic_ssim for _, _, _, meta in processed_records]

    mean_psnr = float(np.mean(psnr_values))
    std_psnr = float(np.std(psnr_values))
    mean_ssim = float(np.mean(ssim_values))
    std_ssim = float(np.std(ssim_values))

    logger.info("=" * 80)
    logger.info("[AUDIT] PIPELINE AUDIT REPORT:")
    logger.info(f"  - Total images processed : {total_imgs}")
    logger.info(f"  - Total elapsed time     : {elapsed_time:.2f} seconds ({total_imgs/elapsed_time:.1f} imgs/sec)")
    logger.info(f"  - Initial Degraded PSNR  : {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    logger.info(f"  - Initial Degraded SSIM  : {mean_ssim:.4f} ± {std_ssim:.4f}")
    logger.info(f"  - Output directories     : {config.output_dir}/HR và {config.output_dir}/LR_{config.scale_factor}x")
    logger.info(f"  - Dry run mode           : {config.dry_run}")
    logger.info("=" * 80)


# ------------------------------------------------------------------------------
# ORCHESTRATION PIPELINE
# ------------------------------------------------------------------------------
def run_degradation_pipeline(config: DegradationConfig) -> None:
    """Điều phối toàn bộ quy trình 5 bước của pipeline."""
    start_time = time.time()
    logger.info(f"[STAGE] Starting Physical Degradation Pipeline with seed={config.seed}")

    # 1. Extract
    image_paths = extract_raw_image_paths(config)

    # Khởi tạo bộ sinh ngẫu nhiên tái lập cho sigma
    sigma_rng = np.random.default_rng(config.seed)

    processed_records = []
    failed_count = 0

    # 2 & 3: Validate & Transform
    for idx, path in enumerate(image_paths):
        is_valid, hr_arr, msg = validate_and_normalize_raw_image(path, config.target_hr_size)
        if not is_valid or hr_arr is None:
            logger.warning(f"[VALIDATE] Skipped {path.name}: {msg}")
            failed_count += 1
            continue

        # Sinh sigma ngẫu nhiên trong khoảng [sigma_min, sigma_max]
        sigma = float(sigma_rng.uniform(config.sigma_min, config.sigma_max))
        image_seed = config.seed + idx

        # Transform
        hr_img, lr_img, psnr_db, ssim_val = transform_single_image(hr_arr, sigma, image_seed, config)

        metadata = DegradationMetadata(
            filename=path.name,
            hr_shape=list(hr_img.shape),
            lr_shape=list(lr_img.shape),
            blur_sigma=round(sigma, 4),
            kernel_size=compute_kernel_size(sigma),
            seed_used=image_seed,
            bicubic_psnr_db=round(psnr_db, 4),
            bicubic_ssim=round(ssim_val, 4),
        )

        processed_records.append((path, hr_img, lr_img, metadata))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            logger.info(f"[TRANSFORM] Processed {idx + 1}/{len(image_paths)} images...")

    # 4. Load
    load_degraded_dataset(processed_records, config)

    # 5. Audit
    elapsed = time.time() - start_time
    audit_pipeline_run(processed_records, elapsed, config)


# ------------------------------------------------------------------------------
# CLI INTERFACE
# ------------------------------------------------------------------------------
def parse_args() -> DegradationConfig:
    parser = argparse.ArgumentParser(
        description="Pipeline mô phỏng suy thoái vật lý quang học và cảm biến y tế (Request 1)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="code software/test_images",
        help="Đường dẫn thư mục ảnh gốc",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/degraded_testset",
        help="Đường dẫn thư mục lưu cặp ảnh",
    )
    parser.add_argument("--scale", type=int, default=2, help="Hệ số hạ mẫu không gian (mặc định 2)")
    parser.add_argument("--sigma-min", type=float, default=0.5, help="Độ lệch chuẩn blur nhỏ nhất")
    parser.add_argument("--sigma-max", type=float, default=1.5, help="Độ lệch chuẩn blur lớn nhất")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên để tái lập hoàn toàn (100%%)")
    parser.add_argument("--num-images", type=int, default=None, help="Giới hạn số lượng ảnh xử lý")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm không ghi file vào đĩa")

    args = parser.parse_args()

    return DegradationConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        scale_factor=args.scale,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        seed=args.seed,
        num_images=args.num_images,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_degradation_pipeline(cfg)
