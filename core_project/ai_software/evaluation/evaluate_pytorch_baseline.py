#!/usr/bin/env python3
"""
================================================================================
Evaluation Pipeline for PyTorch Float32 Baseline Models (Task T1.6)
Target Architecture: Compact SRCNN (1.649 params) vs SRCNN Original (8.129 params) vs Bicubic
Evaluation Dataset: Sealed NIH Chest X-ray Medical Testset (338 LR-HR pairs)
================================================================================
Mục đích:
  Đo đạc đối chứng khách quan toàn diện chất lượng phục hồi siêu phân giải ảnh y tế
  trên tập test niêm phong theo quy chuẩn 5 tầng:
    [EXTRACT]   -> Nạp 338 cặp ảnh LR (512x512) và HR (1024x1024) kèm metadata.
    [VALIDATE]  -> Kiểm tra cặp ảnh, nạp 2 checkpoints Float32 và xác thực số tham số.
    [TRANSFORM] -> Tiền xử lý Pre-upsampling Bicubic 2x và chuẩn hóa tensor Float32 [0.0, 1.0].
    [EVALUATE]  -> Suy luận mô hình và tính toán toàn bộ danh mục chỉ số (PSNR, SSIM,
                   MSE, RMSE, MAE, EPI, Mean, STD, Delta Gains, Clinical Tiers, Latency).
    [AUDIT]     -> Tổng hợp thống kê mô tả (Mean ± Std, Median, IQR, % Gain > 0)
                   và kết xuất ra file Excel đa bảng, CSV và JSON.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

# ------------------------------------------------------------------------------
# SETUP PYTHONPATH FOR SUBMODULES
# ------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
CORE_PROJECT_DIR = CURRENT_DIR.parent.parent
AI_SOFTWARE_DIR = CURRENT_DIR.parent

if str(CORE_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_PROJECT_DIR))
if str(AI_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SOFTWARE_DIR))

from ai_software.models.models_srcnn import Compact_SRCNN, SRCNN_Original
from benchmarks_reports.metrics_calculator import (
    FullReferenceMetrics,
    classify_clinical_tier,
    compute_descriptive_stats,
    compute_positive_gain_rate,
    evaluate_pair,
)

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Rule-03: Prefix standard)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateBaseline")

# ------------------------------------------------------------------------------
# CONSTANTS & DEFAULTS
# ------------------------------------------------------------------------------
DEFAULT_TESTSET_DIR = CORE_PROJECT_DIR / "data" / "degraded_testset"
DEFAULT_CHECKPOINT_COMPACT = AI_SOFTWARE_DIR / "checkpoints" / "compact_srcnn_float32.pth"
DEFAULT_CHECKPOINT_ORIGINAL = AI_SOFTWARE_DIR / "checkpoints" / "srcnn_original_float32.pth"
DEFAULT_OUTPUT_DIR = CORE_PROJECT_DIR / "benchmarks_reports" / "deliverables_export"
SCALE_FACTOR = 2


# ------------------------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------------------------
@dataclass
class SingleEvaluationResult:
    """Kết quả đo đạc toàn diện của 1 bức ảnh qua 3 mô hình."""

    image_id: str
    noise_sigma: float
    degraded_psnr: float

    # Bicubic Baseline
    bicubic_psnr: float
    bicubic_ssim: float
    bicubic_mse: float
    bicubic_rmse: float
    bicubic_mae: float
    bicubic_epi: float
    bicubic_mean: float
    bicubic_std: float
    bicubic_tier: str
    bicubic_latency_ms: float

    # SRCNN Original Float32
    orig_psnr: float
    orig_ssim: float
    orig_mse: float
    orig_rmse: float
    orig_mae: float
    orig_epi: float
    orig_mean: float
    orig_std: float
    orig_tier: str
    orig_latency_ms: float
    orig_delta_psnr: float
    orig_delta_ssim: float
    orig_delta_rmse: float
    orig_delta_epi: float

    # Compact SRCNN Float32
    compact_psnr: float
    compact_ssim: float
    compact_mse: float
    compact_rmse: float
    compact_mae: float
    compact_epi: float
    compact_mean: float
    compact_std: float
    compact_tier: str
    compact_latency_ms: float
    compact_delta_psnr: float
    compact_delta_ssim: float
    compact_delta_rmse: float
    compact_delta_epi: float


# ------------------------------------------------------------------------------
# PIPELINE STEP 1: EXTRACT
# ------------------------------------------------------------------------------
def extract_testset_pairs(
    testset_dir: Path,
    max_samples: Optional[int] = None,
) -> Tuple[List[Tuple[Path, Path, Dict[str, Any]]], Dict[str, Any]]:
    """
    [EXTRACT] Quét và nạp danh sách các cặp ảnh LR (512x512) và HR (1024x1024)
    cùng file metadata cấu hình suy thoái.
    """
    logger.info(f"[EXTRACT] Quét thư mục testset tại: {testset_dir}")
    hr_dir = testset_dir / "HR"
    lr_dir = testset_dir / "LR_2x"
    meta_path = testset_dir / "metadata.json"

    if not hr_dir.exists() or not lr_dir.exists():
        raise FileNotFoundError(f"[EXTRACT] Thư mục HR hoặc LR_2x không tồn tại trong {testset_dir}")

    metadata_dict: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        logger.info(f"[EXTRACT] Đã nạp metadata chứa {len(metadata_dict.get('images', []))} bản ghi.")

    meta_lookup = {item["filename"]: item for item in metadata_dict.get("images", [])}

    hr_files = sorted(list(hr_dir.glob("*.png")) + list(hr_dir.glob("*.jpg")))
    pairs: List[Tuple[Path, Path, Dict[str, Any]]] = []

    for hr_path in hr_files:
        lr_path = lr_dir / hr_path.name
        if lr_path.exists():
            item_meta = meta_lookup.get(hr_path.name, {})
            pairs.append((lr_path, hr_path, item_meta))
        else:
            logger.warning(f"[EXTRACT] Không tìm thấy ảnh LR tương ứng cho: {hr_path.name}")

    if max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]
        logger.info(f"[EXTRACT] Giới hạn số lượng mẫu đánh giá: {len(pairs)} cặp ảnh.")
    else:
        logger.info(f"[EXTRACT] Nạp thành công {len(pairs)} cặp ảnh LR-HR hoàn chỉnh.")

    return pairs, metadata_dict


# ------------------------------------------------------------------------------
# PIPELINE STEP 2: VALIDATE
# ------------------------------------------------------------------------------
def validate_and_load_models(
    ckpt_compact_path: Path,
    ckpt_orig_path: Path,
    device: torch.device,
) -> Tuple[Compact_SRCNN, SRCNN_Original]:
    """
    [VALIDATE] Nạp trọng số mô hình PyTorch, xác nhận số lượng tham số kiến trúc:
      - Compact SRCNN: Đúng 1.649 tham số (khớp RTL phần cứng).
      - SRCNN Original: Đúng 8.129 tham số.
    """
    logger.info("[VALIDATE] Khởi tạo và nạp trọng số 2 mô hình PyTorch Float32...")

    if not ckpt_compact_path.exists():
        raise FileNotFoundError(f"[VALIDATE] Không tìm thấy checkpoint Compact SRCNN tại: {ckpt_compact_path}")
    if not ckpt_orig_path.exists():
        raise FileNotFoundError(f"[VALIDATE] Không tìm thấy checkpoint SRCNN Original tại: {ckpt_orig_path}")

    # 1. Compact SRCNN
    model_compact = Compact_SRCNN()
    compact_ckpt = torch.load(ckpt_compact_path, map_location="cpu")
    compact_weights = compact_ckpt["state_dict"] if "state_dict" in compact_ckpt else compact_ckpt
    model_compact.load_state_dict(compact_weights)
    model_compact.to(device)
    model_compact.eval()

    param_count_compact = sum(p.numel() for p in model_compact.parameters())
    assert param_count_compact == 1649, f"[VALIDATE] Số tham số Compact SRCNN không khớp: {param_count_compact} != 1649"
    logger.info(f"[VALIDATE] Compact SRCNN đã sẵn sàng: {param_count_compact} tham số (Best Val PSNR: {compact_ckpt.get('best_psnr', 'N/A'):.2f} dB).")

    # 2. SRCNN Original
    model_orig = SRCNN_Original()
    orig_ckpt = torch.load(ckpt_orig_path, map_location="cpu")
    orig_weights = orig_ckpt["state_dict"] if "state_dict" in orig_ckpt else orig_ckpt
    model_orig.load_state_dict(orig_weights)
    model_orig.to(device)
    model_orig.eval()

    param_count_orig = sum(p.numel() for p in model_orig.parameters())
    assert param_count_orig == 8129, f"[VALIDATE] Số tham số SRCNN Original không khớp: {param_count_orig} != 8129"
    logger.info(f"[VALIDATE] SRCNN Original đã sẵn sàng: {param_count_orig} tham số.")

    return model_compact, model_orig


# ------------------------------------------------------------------------------
# PIPELINE STEP 3: TRANSFORM
# ------------------------------------------------------------------------------
def transform_input_pair(
    lr_path: Path,
    hr_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Image.Image, torch.Tensor]:
    """
    [TRANSFORM] Tiền xử lý ảnh theo chuẩn quy chuẩn SRCNN:
      1. Đọc ảnh LR và HR dưới dạng ảnh xám (Grayscale Mode 'L').
      2. Phóng đại Bicubic 2x từ LR (512x512) lên kích thước HR (1024x1024).
      3. Chuyển đổi sang Float32 Tensor [1, 1, 1024, 1024] dải [0.0, 1.0].
      4. Trả về mảng numpy uint8 của HR, Bicubic 2x và tensor đầu vào mô hình.
    """
    with Image.open(hr_path) as im_hr:
        hr_gray = im_hr.convert("L")
        arr_hr = np.array(hr_gray, dtype=np.uint8)

    with Image.open(lr_path) as im_lr:
        lr_gray = im_lr.convert("L")

    # Pre-upsampling Bicubic 2x
    w_target, h_target = arr_hr.shape[1], arr_hr.shape[0]
    bicubic_img = lr_gray.resize((w_target, h_target), Image.Resampling.BICUBIC)
    arr_bicubic = np.array(bicubic_img, dtype=np.uint8)

    # Chuyển sang Float32 Tensor [0.0, 1.0]
    tensor_input = torch.from_numpy(arr_bicubic).float().unsqueeze(0).unsqueeze(0) / 255.0

    return arr_hr, arr_bicubic, bicubic_img, tensor_input


# ------------------------------------------------------------------------------
# PIPELINE STEP 4: EVALUATE
# ------------------------------------------------------------------------------
def run_single_inference(
    model: nn.Module,
    tensor_in: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """Chạy suy luận Float32 trên 1 ảnh và đo độ trễ phần mềm (ms)."""
    tensor_dev = tensor_in.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        out_tensor = model(tensor_dev)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Kẹp biên [0.0, 1.0] và chuyển về uint8 [0, 255]
    out_arr = out_tensor.squeeze().cpu().numpy()
    out_arr_clipped = np.clip(out_arr * 255.0, 0.0, 255.0).astype(np.uint8)

    return out_arr_clipped, elapsed_ms


def evaluate_all_test_samples(
    pairs: List[Tuple[Path, Path, Dict[str, Any]]],
    model_compact: Compact_SRCNN,
    model_orig: SRCNN_Original,
    device: torch.device,
) -> List[SingleEvaluationResult]:
    """
    [EVALUATE] Duyệt qua toàn bộ tập test, đánh giá 3 mô hình và tính toán toàn bộ chỉ số.
    """
    logger.info(f"[EVALUATE] Bắt đầu đánh giá trên {len(pairs)} cặp ảnh y tế...")
    results: List[SingleEvaluationResult] = []

    # Warm-up mô hình 3 lần trước khi đo thời gian
    dummy = torch.randn(1, 1, 1024, 1024, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model_compact(dummy)
            _ = model_orig(dummy)

    total_start = time.perf_counter()

    for idx, (lr_path, hr_path, meta) in enumerate(pairs, 1):
        # 1. Transform
        arr_hr, arr_bicubic, _, tensor_in = transform_input_pair(lr_path, hr_path)

        # 2. Bicubic Baseline
        start_bic = time.perf_counter()
        _ = Image.fromarray(arr_bicubic)
        bic_latency_ms = (time.perf_counter() - start_bic) * 1000.0
        metrics_bic = evaluate_pair(arr_bicubic, arr_hr)
        bic_tier = classify_clinical_tier(metrics_bic.psnr, metrics_bic.ssim)

        # 3. SRCNN Original Float32
        arr_orig, orig_latency_ms = run_single_inference(model_orig, tensor_in, device)
        metrics_orig = evaluate_pair(arr_orig, arr_hr)
        orig_tier = classify_clinical_tier(metrics_orig.psnr, metrics_orig.ssim)

        # 4. Compact SRCNN Float32
        arr_compact, compact_latency_ms = run_single_inference(model_compact, tensor_in, device)
        metrics_compact = evaluate_pair(arr_compact, arr_hr)
        compact_tier = classify_clinical_tier(metrics_compact.psnr, metrics_compact.ssim)

        # Metadata
        noise_sigma = float(meta.get("sigma", 1.0))
        deg_psnr = float(meta.get("degraded_psnr", 0.0))

        # Lưu trữ kết quả
        item_res = SingleEvaluationResult(
            image_id=hr_path.name,
            noise_sigma=noise_sigma,
            degraded_psnr=deg_psnr,
            # Bicubic
            bicubic_psnr=metrics_bic.psnr,
            bicubic_ssim=metrics_bic.ssim,
            bicubic_mse=metrics_bic.mse,
            bicubic_rmse=metrics_bic.rmse,
            bicubic_mae=metrics_bic.mae,
            bicubic_epi=metrics_bic.epi,
            bicubic_mean=metrics_bic.img_mean,
            bicubic_std=metrics_bic.img_std,
            bicubic_tier=bic_tier,
            bicubic_latency_ms=bic_latency_ms,
            # Original
            orig_psnr=metrics_orig.psnr,
            orig_ssim=metrics_orig.ssim,
            orig_mse=metrics_orig.mse,
            orig_rmse=metrics_orig.rmse,
            orig_mae=metrics_orig.mae,
            orig_epi=metrics_orig.epi,
            orig_mean=metrics_orig.img_mean,
            orig_std=metrics_orig.img_std,
            orig_tier=orig_tier,
            orig_latency_ms=orig_latency_ms,
            orig_delta_psnr=metrics_orig.psnr - metrics_bic.psnr,
            orig_delta_ssim=metrics_orig.ssim - metrics_bic.ssim,
            orig_delta_rmse=metrics_bic.rmse - metrics_orig.rmse,
            orig_delta_epi=metrics_orig.epi - metrics_bic.epi,
            # Compact
            compact_psnr=metrics_compact.psnr,
            compact_ssim=metrics_compact.ssim,
            compact_mse=metrics_compact.mse,
            compact_rmse=metrics_compact.rmse,
            compact_mae=metrics_compact.mae,
            compact_epi=metrics_compact.epi,
            compact_mean=metrics_compact.img_mean,
            compact_std=metrics_compact.img_std,
            compact_tier=compact_tier,
            compact_latency_ms=compact_latency_ms,
            compact_delta_psnr=metrics_compact.psnr - metrics_bic.psnr,
            compact_delta_ssim=metrics_compact.ssim - metrics_bic.ssim,
            compact_delta_rmse=metrics_bic.rmse - metrics_compact.rmse,
            compact_delta_epi=metrics_compact.epi - metrics_bic.epi,
        )
        results.append(item_res)

        if idx % 50 == 0 or idx == len(pairs):
            logger.info(
                f"[EVALUATE] Đã xử lý {idx:3d}/{len(pairs)} ảnh: "
                f"Compact PSNR={metrics_compact.psnr:.2f} dB, "
                f"Bicubic PSNR={metrics_bic.psnr:.2f} dB, "
                f"Compact Gain={item_res.compact_delta_psnr:+.2f} dB"
            )

    total_time = time.perf_counter() - total_start
    logger.info(f"[EVALUATE] Hoàn thành đo đạc toàn bộ {len(pairs)} ảnh trong {total_time:.2f}s.")
    return results


# ------------------------------------------------------------------------------
# PIPELINE STEP 5: AUDIT & EXPORT
# ------------------------------------------------------------------------------
def build_summary_table(results: List[SingleEvaluationResult]) -> pd.DataFrame:
    """Xây dựng bảng tổng hợp đối đầu 3 mô hình kèm Mean ± Std và các độ phân tán."""
    n_total = len(results)

    # Trích xuất mảng dữ liệu
    bic_psnr = [r.bicubic_psnr for r in results]
    bic_ssim = [r.bicubic_ssim for r in results]
    bic_mse = [r.bicubic_mse for r in results]
    bic_rmse = [r.bicubic_rmse for r in results]
    bic_mae = [r.bicubic_mae for r in results]
    bic_epi = [r.bicubic_epi for r in results]
    bic_mean = [r.bicubic_mean for r in results]
    bic_std = [r.bicubic_std for r in results]
    bic_lat = [r.bicubic_latency_ms for r in results]

    orig_psnr = [r.orig_psnr for r in results]
    orig_ssim = [r.orig_ssim for r in results]
    orig_mse = [r.orig_mse for r in results]
    orig_rmse = [r.orig_rmse for r in results]
    orig_mae = [r.orig_mae for r in results]
    orig_epi = [r.orig_epi for r in results]
    orig_mean = [r.orig_mean for r in results]
    orig_std = [r.orig_std for r in results]
    orig_lat = [r.orig_latency_ms for r in results]
    orig_gain_psnr = [r.orig_delta_psnr for r in results]
    orig_gain_ssim = [r.orig_delta_ssim for r in results]

    comp_psnr = [r.compact_psnr for r in results]
    comp_ssim = [r.compact_ssim for r in results]
    comp_mse = [r.compact_mse for r in results]
    comp_rmse = [r.compact_rmse for r in results]
    comp_mae = [r.compact_mae for r in results]
    comp_epi = [r.compact_epi for r in results]
    comp_mean = [r.compact_mean for r in results]
    comp_std = [r.compact_std for r in results]
    comp_lat = [r.compact_latency_ms for r in results]
    comp_gain_psnr = [r.compact_delta_psnr for r in results]
    comp_gain_ssim = [r.compact_delta_ssim for r in results]

    # Tính Descriptive Stats
    s_bic_p = compute_descriptive_stats(bic_psnr)
    s_bic_s = compute_descriptive_stats(bic_ssim)
    s_bic_rmse = compute_descriptive_stats(bic_rmse)
    s_bic_mae = compute_descriptive_stats(bic_mae)
    s_bic_epi = compute_descriptive_stats(bic_epi)

    s_orig_p = compute_descriptive_stats(orig_psnr)
    s_orig_s = compute_descriptive_stats(orig_ssim)
    s_orig_rmse = compute_descriptive_stats(orig_rmse)
    s_orig_mae = compute_descriptive_stats(orig_mae)
    s_orig_epi = compute_descriptive_stats(orig_epi)
    s_orig_gp = compute_descriptive_stats(orig_gain_psnr)
    s_orig_gs = compute_descriptive_stats(orig_gain_ssim)

    s_comp_p = compute_descriptive_stats(comp_psnr)
    s_comp_s = compute_descriptive_stats(comp_ssim)
    s_comp_rmse = compute_descriptive_stats(comp_rmse)
    s_comp_mae = compute_descriptive_stats(comp_mae)
    s_comp_epi = compute_descriptive_stats(comp_epi)
    s_comp_gp = compute_descriptive_stats(comp_gain_psnr)
    s_comp_gs = compute_descriptive_stats(comp_gain_ssim)

    data_summary = [
        {
            "Model Name": "Bicubic Interpolation 2x",
            "Param Count": 0,
            "PSNR Mean ± Std (dB)": f"{s_bic_p.mean:.4f} ± {s_bic_p.std:.4f}",
            "PSNR Median (IQR)": f"{s_bic_p.median:.4f} ({s_bic_p.iqr:.4f})",
            "SSIM Mean ± Std": f"{s_bic_s.mean:.4f} ± {s_bic_s.std:.4f}",
            "SSIM Median (IQR)": f"{s_bic_s.median:.4f} ({s_bic_s.iqr:.4f})",
            "RMSE Mean ± Std": f"{s_bic_rmse.mean:.4f} ± {s_bic_rmse.std:.4f}",
            "MAE Mean ± Std": f"{s_bic_mae.mean:.4f} ± {s_bic_mae.std:.4f}",
            "EPI Mean ± Std": f"{s_bic_epi.mean:.4f} ± {s_bic_epi.std:.4f}",
            "Output Mean": f"{np.mean(bic_mean):.2f}",
            "Output STD": f"{np.mean(bic_std):.2f}",
            "Delta PSNR Mean ± Std (dB)": "0.0000 ± 0.0000 (Baseline)",
            "Delta SSIM Mean ± Std": "0.0000 ± 0.0000 (Baseline)",
            "% PSNR Gain > 0": "0.00% (Baseline)",
            "% SSIM Gain > 0": "0.00% (Baseline)",
            "Inference Latency (ms)": f"{np.mean(bic_lat):.2f} ± {np.std(bic_lat):.2f}",
        },
        {
            "Model Name": "SRCNN Original Float32",
            "Param Count": 8129,
            "PSNR Mean ± Std (dB)": f"{s_orig_p.mean:.4f} ± {s_orig_p.std:.4f}",
            "PSNR Median (IQR)": f"{s_orig_p.median:.4f} ({s_orig_p.iqr:.4f})",
            "SSIM Mean ± Std": f"{s_orig_s.mean:.4f} ± {s_orig_s.std:.4f}",
            "SSIM Median (IQR)": f"{s_orig_s.median:.4f} ({s_orig_s.iqr:.4f})",
            "RMSE Mean ± Std": f"{s_orig_rmse.mean:.4f} ± {s_orig_rmse.std:.4f}",
            "MAE Mean ± Std": f"{s_orig_mae.mean:.4f} ± {s_orig_mae.std:.4f}",
            "EPI Mean ± Std": f"{s_orig_epi.mean:.4f} ± {s_orig_epi.std:.4f}",
            "Output Mean": f"{np.mean(orig_mean):.2f}",
            "Output STD": f"{np.mean(orig_std):.2f}",
            "Delta PSNR Mean ± Std (dB)": f"{s_orig_gp.mean:+.4f} ± {s_orig_gp.std:.4f}",
            "Delta SSIM Mean ± Std": f"{s_orig_gs.mean:+.4f} ± {s_orig_gs.std:.4f}",
            "% PSNR Gain > 0": f"{compute_positive_gain_rate(orig_gain_psnr):.2f}%",
            "% SSIM Gain > 0": f"{compute_positive_gain_rate(orig_gain_ssim):.2f}%",
            "Inference Latency (ms)": f"{np.mean(orig_lat):.2f} ± {np.std(orig_lat):.2f}",
        },
        {
            "Model Name": "Compact SRCNN Float32",
            "Param Count": 1649,
            "PSNR Mean ± Std (dB)": f"{s_comp_p.mean:.4f} ± {s_comp_p.std:.4f}",
            "PSNR Median (IQR)": f"{s_comp_p.median:.4f} ({s_comp_p.iqr:.4f})",
            "SSIM Mean ± Std": f"{s_comp_s.mean:.4f} ± {s_comp_s.std:.4f}",
            "SSIM Median (IQR)": f"{s_comp_s.median:.4f} ({s_comp_s.iqr:.4f})",
            "RMSE Mean ± Std": f"{s_comp_rmse.mean:.4f} ± {s_comp_rmse.std:.4f}",
            "MAE Mean ± Std": f"{s_comp_mae.mean:.4f} ± {s_comp_mae.std:.4f}",
            "EPI Mean ± Std": f"{s_comp_epi.mean:.4f} ± {s_comp_epi.std:.4f}",
            "Output Mean": f"{np.mean(comp_mean):.2f}",
            "Output STD": f"{np.mean(comp_std):.2f}",
            "Delta PSNR Mean ± Std (dB)": f"{s_comp_gp.mean:+.4f} ± {s_comp_gp.std:.4f}",
            "Delta SSIM Mean ± Std": f"{s_comp_gs.mean:+.4f} ± {s_comp_gs.std:.4f}",
            "% PSNR Gain > 0": f"{compute_positive_gain_rate(comp_gain_psnr):.2f}%",
            "% SSIM Gain > 0": f"{compute_positive_gain_rate(comp_gain_ssim):.2f}%",
            "Inference Latency (ms)": f"{np.mean(comp_lat):.2f} ± {np.std(comp_lat):.2f}",
        },
    ]

    return pd.DataFrame(data_summary)


def build_clinical_tiers_table(results: List[SingleEvaluationResult]) -> pd.DataFrame:
    """Xây dựng bảng phân tầng cấp bậc lâm sàng 4 bậc y tế cho cả 3 mô hình."""
    tiers = ["EXCELLENT", "GOOD", "ACCEPTABLE", "UNACCEPTABLE"]
    descriptions = [
        "Xuất sắc (PSNR ≥ 40 dB & SSIM ≥ 0.95)",
        "Tốt (35 ≤ PSNR < 40 dB)",
        "Đạt chuẩn lâm sàng (30 ≤ PSNR < 35 dB)",
        "Không đạt chuẩn (PSNR < 30 dB)",
    ]

    total = len(results)
    bic_counts = {t: sum(1 for r in results if r.bicubic_tier == t) for t in tiers}
    orig_counts = {t: sum(1 for r in results if r.orig_tier == t) for t in tiers}
    comp_counts = {t: sum(1 for r in results if r.compact_tier == t) for t in tiers}

    rows = []
    for t, desc in zip(tiers, descriptions):
        rows.append(
            {
                "Clinical Tier": t,
                "Tier Definition": desc,
                "Bicubic Count": bic_counts[t],
                "Bicubic Pct (%)": f"{(bic_counts[t] / total) * 100.0:.2f}%",
                "SRCNN Orig Count": orig_counts[t],
                "SRCNN Orig Pct (%)": f"{(orig_counts[t] / total) * 100.0:.2f}%",
                "Compact SRCNN Count": comp_counts[t],
                "Compact SRCNN Pct (%)": f"{(comp_counts[t] / total) * 100.0:.2f}%",
            }
        )

    return pd.DataFrame(rows)


def build_detailed_table(results: List[SingleEvaluationResult]) -> pd.DataFrame:
    """Xây dựng bảng dữ liệu chi tiết từng bức ảnh."""
    data = []
    for r in results:
        data.append(
            {
                "Image_ID": r.image_id,
                "Noise_Sigma": r.noise_sigma,
                "Degraded_PSNR_dB": r.degraded_psnr,
                # Bicubic
                "Bicubic_PSNR_dB": r.bicubic_psnr,
                "Bicubic_SSIM": r.bicubic_ssim,
                "Bicubic_RMSE": r.bicubic_rmse,
                "Bicubic_MAE": r.bicubic_mae,
                "Bicubic_EPI": r.bicubic_epi,
                "Bicubic_Mean": r.bicubic_mean,
                "Bicubic_STD": r.bicubic_std,
                "Bicubic_Tier": r.bicubic_tier,
                # SRCNN Original
                "Orig_PSNR_dB": r.orig_psnr,
                "Orig_SSIM": r.orig_ssim,
                "Orig_RMSE": r.orig_rmse,
                "Orig_MAE": r.orig_mae,
                "Orig_EPI": r.orig_epi,
                "Orig_Mean": r.orig_mean,
                "Orig_STD": r.orig_std,
                "Orig_Tier": r.orig_tier,
                "Orig_Delta_PSNR_dB": r.orig_delta_psnr,
                "Orig_Delta_SSIM": r.orig_delta_ssim,
                "Orig_Delta_RMSE": r.orig_delta_rmse,
                "Orig_Delta_EPI": r.orig_delta_epi,
                # Compact SRCNN
                "Compact_PSNR_dB": r.compact_psnr,
                "Compact_SSIM": r.compact_ssim,
                "Compact_RMSE": r.compact_rmse,
                "Compact_MAE": r.compact_mae,
                "Compact_EPI": r.compact_epi,
                "Compact_Mean": r.compact_mean,
                "Compact_STD": r.compact_std,
                "Compact_Tier": r.compact_tier,
                "Compact_Delta_PSNR_dB": r.compact_delta_psnr,
                "Compact_Delta_SSIM": r.compact_delta_ssim,
                "Compact_Delta_RMSE": r.compact_delta_rmse,
                "Compact_Delta_EPI": r.compact_delta_epi,
            }
        )
    return pd.DataFrame(data)


def export_deliverables(
    df_summary: pd.DataFrame,
    df_clinical: pd.DataFrame,
    df_detailed: pd.DataFrame,
    metadata_env: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """
    [AUDIT] Xuất bản kết quả ra 3 file chuẩn:
      1. baseline_psnr_ssim_comparison.xlsx (Đa bảng, định dạng chuyên nghiệp)
      2. baseline_psnr_ssim_comparison.csv
      3. baseline_psnr_ssim_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = output_dir / "baseline_psnr_ssim_comparison.xlsx"
    csv_path = output_dir / "baseline_psnr_ssim_comparison.csv"
    json_path = output_dir / "baseline_psnr_ssim_summary.json"

    logger.info(f"[AUDIT] Xuất file Excel tại: {xlsx_path}")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary_Overview", index=False)
        df_clinical.to_excel(writer, sheet_name="Clinical_Grading", index=False)
        df_detailed.to_excel(writer, sheet_name="Detailed_Per_Image", index=False)

        # Metadata sheet
        df_meta = pd.DataFrame(list(metadata_env.items()), columns=["Config Property", "Value"])
        df_meta.to_excel(writer, sheet_name="Metadata_Config", index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = col[0].column_letter
                ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    logger.info(f"[AUDIT] Xuất file CSV chi tiết tại: {csv_path}")
    df_detailed.to_csv(csv_path, index=False)

    logger.info(f"[AUDIT] Xuất file JSON tóm tắt tại: {json_path}")
    json_data = {
        "metadata": metadata_env,
        "summary_table": df_summary.to_dict(orient="records"),
        "clinical_tiers": df_clinical.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return xlsx_path, csv_path, json_path


# ------------------------------------------------------------------------------
# CLI & MAIN ENTRY POINT
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đo đạc chỉ số Float32 Baseline trên tập test đối chứng (Task T1.6)"
    )
    parser.add_argument(
        "--testset-dir",
        type=Path,
        default=DEFAULT_TESTSET_DIR,
        help="Đường dẫn thư mục tập test chứa HR/ và LR_2x/",
    )
    parser.add_argument(
        "--checkpoint-compact",
        type=Path,
        default=DEFAULT_CHECKPOINT_COMPACT,
        help="Đường dẫn checkpoint Compact SRCNN Float32",
    )
    parser.add_argument(
        "--checkpoint-original",
        type=Path,
        default=DEFAULT_CHECKPOINT_ORIGINAL,
        help="Đường dẫn checkpoint SRCNN Original Float32",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục xuất file Excel, CSV và JSON bàn giao",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Thiết bị tính toán PyTorch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chế độ chạy thử nhanh với số mẫu tối thiểu (5 mẫu)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số lượng mẫu đánh giá",
    )
    args = parser.parse_args()

    # Xác định Device
    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    logger.info("================================================================================")
    logger.info("KHỞI CHẠY PIPELINE ĐÁNH GIÁ ĐỐI CHỨNG FLOAT32 BASELINE (TASK T1.6)")
    logger.info("================================================================================")
    logger.info(f"Target Platform : PyTorch {torch.__version__} trên thiết bị: {dev}")
    logger.info(f"Testset Dir     : {args.testset_dir}")
    logger.info(f"Output Dir      : {args.output_dir}")
    logger.info(f"Dry-run Mode    : {args.dry_run}")

    num_samples = 5 if args.dry_run else args.max_samples

    # Step 1: EXTRACT
    pairs, meta_dict = extract_testset_pairs(args.testset_dir, max_samples=num_samples)

    # Step 2: VALIDATE
    model_compact, model_orig = validate_and_load_models(
        args.checkpoint_compact, args.checkpoint_original, dev
    )

    # Step 3 & 4: TRANSFORM & EVALUATE
    results = evaluate_all_test_samples(pairs, model_compact, model_orig, dev)

    # Step 5: AUDIT
    df_summary = build_summary_table(results)
    df_clinical = build_clinical_tiers_table(results)
    df_detailed = build_detailed_table(results)

    meta_env = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "device": str(dev),
        "total_test_images": len(results),
        "compact_checkpoint": str(args.checkpoint_compact),
        "orig_checkpoint": str(args.checkpoint_original),
        "scale_factor": SCALE_FACTOR,
        "metrics_evaluated": [
            "PSNR",
            "SSIM",
            "MSE",
            "RMSE",
            "MAE",
            "EPI (Edge Preservation Index)",
            "Output Mean",
            "Output STD",
            "Delta PSNR",
            "Delta SSIM",
            "Delta RMSE",
            "Delta EPI",
            "% PSNR Gain > 0",
            "% SSIM Gain > 0",
            "Clinical Tiers (4 tiers)",
            "Inference Latency ms",
        ],
    }

    xlsx_path, csv_path, json_path = export_deliverables(
        df_summary, df_clinical, df_detailed, meta_env, args.output_dir
    )

    logger.info("================================================================================")
    logger.info("BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU 3 MỨC:")
    logger.info("================================================================================")
    print(df_summary.to_string(index=False))
    logger.info("--------------------------------------------------------------------------------")
    logger.info("PHÂN TẦNG CẤP BẬC LÂM SÀNG Y TẾ (CLINICAL TIERS):")
    logger.info("--------------------------------------------------------------------------------")
    print(df_clinical.to_string(index=False))
    logger.info("================================================================================")
    logger.info(f"✅ HOÀN TẤT ĐO ĐẠC: Các file báo cáo đã được lưu tại:")
    logger.info(f"  - Excel : {xlsx_path}")
    logger.info(f"  - CSV   : {csv_path}")
    logger.info(f"  - JSON  : {json_path}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
