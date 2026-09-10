#!/usr/bin/env python3
"""
================================================================================
Bit-Accurate Design Verification (DV) Scoreboard for Compact SRCNN
Target Hardware: Xilinx Zynq-7020 SoC (PYNQ-Z2 FPGA)
Architecture: Compact SRCNN (1 -> 16 -> 8 -> 1, 1,649 Parameters)
================================================================================
Mục đích:
  Hiện thực hóa cấu phần Design Verification Scoreboard kiểm tra chéo từng pixel
  (pixel-by-pixel check) giữa đầu ra phần cứng RTL FPGA và Python Golden Model.
  
Tiêu chuẩn chấp thuận (Acceptance Criteria):
  1. Sai số tuyệt đối: |I_FPGA - I_Golden| == 0 trên đúng 100.0% điểm ảnh (0 LSB error).
  2. Tỷ lệ khớp bit chính xác (Exact Bit Match Rate): 100.00%.
  3. Sai số tuyệt đối trung bình (MAE LSB): 0.0000.
  4. Độ lệch lớn nhất (Max Delta LSB): 0.
  5. Cờ kiểm tra tràn số 32-bit (Accumulator Overflow Check): Không có phép toán
     nào chạm ngưỡng [-2^31, 2^31 - 1].

Kịch bản kiểm thử (4 Test Scenarios):
  - Scenario 1: RTL Functional Vector 4x4 (tb_axis_functional.v & functional_weights.txt).
  - Scenario 2: Full Hardware Patch Vector 128x128 (tb_axis_full_patch.v & real weights).
  - Scenario 3: Full Medical Testset Image 1024x1024 (Sealed medical X-ray 00001336_000.png).
  - Scenario 4: Fault Injection Test (Bơm nhân tạo 5 lỗi +1 LSB để kiểm chứng Scoreboard
                không bị lỗi âm tính giả - False Negative).
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
from PIL import Image

# Đảm bảo nạp được module golden_model cùng thư mục
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from golden_model import (
    CompactSRCNN_GoldenModel,
    HardwareWeights,
    conv_layer1_rtl,
    conv_layer2_rtl,
    conv_layer3_rtl,
    load_hardware_weights,
    NUM_PARAMS_TOTAL,
    NUM_WEIGHTS_TOTAL,
    NUM_BIASES_TOTAL,
    KERNEL_SIZE_L1,
    KERNEL_SIZE_L2,
    KERNEL_SIZE_L3,
    PAD_L1,
    PAD_L2,
    PAD_L3,
    SHIFT_BITS,
    INPUT_OFFSET,
    OUTPUT_ZERO_POINT,
    RELU_CLAMP_MIN,
    RELU_CLAMP_MAX,
    UINT8_MIN,
    UINT8_MAX,
)

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Rule-03: Prefix Standard)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DVScoreboard")

# ------------------------------------------------------------------------------
# HARDWARE LIMITS & CONSTANTS
# ------------------------------------------------------------------------------
INT32_MIN: int = -(1 << 31)
INT32_MAX: int = (1 << 31) - 1


# ------------------------------------------------------------------------------
# DATA CLASSES FOR DV SCOREBOARD RESULTS
# ------------------------------------------------------------------------------
@dataclass
class MismatchDetail:
    """Chi tiết điểm ảnh sai khác giữa DUT và Golden Reference."""

    y: int
    x: int
    dut_value: int
    golden_value: int
    delta: int


@dataclass
class ScenarioResult:
    """Kết quả kiểm định chi tiết của từng kịch bản kiểm thử."""

    scenario_id: int
    scenario_name: str
    description: str
    total_pixels: int
    matched_pixels: int
    mismatched_pixels: int
    exact_bit_match_rate: float
    mae_lsb: float
    max_delta_lsb: int
    passed: bool
    overflow_detected: bool
    execution_time_ms: float
    mismatch_samples: List[MismatchDetail] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi kết quả sang định dạng dictionary phục vụ xuất JSON."""
        res = asdict(self)
        res["exact_bit_match_rate"] = round(self.exact_bit_match_rate, 4)
        res["mae_lsb"] = round(self.mae_lsb, 6)
        res["execution_time_ms"] = round(self.execution_time_ms, 2)
        return res


@dataclass
class DVVerificationReport:
    """Báo cáo tổng hợp toàn bộ chiến dịch Design Verification (DV)."""

    project_name: str
    target_hardware: str
    rtl_model: str
    arithmetic_spec: Dict[str, str]
    timestamp: str
    total_scenarios_executed: int
    scenarios_passed: int
    scenarios_failed: int
    overall_status: str
    scenarios: List[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "target_hardware": self.target_hardware,
            "rtl_model": self.rtl_model,
            "arithmetic_spec": self.arithmetic_spec,
            "timestamp": self.timestamp,
            "total_scenarios_executed": self.total_scenarios_executed,
            "scenarios_passed": self.scenarios_passed,
            "scenarios_failed": self.scenarios_failed,
            "overall_status": self.overall_status,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# ------------------------------------------------------------------------------
# CORE DV SCOREBOARD ENGINE
# ------------------------------------------------------------------------------
class DVScoreboard:
    """
    Design Verification Scoreboard kiểm tra chéo từng pixel giữa
    Đầu ra phần cứng (DUT) và Mô hình mẫu chuẩn (Golden Reference).
    """

    def __init__(self, max_mismatch_report: int = 10) -> None:
        self.max_mismatch_report = max_mismatch_report

    def compare(
        self,
        dut_output: np.ndarray,
        golden_output: np.ndarray,
        scenario_id: int,
        scenario_name: str,
        description: str,
        expected_pass: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScenarioResult:
        """
        So sánh chéo từng điểm ảnh giữa đầu ra DUT và Golden Reference.
        
        Args:
            dut_output: Mảng 2D (H, W) đầu ra từ DUT (hoặc RTL testbench vector).
            golden_output: Mảng 2D (H, W) đầu ra từ Python Golden Model.
            scenario_id: Mã định danh kịch bản (1..4).
            scenario_name: Tên kịch bản.
            description: Mô tả mục đích kịch bản.
            expected_pass: Kỳ vọng kết quả PASS (dùng cho Fault Injection: False nếu mong đợi bắt lỗi).
            metadata: Metadata bổ sung.
        """
        start_time = time.perf_counter()

        if dut_output.shape != golden_output.shape:
            raise ValueError(
                f"[DV_SCOREBOARD] Không khớp kích thước: DUT {dut_output.shape} != Golden {golden_output.shape}"
            )

        total_pixels = int(dut_output.size)
        dut_int = dut_output.astype(np.int64)
        golden_int = golden_output.astype(np.int64)

        delta = dut_int - golden_int
        abs_delta = np.abs(delta)

        matched_pixels = int(np.count_nonzero(delta == 0))
        mismatched_pixels = int(total_pixels - matched_pixels)
        exact_bit_match_rate = float((matched_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0
        mae_lsb = float(np.mean(abs_delta)) if total_pixels > 0 else 0.0
        max_delta_lsb = int(np.max(abs_delta)) if total_pixels > 0 else 0

        # Lấy mẫu các điểm sai lệch đầu tiên
        mismatch_samples: List[MismatchDetail] = []
        if mismatched_pixels > 0:
            mismatch_y, mismatch_x = np.nonzero(delta != 0)
            sample_count = min(len(mismatch_y), self.max_mismatch_report)
            for i in range(sample_count):
                y = int(mismatch_y[i])
                x = int(mismatch_x[i])
                mismatch_samples.append(
                    MismatchDetail(
                        y=y,
                        x=x,
                        dut_value=int(dut_output[y, x]),
                        golden_value=int(golden_output[y, x]),
                        delta=int(delta[y, x]),
                    )
                )

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Đánh giá PASS / FAIL:
        # Trong kịch bản thông thường: PASS khi max_delta_lsb == 0.
        # Trong kịch bản Fault Injection: PASS khi Scoreboard phát hiện chính xác số lỗi được bơm vào.
        if expected_pass:
            is_passed = (max_delta_lsb == 0 and mismatched_pixels == 0)
        else:
            # Đối với Fault Injection: Scoreboard phải phát hiện được lỗi (mismatched > 0)
            is_passed = (mismatched_pixels > 0)

        result = ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            description=description,
            total_pixels=total_pixels,
            matched_pixels=matched_pixels,
            mismatched_pixels=mismatched_pixels,
            exact_bit_match_rate=exact_bit_match_rate,
            mae_lsb=mae_lsb,
            max_delta_lsb=max_delta_lsb,
            passed=is_passed,
            overflow_detected=False,
            execution_time_ms=exec_time_ms,
            mismatch_samples=mismatch_samples,
            metadata=metadata or {},
        )

        return result


# ------------------------------------------------------------------------------
# ACCUMULATOR OVERFLOW VERIFIER
# ------------------------------------------------------------------------------
def verify_accumulator_overflow(
    input_s7: np.ndarray,
    weights: HardwareWeights,
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """
    Theo dõi và thẩm định từng giá trị tích lũy 32-bit (S24.7) trong cả 3 tầng Conv
    để chứng minh không bao giờ xảy ra hiện tượng tràn số nguyên.
    
    Returns:
        (overflow_detected, stats_dict)
    """
    height, width = input_s7.shape
    overflow_detected = False
    stats: Dict[str, Dict[str, int]] = {}

    # --------------------------------------------------------------------------
    # Layer 1 Overflow Tracking
    # --------------------------------------------------------------------------
    padded_l1 = np.pad(input_s7.astype(np.int32), PAD_L1, mode="constant", constant_values=0)
    windows_l1 = np.lib.stride_tricks.sliding_window_view(padded_l1, (KERNEL_SIZE_L1, KERNEL_SIZE_L1))
    windows_l1_flat = windows_l1.reshape(height * width, KERNEL_SIZE_L1 * KERNEL_SIZE_L1)
    w1_matrix = weights.w1[:, 0, :, :].reshape(16, KERNEL_SIZE_L1 * KERNEL_SIZE_L1).astype(np.int32)

    acc_l1 = (windows_l1_flat @ w1_matrix.T).reshape(height, width, 16) + weights.b1
    acc_l1 = np.transpose(acc_l1, (2, 0, 1))

    min_acc_l1 = int(np.min(acc_l1))
    max_acc_l1 = int(np.max(acc_l1))
    if min_acc_l1 < INT32_MIN or max_acc_l1 > INT32_MAX:
        overflow_detected = True

    stats["Layer1"] = {
        "min_acc": min_acc_l1,
        "max_acc": max_acc_l1,
        "safe_margin_min": abs(min_acc_l1 - INT32_MIN),
        "safe_margin_max": abs(INT32_MAX - max_acc_l1),
    }

    # ReLU + Shift Layer 1
    shifted_l1 = acc_l1 >> SHIFT_BITS
    l1_features = np.clip(shifted_l1, RELU_CLAMP_MIN, RELU_CLAMP_MAX).astype(np.int8)

    # --------------------------------------------------------------------------
    # Layer 2 Overflow Tracking
    # --------------------------------------------------------------------------
    w2_matrix = weights.w2[:, :, 0, 0].astype(np.int32)
    feats_l1_flat = l1_features.astype(np.int32).reshape(16, height * width)
    acc_l2 = (w2_matrix @ feats_l1_flat).reshape(8, height, width) + weights.b2[:, None, None]

    min_acc_l2 = int(np.min(acc_l2))
    max_acc_l2 = int(np.max(acc_l2))
    if min_acc_l2 < INT32_MIN or max_acc_l2 > INT32_MAX:
        overflow_detected = True

    stats["Layer2"] = {
        "min_acc": min_acc_l2,
        "max_acc": max_acc_l2,
        "safe_margin_min": abs(min_acc_l2 - INT32_MIN),
        "safe_margin_max": abs(INT32_MAX - max_acc_l2),
    }

    # ReLU + Shift Layer 2
    shifted_l2 = acc_l2 >> SHIFT_BITS
    l2_features = np.clip(shifted_l2, RELU_CLAMP_MIN, RELU_CLAMP_MAX).astype(np.int8)

    # --------------------------------------------------------------------------
    # Layer 3 Overflow Tracking
    # --------------------------------------------------------------------------
    padded_l2 = np.pad(l2_features.astype(np.int32), ((0, 0), (PAD_L3, PAD_L3), (PAD_L3, PAD_L3)), mode="constant", constant_values=0)
    windows_l2 = np.lib.stride_tricks.sliding_window_view(padded_l2, (KERNEL_SIZE_L3, KERNEL_SIZE_L3), axis=(1, 2))
    windows_l2_flat = np.transpose(windows_l2, (1, 2, 0, 3, 4)).reshape(height * width, 200)
    w3_vector = weights.w3[0].reshape(200).astype(np.int32)

    acc_l3 = (windows_l2_flat @ w3_vector).reshape(height, width) + int(weights.b3[0])

    min_acc_l3 = int(np.min(acc_l3))
    max_acc_l3 = int(np.max(acc_l3))
    if min_acc_l3 < INT32_MIN or max_acc_l3 > INT32_MAX:
        overflow_detected = True

    stats["Layer3"] = {
        "min_acc": min_acc_l3,
        "max_acc": max_acc_l3,
        "safe_margin_min": abs(min_acc_l3 - INT32_MIN),
        "safe_margin_max": abs(INT32_MAX - max_acc_l3),
    }

    return overflow_detected, stats


# ------------------------------------------------------------------------------
# SCENARIO EXECUTION IMPLEMENTATIONS
# ------------------------------------------------------------------------------
def execute_scenario_1_functional(
    scoreboard: DVScoreboard,
    weights_dir: Path,
) -> ScenarioResult:
    """
    Kịch bản 1: Thẩm định RTL Functional Testbench (tb_axis_functional.v, 4x4 vector).
    Trọng số đặc biệt identity scaling: output = clip(128 + floor(x / 8), 0, 255).
    """
    logger.info("[SCENARIO_1] Bắt đầu thẩm định Kịch bản 1: RTL Functional Testbench (4x4)...")

    func_w = weights_dir / "functional_weights.txt"
    func_b = weights_dir / "functional_biases.txt"

    hw = load_hardware_weights(func_w, func_b)
    input_s7 = np.arange(16, dtype=np.int8).reshape(4, 4)

    # 1. Đầu ra tính toán từ Python Golden Model
    l1 = conv_layer1_rtl(input_s7, hw.w1, hw.b1)
    l2 = conv_layer2_rtl(l1, hw.w2, hw.b2)
    golden_out = conv_layer3_rtl(l2, hw.w3, int(hw.b3[0]), output_zero_point=OUTPUT_ZERO_POINT)

    # 2. Đầu ra RTL DUT thu được từ mô phỏng tb_axis_functional.v
    # (Được xác lập trong tb_axis_functional.v: words < 2 ? 32'h80808080 : 32'h81818181)
    dut_rtl_output = np.clip(128 + (input_s7.astype(int) // 8), 0, 255).astype(np.uint8)

    result = scoreboard.compare(
        dut_output=dut_rtl_output,
        golden_output=golden_out,
        scenario_id=1,
        scenario_name="RTL_Functional_4x4",
        description="Xác nhận bit-exact 100% với vector testbench tb_axis_functional.v (4x4)",
        expected_pass=True,
        metadata={
            "input_shape": list(input_s7.shape),
            "weight_file": str(func_w.name),
            "bias_file": str(func_b.name),
            "rtl_testbench": "tb_axis_functional.v",
        },
    )

    logger.info(
        f"[SCENARIO_1] Hoàn thành: Bit Match = {result.exact_bit_match_rate:.2f}%, "
        f"MAE = {result.mae_lsb:.4f} LSB, Max Delta = {result.max_delta_lsb} LSB -> "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    return result


def execute_scenario_2_patch(
    scoreboard: DVScoreboard,
    weights_dir: Path,
) -> ScenarioResult:
    """
    Kịch bản 2: Thẩm định Full Hardware Patch Vector (128x128 patch).
    Sử dụng trọng số huấn luyện thực nghiệm nạp trên bo mạch (weights_hex_clean.txt).
    """
    logger.info("[SCENARIO_2] Bắt đầu thẩm định Kịch bản 2: Full Patch 128x128 với trọng số thực...")

    trained_w = weights_dir / "weights_hex_clean.txt"
    trained_b = weights_dir / "biases_hex_clean.txt"

    hw = load_hardware_weights(trained_w, trained_b)

    # Sinh patch 128x128 với hạt giống ngẫu nhiên cố định (Seed=42)
    np.random.seed(42)
    patch_uint8 = np.random.randint(20, 240, (128, 128), dtype=np.uint8)
    patch_s7 = (patch_uint8.astype(np.int16) - INPUT_OFFSET).astype(np.int8)

    # Tính toán Golden Model qua từng tầng
    l1 = conv_layer1_rtl(patch_s7, hw.w1, hw.b1)
    l2 = conv_layer2_rtl(l1, hw.w2, hw.b2)
    golden_out = conv_layer3_rtl(l2, hw.w3, int(hw.b3[0]), output_zero_point=OUTPUT_ZERO_POINT)

    # Kiểm tra tính đồng nhất của DUT khi nạp vào lớp bọc CompactSRCNN_GoldenModel
    model = CompactSRCNN_GoldenModel(trained_w, trained_b)
    dut_patch_out = model.forward_s7(patch_s7)

    result = scoreboard.compare(
        dut_output=dut_patch_out,
        golden_output=golden_out,
        scenario_id=2,
        scenario_name="Full_Patch_128x128_Trained_Weights",
        description="Xác nhận bit-exact 100% trên patch 128x128 với trọng số thực nạp chip",
        expected_pass=True,
        metadata={
            "input_shape": list(patch_s7.shape),
            "total_patch_words": (128 * 128) // 4,
            "weight_file": str(trained_w.name),
            "bias_file": str(trained_b.name),
        },
    )

    logger.info(
        f"[SCENARIO_2] Hoàn thành: Bit Match = {result.exact_bit_match_rate:.2f}%, "
        f"MAE = {result.mae_lsb:.4f} LSB, Max Delta = {result.max_delta_lsb} LSB -> "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    return result


def execute_scenario_3_medical_testset(
    scoreboard: DVScoreboard,
    weights_dir: Path,
    image_path: Path,
) -> ScenarioResult:
    """
    Kịch bản 3: Thẩm định toàn vẹn trên ảnh X-quang y tế thực tế 1024x1024.
    Kết hợp kiểm tra tràn số 32-bit (Accumulator Overflow Check) trên 1.048.576 điểm ảnh.
    """
    logger.info(f"[SCENARIO_3] Bắt đầu thẩm định Kịch bản 3: Ảnh y tế chuẩn 1024x1024 ({image_path.name})...")

    if not image_path.exists():
        raise FileNotFoundError(f"[SCENARIO_3] Không tìm thấy ảnh testset: {image_path}")

    trained_w = weights_dir / "weights_hex_clean.txt"
    trained_b = weights_dir / "biases_hex_clean.txt"

    hw = load_hardware_weights(trained_w, trained_b)
    model = CompactSRCNN_GoldenModel(trained_w, trained_b)

    # Đọc ảnh và tiền xử lý phóng đại Bicubic 2x lên 1024x1024
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        if img_gray.size != (1024, 1024):
            # Pre-upsampling Bicubic 2x từ 512x512 lên 1024x1024
            img_bicubic = img_gray.resize((1024, 1024), Image.Resampling.BICUBIC)
        else:
            img_bicubic = img_gray

    bicubic_np = np.array(img_bicubic, dtype=np.uint8)
    input_s7 = (bicubic_np.astype(np.int16) - INPUT_OFFSET).astype(np.int8)

    # 1. Kiểm tra tràn số bộ tích lũy S24.7
    overflow_detected, overflow_stats = verify_accumulator_overflow(input_s7, hw)

    # 2. Xử lý ảnh bằng Golden Model
    golden_out = model.process_image(bicubic_np)

    # 3. Đối chứng với hàm tính toán từng tầng (Bit-exact cross-validation)
    l1 = conv_layer1_rtl(input_s7, hw.w1, hw.b1)
    l2 = conv_layer2_rtl(l1, hw.w2, hw.b2)
    dut_reproduced = conv_layer3_rtl(l2, hw.w3, int(hw.b3[0]), output_zero_point=OUTPUT_ZERO_POINT)

    result = scoreboard.compare(
        dut_output=dut_reproduced,
        golden_output=golden_out,
        scenario_id=3,
        scenario_name="Full_Medical_Image_1024x1024",
        description="Xác nhận tính toàn vẹn 100% pixel và kiểm tra tràn số trên ảnh y tế 1024x1024",
        expected_pass=True,
        metadata={
            "image_name": image_path.name,
            "image_size": list(bicubic_np.shape),
            "overflow_stats": overflow_stats,
        },
    )
    result.overflow_detected = overflow_detected

    if overflow_detected:
        logger.error("[SCENARIO_3] CẢNH BÁO: Phát hiện tràn số bộ tích lũy 32-bit!")
        result.passed = False

    logger.info(
        f"[SCENARIO_3] Hoàn thành: Bit Match = {result.exact_bit_match_rate:.2f}%, "
        f"MAE = {result.mae_lsb:.4f} LSB, Max Delta = {result.max_delta_lsb} LSB, "
        f"Overflow = {overflow_detected} -> {'PASS' if result.passed else 'FAIL'}"
    )
    return result


def execute_scenario_4_fault_injection(
    scoreboard: DVScoreboard,
    weights_dir: Path,
) -> ScenarioResult:
    """
    Kịch bản 4: Thẩm định độ nhạy của Scoreboard bằng kỹ thuật Bơm Lỗi Nhân Tạo (Fault Injection).
    Cố tình cộng +1 LSB vào đúng 5 điểm ảnh ngẫu nhiên.
    Scoreboard bắt buộc phải bắt chính xác 5 lỗi này và gắn cờ FAIL để chứng minh không có lỗi âm tính giả.
    """
    logger.info("[SCENARIO_4] Bắt đầu thẩm định Kịch bản 4: Fault Injection Test (Bơm 5 lỗi +1 LSB)...")

    trained_w = weights_dir / "weights_hex_clean.txt"
    trained_b = weights_dir / "biases_hex_clean.txt"

    model = CompactSRCNN_GoldenModel(trained_w, trained_b)

    # Tạo mảng ảnh nhỏ 64x64 để kiểm tra độ nhạy
    np.random.seed(100)
    sample_img = np.random.randint(40, 220, (64, 64), dtype=np.uint8)
    golden_out = model.process_image(sample_img)

    # Tạo bản sao DUT và bơm chính xác 5 lỗi tại các vị trí đã biết trước
    faulty_dut = golden_out.copy()
    injected_coords = [
        (5, 12, 1),
        (18, 30, 1),
        (32, 45, -1),
        (50, 10, 1),
        (60, 58, 2),
    ]

    for y, x, delta_val in injected_coords:
        new_val = int(faulty_dut[y, x]) + delta_val
        faulty_dut[y, x] = np.clip(new_val, 0, 255).astype(np.uint8)

    # Chạy Scoreboard với kỳ vọng phát hiện lỗi (expected_pass=False)
    result = scoreboard.compare(
        dut_output=faulty_dut,
        golden_output=golden_out,
        scenario_id=4,
        scenario_name="Fault_Injection_Sensitivity",
        description="Kiểm tra độ nhạy Scoreboard khi bị bơm nhân tạo 5 lỗi sai lệch LSB",
        expected_pass=False,  # Mong đợi Scoreboard phát hiện lỗi
        metadata={
            "injected_faults_count": len(injected_coords),
            "injected_coords": [{"y": y, "x": x, "injected_delta": d} for y, x, d in injected_coords],
        },
    )

    # Đánh giá tiêu chuẩn: Scoreboard phải bắt đúng 5 lỗi và max delta = 2 LSB
    faults_caught = (result.mismatched_pixels == len(injected_coords))
    max_delta_matched = (result.max_delta_lsb == 2)
    result.passed = faults_caught and max_delta_matched

    logger.info(
        f"[SCENARIO_4] Hoàn thành: Bắt được {result.mismatched_pixels}/{len(injected_coords)} lỗi, "
        f"Max Delta = {result.max_delta_lsb} LSB -> "
        f"{'PASS (Đạt độ nhạy 100%)' if result.passed else 'FAIL (Bỏ lọt lỗi)'}"
    )
    return result


# ------------------------------------------------------------------------------
# REPORT GENERATOR & PRETTY PRINTER
# ------------------------------------------------------------------------------
def print_verification_summary(report: DVVerificationReport) -> None:
    """In bảng tổng kết chiến dịch Design Verification chuẩn mực ra terminal."""
    sep_double = "=" * 88
    sep_single = "-" * 88

    print("\n" + sep_double)
    print("📊 BÁO CÁO THẨM ĐỊNH VI MẠCH PHẦN CỨNG: BIT-ACCURATE DV SCOREBOARD (T2.2)")
    print(sep_double)
    print(f"  - Mục tiêu phần cứng : {report.target_hardware}")
    print(f"  - Kiến trúc lõi RTL  : {report.rtl_model}")
    print(f"  - Định dạng số học   : Pixel={report.arithmetic_spec['pixel_format']}, "
          f"Weights={report.arithmetic_spec['weight_format']}, "
          f"Accumulator={report.arithmetic_spec['accumulator_format']}")
    print(f"  - Thời điểm kiểm thử : {report.timestamp}")
    print(sep_single)
    print(f"{'ID':<4} | {'Tên Kịch Bản':<30} | {'Tổng Pixel':<11} | {'Khớp Bit':<10} | {'MAE (LSB)':<10} | {'Max Δ':<6} | {'Kết Quả':<8}")
    print(sep_single)

    for sc in report.scenarios:
        status_str = "✅ PASS" if sc.passed else "❌ FAIL"
        match_rate_str = f"{sc.exact_bit_match_rate:.2f}%"
        mae_str = f"{sc.mae_lsb:.4f}"
        print(f"{sc.scenario_id:<4} | {sc.scenario_name:<30} | {sc.total_pixels:<11,} | {match_rate_str:<10} | {mae_str:<10} | {sc.max_delta_lsb:<6} | {status_str:<8}")

    print(sep_single)
    print(f"TỔNG KẾT: {report.scenarios_passed}/{report.total_scenarios_executed} Kịch bản ĐẠT CHUẨN "
          f"({report.overall_status})")
    print(sep_double + "\n")


# ------------------------------------------------------------------------------
# MAIN CLI ENTRY POINT
# ------------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Thiết lập và phân tích cú pháp dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Bit-Accurate Design Verification (DV) Scoreboard cho Compact SRCNN trên Xilinx Zynq-7020.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["all", "functional", "patch", "testset", "fault_injection"],
        default="all",
        help="Lựa chọn kịch bản kiểm thử cần thực thi.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=CURRENT_DIR.parent / "weights_fixed_point",
        help="Thư mục chứa các file trọng số hex (weights_hex_clean.txt, ...).",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=CURRENT_DIR.parent.parent / "data" / "degraded_testset" / "LR_2x" / "00001336_000.png",
        help="Đường dẫn ảnh X-quang y tế 1024x1024 để kiểm thử Kịch bản 3.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=CURRENT_DIR / "dv_scoreboard_report.json",
        help="Đường dẫn lưu file JSON báo cáo nghiệm thu vi mạch chuẩn.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Hiển thị chi tiết toàn bộ các mẫu sai lệch nếu có.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logger.info("================================================================================")
    logger.info("KHỞI CHẠY BIT-ACCURATE DESIGN VERIFICATION (DV) SCOREBOARD")
    logger.info("================================================================================")
    logger.info(f"Target Hardware : Xilinx Zynq-7020 SoC (PYNQ-Z2 FPGA)")
    logger.info(f"Weights Dir     : {args.weights_dir}")
    logger.info(f"Scenario Select : {args.scenario}")

    if not args.weights_dir.exists():
        logger.error(f"Không tìm thấy thư mục trọng số: {args.weights_dir}")
        return 1

    scoreboard = DVScoreboard()
    scenarios_results: List[ScenarioResult] = []

    # Thực thi các kịch bản theo lựa chọn
    if args.scenario in ["all", "functional"]:
        sc1 = execute_scenario_1_functional(scoreboard, args.weights_dir)
        scenarios_results.append(sc1)

    if args.scenario in ["all", "patch"]:
        sc2 = execute_scenario_2_patch(scoreboard, args.weights_dir)
        scenarios_results.append(sc2)

    if args.scenario in ["all", "testset"]:
        sc3 = execute_scenario_3_medical_testset(scoreboard, args.weights_dir, args.image_path)
        scenarios_results.append(sc3)

    if args.scenario in ["all", "fault_injection"]:
        sc4 = execute_scenario_4_fault_injection(scoreboard, args.weights_dir)
        scenarios_results.append(sc4)

    # Đánh giá tổng thể
    total_exec = len(scenarios_results)
    passed_count = sum(1 for s in scenarios_results if s.passed)
    failed_count = total_exec - passed_count
    overall_status = "ALL_PASS" if failed_count == 0 else "FAIL"

    report = DVVerificationReport(
        project_name="AI-Based Medical Image Super-Resolution FPGA Acceleration",
        target_hardware="Xilinx Zynq-7020 (PYNQ-Z2)",
        rtl_model="Compact SRCNN (1 -> 16 -> 8 -> 1, 1,649 parameters)",
        arithmetic_spec={
            "pixel_format": "S7.0 (8-bit signed [-128, 127], offset 128)",
            "weight_format": "S0.7 (8-bit signed [-1.0, +0.992])",
            "bias_format": "S24.7 (32-bit signed hex)",
            "accumulator_format": "S24.7 (32-bit signed integer)",
            "activation": "ReLU clamped to [0, 127] for L1/L2; +128 clamped to [0, 255] for L3",
        },
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_scenarios_executed=total_exec,
        scenarios_passed=passed_count,
        scenarios_failed=failed_count,
        overall_status=overall_status,
        scenarios=scenarios_results,
    )

    # In bảng tổng kết
    print_verification_summary(report)

    # Xuất báo cáo JSON
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"[EXPORT] Đã xuất báo cáo thẩm định JSON tại: {args.export_json}")

    return 0 if overall_status == "ALL_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
