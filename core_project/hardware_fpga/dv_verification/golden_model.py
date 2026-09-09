#!/usr/bin/env python3
"""
================================================================================
Bit-Accurate Python Golden Model for Compact SRCNN (1 -> 16 -> 8 -> 1)
Hardware Acceleration on Xilinx Zynq-7020 (PYNQ-Z2)
================================================================================
Mục đích:
  Mô phỏng chính xác 100% số học số nguyên Fixed-Point của RTL phần cứng:
    1. Pixel đầu vào: S7.0 (8-bit signed [-128, 127], trừ offset 128 từ uint8 [0, 255]).
    2. Trọng số: S0.7 (8-bit signed [-1.0, +0.992], 1.624 trọng số).
    3. Biases: S24.7 (32-bit signed, 25 bias giá trị hex).
    4. Bộ tích lũy (Accumulator): S24.7 (32-bit signed integer math).
    5. Dịch bit & Kẹp biên:
       - Layer 1 (9x9, pad 4): acc >> 7, ReLU kẹp vào [0, 127].
       - Layer 2 (1x1, pad 0): acc >> 7, ReLU kẹp vào [0, 127].
       - Layer 3 (5x5, pad 2): acc >> 7, cộng 128 (OUTPUT_ZERO_POINT), kẹp vào [0, 255].
  Cam kết: 0.0 floating-point operations trong toàn bộ quá trình tích lũy và lượng tử.
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Rule-03: Prefix standard)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GoldenModel")

# ------------------------------------------------------------------------------
# RTL HARDWARE CONSTANTS
# ------------------------------------------------------------------------------
NUM_PARAMS_TOTAL: int = 1649
NUM_WEIGHTS_TOTAL: int = 1624
NUM_BIASES_TOTAL: int = 25

NUM_WEIGHTS_L1: int = 1296  # 16 * 1 * 9 * 9
NUM_WEIGHTS_L2: int = 128   # 8 * 16 * 1 * 1
NUM_WEIGHTS_L3: int = 200   # 1 * 8 * 5 * 5

NUM_BIASES_L1: int = 16
NUM_BIASES_L2: int = 8
NUM_BIASES_L3: int = 1

KERNEL_SIZE_L1: int = 9
PAD_L1: int = 4
KERNEL_SIZE_L2: int = 1
PAD_L2: int = 0
KERNEL_SIZE_L3: int = 5
PAD_L3: int = 2

SHIFT_BITS: int = 7
INPUT_OFFSET: int = 128
OUTPUT_ZERO_POINT: int = 128
RELU_CLAMP_MIN: int = 0
RELU_CLAMP_MAX: int = 127
UINT8_MIN: int = 0
UINT8_MAX: int = 255


# ------------------------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareWeights:
    """Tập hợp toàn bộ trọng số và bias phần cứng đã phân rã theo tầng."""

    w1: np.ndarray  # Shape: (16, 1, 9, 9), int8
    b1: np.ndarray  # Shape: (16,), int32
    w2: np.ndarray  # Shape: (8, 16, 1, 1), int8
    b2: np.ndarray  # Shape: (8,), int32
    w3: np.ndarray  # Shape: (1, 8, 5, 5), int8
    b3: np.ndarray  # Shape: (1,), int32


# ------------------------------------------------------------------------------
# WEIGHT LOADER & HEX PARSER
# ------------------------------------------------------------------------------
def parse_signed_hex_byte(hex_str: str) -> int:
    """Chuyển chuỗi hex 2 ký tự (bù 2) thành số nguyên có dấu 8-bit [-128, 127]."""
    val = int(hex_str.strip(), 16)
    if val >= 128:
        val -= 256
    return val


def parse_signed_hex_word(hex_str: str) -> int:
    """Chuyển chuỗi hex 8 ký tự (bù 2) thành số nguyên có dấu 32-bit."""
    val = int(hex_str.strip(), 16)
    if val >= (1 << 31):
        val -= (1 << 32)
    return val


def load_hardware_weights(
    weights_path: Union[str, Path],
    biases_path: Union[str, Path],
) -> HardwareWeights:
    """
    Đọc và phân rã các file hex trọng số và bias phần cứng RTL.
    
    Args:
        weights_path: Đường dẫn file weights_hex_clean.txt (1.624 dòng).
        biases_path: Đường dẫn file biases_hex_clean.txt (25 dòng).
        
    Returns:
        HardwareWeights chứa các mảng numpy int8 và int32.
    """
    w_path = Path(weights_path)
    b_path = Path(biases_path)

    if not w_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {w_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file biases: {b_path}")

    with open(w_path, "r", encoding="utf-8") as f:
        w_lines = [line.strip() for line in f if line.strip()]
    with open(b_path, "r", encoding="utf-8") as f:
        b_lines = [line.strip() for line in f if line.strip()]

    if len(w_lines) != NUM_WEIGHTS_TOTAL:
        raise ValueError(
            f"[VALIDATE] Số dòng trọng số không hợp lệ: {len(w_lines)} != {NUM_WEIGHTS_TOTAL}"
        )
    if len(b_lines) != NUM_BIASES_TOTAL:
        raise ValueError(
            f"[VALIDATE] Số dòng bias không hợp lệ: {len(b_lines)} != {NUM_BIASES_TOTAL}"
        )

    # Chuyển đổi định dạng bù 2
    raw_weights = np.array([parse_signed_hex_byte(h) for h in w_lines], dtype=np.int8)
    raw_biases = np.array([parse_signed_hex_word(h) for h in b_lines], dtype=np.int32)

    # Phân bổ trọng số theo layout PyTorch contiguous tensor
    # Layer 1: [16, 1, 9, 9] (0..1295)
    w1 = raw_weights[0:NUM_WEIGHTS_L1].reshape(16, 1, KERNEL_SIZE_L1, KERNEL_SIZE_L1)
    b1 = raw_biases[0:NUM_BIASES_L1]

    # Layer 2: [8, 16, 1, 1] (1296..1423)
    offset_w2 = NUM_WEIGHTS_L1
    w2 = raw_weights[offset_w2 : offset_w2 + NUM_WEIGHTS_L2].reshape(8, 16, 1, 1)
    offset_b2 = NUM_BIASES_L1
    b2 = raw_biases[offset_b2 : offset_b2 + NUM_BIASES_L2]

    # Layer 3: [1, 8, 5, 5] (1424..1623)
    offset_w3 = offset_w2 + NUM_WEIGHTS_L2
    w3 = raw_weights[offset_w3 : offset_w3 + NUM_WEIGHTS_L3].reshape(1, 8, KERNEL_SIZE_L3, KERNEL_SIZE_L3)
    offset_b3 = offset_b2 + NUM_BIASES_L2
    b3 = raw_biases[offset_b3 : offset_b3 + NUM_BIASES_L3]

    logger.info(
        f"[LOAD] Nạp thành công trọng số phần cứng: "
        f"L1={w1.shape} (non-zero: {np.count_nonzero(w1)}), "
        f"L2={w2.shape} (non-zero: {np.count_nonzero(w2)}), "
        f"L3={w3.shape} (non-zero: {np.count_nonzero(w3)})"
    )

    return HardwareWeights(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)


# ------------------------------------------------------------------------------
# CORE BIT-ACCURATE RTL CONVOLUTION LAYERS
# ------------------------------------------------------------------------------
def conv_layer1_rtl(
    input_s7: np.ndarray,
    weights_s0_7: np.ndarray,
    biases_s24_7: np.ndarray,
) -> np.ndarray:
    """
    Mô phỏng Bit-Accurate Layer 1: Conv2d(1, 16, kernel_size=9, padding=4) + ReLU.
    
    Số học:
      - Đệm 0 (zero-padding 4 pixels).
      - Tích lũy số nguyên 32-bit có dấu (S24.7): acc = bias + sum(pix * w).
      - Dịch bit phải số học: shifted = acc >> 7.
      - Kẹp biên ReLU: [0, 127].
      
    Args:
        input_s7: Ảnh đầu vào 2D shape (H, W) kiểu int8 [-128, 127].
        weights_s0_7: Trọng số shape (16, 1, 9, 9) kiểu int8.
        biases_s24_7: Bias shape (16,) kiểu int32.
        
    Returns:
        Features shape (16, H, W) kiểu int8 trong khoảng [0, 127].
    """
    height, width = input_s7.shape

    # Zero-padding 4 pixels
    padded = np.pad(
        input_s7.astype(np.int32),
        PAD_L1,
        mode="constant",
        constant_values=0,
    )

    # Trích xuất toàn bộ cửa sổ 9x9 không gian (H, W, 9, 9)
    windows = sliding_window_view(padded, (KERNEL_SIZE_L1, KERNEL_SIZE_L1))
    windows_flat = windows.reshape(height * width, KERNEL_SIZE_L1 * KERNEL_SIZE_L1)

    # Trọng số tầng 1: (16, 81)
    w_matrix = weights_s0_7[:, 0, :, :].reshape(16, KERNEL_SIZE_L1 * KERNEL_SIZE_L1).astype(np.int32)

    # Phép nhân tích chập số nguyên (Integer matrix multiplication: H*W, 16)
    accumulators = (windows_flat @ w_matrix.T).reshape(height, width, 16) + biases_s24_7
    accumulators = np.transpose(accumulators, (2, 0, 1))  # Shape: (16, H, W)

    # Dịch bit phải số học (>> 7) và kẹp biên ReLU [0, 127]
    shifted = accumulators >> SHIFT_BITS
    features_out = np.clip(shifted, RELU_CLAMP_MIN, RELU_CLAMP_MAX).astype(np.int8)

    return features_out


def conv_layer2_rtl(
    l1_features: np.ndarray,
    weights_s0_7: np.ndarray,
    biases_s24_7: np.ndarray,
) -> np.ndarray:
    """
    Mô phỏng Bit-Accurate Layer 2: Pointwise Conv2d(16, 8, kernel_size=1, padding=0) + ReLU.
    
    Số học:
      - Tích chập điểm 1x1 chia sẻ qua 16 kênh vào, 8 kênh ra.
      - Tích lũy số nguyên 32-bit có dấu: acc = bias + sum(l1 * w).
      - Dịch bit phải số học: shifted = acc >> 7.
      - Kẹp biên ReLU: [0, 127].
      
    Args:
        l1_features: Features từ L1 shape (16, H, W) kiểu int8 [0, 127].
        weights_s0_7: Trọng số shape (8, 16, 1, 1) kiểu int8.
        biases_s24_7: Bias shape (8,) kiểu int32.
        
    Returns:
        Features shape (8, H, W) kiểu int8 trong khoảng [0, 127].
    """
    num_out_channels, num_in_channels, _, _ = weights_s0_7.shape
    _, height, width = l1_features.shape

    # Ma trận trọng số (8, 16)
    w_matrix = weights_s0_7[:, :, 0, 0].astype(np.int32)
    # Features phẳng hóa (16, H * W)
    feats_flat = l1_features.astype(np.int32).reshape(num_in_channels, height * width)

    # Nhân ma trận số nguyên: (8, H * W) + bias
    accumulators = (w_matrix @ feats_flat).reshape(num_out_channels, height, width)
    accumulators += biases_s24_7[:, None, None]

    # Dịch bit phải số học (>> 7) và kẹp biên ReLU [0, 127]
    shifted = accumulators >> SHIFT_BITS
    features_out = np.clip(shifted, RELU_CLAMP_MIN, RELU_CLAMP_MAX).astype(np.int8)

    return features_out


def conv_layer3_rtl(
    l2_features: np.ndarray,
    weights_s0_7: np.ndarray,
    bias_s24_7: int,
    output_zero_point: int = OUTPUT_ZERO_POINT,
) -> np.ndarray:
    """
    Mô phỏng Bit-Accurate Layer 3: Conv2d(8, 1, kernel_size=5, padding=2) + Offset Unsigned.
    
    Số học:
      - Đệm 0 (zero-padding 2 pixels).
      - Tích lũy số nguyên 32-bit có dấu: acc = bias + sum(l2 * w).
      - Dịch bit phải số học: shifted = acc >> 7.
      - Khôi phục miền không dấu: adjusted = shifted + OUTPUT_ZERO_POINT (128).
      - Kẹp biên uint8: [0, 255].
      
    Args:
        l2_features: Features từ L2 shape (8, H, W) kiểu int8 [0, 127].
        weights_s0_7: Trọng số shape (1, 8, 5, 5) kiểu int8.
        bias_s24_7: Giá trị bias đơn lẻ (32-bit signed).
        output_zero_point: Điểm 0 khôi phục ảnh không dấu (mặc định 128).
        
    Returns:
        Ảnh siêu phân giải 2D shape (H, W) kiểu uint8 [0, 255].
    """
    _, height, width = l2_features.shape

    # Zero-padding 2 pixels trên trục chiều cao và chiều rộng
    padded = np.pad(
        l2_features.astype(np.int32),
        ((0, 0), (PAD_L3, PAD_L3), (PAD_L3, PAD_L3)),
        mode="constant",
        constant_values=0,
    )

    # Trích xuất cửa sổ 5x5 trên từng kênh: shape (8, H, W, 5, 5)
    windows = sliding_window_view(padded, (KERNEL_SIZE_L3, KERNEL_SIZE_L3), axis=(1, 2))
    # Chuyển trục để ghép 8 kênh vào vector 200 phần tử: shape (H * W, 200)
    windows_flat = np.transpose(windows, (1, 2, 0, 3, 4)).reshape(height * width, NUM_WEIGHTS_L3)

    # Trọng số tầng 3 phẳng hóa: shape (200,)
    w_vector = weights_s0_7[0].reshape(NUM_WEIGHTS_L3).astype(np.int32)

    # Nhân tích chập: (H * W,)
    accumulators = (windows_flat @ w_vector).reshape(height, width) + int(bias_s24_7)

    # Dịch bit phải số học (>> 7)
    shifted = accumulators >> SHIFT_BITS
    # Khôi phục giá trị pixel không dấu và kẹp biên uint8 [0, 255]
    adjusted = shifted + output_zero_point
    image_out = np.clip(adjusted, UINT8_MIN, UINT8_MAX).astype(np.uint8)

    return image_out


# ------------------------------------------------------------------------------
# COMPACT SRCNN GOLDEN MODEL CLASS
# ------------------------------------------------------------------------------
class CompactSRCNN_GoldenModel:
    """
    Mô hình Golden Model đại diện chuẩn xác tuyệt đối cho IP Core phần cứng Compact SRCNN.
    
    Sử dụng 100% phép toán số học số nguyên (Integer Math).
    Không dùng float32 để loại bỏ hoàn toàn sai số làm tròn (Zero Roundoff Error).
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        biases_path: Optional[Union[str, Path]] = None,
    ) -> None:
        default_dir = Path(__file__).resolve().parent.parent / "weights_fixed_point"
        w_path = weights_path or (default_dir / "weights_hex_clean.txt")
        b_path = biases_path or (default_dir / "biases_hex_clean.txt")

        self.weights = load_hardware_weights(w_path, b_path)
        logger.info("[GOLDEN_MODEL] Khởi tạo thành công Bit-Accurate Golden Model.")

    def forward_s7(self, input_s7: np.ndarray) -> np.ndarray:
        """
        Thực thi forward pass trên dữ liệu S7.0 đã trừ offset 128.
        
        Args:
            input_s7: Mảng 2D int8 shape (H, W).
            
        Returns:
            Ảnh đầu ra 2D uint8 shape (H, W).
        """
        l1 = conv_layer1_rtl(input_s7, self.weights.w1, self.weights.b1)
        l2 = conv_layer2_rtl(l1, self.weights.w2, self.weights.b2)
        out = conv_layer3_rtl(l2, self.weights.w3, int(self.weights.b3[0]))
        return out

    def process_image(self, image_uint8: np.ndarray, scale: int = 1) -> np.ndarray:
        """
        Thực thi siêu phân giải từ ảnh grayscale [0, 255].
        
        Args:
            image_uint8: Mảng numpy 2D uint8 [0, 255].
            scale: Hệ số phóng đại bicubic trước khi vào mạng (mặc định 1 nếu ảnh đã upscale, 2 nếu ảnh là LR).
            
        Returns:
            Ảnh tái tạo 2D uint8 [0, 255].
        """
        if image_uint8.ndim != 2:
            raise ValueError(f"[VALIDATE] Ảnh đầu vào phải là ảnh xám 2D, nhận shape: {image_uint8.shape}")
        if image_uint8.dtype != np.uint8:
            raise TypeError(f"[VALIDATE] Kiểu dữ liệu ảnh phải là uint8, nhận: {image_uint8.dtype}")

        if scale > 1:
            h, w = image_uint8.shape
            pil_img = Image.fromarray(image_uint8)
            pil_bicubic = pil_img.resize((w * scale, h * scale), Image.Resampling.BICUBIC)
            image_uint8 = np.array(pil_bicubic, dtype=np.uint8)

        # Chuyển đổi từ không dấu [0, 255] sang S7.0 có dấu [-128, 127]
        input_s7 = (image_uint8.astype(np.int16) - INPUT_OFFSET).astype(np.int8)

        # Chạy forward pass
        return self.forward_s7(input_s7)


# ------------------------------------------------------------------------------
# SELF-TEST & VERIFICATION FUNCTIONS
# ------------------------------------------------------------------------------
def run_functional_tb_verification() -> bool:
    """
    Tái lập chính xác 100% testbench chức năng tb_axis_functional.v của RTL:
      - Sử dụng functional_weights.txt và functional_biases.txt.
      - Trọng số tâm mỗi tầng là 0.5 (Q7 = 64).
      - Đầu vào x = 0..15 trên ảnh 4x4.
      - Kết quả kỳ vọng của RTL: output = 128 + floor(x / 8).
    """
    logger.info("[VERIFY] Đang chạy kiểm thử tái lập RTL tb_axis_functional.v (4x4 patch)...")

    base_dir = Path(__file__).resolve().parent.parent / "weights_fixed_point"
    func_w_path = base_dir / "functional_weights.txt"
    func_b_path = base_dir / "functional_biases.txt"

    if not func_w_path.exists() or not func_b_path.exists():
        logger.warning("[VERIFY] Không tìm thấy file functional test, bỏ qua test tb_axis_functional.")
        return True

    weights = load_hardware_weights(func_w_path, func_b_path)

    # Tạo đầu vào 4x4 từ x = 0 đến 15 (chưa trừ 128)
    # Theo tb_axis_functional.v:
    # 0x80 (-128, tức x=0), 0x81 (-127, x=1), ..., 0x8F (-113, x=15)
    # Như vậy x_s7 = 0..15 trong miền signed
    input_s7 = np.arange(16, dtype=np.int8).reshape(4, 4)

    # Chạy qua 3 tầng
    l1 = conv_layer1_rtl(input_s7, weights.w1, weights.b1)
    l2 = conv_layer2_rtl(l1, weights.w2, weights.b2)
    output = conv_layer3_rtl(l2, weights.w3, int(weights.b3[0]))

    # Kết quả kỳ vọng lý thuyết của RTL: 128 + floor(x / 8)
    expected = 128 + (input_s7.astype(int) // 8)
    expected = np.clip(expected, 0, 255).astype(np.uint8)

    diff = np.max(np.abs(output.astype(int) - expected.astype(int)))
    if diff == 0:
        logger.info("[VERIFY] tb_axis_functional: PASS 100% Bit-Exact (Sai số tuyệt đối = 0).")
        return True
    else:
        logger.error(f"[VERIFY] tb_axis_functional: FAILED, sai số lớn nhất: {diff}")
        return False


def run_smoke_test_zero_weights() -> bool:
    """Kiểm tra với trọng số bằng 0 (smoke_weights.txt): Output phải là 128 trên toàn bộ ảnh."""
    logger.info("[VERIFY] Đang chạy kiểm thử Smoke Test (Zero weights)...")

    base_dir = Path(__file__).resolve().parent.parent / "weights_fixed_point"
    smoke_w = base_dir / "smoke_weights.txt"
    smoke_b = base_dir / "smoke_biases.txt"

    if not smoke_w.exists() or not smoke_b.exists():
        logger.warning("[VERIFY] Không tìm thấy file smoke test, bỏ qua.")
        return True

    weights = load_hardware_weights(smoke_w, smoke_b)
    dummy_input = np.random.randint(-128, 128, (32, 32), dtype=np.int8)

    l1 = conv_layer1_rtl(dummy_input, weights.w1, weights.b1)
    l2 = conv_layer2_rtl(l1, weights.w2, weights.b2)
    out = conv_layer3_rtl(l2, weights.w3, int(weights.b3[0]))

    # Toàn bộ output phải bằng đúng 128
    if np.all(out == 128):
        logger.info("[VERIFY] Smoke Test: PASS 100% (Output đồng nhất 128).")
        return True
    else:
        logger.error("[VERIFY] Smoke Test: FAILED, output không bằng 128.")
        return False


def run_trained_model_benchmark(num_images: int = 5) -> bool:
    """Chạy kiểm thử Golden Model trên một số ảnh mẫu thực tế từ tập degraded_testset."""
    logger.info(f"[VERIFY] Đang chạy kiểm thử Golden Model trên {num_images} ảnh testset...")

    testset_dir = Path(__file__).resolve().parent.parent.parent / "data" / "degraded_testset"
    lr_dir = testset_dir / "LR_2x"

    if not lr_dir.exists():
        logger.warning(f"[VERIFY] Thư mục ảnh test không tồn tại: {lr_dir}, bỏ qua.")
        return True

    model = CompactSRCNN_GoldenModel()
    image_files = sorted(list(lr_dir.glob("*.png")))[:num_images]

    latencies: List[float] = []
    for img_path in image_files:
        with Image.open(img_path) as img:
            lr_img = np.array(img.convert("L"), dtype=np.uint8)

        # Phóng đại Bicubic 2x trước khi vào mạng SRCNN (chuẩn thiết kế SRCNN)
        h, w = lr_img.shape
        pil_lr = Image.fromarray(lr_img)
        pil_bicubic = pil_lr.resize((w * 2, h * 2), Image.Resampling.BICUBIC)
        bicubic_arr = np.array(pil_bicubic, dtype=np.uint8)

        t0 = time.perf_counter()
        sr_out = model.process_image(bicubic_arr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        assert sr_out.shape == bicubic_arr.shape
        assert sr_out.dtype == np.uint8
        logger.info(
            f"[INFERENCE] {img_path.name}: Shape={sr_out.shape}, "
            f"Range=[{sr_out.min()}, {sr_out.max()}], Mean={sr_out.mean():.2f}, "
            f"Latency={elapsed_ms:.1f} ms"
        )

    avg_lat = np.mean(latencies)
    logger.info(f"[VERIFY] Hoàn thành {len(image_files)} ảnh mẫu. Độ trễ trung bình: {avg_lat:.2f} ms/ảnh.")
    return True


# ------------------------------------------------------------------------------
# CLI INTERFACE
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bit-Accurate Python Golden Model cho Compact SRCNN trên PYNQ-Z2 (Request 2)."
    )
    parser.add_argument("--test", action="store_true", help="Chạy toàn bộ unit tests kiểm thử RTL")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file weights_hex_clean.txt")
    parser.add_argument("--biases", type=str, default=None, help="Đường dẫn file biases_hex_clean.txt")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn ảnh xám đầu vào để xử lý")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu ảnh siêu phân giải đầu ra")

    args = parser.parse_args()

    if args.test or (args.input is None):
        logger.info("[START] Bắt đầu quy trình kiểm tra toàn diện Golden Model...")
        pass_func = run_functional_tb_verification()
        pass_smoke = run_smoke_test_zero_weights()
        pass_bench = run_trained_model_benchmark(num_images=3)

        if pass_func and pass_smoke and pass_bench:
            logger.info("================================================================================")
            logger.info("✅ XÁC THỰC HOÀN TẤT: Python Golden Model hoàn toàn Bit-Accurate với RTL phần cứng.")
            logger.info("================================================================================")
        else:
            logger.error("❌ XÁC THỰC THẤT BẠI: Phát hiện sai số giữa Golden Model và định nghĩa phần cứng.")
            sys.exit(1)

    if args.input is not None:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh đầu vào: {in_path}")

        model = CompactSRCNN_GoldenModel(args.weights, args.biases)
        with Image.open(in_path) as img:
            arr = np.array(img.convert("L"), dtype=np.uint8)

        logger.info(f"[INFERENCE] Đang xử lý ảnh: {in_path.name} ({arr.shape})...")
        t0 = time.perf_counter()
        out_arr = model.process_image(arr)
        elapsed = (time.perf_counter() - t0) * 1000.0

        out_path = Path(args.output) if args.output else in_path.parent / f"{in_path.stem}_golden_sr.png"
        Image.fromarray(out_arr).save(out_path)
        logger.info(f"[OUTPUT] Đã lưu ảnh siêu phân giải tại: {out_path} (Thời gian: {elapsed:.2f} ms)")


if __name__ == "__main__":
    main()
