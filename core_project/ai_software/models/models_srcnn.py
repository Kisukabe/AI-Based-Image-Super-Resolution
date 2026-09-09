#!/usr/bin/env python3
"""
models_srcnn.py
================================================================================
Mục đích:
    Định nghĩa 2 kiến trúc PyTorch Float32 cho bài toán siêu phân giải ảnh y tế
    theo yêu cầu mục 1.1 trong Request1.txt:
    1. SRCNN_Original (1 -> 64 -> 32 -> 1): Đúng 8.129 tham số (Mô hình đối chứng baseline).
    2. Compact_SRCNN (1 -> 16 -> 8 -> 1): Đúng 1.649 tham số (Mô hình khớp 100% với RTL FPGA).

Dependencies:
    - python >= 3.8
    - torch >= 1.10.0

Cách chạy kiểm thử độc lập:
    python3 "code software/scripts/models_srcnn.py"
================================================================================
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN_Original(nn.Module):
    """
    Kiến trúc SRCNN gốc chuẩn Dong et al. cho ảnh 1 kênh xám (Grayscale).
    Tổng số tham số: đúng 8.129 tham số (Float32).

    Cấu trúc:
        Layer 1 (Patch extraction): Conv2d(1, 64, kernel_size=9, padding=4) + ReLU
            - Weights: 64 * 1 * 9 * 9 = 5.184
            - Biases: 64
            - Subtotal: 5.248
        Layer 2 (Non-linear mapping): Conv2d(64, 32, kernel_size=1, padding=0) + ReLU
            - Weights: 32 * 64 * 1 * 1 = 2.048
            - Biases: 32
            - Subtotal: 2.080
        Layer 3 (Reconstruction): Conv2d(32, 1, kernel_size=5, padding=2)
            - Weights: 1 * 32 * 5 * 5 = 800
            - Biases: 1
            - Subtotal: 801
        Total: 5.248 + 2.080 + 801 = 8.129 tham số.
    """

    EXPECTED_PARAMS: int = 8129

    def __init__(self, pre_upsample: bool = False, upscale_factor: int = 2) -> None:
        """
        Khởi tạo mô hình SRCNN gốc.

        Args:
            pre_upsample: Nếu True, tự động nội suy Bicubic trước khi đưa vào Conv.
                          Nếu False, đầu vào phải là ảnh đã được phóng to sẵn (pre-interpolated).
            upscale_factor: Hệ số phóng đại nếu pre_upsample=True (mặc định 2).
        """
        super().__init__()
        self.pre_upsample = pre_upsample
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2, bias=True)

        self._verify_param_count()

    def _verify_param_count(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        if total_params != self.EXPECTED_PARAMS:
            raise ValueError(
                f"[SRCNN_Original] Sai lệch số lượng tham số: Thực tế {total_params:,}, "
                f"Kỳ vọng {self.EXPECTED_PARAMS:,}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mạng SRCNN gốc.

        Args:
            x: Tensor đầu vào có kích thước [Batch, 1, H, W].

        Returns:
            Tensor siêu phân giải có kích thước [Batch, 1, H_out, W_out].
        """
        if self.pre_upsample and self.upscale_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=float(self.upscale_factor),
                mode="bicubic",
                align_corners=False,
            )

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class Compact_SRCNN(nn.Module):
    """
    Kiến trúc Compact SRCNN rút gọn kênh, tối ưu hóa để triển khai trên phần cứng FPGA.
    Cấu trúc khớp 100% với lõi phần cứng RTL srcnn_top_core.v trên Xilinx Zynq-7020.
    Tổng số tham số: đúng 1.649 tham số (Float32).

    Cấu trúc:
        Layer 1: Conv2d(1, 16, kernel_size=9, padding=4) + ReLU
            - Weights: 16 * 1 * 9 * 9 = 1.296
            - Biases: 16
            - Subtotal: 1.312
        Layer 2: Conv2d(16, 8, kernel_size=1, padding=0) + ReLU
            - Weights: 8 * 16 * 1 * 1 = 128
            - Biases: 8
            - Subtotal: 136
        Layer 3: Conv2d(8, 1, kernel_size=5, padding=2)
            - Weights: 1 * 8 * 5 * 5 = 200
            - Biases: 1
            - Subtotal: 201
        Total: 1.312 + 136 + 201 = 1.649 tham số.
    """

    EXPECTED_PARAMS: int = 1649

    def __init__(self, pre_upsample: bool = False, upscale_factor: int = 2) -> None:
        """
        Khởi tạo mô hình Compact SRCNN.

        Args:
            pre_upsample: Nếu True, tự động nội suy Bicubic trước khi đưa vào Conv.
                          Nếu False, đầu vào phải là ảnh đã được phóng to sẵn.
            upscale_factor: Hệ số phóng đại nếu pre_upsample=True (mặc định 2).
        """
        super().__init__()
        self.pre_upsample = pre_upsample
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, padding=4, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, padding=0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, padding=2, bias=True)

        self._verify_param_count()

    def _verify_param_count(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        if total_params != self.EXPECTED_PARAMS:
            raise ValueError(
                f"[Compact_SRCNN] Sai lệch số lượng tham số: Thực tế {total_params:,}, "
                f"Kỳ vọng {self.EXPECTED_PARAMS:,}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mạng Compact SRCNN.

        Args:
            x: Tensor đầu vào có kích thước [Batch, 1, H, W].

        Returns:
            Tensor siêu phân giải có kích thước [Batch, 1, H_out, W_out].
        """
        if self.pre_upsample and self.upscale_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=float(self.upscale_factor),
                mode="bicubic",
                align_corners=False,
            )

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


def get_model_summary(model: nn.Module) -> dict[str, int]:
    """
    Trích xuất bảng phân tích chi tiết trọng số và bias của từng tầng.

    Args:
        model: Đối tượng PyTorch nn.Module.

    Returns:
        Dictionary chứa tên tham số và kích thước số lượng.
    """
    summary = {}
    total = 0
    for name, param in model.named_parameters():
        num = param.numel()
        summary[name] = num
        total += num
    summary["TOTAL_PARAMETERS"] = total
    return summary


if __name__ == "__main__":
    print("=" * 80)
    print("KIỂM THỬ ĐỘC LẬP KIẾN TRÚC MẠNG PYTORCH CHO NHIỆM VỤ 1.1")
    print("=" * 80)

    # 1. Kiểm tra SRCNN gốc (8.129 tham số)
    srcnn_orig = SRCNN_Original(pre_upsample=False)
    summary_orig = get_model_summary(srcnn_orig)
    print(f"\n[1] SRCNN_Original (1 -> 64 -> 32 -> 1):")
    for k, v in summary_orig.items():
        print(f"  - {k:<25}: {v:>6,}")
    assert summary_orig["TOTAL_PARAMETERS"] == 8129, "SRCNN gốc phải có đúng 8.129 tham số"

    # Test forward với tensor kích thước chuẩn X-quang
    dummy_input = torch.randn(1, 1, 1024, 1024)
    with torch.no_grad():
        out_orig = srcnn_orig(dummy_input)
    print(f"  > Dummy Input shape : {list(dummy_input.shape)}")
    print(f"  > Output shape       : {list(out_orig.shape)} (Bảo toàn kích thước không gian)")
    assert out_orig.shape == dummy_input.shape, "Shape đầu ra phải bằng shape đầu vào"

    # 2. Kiểm tra Compact SRCNN (1.649 tham số)
    compact_srcnn = Compact_SRCNN(pre_upsample=False)
    summary_compact = get_model_summary(compact_srcnn)
    print(f"\n[2] Compact_SRCNN (1 -> 16 -> 8 -> 1 - Khớp RTL):")
    for k, v in summary_compact.items():
        print(f"  - {k:<25}: {v:>6,}")
    assert summary_compact["TOTAL_PARAMETERS"] == 1649, "Compact SRCNN phải có đúng 1.649 tham số"

    with torch.no_grad():
        out_compact = compact_srcnn(dummy_input)
    print(f"  > Dummy Input shape : {list(dummy_input.shape)}")
    print(f"  > Output shape       : {list(out_compact.shape)} (Bảo toàn kích thước không gian)")
    assert out_compact.shape == dummy_input.shape, "Shape đầu ra phải bằng shape đầu vào"

    # 3. Kiểm tra tính năng pre_upsample
    compact_with_upsample = Compact_SRCNN(pre_upsample=True, upscale_factor=2)
    lr_input = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        out_from_lr = compact_with_upsample(lr_input)
    print(f"\n[3] Kiểm tra Pre-upsample 2x (512x512 -> 1024x1024):")
    print(f"  > LR Input shape    : {list(lr_input.shape)}")
    print(f"  > HR Output shape   : {list(out_from_lr.shape)}")
    assert out_from_lr.shape == (1, 1, 1024, 1024), "Output phải có kích thước 1024x1024 khi upscale 2x"

    print("\n" + "=" * 80)
    print("XÁC NHẬN: TẤT CẢ UNIT TESTS VỀ THAM SỐ VÀ TENSOR SHAPE ĐÃ PASS 100%")
    print("=" * 80)
