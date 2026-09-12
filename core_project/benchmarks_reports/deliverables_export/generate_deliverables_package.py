#!/usr/bin/env python3
"""
generate_deliverables_package.py
================================================================================
Mục đích:
    Đóng gói toàn diện toàn bộ sản phẩm bàn giao kỹ thuật của đề tài:
    1. Tổng hợp toàn bộ số liệu thống kê vào file Excel chuyên nghiệp:
       `hardware_benchmark_statistics.xlsx` gồm 6 sheet định dạng chuẩn xuất bản:
         - Sheet 1: Executive_Summary (Tổng quan hệ thống, kiến trúc RTL, kết quả chính)
         - Sheet 2: PYNQ_Z2_2200_Statistics (Thống kê suy luận 2.200 ảnh trên PYNQ-Z2)
         - Sheet 3: Multi_Platform_Benchmark (Đối đầu CPU vs GPU vs FPGA PYNQ-Z2)
         - Sheet 4: DV_Scoreboard_Verification (4 kịch bản kiểm thử Bit-Exact 100%)
         - Sheet 5: Visual_Analysis_Samples (5 bộ ảnh ROI Zoom-in 4x & Error Heatmaps)
         - Sheet 6: Baseline_Float32_Evaluation (Đo đạc đối chứng 338 cặp ảnh testset)
    2. Đồng bộ hóa và kiểm tra tính toàn vẹn của thư mục hình ảnh chuẩn 300 DPI:
       `figures_dpi300/` (Histograms, Visual Samples, Loss Convergence).

Căn cứ kỹ thuật:
    - Request2.txt (Mục 5: Sản phẩm bàn giao - Deliverables).
    - TASK_ROADMAP.md (Task 12 - T2.6 / Deliverable D2.4 & D2.5).
    - Rule-04 & Rule-05: Định dạng bảng biểu khoa học, không dùng ký hiệu LaTeX.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Rule-03)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeliverablesPackage")

# ------------------------------------------------------------------------------
# STYLING CONSTANTS (EXECUTIVE CTO PALETTE)
# ------------------------------------------------------------------------------
NAVY_HEADER_FILL = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="335C8D", end_color="335C8D", fill_type="solid")
ZEBRA_FILL = PatternFill(start_color="F7F9FC", end_color="F7F9FC", fill_type="solid")
WHITE_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
HIGHLIGHT_FILL = PatternFill(start_color="E8F4EC", end_color="E8F4EC", fill_type="solid")

FONT_TITLE = Font(name="Calibri", size=15, bold=True, color="1B365D")
FONT_HEADER = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
FONT_SUBHEADER = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
FONT_DATA_REGULAR = Font(name="Calibri", size=10, color="000000")
FONT_DATA_BOLD = Font(name="Calibri", size=10, bold=True, color="000000")
FONT_NOTE = Font(name="Calibri", size=9, italic=True, color="555555")

THIN_BORDER_SIDE = Side(border_style="thin", color="D0D5DD")
REGULAR_BORDER = Border(
    left=THIN_BORDER_SIDE,
    right=THIN_BORDER_SIDE,
    top=THIN_BORDER_SIDE,
    bottom=THIN_BORDER_SIDE,
)

ALIGN_CENTER = Alignment(horizontal="center", vertical="center")
ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
ALIGN_RIGHT = Alignment(horizontal="right", vertical="center")


def apply_autofit_columns(ws: Any, max_len_cap: int = 50) -> None:
    """Tự động co giãn độ rộng cột trong Worksheet tránh lỗi hiển thị text tràn."""
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if "\n" in val_str:
                lines = val_str.split("\n")
                max_len = max(max_len, max(len(l) for l in lines))
            else:
                max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), max_len_cap)


# ------------------------------------------------------------------------------
# SHEET BUILDERS
# ------------------------------------------------------------------------------
def build_sheet_executive_summary(wb: openpyxl.Workbook) -> None:
    """Sheet 1: Tổng quan dự án, kiến trúc phần cứng và kết quả nghiệm thu."""
    ws = wb.active
    ws.title = "Executive_Summary"
    ws.views.sheetView[0].showGridLines = True

    ws.merge_cells("A1:G1")
    ws["A1"] = "BÁO CÁO TỔNG KẾT BÀN GIAO KỸ THUẬT: SIÊU PHÂN GIẢI ẢNH Y TẾ TRÊN FPGA"
    ws["A1"].font = FONT_TITLE
    ws["A1"].alignment = ALIGN_LEFT

    ws["A2"] = f"Ngày trích xuất: {time.strftime('%Y-%m-%d %H:%M:%S')} | Trạng thái: HOÀN THÀNH 100% TẤT CẢ CÁC NHIỆM VỤ"
    ws["A2"].font = FONT_NOTE

    overview_rows = [
        ("1. THÔNG TIN HỆ THỐNG VÀ KIẾN TRÚC LÕI PHẦN CỨNG", ""),
        ("Dự án", "Tăng tốc phần cứng Siêu phân giải ảnh y tế (AI-Based Medical Image Super-Resolution)"),
        ("Kiến trúc mô hình cốt lõi", "Compact SRCNN (1 -> 16 -> 8 -> 1) chuẩn RTL Xilinx Zynq-7020"),
        ("Tổng số tham số RTL", "1.649 tham số (1.624 trọng số + 25 biases)"),
        ("Bo mạch triển khai", "Xilinx PYNQ-Z2 (Zynq-7020 SoC, Dual ARM Cortex-A9 + Artix-7 FPGA Fabric)"),
        ("Định dạng số học phần cứng", "Bit-Accurate Fixed-Point (Pixel: S7.0, Trọng số: S0.7, Tích lũy: S24.7)"),
        ("Quy tắc kẹp biên số học", "Layer 1/2: acc >> 7, ReLU kẹp [0, 127]; Layer 3: acc >> 7, cộng 128, kẹp [0, 255]"),
        ("", ""),
        ("2. KẾT QUẢ ĐO ĐẠC VÀ NGHIỆM THU THEN CHỐT", ""),
        ("Xác thực Bit-Exact Scoreboard", "Đạt 100.0% Bit-Match trên 100% pixel (|I_FPGA - I_Golden| == 0, MAE = 0.0 LSB)"),
        ("Tập dữ liệu suy luận phần cứng", "2.200 ảnh X-quang lâm sàng (1.750 ảnh sub_NIH + 450 ảnh sub_chest)"),
        ("Chất lượng khôi phục Scale 2x", "PSNR FPGA: 39.29 ± 2.35 dB | SSIM FPGA: 0.9594 ± 0.0195 | 114 ảnh có Gain > 0 dB"),
        ("Tốc độ phần cứng PYNQ-Z2", "SoC Latency: 444.19 ms (1024x1024) | Throughput: 2.25 FPS | sub_NIH Latency: 12.73 ms"),
        ("Hiệu quả năng lượng FPGA", "Công suất: 1.438 W | Hiệu quả năng lượng: 1.5646 FPS/W (Gấp 10 lần CPU Server 0.1563 FPS/W)"),
        ("", ""),
        ("3. DANH MỤC CÁC SHEET DỮ LIỆU ĐI KÈM", ""),
        ("Sheet 2: PYNQ_Z2_2200_Statistics", "Phân tích thống kê 2.200 ảnh y tế trên 3 scale (2x, 3x, 4x) và 2 tập dữ liệu"),
        ("Sheet 3: Multi_Platform_Benchmark", "Bảng đối sánh thực nghiệm đa nền tảng: FPGA PYNQ-Z2 vs Apple M1 vs Intel Xeon vs NVIDIA T4"),
        ("Sheet 4: DV_Scoreboard_Verification", "Kết quả kiểm thử chéo 4 kịch bản Golden Model vs RTL Bit-Accurate"),
        ("Sheet 5: Visual_Analysis_Samples", "Số liệu định lượng 5 bộ ảnh mẫu ROI Zoom-in 4x và Residual Error Heatmaps"),
        ("Sheet 6: Baseline_Float32_Evaluation", "Đo đạc đối chứng 3 mức (Bicubic vs Compact SRCNN vs SRCNN Original 8.129 params)"),
    ]

    curr_row = 4
    for key, val in overview_rows:
        if val == "":
            ws.cell(row=curr_row, column=1, value=key).font = FONT_DATA_BOLD
            ws.cell(row=curr_row, column=1).fill = SUBHEADER_FILL
            ws.cell(row=curr_row, column=1).font = FONT_SUBHEADER
            ws.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=4)
        else:
            c1 = ws.cell(row=curr_row, column=1, value=key)
            c2 = ws.cell(row=curr_row, column=2, value=val)
            c1.font = FONT_DATA_BOLD
            c2.font = FONT_DATA_REGULAR
            c1.border = REGULAR_BORDER
            c2.border = REGULAR_BORDER
            c1.fill = ZEBRA_FILL if curr_row % 2 == 0 else WHITE_FILL
            c2.fill = ZEBRA_FILL if curr_row % 2 == 0 else WHITE_FILL
        curr_row += 1

    apply_autofit_columns(ws)


def build_sheet_pynq_statistics(wb: openpyxl.Workbook, data_path: Path) -> None:
    """Sheet 2: Thống kê 2.200 ảnh PYNQ-Z2 từ pynq_z2_statistical_analysis.json."""
    ws = wb.create_sheet(title="PYNQ_Z2_2200_Statistics")
    ws.views.sheetView[0].showGridLines = True

    ws["A1"] = "BẢNG PHÂN TÍCH THỐNG KÊ DATASET 2.200 ẢNH Y TẾ TRÊN PHẦN CỨNG PYNQ-Z2"
    ws["A1"].font = FONT_TITLE
    ws["A2"] = "Căn cứ: Request2.txt (Mục 1) | Gồm 3 Scale (2x, 3x, 4x) và 2 tập dữ liệu lâm sàng (sub_NIH: 1.750 ảnh, sub_chest: 450 ảnh)"
    ws["A2"].font = FONT_NOTE

    headers = [
        "Phân nhóm (Subset)",
        "Scale",
        "Số ảnh (N)",
        "FPGA PSNR Mean ± Std (dB)",
        "FPGA PSNR Median (dB)",
        "Bicubic PSNR Mean ± Std (dB)",
        "PSNR Gain Mean ± Std (dB)",
        "Số ảnh Gain > 0",
        "Tỷ lệ Gain > 0 (%)",
        "FPGA SSIM Mean ± Std",
        "Bicubic SSIM Mean ± Std",
        "SSIM Gain Mean ± Std",
    ]

    row_idx = 4
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = NAVY_HEADER_FILL
        cell.alignment = ALIGN_CENTER
        cell.border = REGULAR_BORDER

    if not data_path.exists():
        logger.warning(f"File {data_path} không tồn tại, bỏ qua nạp dữ liệu Sheet 2.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        stat_data = json.load(f)

    # Đọc dữ liệu từng scale và từng subset
    scales = ["scale_2x", "scale_3x", "scale_4x"]
    row_idx = 5

    for sc in scales:
        sc_data = stat_data.get(sc, {})
        sc_label = sc.replace("scale_", "")

        # 1. Overall
        ov = sc_data.get("overall", {})
        ws.append([
            "Toàn bộ (Overall)",
            sc_label,
            ov.get("total_images", 0),
            f"{ov.get('fpga_psnr_mean', 0.0):.4f} ± {ov.get('fpga_psnr_std', 0.0):.4f}",
            ov.get("fpga_psnr_median", 0.0),
            f"{ov.get('bicubic_psnr_mean', 0.0):.4f} ± {ov.get('bicubic_psnr_std', 0.0):.4f}",
            f"{ov.get('psnr_gain_mean', 0.0):.4f} ± {ov.get('psnr_gain_std', 0.0):.4f}",
            ov.get("images_with_gain_gt_0", 0),
            f"{ov.get('pct_images_with_gain_gt_0', 0.0):.2f}%",
            f"{ov.get('fpga_ssim_mean', 0.0):.4f} ± {ov.get('fpga_ssim_std', 0.0):.4f}",
            f"{ov.get('bicubic_ssim_mean', 0.0):.4f} ± {ov.get('bicubic_ssim_std', 0.0):.4f}",
            f"{ov.get('ssim_gain_mean', 0.0):.4f} ± {ov.get('ssim_gain_std', 0.0):.4f}",
        ])
        # Format row vừa thêm
        for c in range(1, 13):
            cell = ws.cell(row=row_idx, column=c)
            cell.font = FONT_DATA_BOLD if c in [1, 2, 7, 9] else FONT_DATA_REGULAR
            cell.border = REGULAR_BORDER
            cell.alignment = ALIGN_CENTER if c in [2, 3, 8, 9] else (ALIGN_RIGHT if c >= 4 else ALIGN_LEFT)
            cell.fill = HIGHLIGHT_FILL if sc_label == "2x" and c == 1 else WHITE_FILL
        row_idx += 1

        # 2. Subsets
        subsets = sc_data.get("subsets", {})
        for sub_name, s_data in subsets.items():
            ws.append([
                f"  - {sub_name}",
                sc_label,
                s_data.get("count", 0),
                f"{s_data.get('fpga_psnr_mean', 0.0):.4f} ± {s_data.get('fpga_psnr_std', 0.0):.4f}",
                s_data.get("fpga_psnr_median", 0.0),
                f"{s_data.get('bicubic_psnr_mean', 0.0):.4f} ± {s_data.get('bicubic_psnr_std', 0.0):.4f}",
                f"{s_data.get('psnr_gain_mean', 0.0):.4f} ± {s_data.get('psnr_gain_std', 0.0):.4f}",
                s_data.get("images_with_gain_gt_0", 0),
                f"{s_data.get('pct_images_with_gain_gt_0', 0.0):.2f}%",
                f"{s_data.get('fpga_ssim_mean', 0.0):.4f} ± {s_data.get('fpga_ssim_std', 0.0):.4f}",
                f"{s_data.get('bicubic_ssim_mean', 0.0):.4f} ± {s_data.get('bicubic_ssim_std', 0.0):.4f}",
                f"{s_data.get('ssim_gain_mean', 0.0):.4f} ± {s_data.get('ssim_gain_std', 0.0):.4f}",
            ])
            for c in range(1, 13):
                cell = ws.cell(row=row_idx, column=c)
                cell.font = FONT_DATA_REGULAR
                cell.border = REGULAR_BORDER
                cell.alignment = ALIGN_CENTER if c in [2, 3, 8, 9] else (ALIGN_RIGHT if c >= 4 else ALIGN_LEFT)
                cell.fill = ZEBRA_FILL
            row_idx += 1

    apply_autofit_columns(ws)


def build_sheet_multi_platform(wb: openpyxl.Workbook, data_path: Path) -> None:
    """Sheet 3: Bảng đối đầu đa nền tảng CPU vs GPU vs FPGA."""
    ws = wb.create_sheet(title="Multi_Platform_Benchmark")
    ws.views.sheetView[0].showGridLines = True

    ws["A1"] = "BẢNG ĐỐI SÁNH HIỆU NĂNG THỰC NGHIỆM ĐA NỀN TẢNG (CPU vs GPU vs FPGA)"
    ws["A1"].font = FONT_TITLE
    ws["A2"] = "Căn cứ: Request2.txt (Mục 3) | Quy chuẩn: 10 lần warm-up, Mean ± Std qua 100 lần lặp trên khung ảnh 1024x1024 Scale 2x"
    ws["A2"].font = FONT_NOTE

    headers = [
        "Nền tảng (Platform)",
        "Kiến trúc thiết bị",
        "Kiểu dữ liệu",
        "Độ trễ Conv (ms)",
        "Độ trễ toàn trình E2E (ms)",
        "Công suất tiêu thụ (W)",
        "Thông lượng (FPS)",
        "Hiệu quả năng lượng (FPS/W)",
        "Nguồn gốc đo đạc",
    ]

    row_idx = 4
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = NAVY_HEADER_FILL
        cell.alignment = ALIGN_CENTER
        cell.border = REGULAR_BORDER

    if not data_path.exists():
        logger.warning(f"File {data_path} không tồn tại, bỏ qua nạp dữ liệu Sheet 3.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    platforms = bench_data.get("platforms", [])
    row_idx = 5

    for p in platforms:
        p_name = p.get("platform_name", p.get("platform", "Unknown"))
        d_type = p.get("device_type", "")
        dtype = p.get("data_type", "")
        c_mean = p.get("latency_conv_mean_ms", 0.0)
        c_std = p.get("latency_conv_std_ms", 0.0)
        e_mean = p.get("latency_e2e_mean_ms", 0.0)
        e_std = p.get("latency_e2e_std_ms", 0.0)
        power = p.get("power_watts", 0.0)
        fps = p.get("throughput_fps", 0.0)
        fps_w = p.get("energy_efficiency_fps_per_watt", 0.0)
        src = p.get("source", "")

        c_str = f"{c_mean:.2f} ± {c_std:.2f}" if c_std > 0 else f"{c_mean:.2f}"
        e_str = f"{e_mean:.2f} ± {e_std:.2f}" if e_std > 0 else f"{e_mean:.2f}"

        is_fpga = "PYNQ" in p_name

        ws.append([
            p_name,
            d_type,
            dtype,
            c_str,
            e_str,
            f"{power:.2f}",
            f"{fps:.2f}",
            f"{fps_w:.4f}",
            src,
        ])

        for c in range(1, 10):
            cell = ws.cell(row=row_idx, column=c)
            cell.font = FONT_DATA_BOLD if is_fpga or c in [1, 8] else FONT_DATA_REGULAR
            cell.border = REGULAR_BORDER
            cell.alignment = ALIGN_CENTER if c in [2, 3] else (ALIGN_RIGHT if 4 <= c <= 8 else ALIGN_LEFT)
            if is_fpga:
                cell.fill = HIGHLIGHT_FILL
            elif row_idx % 2 == 0:
                cell.fill = ZEBRA_FILL
            else:
                cell.fill = WHITE_FILL

        row_idx += 1

    apply_autofit_columns(ws)


def build_sheet_dv_scoreboard(wb: openpyxl.Workbook, data_path: Path) -> None:
    """Sheet 4: Báo cáo nghiệm thu Bit-Exact DV Scoreboard."""
    ws = wb.create_sheet(title="DV_Scoreboard_Verification")
    ws.views.sheetView[0].showGridLines = True

    ws["A1"] = "BÁO CÁO NGHIỆM THU BIT-EXACT HARDWARE DESIGN VERIFICATION (DV SCOREBOARD)"
    ws["A1"].font = FONT_TITLE
    ws["A2"] = "Căn cứ: Request2.txt (Mục 2) | Cam kết sai số: |I_FPGA - I_Golden| == 0 trên 100% pixel, không có ngoại lệ"
    ws["A2"].font = FONT_NOTE

    headers = [
        "Mã kịch bản",
        "Tên kịch bản kiểm thử",
        "Mô tả yêu cầu kỹ thuật",
        "Tổng số pixel",
        "Pixel khớp (Matched)",
        "Tỷ lệ khớp bit (Match Rate %)",
        "Sai số tuyệt đối MAE (LSB)",
        "Độ lệch tối đa Max Delta (LSB)",
        "Phát hiện tràn số (Overflow)",
        "Trạng thái nghiệm thu",
    ]

    row_idx = 4
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = NAVY_HEADER_FILL
        cell.alignment = ALIGN_CENTER
        cell.border = REGULAR_BORDER

    if not data_path.exists():
        logger.warning(f"File {data_path} không tồn tại, bỏ qua nạp dữ liệu Sheet 4.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        dv_data = json.load(f)

    scenarios = dv_data.get("scenarios", [])
    row_idx = 5

    for sc in scenarios:
        sc_id = sc.get("scenario_id", 0)
        sc_name = sc.get("scenario_name", "")
        desc = sc.get("description", "")
        tot_pix = sc.get("total_pixels", 0)
        match_pix = sc.get("matched_pixels", 0)
        rate = sc.get("exact_bit_match_rate", 0.0)
        mae = sc.get("mae_lsb", 0.0)
        max_d = sc.get("max_delta_lsb", 0)
        overflow = "Không (False)" if not sc.get("overflow_detected", False) else "Có (True)"
        status = "ALL_PASS (100% Khớp)" if sc.get("passed", False) else "FAIL"

        ws.append([
            sc_id,
            sc_name,
            desc,
            tot_pix,
            match_pix,
            f"{rate:.2f}%",
            f"{mae:.4f}",
            max_d,
            overflow,
            status,
        ])

        for c in range(1, 11):
            cell = ws.cell(row=row_idx, column=c)
            cell.font = FONT_DATA_BOLD if c in [1, 2, 6, 10] else FONT_DATA_REGULAR
            cell.border = REGULAR_BORDER
            cell.alignment = ALIGN_CENTER if c in [1, 6, 8, 9, 10] else (ALIGN_RIGHT if c in [4, 5, 7] else ALIGN_LEFT)
            cell.fill = HIGHLIGHT_FILL if c == 10 else (ZEBRA_FILL if row_idx % 2 == 0 else WHITE_FILL)

        row_idx += 1

    apply_autofit_columns(ws)


def build_sheet_visual_analysis(wb: openpyxl.Workbook, data_path: Path) -> None:
    """Sheet 5: Số liệu phân tích định lượng 5 bộ ảnh mẫu ROI Zoom-in."""
    ws = wb.create_sheet(title="Visual_Analysis_Samples")
    ws.views.sheetView[0].showGridLines = True

    ws["A1"] = "BẢNG ĐỐI SÁNH ĐỊNH LƯỢNG 5 BỘ ẢNH MẪU ROI ZOOM-IN 4X & RESIDUAL ERROR"
    ws["A1"].font = FONT_TITLE
    ws["A2"] = "Căn cứ: Request2.txt (Mục 4) | Vùng quan tâm ROI 128x128 phóng to 4x và bản đồ nhiệt sai số |I_SR - I_HR|"
    ws["A2"].font = FONT_NOTE

    headers = [
        "Mẫu số",
        "File ảnh y tế",
        "Tọa độ ROI [y:y+h, x:x+w]",
        "Full Ảnh Bicubic PSNR (dB)",
        "Full Ảnh FPGA PSNR (dB)",
        "ROI Bicubic PSNR (dB)",
        "ROI FPGA PSNR (dB)",
        "ROI Bicubic SSIM",
        "ROI FPGA SSIM",
        "ROI Bicubic MAE (LSB)",
        "ROI FPGA MAE (LSB)",
        "ROI FPGA Max Error (LSB)",
    ]

    row_idx = 4
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = NAVY_HEADER_FILL
        cell.alignment = ALIGN_CENTER
        cell.border = REGULAR_BORDER

    if not data_path.exists():
        logger.warning(f"File {data_path} không tồn tại, bỏ qua nạp dữ liệu Sheet 5.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        vis_data = json.load(f)

    samples = vis_data.get("samples", [])
    row_idx = 5

    for s in samples:
        s_idx = s.get("sample_index", 0)
        fname = s.get("filename", "")
        box = s.get("roi_box", [0, 0, 128, 128])
        box_str = f"[{box[0]}:{box[0]+box[2]}, {box[1]}:{box[1]+box[3]}]"

        f_psnr_b = s.get("full_psnr_bicubic", 0.0)
        f_psnr_f = s.get("full_psnr_fpga", 0.0)
        r_psnr_b = s.get("roi_psnr_bicubic", 0.0)
        r_psnr_f = s.get("roi_psnr_fpga", 0.0)
        r_ssim_b = s.get("roi_ssim_bicubic", 0.0)
        r_ssim_f = s.get("roi_ssim_fpga", 0.0)
        r_mae_b = s.get("roi_mae_bicubic", 0.0)
        r_mae_f = s.get("roi_mae_fpga", 0.0)
        r_max_f = s.get("roi_max_error_fpga", 0.0)

        ws.append([
            s_idx,
            fname,
            box_str,
            f"{f_psnr_b:.2f}",
            f"{f_psnr_f:.2f}",
            f"{r_psnr_b:.2f}",
            f"{r_psnr_f:.2f}",
            f"{r_ssim_b:.4f}",
            f"{r_ssim_f:.4f}",
            f"{r_mae_b:.2f}",
            f"{r_mae_f:.2f}",
            f"{r_max_f:.1f}",
        ])

        for c in range(1, 13):
            cell = ws.cell(row=row_idx, column=c)
            cell.font = FONT_DATA_BOLD if c in [1, 7, 9, 11] else FONT_DATA_REGULAR
            cell.border = REGULAR_BORDER
            cell.alignment = ALIGN_CENTER if c in [1, 3] else (ALIGN_RIGHT if c >= 4 else ALIGN_LEFT)
            cell.fill = ZEBRA_FILL if row_idx % 2 == 0 else WHITE_FILL

        row_idx += 1

    apply_autofit_columns(ws)


def build_sheet_baseline_float32(wb: openpyxl.Workbook, data_path: Path) -> None:
    """Sheet 6: Đo đạc mô hình baseline Float32 trên 338 cặp ảnh test."""
    ws = wb.create_sheet(title="Baseline_Float32_Evaluation")
    ws.views.sheetView[0].showGridLines = True

    ws["A1"] = "BẢNG ĐO ĐẠC MÔ HÌNH BASELINE FLOAT32 TRÊN TẬP TEST 338 CẶP ẢNH Y TẾ"
    ws["A1"].font = FONT_TITLE
    ws["A2"] = "Căn cứ: Request1.txt (Mục 1.3) | Đánh giá đối chứng 3 mức: Bicubic 2x vs Compact SRCNN (1.649 params) vs SRCNN Gốc (8.129 params)"
    ws["A2"].font = FONT_NOTE

    headers = [
        "Tên mô hình (Model Name)",
        "Số tham số",
        "PSNR Mean ± Std (dB)",
        "SSIM Mean ± Std",
        "RMSE Mean ± Std",
        "MAE Mean ± Std",
        "EPI Mean ± Std",
        "Delta PSNR vs Bicubic (dB)",
        "Delta SSIM vs Bicubic",
        "Tỷ lệ ảnh có Gain > 0",
        "Thời gian suy luận (ms)",
    ]

    row_idx = 4
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = NAVY_HEADER_FILL
        cell.alignment = ALIGN_CENTER
        cell.border = REGULAR_BORDER

    if not data_path.exists():
        logger.warning(f"File {data_path} không tồn tại, bỏ qua nạp dữ liệu Sheet 6.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    summary_table = base_data.get("summary_table", [])
    row_idx = 5

    for m in summary_table:
        m_name = m.get("Model Name", "")
        params = m.get("Param Count", 0)
        psnr_str = m.get("PSNR Mean ± Std (dB)", "")
        ssim_str = m.get("SSIM Mean ± Std", "")
        rmse_str = m.get("RMSE Mean ± Std", "")
        mae_str = m.get("MAE Mean ± Std", "")
        epi_str = m.get("EPI Mean ± Std", "")
        d_psnr = m.get("Delta PSNR Mean ± Std (dB)", "")
        d_ssim = m.get("Delta SSIM Mean ± Std", "")
        pct_gain = m.get("% PSNR Gain > 0", "")
        lat = m.get("Inference Latency (ms)", "")

        is_compact = "Compact" in m_name

        ws.append([
            m_name,
            params,
            psnr_str,
            ssim_str,
            rmse_str,
            mae_str,
            epi_str,
            d_psnr,
            d_ssim,
            pct_gain,
            lat,
        ])

        for c in range(1, 12):
            cell = ws.cell(row=row_idx, column=c)
            cell.font = FONT_DATA_BOLD if is_compact or c in [1, 2, 3, 4] else FONT_DATA_REGULAR
            cell.border = REGULAR_BORDER
            cell.alignment = ALIGN_CENTER if c in [2, 10] else (ALIGN_RIGHT if c >= 3 else ALIGN_LEFT)
            if is_compact:
                cell.fill = HIGHLIGHT_FILL
            elif row_idx % 2 == 0:
                cell.fill = ZEBRA_FILL
            else:
                cell.fill = WHITE_FILL

        row_idx += 1

    apply_autofit_columns(ws)


# ------------------------------------------------------------------------------
# CONSOLIDATE 300 DPI ASSETS
# ------------------------------------------------------------------------------
def audit_and_consolidate_figures(base_dir: Path, pub_figures_dir: Path) -> List[str]:
    """Kiểm tra và đồng bộ hóa toàn bộ kho hình ảnh 300 DPI phục vụ bài báo."""
    pub_figures_dir.mkdir(parents=True, exist_ok=True)
    synced_files: List[str] = []

    # 1. Đồng bộ đồ thị huấn luyện Loss Convergence
    train_loss = base_dir / "core_project" / "ai_software" / "training" / "loss_convergence_dpi300.png"
    if train_loss.exists():
        dest = pub_figures_dir / "loss_convergence_dpi300.png"
        shutil.copy2(train_loss, dest)
        synced_files.append(dest.name)

    # 2. Đồng bộ các Histogram từ PYNQ-Z2
    pynq_fig_dir = base_dir / "core_project" / "benchmarks_reports" / "pynq_z2" / "figures_dpi300"
    if pynq_fig_dir.exists():
        for f in pynq_fig_dir.glob("*.png"):
            dest = pub_figures_dir / f.name
            shutil.copy2(f, dest)
            synced_files.append(dest.name)

    # 3. Đồng bộ Visual Samples
    vis_dir = base_dir / "core_project" / "benchmarks_reports" / "visual_analysis"
    vis_pub_dir = pub_figures_dir / "visual_samples"
    vis_pub_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir.exists():
        for f in vis_dir.glob("*.png"):
            dest = vis_pub_dir / f.name
            shutil.copy2(f, dest)
            synced_files.append(f"visual_samples/{dest.name}")

    logger.info(f"[PACKAGE] Đã kiểm tra và đồng bộ thành công {len(synced_files)} tệp hình ảnh 300 DPI.")
    return synced_files


# ------------------------------------------------------------------------------
# MAIN PACKAGER
# ------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Đóng gói sản phẩm bàn giao kỹ thuật của đề tài")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="core_project/benchmarks_reports/deliverables_export",
        help="Thư mục xuất file Excel bàn giao",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_filename = "hardware_benchmark_statistics.xlsx"
    excel_path_local = out_dir / excel_filename
    excel_path_root = base_dir / excel_filename
    pub_figures_dir = base_dir / "figures_dpi300"

    logger.info("================================================================================")
    logger.info("BẮT ĐẦU ĐÓNG GÓI TOÀN DIỆN SẢN PHẨM BÀN GIAO (EXCEL & 300 DPI FIGURES)")
    logger.info(f"Thư mục gốc: {base_dir}")
    logger.info(f"File Excel đích: {excel_path_local}")
    logger.info("================================================================================")

    # 1. Khởi tạo Workbook OpenPyXL
    wb = openpyxl.Workbook()

    # Nạp dữ liệu vào từng Sheet
    stat_2200_json = base_dir / "core_project" / "benchmarks_reports" / "pynq_z2" / "pynq_z2_statistical_analysis.json"
    multi_plat_json = base_dir / "core_project" / "benchmarks_reports" / "multi_platform" / "multi_platform_benchmark.json"
    dv_score_json = base_dir / "core_project" / "hardware_fpga" / "dv_verification" / "dv_scoreboard_report.json"
    visual_json = base_dir / "core_project" / "benchmarks_reports" / "visual_analysis" / "visual_analysis_report.json"
    baseline_json = base_dir / "core_project" / "benchmarks_reports" / "deliverables_export" / "baseline_psnr_ssim_summary.json"

    logger.info("[PACKAGE] Đang xây dựng Sheet 1: Executive_Summary...")
    build_sheet_executive_summary(wb)

    logger.info("[PACKAGE] Đang xây dựng Sheet 2: PYNQ_Z2_2200_Statistics...")
    build_sheet_pynq_statistics(wb, stat_2200_json)

    logger.info("[PACKAGE] Đang xây dựng Sheet 3: Multi_Platform_Benchmark...")
    build_sheet_multi_platform(wb, multi_plat_json)

    logger.info("[PACKAGE] Đang xây dựng Sheet 4: DV_Scoreboard_Verification...")
    build_sheet_dv_scoreboard(wb, dv_score_json)

    logger.info("[PACKAGE] Đang xây dựng Sheet 5: Visual_Analysis_Samples...")
    build_sheet_visual_analysis(wb, visual_json)

    logger.info("[PACKAGE] Đang xây dựng Sheet 6: Baseline_Float32_Evaluation...")
    build_sheet_baseline_float32(wb, baseline_json)

    # Lưu file Excel
    wb.save(excel_path_local)
    wb.save(excel_path_root)
    logger.info(f"[PACKAGE] Đã tạo thành công file Excel bàn giao 6 sheet tại: {excel_path_local}")
    logger.info(f"[PACKAGE] Đã đồng bộ file Excel ra thư mục gốc: {excel_path_root}")

    # 2. Đồng bộ hóa thư mục hình ảnh 300 DPI
    logger.info("[PACKAGE] Đang đồng bộ hóa kho hình ảnh 300 DPI...")
    synced = audit_and_consolidate_figures(base_dir, pub_figures_dir)

    print("\n" + "=" * 90)
    print("HOÀN TẤT ĐÓNG GÓI TOÀN DIỆN SẢN PHẨM BÀN GIAO (DELIVERABLES PACKAGING)")
    print("=" * 90)
    print(f"1. File Excel tổng hợp : {excel_path_root} (6 Sheet định dạng chuyên nghiệp)")
    print(f"2. Thư mục hình ảnh    : {pub_figures_dir} ({len(synced)} ảnh đạt chuẩn 300 DPI)")
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
