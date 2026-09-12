#!/usr/bin/env python3
"""
================================================================================
Merge Performance Comparison Reports Script
Dự án: Siêu phân giải ảnh y tế AI-Based Image Super-Resolution
================================================================================
Mục đích:
  Gộp 2 báo cáo so sánh hiệu năng:
    1. So sánh hiệu năng Bicubic và 6 Model phần mềm (SRCNN, ESPCN, FSRCNN, VDSR, EDSR, SRGAN)
    2. So sánh hiệu năng đối đầu phần cứng (Bicubic vs SRCNN RTL Q7 vs Swift-SRGAN Q7)
  Thành 1 file PDF báo cáo tổng hợp duy nhất có trang bìa, mục lục và bookmark điều hướng.
================================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import pymupdf

CURRENT_DIR = Path(__file__).resolve().parent
OUT_DIR = CURRENT_DIR / "output"
OUT_PDF = OUT_DIR / "so_sanh_hieu_nang_tong_hop.pdf"

CHARTS_PART1 = [
    OUT_DIR / "nb1_sub_NIH_scale2x.png",
    OUT_DIR / "nb1_sub_NIH_scale3x.png",
    OUT_DIR / "nb1_sub_NIH_scale4x.png",
    OUT_DIR / "nb1_sub_chest_scale2x.png",
    OUT_DIR / "nb1_sub_chest_scale3x.png",
    OUT_DIR / "nb1_sub_chest_scale4x.png",
]

CHARTS_PART2 = [
    OUT_DIR / "nb2_sub_NIH_scale2x.png",
    OUT_DIR / "nb2_sub_NIH_scale3x.png",
    OUT_DIR / "nb2_sub_NIH_scale4x.png",
    OUT_DIR / "nb2_sub_chest_scale2x.png",
    OUT_DIR / "nb2_sub_chest_scale3x.png",
    OUT_DIR / "nb2_sub_chest_scale4x.png",
]


def create_master_cover_page() -> plt.Figure:
    """Tạo trang bìa chính và bảng mục lục tổng hợp hai phân hệ."""
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    fig.patch.set_facecolor("#f8fafd")
    ax.set_facecolor("#f8fafd")
    ax.axis("off")

    # Header
    fig.text(
        0.5,
        0.955,
        "BÁO CÁO TỔNG HỢP SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH SIÊU PHÂN GIẢI",
        ha="center",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.918,
        "Đánh giá toàn diện: Bicubic | 6 Mô hình Phần mềm (SRCNN Model: 1-64-32-1) | Tăng tốc Phần cứng FPGA RTL (Compact SRCNN: 1-16-8-1, Q7)",
        ha="center",
        va="top",
        fontsize=10.0,
        style="italic",
        color="#2b6cb0",
        fontfamily="DejaVu Sans",
    )

    # Architecture Distinction Callout Box
    rect = patches.FancyBboxPatch(
        (0.08, 0.772),
        0.84,
        0.112,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        edgecolor="#2b6cb0",
        facecolor="#eef5fc",
        linewidth=1.3,
        transform=fig.transFigure,
    )
    fig.patches.append(rect)

    fig.text(
        0.095,
        0.868,
        "ĐẶC TẢ PHÂN BIỆT KIẾN TRÚC SRCNN (PHẦN MỀM vs PHẦN CỨNG RTL):",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#0f2b48",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.838,
        "• SRCNN Model (Phần mềm Baseline): Kiến trúc 1 -> 64 -> 32 -> 1 (Tổng 8.129 tham số, FP32, trích xuất đặc trưng sâu 64 và 32 kênh)",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.810,
        "• Compact SRCNN RTL (Phần cứng FPGA): Kiến trúc thu gọn 1 -> 16 -> 8 -> 1 (Tổng 1.649 tham số, Fixed-Point Q7 S7.0/S0.7/S24.7)",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.783,
        "  -> Tối ưu hóa triệt để tài nguyên DSP slice / BRAM / LUT trên chip Xilinx Zynq-7020 (PYNQ-Z2), đạt thông lượng thời gian thực.",
        ha="left",
        va="top",
        fontsize=8.5,
        style="italic",
        color="#4a5568",
        fontfamily="DejaVu Sans",
    )

    # TOC Header
    fig.text(
        0.08,
        0.735,
        "MỤC LỤC TỔNG HỢP (TABLE OF CONTENTS):",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#222222",
        fontfamily="DejaVu Sans",
    )

    toc_items = [
        ("part", "PHẦN I: SO SÁNH HIỆU NĂNG BICUBIC VÀ 6 MODEL PHẦN MỀM (SRCNN Model [1-64-32-1, 8.129 params], ESPCN, FSRCNN, VDSR, EDSR, SRGAN)", None, 0.696),
        ("sec", "  1. Tập dữ liệu sub_NIH (NIH ChestX-ray14)", None, 0.662),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 2, 0.632),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 3, 0.603),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 4, 0.574),
        ("sec", "  2. Tập dữ liệu sub_chest (Chest X-ray Clinical)", None, 0.540),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 5, 0.510),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 6, 0.481),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 7, 0.452),
        ("part", "PHẦN II: SO SÁNH HIỆU NĂNG ĐỐI ĐẦU PHẦN CỨNG (BICUBIC vs COMPACT SRCNN RTL [1-16-8-1, 1.649 params, Q7])", None, 0.405),
        ("sec", "  1. Tập dữ liệu sub_NIH (NIH ChestX-ray14)", None, 0.371),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 8, 0.341),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 9, 0.312),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 10, 0.283),
        ("sec", "  2. Tập dữ liệu sub_chest (Chest X-ray Clinical)", None, 0.249),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 11, 0.219),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 12, 0.190),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 13, 0.161),
    ]

    for itype, text, page, y in toc_items:
        if itype == "part":
            fig.text(0.08, y, text, ha="left", va="center", fontsize=9.8, fontweight="bold", color="#0f2b48", fontfamily="DejaVu Sans")
        elif itype == "sec":
            fig.text(0.10, y, text, ha="left", va="center", fontsize=9.2, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")
        else:
            fig.text(0.12, y, text, ha="left", va="center", fontsize=8.8, color="#333333", fontfamily="DejaVu Sans")
            fig.lines.append(
                plt.Line2D([0.58, 0.84], [y, y], transform=fig.transFigure, color="#bbbbbb", linestyle=":", linewidth=0.9)
            )
            fig.text(0.89, y, f"Trang {page}", ha="right", va="center", fontsize=8.8, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")

    fig.text(
        0.5,
        0.035,
        "AI-Based Image Super-Resolution for Medical Imaging | FPGA Xilinx Zynq-7020 Acceleration",
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#777777",
        fontfamily="DejaVu Sans",
    )
    return fig


def generate_merged_pdf(output_path: Path = OUT_PDF) -> Path:
    """Tạo file PDF tổng hợp kèm bookmarks điều hướng chuyên nghiệp."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_charts = CHARTS_PART1 + CHARTS_PART2

    # Render PDF pages
    with PdfPages(str(output_path)) as pdf:
        cover_fig = create_master_cover_page()
        pdf.savefig(cover_fig, bbox_inches="tight", dpi=130)
        plt.close(cover_fig)

        for p in all_charts:
            if p.exists():
                img = mpimg.imread(str(p))
                fig, ax = plt.subplots(figsize=(12.0, img.shape[0] / img.shape[1] * 12.0))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight", dpi=130)
                plt.close(fig)
            else:
                print(f"Cảnh báo: Không tìm thấy ảnh biểu đồ {p.name}")

    # Gắn bookmark điều hướng tương tác
    doc = pymupdf.open(str(output_path))
    toc = [
        [1, "Trang Bìa & Mục Lục Tổng Hợp", 1],
        [1, "PHẦN I: So Sánh 6 Model Phần Mềm (SRCNN Model: 1-64-32-1 [8.129 params])", 2],
        [2, "Tập dữ liệu sub_NIH - Scale 2x", 2],
        [2, "Tập dữ liệu sub_NIH - Scale 3x", 3],
        [2, "Tập dữ liệu sub_NIH - Scale 4x", 4],
        [2, "Tập dữ liệu sub_chest - Scale 2x", 5],
        [2, "Tập dữ liệu sub_chest - Scale 3x", 6],
        [2, "Tập dữ liệu sub_chest - Scale 4x", 7],
        [1, "PHẦN II: Đối Đầu Phần Cứng (Bicubic vs Compact SRCNN RTL: 1-16-8-1 [1.649 params, Q7])", 8],
        [2, "Tập dữ liệu sub_NIH - Scale 2x", 8],
        [2, "Tập dữ liệu sub_NIH - Scale 3x", 9],
        [2, "Tập dữ liệu sub_NIH - Scale 4x", 10],
        [2, "Tập dữ liệu sub_chest - Scale 2x", 11],
        [2, "Tập dữ liệu sub_chest - Scale 3x", 12],
        [2, "Tập dữ liệu sub_chest - Scale 4x", 13],
    ]
    doc.set_toc(toc)
    temp_path = output_path.with_suffix(".tmp.pdf")
    doc.save(str(temp_path))
    doc.close()
    temp_path.replace(output_path)

    print(f"Đã xuất thành công file PDF tổng hợp: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_path


if __name__ == "__main__":
    generate_merged_pdf()
