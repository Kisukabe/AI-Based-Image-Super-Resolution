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
    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    fig.patch.set_facecolor("#f8fafd")
    ax.set_facecolor("#f8fafd")
    ax.axis("off")

    # Header
    fig.text(
        0.5,
        0.94,
        "BÁO CÁO TỔNG HỢP SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH SIÊU PHÂN GIẢI",
        ha="center",
        va="top",
        fontsize=14.5,
        fontweight="bold",
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.895,
        "Đánh giá toàn diện: Bicubic | 6 Mô hình Phần mềm (Software) | Tăng tốc Phần cứng FPGA RTL (Q7)",
        ha="center",
        va="top",
        fontsize=11.0,
        style="italic",
        color="#2b6cb0",
        fontfamily="DejaVu Sans",
    )

    # Divider
    fig.lines.append(
        plt.Line2D([0.08, 0.92], [0.865, 0.865], transform=fig.transFigure, color="#1a3a5c", linewidth=1.8)
    )

    # TOC Header
    fig.text(
        0.08,
        0.83,
        "MỤC LỤC TỔNG HỢP (TABLE OF CONTENTS):",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color="#222222",
        fontfamily="DejaVu Sans",
    )

    toc_items = [
        ("part", "PHẦN I: SO SÁNH HIỆU NĂNG BICUBIC VÀ 6 MODEL PHẦN MỀM (SRCNN, ESPCN, FSRCNN, VDSR, EDSR, SRGAN)", None, 0.77),
        ("sec", "  1. Tập dữ liệu sub_NIH (NIH ChestX-ray14)", None, 0.725),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 2, 0.685),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 3, 0.650),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 4, 0.615),
        ("sec", "  2. Tập dữ liệu sub_chest (Chest X-ray Clinical)", None, 0.570),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 5, 0.530),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 6, 0.495),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 7, 0.460),
        ("part", "PHẦN II: SO SÁNH HIỆU NĂNG ĐỐI ĐẦU PHẦN CỨNG (BICUBIC vs SRCNN RTL Q7 vs SWIFT-SRGAN Q7)", None, 0.395),
        ("sec", "  1. Tập dữ liệu sub_NIH (NIH ChestX-ray14)", None, 0.350),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 8, 0.310),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 9, 0.275),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 10, 0.240),
        ("sec", "  2. Tập dữ liệu sub_chest (Chest X-ray Clinical)", None, 0.195),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 11, 0.155),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 12, 0.120),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 13, 0.085),
    ]

    for itype, text, page, y in toc_items:
        if itype == "part":
            fig.text(0.08, y, text, ha="left", va="center", fontsize=10.5, fontweight="bold", color="#0f2b48", fontfamily="DejaVu Sans")
        elif itype == "sec":
            fig.text(0.10, y, text, ha="left", va="center", fontsize=10.0, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")
        else:
            fig.text(0.12, y, text, ha="left", va="center", fontsize=9.5, color="#333333", fontfamily="DejaVu Sans")
            fig.lines.append(
                plt.Line2D([0.55, 0.84], [y, y], transform=fig.transFigure, color="#bbbbbb", linestyle=":", linewidth=1.0)
            )
            fig.text(0.89, y, f"Trang {page}", ha="right", va="center", fontsize=9.5, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")

    fig.text(
        0.5,
        0.03,
        "AI-Based Image Super-Resolution for Medical Imaging | FPGA Xilinx Zynq-7020 Acceleration",
        ha="center",
        va="bottom",
        fontsize=9.0,
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
        [1, "PHẦN I: So Sánh Hiệu Năng Bicubic và 6 Model Phần Mềm", 2],
        [2, "Tập dữ liệu sub_NIH - Scale 2x", 2],
        [2, "Tập dữ liệu sub_NIH - Scale 3x", 3],
        [2, "Tập dữ liệu sub_NIH - Scale 4x", 4],
        [2, "Tập dữ liệu sub_chest - Scale 2x", 5],
        [2, "Tập dữ liệu sub_chest - Scale 3x", 6],
        [2, "Tập dữ liệu sub_chest - Scale 4x", 7],
        [1, "PHẦN II: So Sánh Hiệu Năng Đối Đầu Phần Cứng (Bicubic vs SRCNN RTL Q7 vs Swift-SRGAN Q7)", 8],
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
