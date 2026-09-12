#!/usr/bin/env python3
"""
================================================================================
Merge Performance Comparison Reports Script (Software Only)
Dự án: Siêu phân giải ảnh y tế AI-Based Image Super-Resolution
================================================================================
Mục đích:
  Xuất báo cáo PDF so sánh hiệu năng của Bicubic Baseline cùng 6 mô hình phần mềm:
  SRCNN (1-64-32-1, 8.129 tham số - đang infer), ESPCN, FSRCNN, VDSR, EDSR, SRGAN
  qua 3 scale (2x, 3x, 4x) trên 2 tập dữ liệu (sub_NIH, sub_chest).
  (Phần cứng đã được tách riêng theo yêu cầu cấu trúc báo cáo).
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

CHARTS_SOFTWARE = [
    OUT_DIR / "nb1_sub_NIH_scale2x.png",
    OUT_DIR / "nb1_sub_NIH_scale3x.png",
    OUT_DIR / "nb1_sub_NIH_scale4x.png",
    OUT_DIR / "nb1_sub_chest_scale2x.png",
    OUT_DIR / "nb1_sub_chest_scale3x.png",
    OUT_DIR / "nb1_sub_chest_scale4x.png",
]


def create_master_cover_page() -> plt.Figure:
    """Tạo trang bìa chính và bảng mục lục so sánh các mô hình phần mềm."""
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    fig.patch.set_facecolor("#f8fafd")
    ax.set_facecolor("#f8fafd")
    ax.axis("off")

    # Header
    fig.text(
        0.5,
        0.955,
        "BÁO CÁO TỔNG HỢP SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH PHẦN MỀM SIÊU PHÂN GIẢI",
        ha="center",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.920,
        "Đánh giá đối chứng thực nghiệm: Bicubic Baseline và 6 Mô hình Phần mềm Deep Learning (PyTorch GPU CUDA)",
        ha="center",
        va="top",
        fontsize=10.0,
        style="italic",
        color="#2b6cb0",
        fontfamily="DejaVu Sans",
    )

    # Architectural Overview Box
    rect = patches.FancyBboxPatch(
        (0.08, 0.705),
        0.84,
        0.185,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        edgecolor="#2b6cb0",
        facecolor="#eef5fc",
        linewidth=1.3,
        transform=fig.transFigure,
    )
    fig.patches.append(rect)

    fig.text(
        0.095,
        0.875,
        "DANH MỤC CÁC MÔ HÌNH ĐỐI CHỨNG THỰC NGHIỆM:",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#0f2b48",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.852,
        "1. Bicubic (Baseline): Thuật toán nội suy đa thức bậc ba chuẩn hóa (không tham số học).",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.830,
        "2. SRCNN Model: Kiến trúc nguyên bản 1 -> 64 -> 32 -> 1 (8.129 tham số, FP32) — [Đang infer trên Kaggle GPU].",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#b07700",
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.808,
        "3. ESPCN: Kiến trúc Sub-Pixel Convolution (PixelShuffle) tăng tốc tái tạo ảnh siêu phân giải.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.786,
        "4. FSRCNN: Mạng SRCNN cải tiến co hẹp số chiều đặc trưng (Shrinking) và mở rộng (Expanding).",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.764,
        "5. VDSR: Mạng rất sâu 20 tầng tích chập học phần dư (Residual Learning) với gradient clipping.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.742,
        "6. EDSR: Enhanced Deep Residual Networks (8 khối ResBlock, 64 kênh đặc trưng chiều sâu).",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.720,
        "7. SRGAN: Mạng nơ-ron đối kháng tạo sinh tối ưu hóa hàm mất mát thụ cảm trực quan (Perceptual Loss).",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )

    # TOC Header
    fig.text(
        0.08,
        0.675,
        "CẤU TRÚC NỘI DUNG BÁO CÁO (MỤC LỤC):",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#222222",
        fontfamily="DejaVu Sans",
    )

    toc_items = [
        ("sec", "I. TẬP DỮ LIỆU SUB_NIH (NIH ChestX-ray14 — 1.750 ảnh y tế)", None, 0.635),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 2, 0.598),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 3, 0.562),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 4, 0.526),
        ("sec", "II. TẬP DỮ LIỆU SUB_CHEST (Chest X-ray Clinical — 450 ảnh lâm sàng)", None, 0.475),
        ("item", "     • Tỉ lệ phóng đại: Scale 2x", 5, 0.438),
        ("item", "     • Tỉ lệ phóng đại: Scale 3x", 6, 0.402),
        ("item", "     • Tỉ lệ phóng đại: Scale 4x", 7, 0.366),
    ]

    for itype, text, page, y in toc_items:
        if itype == "sec":
            fig.text(0.08, y, text, ha="left", va="center", fontsize=9.8, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")
        else:
            fig.text(0.11, y, text, ha="left", va="center", fontsize=9.0, color="#333333", fontfamily="DejaVu Sans")
            fig.lines.append(
                plt.Line2D([0.52, 0.85], [y, y], transform=fig.transFigure, color="#bbbbbb", linestyle=":", linewidth=1.0)
            )
            fig.text(0.89, y, f"Trang {page}", ha="right", va="center", fontsize=9.0, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")

    fig.text(
        0.5,
        0.035,
        "AI-Based Image Super-Resolution for Medical Imaging | Benchmarking Suite",
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

    with PdfPages(str(output_path)) as pdf:
        cover_fig = create_master_cover_page()
        pdf.savefig(cover_fig, bbox_inches="tight", dpi=140)
        plt.close(cover_fig)

        for p in CHARTS_SOFTWARE:
            if p.exists():
                img = mpimg.imread(str(p))
                fig, ax = plt.subplots(figsize=(12.0, img.shape[0] / img.shape[1] * 12.0))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight", dpi=140)
                plt.close(fig)
            else:
                print(f"Cảnh báo: Không tìm thấy ảnh biểu đồ {p.name}")

    # Gắn bookmark điều hướng tương tác
    doc = pymupdf.open(str(output_path))
    toc = [
        [1, "Trang Bìa & Mục Lục", 1],
        [1, "I. Tập Dữ Liệu sub_NIH", 2],
        [2, "Scale 2x", 2],
        [2, "Scale 3x", 3],
        [2, "Scale 4x", 4],
        [1, "II. Tập Dữ Liệu sub_chest", 5],
        [2, "Scale 2x", 5],
        [2, "Scale 3x", 6],
        [2, "Scale 4x", 7],
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
