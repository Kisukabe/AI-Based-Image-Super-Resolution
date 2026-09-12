#!/usr/bin/env python3
"""
================================================================================
Render Hardware Comparison Charts (Bicubic vs Compact SRCNN RTL Q7)
Dự án: Siêu phân giải ảnh y tế AI-Based Image Super-Resolution
================================================================================
Mục đích:
  Tái tạo 6 biểu đồ so sánh hiệu năng phần cứng trong thư mục output/:
    - nb2_sub_NIH_scale2x.png, nb2_sub_NIH_scale3x.png, nb2_sub_NIH_scale4x.png
    - nb2_sub_chest_scale2x.png, nb2_sub_chest_scale3x.png, nb2_sub_chest_scale4x.png
  Yêu cầu:
    - LOẠI BỎ HOÀN TOÀN Swift-SRGAN ra khỏi bảng đối đầu phần cứng.
    - Chỉ giữ 2 phương pháp: Bicubic (Baseline) vs Compact SRCNN RTL (Q7).
    - Tối ưu hóa độ rộng cột, phông chữ, định dạng làm nổi bật chỉ số tốt nhất.
================================================================================
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ── Đường dẫn thư mục ────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
BENCH_ROOT = PROJECT_ROOT / "legacy_experiments" / "benchmark_results"
OUT_DIR = CURRENT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hằng số thẩm mỹ giao diện ────────────────────────────────────────────────
HEADER_BG = "#1a3a5c"
HEADER_FG = "white"
ROW_ODD = "#eaf1fb"
ROW_EVEN = "#ffffff"
EDGE_COLOR = "#c8d4e3"
PAGE_BG = "#f5f8fc"
TITLE_COLOR = "#1a3a5c"

METRICS = ["PSNR", "SSIM", "MS_SSIM", "LPIPS", "NIQE", "EPI", "MSE", "RMSE", "Latency", "FPS"]
METRIC_LABELS = [
    "PSNR (dB) ↑",
    "SSIM ↑",
    "MS-SSIM ↑",
    "LPIPS ↓",
    "NIQE ↓",
    "EPI ↑",
    "MSE ↓",
    "RMSE ↓",
    "Độ trễ Latency (ms) ↓",
    "Tốc độ thông lượng (FPS) ↑",
]


def fmt_val(metric: str, v) -> str:
    """Định dạng giá trị số hiển thị trên bảng."""
    if v is None:
        return "—"
    try:
        f = float(v)
        if metric == "PSNR":
            return f"{f:.4f} dB"
        elif metric == "Latency":
            return f"{f:.2f} ms"
        elif metric == "FPS":
            return f"{f:.1f} FPS"
        return f"{f:.4f}"
    except Exception:
        return str(v)


def load_bicubic(scale: int) -> dict:
    """Nạp dữ liệu thực nghiệm nội suy Bicubic."""
    fn = BENCH_ROOT / f"hardware/bicubic_{scale}x_benchmark.json"
    with open(fn, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for ds in ["sub_NIH", "sub_chest"]:
        recs = [r for r in d["per_image_results"] if r.get("dataset") == ds and r["status"] == "ok"]
        lat = np.mean([r["latency_ms"] for r in recs])
        out[ds] = {
            "N": len(recs),
            "PSNR": round(float(np.mean([r["psnr_bicubic_db"] for r in recs])), 4),
            "SSIM": round(float(np.mean([r["ssim_bicubic"] for r in recs])), 4),
            "MS_SSIM": round(float(np.mean([r["ms_ssim_bicubic"] for r in recs])), 4),
            "LPIPS": round(float(np.mean([r["lpips_bicubic"] for r in recs])), 4),
            "NIQE": round(float(np.mean([r["niqe_bicubic"] for r in recs])), 4),
            "EPI": round(float(np.mean([r["epi_bicubic"] for r in recs])), 4),
            "MSE": round(float(np.mean([r["mse_bicubic"] for r in recs])), 4),
            "RMSE": round(float(np.mean([r["rmse_bicubic"] for r in recs])), 4),
            "Latency": round(float(lat), 2),
            "FPS": round(1000 / float(lat), 1),
        }
    return out


def load_srcnn_hardware(scale: int) -> dict:
    """Nạp dữ liệu đo đạc của mô hình SRCNN phần cứng."""
    fn = BENCH_ROOT / f"software/SRCNN/srcnn_{scale}x_benchmark.json"
    with open(fn, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for ds in ["sub_NIH", "sub_chest"]:
        recs = [r for r in d["per_image_results"] if r.get("dataset") == ds and r["status"] == "ok"]
        if not recs:
            out[ds] = None
            continue
        lat = np.mean([r["latency_ms"] for r in recs if r.get("latency_ms") is not None])

        def val(r, *keys):
            for k in keys:
                v = r.get(k)
                if v is not None:
                    return float(v)
            return 0.0

        out[ds] = {
            "N": len(recs),
            "PSNR": round(float(np.mean([val(r, "psnr_model_db", "psnr_fpga_db") for r in recs])), 4),
            "SSIM": round(float(np.mean([val(r, "ssim_model", "ssim_fpga") for r in recs])), 4),
            "MS_SSIM": round(float(np.mean([val(r, "msssim_fpga", "msssim_model") for r in recs])), 4),
            "LPIPS": round(float(np.mean([val(r, "lpips_fpga", "lpips_srgan", "lpips") for r in recs])), 4),
            "NIQE": round(float(np.mean([val(r, "niqe_fpga", "niqe_model") for r in recs])), 4),
            "EPI": round(float(np.mean([val(r, "epi") for r in recs])), 4),
            "MSE": round(float(np.mean([val(r, "mse_model", "mse_fpga") for r in recs])), 4),
            "RMSE": round(float(np.mean([val(r, "rmse_model", "rmse_fpga") for r in recs])), 4),
            "Latency": round(float(lat), 2),
            "FPS": round(1000 / float(lat), 1) if lat > 0 else 0.0,
        }
    return out


def build_transposed_table(models: list, model_data_dict: dict, ds: str) -> tuple:
    """Tạo tiêu đề cột và các dòng số liệu bảng ma trận."""
    col_labels = ["Thông số đánh giá"] + models
    table_rows = []
    for m_key, m_label in zip(METRICS, METRIC_LABELS):
        row = [m_label]
        for model in models:
            m_source = model_data_dict.get(model)
            data = m_source.get(ds) if m_source else None
            v = data.get(m_key) if data else None
            row.append(fmt_val(m_key, v))
        table_rows.append(row)
    return col_labels, table_rows


def render_table(title: str, subtitle: str, col_labels: list, table_rows: list, out_path: Path):
    """Vẽ bảng so sánh đối đầu 2 phương pháp với tỷ lệ cột cân đối."""
    models = col_labels[1:]
    col_w = [0.38, 0.31, 0.31]
    fig_w = 10.5
    fig_h = 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)
    ax.axis("off")

    # Tiêu đề & Phụ đề phân cấp
    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=TITLE_COLOR,
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#2b6cb0",
        fontfamily="DejaVu Sans",
    )

    def parse_float(v_str):
        try:
            return float(v_str.replace("dB", "").replace("ms", "").replace("FPS", "").strip())
        except Exception:
            return None

    # Tìm cột tối ưu nhất cho từng dòng
    best_cols_per_row = {}
    for r_idx, row in enumerate(table_rows, start=1):
        label = row[0]
        is_higher = "↑" in label
        valid_items = []
        for c_idx, val_str in enumerate(row[1:], start=1):
            v_num = parse_float(val_str)
            if v_num is not None:
                valid_items.append((c_idx, v_num))
        if valid_items:
            best_v = max(v for _, v in valid_items) if is_higher else min(v for _, v in valid_items)
            best_cols_per_row[r_idx] = [c for c, v in valid_items if np.isclose(v, best_v)]
        else:
            best_cols_per_row[r_idx] = []

    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        colWidths=col_w,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.0)
    tbl.scale(1, 1.95)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold", fontsize=10.5)
        else:
            if c == 0:
                cell.set_facecolor("#e2eaf4")
                cell.set_text_props(fontweight="bold", color="#1a3a5c")
            else:
                is_best = c in best_cols_per_row.get(r, [])
                if is_best:
                    cell.set_text_props(fontweight="bold", color="#0f2b48")
                    cell.set_facecolor("#dce9f8" if r % 2 == 1 else "#eef5fd")
                elif r % 2 == 1:
                    cell.set_facecolor(ROW_ODD)
                else:
                    cell.set_facecolor(ROW_EVEN)

    # Chú thích kỹ thuật học thuật
    note_text = (
        "* Ghi chú: (↑) Giá trị càng cao càng tốt  |  (↓) Giá trị càng thấp càng tốt\n"
        "  In đậm: Chỉ số tối ưu vượt trội  |  Compact SRCNN RTL (Q7): Kiến trúc tối ưu hóa phần cứng FPGA (1-16-8-1, INT8 Q7)"
    )
    fig.text(
        0.05,
        0.025,
        note_text,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#444444",
        style="italic",
        fontfamily="DejaVu Sans",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=PAGE_BG)
    plt.close()
    print(f"  ✓ Đã lưu biểu đồ: {out_path.name}")


def create_hardware_cover_page() -> plt.Figure:
    """Tạo trang bìa và mục lục báo cáo so sánh phần cứng."""
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    fig.patch.set_facecolor("#f8fafd")
    ax.set_facecolor("#f8fafd")
    ax.axis("off")

    # Header title
    fig.text(
        0.5,
        0.92,
        "SO SÁNH HIỆU NĂNG PHẦN CỨNG: BICUBIC vs COMPACT SRCNN RTL (Q7)",
        ha="center",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.860,
        "Bicubic (Baseline) vs Compact SRCNN RTL (FPGA Xilinx Zynq-7020, 1-16-8-1, INT8 Q7)",
        ha="center",
        va="top",
        fontsize=11.0,
        style="italic",
        color="#2b6cb0",
        fontfamily="DejaVu Sans",
    )

    # Divider line
    fig.lines.append(
        plt.Line2D([0.1, 0.9], [0.82, 0.82], transform=fig.transFigure, color="#1a3a5c", linewidth=1.8)
    )

    # TOC Header
    fig.text(
        0.1,
        0.75,
        "CẤU TRÚC NỘI DUNG BÁO CÁO (MỤC LỤC):",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color="#222222",
        fontfamily="DejaVu Sans",
    )

    items = [
        ("sec", "I. TẬP DỮ LIỆU SUB_NIH (NIH ChestX-ray14)", None, 0.67),
        ("item", "1. Tỉ lệ phóng đại: Scale 2x", 2, 0.610),
        ("item", "2. Tỉ lệ phóng đại: Scale 3x", 3, 0.555),
        ("item", "3. Tỉ lệ phóng đại: Scale 4x", 4, 0.500),
        ("sec", "II. TẬP DỮ LIỆU SUB_CHEST (Chest X-ray Clinical)", None, 0.410),
        ("item", "1. Tỉ lệ phóng đại: Scale 2x", 5, 0.350),
        ("item", "2. Tỉ lệ phóng đại: Scale 3x", 6, 0.295),
        ("item", "3. Tỉ lệ phóng đại: Scale 4x", 7, 0.240),
    ]

    for itype, text, page, y in items:
        if itype == "sec":
            fig.text(0.10, y, text, ha="left", va="center", fontsize=11.0, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")
        else:
            fig.text(0.14, y, text, ha="left", va="center", fontsize=10.5, color="#333333", fontfamily="DejaVu Sans")
            fig.lines.append(
                plt.Line2D([0.48, 0.82], [y, y], transform=fig.transFigure, color="#bbbbbb", linestyle=":", linewidth=1.0)
            )
            fig.text(0.86, y, f"Trang {page}", ha="right", va="center", fontsize=10.5, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")

    fig.text(
        0.5,
        0.05,
        "AI-Based Image Super-Resolution for Medical Imaging | FPGA Acceleration",
        ha="center",
        va="bottom",
        fontsize=9.0,
        style="italic",
        color="#777777",
        fontfamily="DejaVu Sans",
    )
    return fig


def generate_hardware_pdf():
    """Xuất file PDF báo cáo đối đầu phần cứng 7 trang."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    ordered_charts = [
        OUT_DIR / "nb2_sub_NIH_scale2x.png",
        OUT_DIR / "nb2_sub_NIH_scale3x.png",
        OUT_DIR / "nb2_sub_NIH_scale4x.png",
        OUT_DIR / "nb2_sub_chest_scale2x.png",
        OUT_DIR / "nb2_sub_chest_scale3x.png",
        OUT_DIR / "nb2_sub_chest_scale4x.png",
    ]

    pdf_targets = [
        OUT_DIR / "so_sanh_hieu_nang_bicubic-srcnn.pdf",
        OUT_DIR / "so_sanh_hieu_nang_bicubic-srcnn-swiftSrgan.pdf",
    ]

    for pdf_path in pdf_targets:
        with PdfPages(str(pdf_path)) as pdf:
            fig_cover = create_hardware_cover_page()
            pdf.savefig(fig_cover, bbox_inches="tight", dpi=130)
            plt.close(fig_cover)

            for png_path in ordered_charts:
                if png_path.exists():
                    img = mpimg.imread(str(png_path))
                    fig, ax = plt.subplots(figsize=(11.5, img.shape[0] / img.shape[1] * 11.5))
                    ax.imshow(img)
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches="tight", dpi=130)
                    plt.close(fig)
        print(f"  ✓ Đã xuất PDF phần cứng: {pdf_path.name}")


def generate_all_hardware_charts():
    """Tạo toàn bộ 6 biểu đồ so sánh đối đầu phần cứng và xuất PDF."""
    print("Bắt đầu sinh lại 6 biểu đồ so sánh phần cứng (LOẠI BỎ SWIFT-SRGAN)...")
    models = ["Bicubic (Baseline)", "Compact SRCNN RTL (Q7)"]

    configs = [
        ("sub_NIH", 2, "I. TẬP DỮ LIỆU SUB_NIH (NIH ChestX-ray14)", "1. Tỉ lệ phóng đại: Scale 2x"),
        ("sub_NIH", 3, "I. TẬP DỮ LIỆU SUB_NIH (NIH ChestX-ray14)", "2. Tỉ lệ phóng đại: Scale 3x"),
        ("sub_NIH", 4, "I. TẬP DỮ LIỆU SUB_NIH (NIH ChestX-ray14)", "3. Tỉ lệ phóng đại: Scale 4x"),
        ("sub_chest", 2, "II. TẬP DỮ LIỆU SUB_CHEST (Chest X-ray Clinical)", "1. Tỉ lệ phóng đại: Scale 2x"),
        ("sub_chest", 3, "II. TẬP DỮ LIỆU SUB_CHEST (Chest X-ray Clinical)", "2. Tỉ lệ phóng đại: Scale 3x"),
        ("sub_chest", 4, "II. TẬP DỮ LIỆU SUB_CHEST (Chest X-ray Clinical)", "3. Tỉ lệ phóng đại: Scale 4x"),
    ]

    for ds, scale, main_title, sub_title in configs:
        model_data = {
            "Bicubic (Baseline)": load_bicubic(scale),
            "Compact SRCNN RTL (Q7)": load_srcnn_hardware(scale),
        }
        col_labels, rows = build_transposed_table(models, model_data, ds)
        out_png = OUT_DIR / f"nb2_{ds}_scale{scale}x.png"
        render_table(main_title, sub_title, col_labels, rows, out_png)

    generate_hardware_pdf()
    print("Hoàn tất sinh 6 biểu đồ và xuất bản PDF phần cứng thành công.")


if __name__ == "__main__":
    generate_all_hardware_charts()
