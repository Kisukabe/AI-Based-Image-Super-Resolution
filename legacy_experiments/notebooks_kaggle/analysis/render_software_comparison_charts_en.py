#!/usr/bin/env python3
"""
================================================================================
Render Software Comparison Charts & PDF Report (English Edition)
Project: AI-Based Medical Image Super-Resolution
================================================================================
Purpose:
  Generate 6 benchmark comparison charts and a publication-quality PDF report
  evaluating Bicubic Baseline and Deep Learning architectures:
    1. Bicubic (Non-parametric baseline)
    2. Compact SRCNN RTL (1-16-8-1, 1,649 parameters, INT8 Q7 Fixed-Point)
    3. SRCNN Original (1-64-32-1, 8,129 parameters, Float32)
    4. ESPCN (Efficient Sub-Pixel Convolution)
    5. FSRCNN (Fast Super-Resolution CNN)
    6. VDSR (20-layer Very Deep Super-Resolution)
    7. EDSR (Enhanced Deep Residual Networks)
    8. SRGAN (Super-Resolution Generative Adversarial Network)
  across 3 magnification scales (2x, 3x, 4x) on 2 medical imaging datasets:
    - sub_NIH (1,750 images from NIH ChestX-ray14)
    - sub_chest (450 images from Clinical Chest X-ray)
  Outputs publication-grade PDF reports with bookmarks and cover page.
================================================================================
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import pymupdf

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
BASE = PROJECT_ROOT / "legacy_experiments/benchmark_results"
OUT_DIR = CURRENT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Scientific Palette & Styling ─────────────────────────────────────────────
HEADER_BG   = '#1a3a5c'
HEADER_FG   = '#ffffff'
ROW_ODD     = '#eaf1fb'
ROW_EVEN    = '#ffffff'
EDGE_COLOR  = '#c8d4e3'
PAGE_BG     = '#f5f8fc'
TITLE_COLOR = '#1a3a5c'
SUB_COLOR   = '#555555'

METRICS = ['PSNR', 'SSIM', 'MS_SSIM', 'LPIPS', 'NIQE', 'EPI', 'MSE', 'RMSE', 'Latency', 'FPS']
METRIC_LABELS = [
    'PSNR (dB) ↑', 'SSIM ↑', 'MS-SSIM ↑', 'LPIPS ↓',
    'NIQE ↓', 'EPI ↑', 'MSE ↓', 'RMSE ↓', 'Latency (ms) ↓', 'Throughput (FPS) ↑'
]

MODELS_LIST = [
    'Bicubic',
    'Compact SRCNN RTL',
    'SRCNN Original',
    'ESPCN',
    'FSRCNN',
    'VDSR',
    'EDSR',
    'SRGAN'
]

PENDING_MODELS = set()


def fmt_val(metric: str, v) -> str:
    """Format numerical metric values with scientific precision and units."""
    if v is None:
        return 'N/A'
    try:
        f = float(v)
        if metric == 'PSNR':
            return f"{f:.4f} dB"
        elif metric == 'Latency':
            return f"{f:.2f} ms"
        elif metric == 'FPS':
            return f"{f:.1f} FPS"
        return f"{f:.4f}"
    except Exception:
        return str(v)


def load_bicubic(scale: int) -> dict:
    """Load Bicubic baseline benchmark data."""
    fn = BASE / f'hardware/bicubic_{scale}x_benchmark.json'
    if not fn.exists():
        fn = PROJECT_ROOT / f'results/hardware/bicubic_{scale}x_benchmark.json'
    with open(fn, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = {}
    for ds in ['sub_NIH', 'sub_chest']:
        recs = [r for r in d['per_image_results'] if r.get('dataset') == ds and r.get('status') == 'ok']
        if not recs:
            out[ds] = None
            continue
        lat = np.mean([r['latency_ms'] for r in recs if r.get('latency_ms') is not None])
        out[ds] = {
            'N': len(recs),
            'PSNR':     round(float(np.mean([r['psnr_bicubic_db'] for r in recs])), 4),
            'SSIM':     round(float(np.mean([r['ssim_bicubic'] for r in recs])), 4),
            'MS_SSIM':  round(float(np.mean([r['ms_ssim_bicubic'] for r in recs])), 4),
            'LPIPS':    round(float(np.mean([r['lpips_bicubic'] for r in recs])), 4),
            'NIQE':     round(float(np.mean([r['niqe_bicubic'] for r in recs])), 4),
            'EPI':      round(float(np.mean([r['epi_bicubic'] for r in recs])), 4),
            'MSE':      round(float(np.mean([r['mse_bicubic'] for r in recs])), 4),
            'RMSE':     round(float(np.mean([r['rmse_bicubic'] for r in recs])), 4),
            'Latency':  round(float(lat), 2) if lat > 0 else 0.0,
            'FPS':      round(1000.0 / float(lat), 1) if lat > 0 else 0.0,
        }
    return out


def load_compact_srcnn_rtl(scale: int) -> dict:
    """Load Compact SRCNN RTL hardware-simulated benchmark data."""
    fn = BASE / f'hardware/compact_srcnn_rtl_{scale}x_benchmark.json'
    if not fn.exists():
        fn = PROJECT_ROOT / f'results/hardware/compact_srcnn_rtl_{scale}x_benchmark.json'
    if not fn.exists():
        return None
    with open(fn, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = {}
    for ds in ['sub_NIH', 'sub_chest']:
        recs = [r for r in d['per_image_results'] if r.get('dataset') == ds and r.get('status') == 'ok']
        if not recs:
            out[ds] = None
            continue
        lat = np.mean([r['latency_ms'] for r in recs if r.get('latency_ms') is not None])

        def val(r, *keys):
            for k in keys:
                v = r.get(k)
                if v is not None:
                    return float(v)
            return 0.0

        out[ds] = {
            'N': len(recs),
            'PSNR':    round(float(np.mean([val(r, 'psnr_model_db', 'psnr_fpga_db') for r in recs])), 4),
            'SSIM':    round(float(np.mean([val(r, 'ssim_model', 'ssim_fpga') for r in recs])), 4),
            'MS_SSIM': round(float(np.mean([val(r, 'msssim_fpga', 'msssim_model') for r in recs])), 4),
            'LPIPS':   round(float(np.mean([val(r, 'lpips_fpga', 'lpips') for r in recs])), 4),
            'NIQE':    round(float(np.mean([val(r, 'niqe_fpga', 'niqe_model') for r in recs])), 4),
            'EPI':     round(float(np.mean([val(r, 'epi') for r in recs])), 4),
            'MSE':     round(float(np.mean([val(r, 'mse_model', 'mse_fpga') for r in recs])), 4),
            'RMSE':    round(float(np.mean([val(r, 'rmse_model', 'rmse_fpga') for r in recs])), 4),
            'Latency': round(float(lat), 2) if lat > 0 else 0.0,
            'FPS':     round(1000.0 / float(lat), 1) if lat > 0 else 0.0,
        }
    return out


def load_software(model_folder: str, scale: int) -> dict:
    """Load benchmark data for software Deep Learning models."""
    fn = BASE / f'software/{model_folder.upper()}/{model_folder.lower()}_{scale}x_benchmark.json'
    if not fn.exists():
        fn = PROJECT_ROOT / f'results/software/{model_folder.upper()}/{model_folder.lower()}_{scale}x_benchmark.json'
    if not fn.exists():
        return None
    with open(fn, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = {}
    for ds in ['sub_NIH', 'sub_chest']:
        recs = [r for r in d['per_image_results'] if r.get('dataset') == ds and r.get('status') == 'ok']
        if not recs:
            out[ds] = None
            continue
        lat = np.mean([r['latency_ms'] for r in recs if r.get('latency_ms') is not None])

        def val(r, *keys):
            for k in keys:
                v = r.get(k)
                if v is not None:
                    return float(v)
            return 0.0

        out[ds] = {
            'N': len(recs),
            'PSNR':    round(float(np.mean([val(r, 'psnr_model_db', 'psnr_fpga_db') for r in recs])), 4),
            'SSIM':    round(float(np.mean([val(r, 'ssim_model', 'ssim_fpga') for r in recs])), 4),
            'MS_SSIM': round(float(np.mean([val(r, 'msssim_fpga', 'msssim_model') for r in recs])), 4),
            'LPIPS':   round(float(np.mean([val(r, 'lpips_fpga', 'lpips_srgan', 'lpips') for r in recs])), 4),
            'NIQE':    round(float(np.mean([val(r, 'niqe_fpga', 'niqe_model') for r in recs])), 4),
            'EPI':     round(float(np.mean([val(r, 'epi') for r in recs])), 4),
            'MSE':     round(float(np.mean([val(r, 'mse_model', 'mse_fpga') for r in recs])), 4),
            'RMSE':    round(float(np.mean([val(r, 'rmse_model', 'rmse_fpga') for r in recs])), 4),
            'Latency': round(float(lat), 2) if lat > 0 else 0.0,
            'FPS':     round(1000.0 / float(lat), 1) if lat > 0 else 0.0,
        }
    return out


def build_transposed_table(models: list, model_data_dict: dict, ds: str):
    """Build standardized 9-column comparison matrix."""
    col_labels = [
        'Metric',
        'Bicubic',
        'Compact SRCNN\nRTL',
        'SRCNN\nOriginal',
        'ESPCN',
        'FSRCNN',
        'VDSR',
        'EDSR',
        'SRGAN'
    ]

    table_rows = []
    for m_key, m_label in zip(METRICS, METRIC_LABELS):
        row = [m_label]
        for m in models:
            m_source = model_data_dict.get(m)
            data = m_source.get(ds) if m_source else None
            v = data.get(m_key) if data else None
            row.append(fmt_val(m_key, v))
        table_rows.append(row)
    return col_labels, table_rows


def render_table(title: str, subtitle: str, col_labels: list, table_rows: list, out_path: Path):
    """Render publication-quality 9-column comparison table (300 DPI)."""
    models = MODELS_LIST
    col_w = [0.16] + [0.105] * 8
    fig_w = 13.0
    fig_h = 7.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)
    ax.axis('off')

    # Title & Subtitle
    if subtitle:
        fig.text(0.5, 0.965, title, ha='center', va='top',
                 fontsize=12.5, fontweight='bold', color=TITLE_COLOR,
                 fontfamily='DejaVu Sans')
        fig.text(0.5, 0.925, subtitle, ha='center', va='top',
                 fontsize=11.0, fontweight='bold', color='#2b6cb0',
                 fontfamily='DejaVu Sans')
    else:
        fig.text(0.5, 0.94, title, ha='center', va='top',
                 fontsize=13.0, fontweight='bold', color=TITLE_COLOR,
                 fontfamily='DejaVu Sans')

    def parse_float(v_str):
        try:
            return float(v_str.replace('dB', '').replace('ms', '').replace('FPS', '').strip())
        except Exception:
            return None

    # Calculate optimal (best) values per metric row
    best_cols_per_row = {}
    for r_idx, row in enumerate(table_rows, start=1):
        label = row[0]
        is_higher = '↑' in label
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

    tbl = ax.table(cellText=table_rows, colLabels=col_labels, colWidths=col_w,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1, 1.95)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight='bold', fontsize=8.8)
        else:
            if c == 0:
                cell.set_facecolor('#e2eaf4')
                cell.set_text_props(fontweight='bold', color='#1a3a5c')
            else:
                is_best = (c in best_cols_per_row.get(r, []))
                if is_best:
                    cell.set_text_props(fontweight='bold', color='#0f2b48')
                    cell.set_facecolor('#dce9f8' if r % 2 == 1 else '#eef5fd')
                elif r % 2 == 1:
                    cell.set_facecolor(ROW_ODD)
                else:
                    cell.set_facecolor(ROW_EVEN)

    # Scientific Footnote
    note_text = (
        "* Notes:  (↑) Higher is better  |  (↓) Lower is better  |  Bold: Optimal metric value across benchmarked models\n"
        "  Architectures: Compact SRCNN RTL (1-16-8-1, 1,649 params, INT8 Q7)  |  SRCNN Original (1-64-32-1, 8,129 params, FP32)"
    )
    fig.text(0.04, 0.020, note_text, ha='left', va='bottom',
             fontsize=8.5, color='#444444', style='italic', fontfamily='DejaVu Sans')

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PAGE_BG)
    plt.close()
    print(f"  ✓ Saved English chart: {out_path.name}")


def create_software_cover_page_en() -> plt.Figure:
    """Create publication cover page and Table of Contents in English."""
    fig, ax = plt.subplots(figsize=(12.0, 8.2))
    fig.patch.set_facecolor("#f8fafd")
    ax.set_facecolor("#f8fafd")
    ax.axis("off")

    # Header title
    fig.text(
        0.5,
        0.960,
        "IMAGE SUPER-RESOLUTION MODELS PERFORMANCE BENCHMARK REPORT",
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.925,
        "Empirical Benchmark: Bicubic Baseline vs. Deep Learning Models (Compact RTL & Software Architectures)",
        ha="center",
        va="top",
        fontsize=9.8,
        style="italic",
        color="#2b6cb0",
        fontfamily="DejaVu Sans",
    )

    # Architectural Overview Box
    rect = patches.FancyBboxPatch(
        (0.08, 0.690),
        0.84,
        0.205,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        edgecolor="#2b6cb0",
        facecolor="#eef5fc",
        linewidth=1.3,
        transform=fig.transFigure,
    )
    fig.patches.append(rect)

    fig.text(
        0.095,
        0.880,
        "BENCHMARKED ARCHITECTURES & EVALUATED MODELS:",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#0f2b48",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.858,
        "1. Bicubic (Baseline): Standard bicubic polynomial interpolation algorithm (non-parametric baseline).",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.836,
        "2. Compact SRCNN RTL: Hardware-compact architecture 1 -> 16 -> 8 -> 1 (1,649 parameters, INT8 Q7).",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.814,
        "3. SRCNN Original: Original CNN baseline architecture 1 -> 64 -> 32 -> 1 (8,129 parameters, FP32).",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.792,
        "4. ESPCN: Efficient Sub-Pixel Convolutional Neural Network (PixelShuffle upscale in LR space).",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.770,
        "5. FSRCNN: Fast Super-Resolution CNN with feature dimension shrinking and expanding layers.",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.748,
        "6. VDSR: Very Deep Super-Resolution network with 20 convolutional layers and residual learning.",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.726,
        "7. EDSR: Enhanced Deep Residual Networks (8 ResBlocks, 64 deep feature channels).",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.095,
        0.704,
        "8. SRGAN: Super-Resolution Generative Adversarial Network optimized for perceptual quality.",
        ha="left",
        va="top",
        fontsize=8.3,
        color="#1a3a5c",
        fontfamily="DejaVu Sans",
    )

    # TOC Header
    fig.text(
        0.08,
        0.660,
        "REPORT OUTLINE (TABLE OF CONTENTS):",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#222222",
        fontfamily="DejaVu Sans",
    )

    toc_items = [
        ("sec", "I. DATASET: SUB_NIH (NIH ChestX-ray14 - 1,750 Medical Images)", None, 0.620),
        ("item", "     • Magnification Factor: Scale 2x", 2, 0.584),
        ("item", "     • Magnification Factor: Scale 3x", 3, 0.548),
        ("item", "     • Magnification Factor: Scale 4x", 4, 0.512),
        ("sec", "II. DATASET: SUB_CHEST (Chest X-ray Clinical - 450 Clinical Images)", None, 0.460),
        ("item", "     • Magnification Factor: Scale 2x", 5, 0.424),
        ("item", "     • Magnification Factor: Scale 3x", 6, 0.388),
        ("item", "     • Magnification Factor: Scale 4x", 7, 0.352),
    ]

    for itype, text, page, y in toc_items:
        if itype == "sec":
            fig.text(0.08, y, text, ha="left", va="center", fontsize=9.8, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")
        else:
            fig.text(0.11, y, text, ha="left", va="center", fontsize=9.0, color="#333333", fontfamily="DejaVu Sans")
            fig.lines.append(
                plt.Line2D([0.52, 0.85], [y, y], transform=fig.transFigure, color="#bbbbbb", linestyle=":", linewidth=1.0)
            )
            fig.text(0.89, y, f"Page {page}", ha="right", va="center", fontsize=9.0, fontweight="bold", color="#1a3a5c", fontfamily="DejaVu Sans")

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


def generate_all_software_charts_and_pdf_en():
    """Execute generation of 6 English charts and publication-grade PDF report."""
    print("Starting generation of 6 English comparison charts and PDF report...")

    datasets = [
        ('sub_NIH', 'I. DATASET: SUB_NIH (NIH CHESTX-RAY14)'),
        ('sub_chest', 'II. DATASET: SUB_CHEST (CLINICAL CHEST X-RAY)'),
    ]

    charts = []
    for ds_key, ds_title in datasets:
        for s_idx, scale in enumerate([2, 3, 4], 1):
            model_data = {
                'Bicubic': load_bicubic(scale),
                'Compact SRCNN RTL': load_compact_srcnn_rtl(scale),
                'SRCNN Original': load_software('SRCNN', scale),
                'ESPCN':  load_software('ESPCN',  scale),
                'FSRCNN': load_software('FSRCNN', scale),
                'VDSR':   load_software('VDSR',   scale),
                'EDSR':   load_software('EDSR',   scale),
                'SRGAN':  load_software('SRGAN',  scale),
            }
            col_labels, rows = build_transposed_table(MODELS_LIST, model_data, ds_key)
            title = ds_title
            subtitle = f"{s_idx}. Magnification Factor: Scale {scale}x"
            out_png = OUT_DIR / f"nb1_{ds_key}_scale{scale}x_en.png"
            render_table(title, subtitle, col_labels, rows, out_png)
            charts.append(out_png)

    # Export English PDF Reports
    pdf_path_en = OUT_DIR / "performance_comparison_report_en.pdf"
    pdf_path_legacy_en = OUT_DIR / "so_sanh_hieu_nang_tong_hop_en.pdf"

    for target_pdf in [pdf_path_en, pdf_path_legacy_en]:
        with PdfPages(str(target_pdf)) as pdf:
            cover_fig = create_software_cover_page_en()
            pdf.savefig(cover_fig, bbox_inches="tight", dpi=140)
            plt.close(cover_fig)

            for p in charts:
                img = mpimg.imread(str(p))
                fig, ax = plt.subplots(figsize=(13.0, img.shape[0] / img.shape[1] * 13.0))
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight", dpi=140)
                plt.close(fig)

        # Attach interactive PDF Bookmarks
        doc = pymupdf.open(str(target_pdf))
        toc = [
            [1, "Cover Page & Table of Contents", 1],
            [1, "I. Dataset: sub_NIH", 2],
            [2, "Scale 2x", 2],
            [2, "Scale 3x", 3],
            [2, "Scale 4x", 4],
            [1, "II. Dataset: sub_chest", 5],
            [2, "Scale 2x", 5],
            [2, "Scale 3x", 6],
            [2, "Scale 4x", 7],
        ]
        doc.set_toc(toc)
        tmp = target_pdf.with_suffix(".tmp.pdf")
        doc.save(str(tmp))
        doc.close()
        tmp.replace(target_pdf)
        print(f"✓ Exported English PDF: {target_pdf.name} ({target_pdf.stat().st_size / 1024 / 1024:.2f} MB)")

    print("Completed English performance comparison suite successfully.")


if __name__ == "__main__":
    generate_all_software_charts_and_pdf_en()
