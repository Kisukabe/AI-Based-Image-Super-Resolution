#!/usr/bin/env python3
"""
================================================================================
Task 2.3: PYNQ-Z2 & RTL Benchmark Statistical Analysis & 300 DPI Histograms
Project: AI-Based Medical Image Super-Resolution
Architecture: Compact SRCNN RTL (1 -> 16 -> 8 -> 1, 1,649 params, INT8 Q7)
================================================================================
Purpose:
  1. Parse benchmark JSON files from PYNQ-Z2 / RTL hardware inference across 
     2,200 medical images (sub_NIH: 1,750 images, sub_chest: 450 images).
  2. Compute rigorous statistical indicators:
     - Mean ± Std for FPGA PSNR, Bicubic PSNR, and PSNR Gain (dB).
     - Percentage (%) of images with PSNR Gain > 0.
     - Mean ± Std for FPGA SSIM, Bicubic SSIM, and SSIM Gain.
     - Breakdown by dataset (sub_NIH vs sub_chest) and overall.
  3. Render publication-quality 300 DPI Histograms:
     - hist_psnr_gain.png: Distribution of PSNR Gain with zero-reference and KDE.
     - hist_ssim.png: Comparison distribution of SSIM (Bicubic vs FPGA).
     - hist_psnr_comparison.png: Absolute PSNR distributions.
     - multiscale_gain_comparison.png: Multi-scale (2x, 3x, 4x) distributions.
  4. Export JSON statistical report and summary tables.
================================================================================
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ANALYZE_KAGGLE_BENCHMARK] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_kaggle_benchmark")

# ── Scientific Palette Constants (300 DPI Standard) ──────────────────────────
COLOR_FPGA = "#1a3a5c"       # Deep Navy Blue (Compact SRCNN RTL)
COLOR_BICUBIC = "#c0392b"    # Brick Red (Bicubic Baseline)
COLOR_GAIN_POS = "#27ae60"   # Forest Green (Positive Gain)
COLOR_GAIN_NEG = "#e74c3c"   # Red (Negative Gain)
COLOR_GRID = "#e2e8f0"       # Subtle slate grid
COLOR_TEXT = "#1e293b"       # Charcoal dark text
BG_COLOR = "#f8fafc"         # Clean Off-White background
DPI_PUBLICATION = 300


def calculate_statistics(values: np.ndarray) -> Dict[str, float]:
    """Calculate descriptive statistics without LaTeX notation."""
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def parse_benchmark_json(json_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse hardware benchmark JSON file and extract valid image records."""
    if not json_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = [r for r in data if r.get("status") == "ok"]
        summary = {}
    elif isinstance(data, dict):
        summary = data.get("summary", {})
        records = [r for r in data.get("per_image_results", []) if r.get("status") == "ok"]
    else:
        raise ValueError(f"Unrecognized benchmark JSON format in {json_path}")

    logger.info(f"Loaded {len(records)} valid records from {json_path.name}")
    return summary, records


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics overall and per dataset."""
    groups = {
        "overall": records,
        "sub_NIH": [r for r in records if r.get("dataset") == "sub_NIH"],
        "sub_chest": [r for r in records if r.get("dataset") == "sub_chest"],
    }

    analysis_result = {}
    for group_name, recs in groups.items():
        if not recs:
            continue
        n_count = len(recs)
        psnr_fpga = np.array([float(r.get("psnr_fpga_db", r.get("psnr_model_db", 0.0))) for r in recs])
        psnr_bicubic = np.array([float(r.get("psnr_bicubic_db", 0.0)) for r in recs])
        psnr_gain = psnr_fpga - psnr_bicubic

        ssim_fpga = np.array([float(r.get("ssim_fpga", r.get("ssim_model", 0.0))) for r in recs])
        ssim_bicubic = np.array([float(r.get("ssim_bicubic", 0.0)) for r in recs])
        ssim_gain = ssim_fpga - ssim_bicubic

        latency = np.array([float(r.get("latency_ms", 0.0)) for r in recs if r.get("latency_ms") is not None])

        pos_gain_count = int(np.sum(psnr_gain > 0))
        pos_gain_pct = (pos_gain_count / n_count) * 100.0 if n_count > 0 else 0.0

        pos_ssim_gain_count = int(np.sum(ssim_gain > 0))
        pos_ssim_gain_pct = (pos_ssim_gain_count / n_count) * 100.0 if n_count > 0 else 0.0

        analysis_result[group_name] = {
            "sample_size": n_count,
            "psnr_fpga_db": calculate_statistics(psnr_fpga),
            "psnr_bicubic_db": calculate_statistics(psnr_bicubic),
            "psnr_gain_db": calculate_statistics(psnr_gain),
            "psnr_gain_positive_count": pos_gain_count,
            "psnr_gain_positive_percentage": round(pos_gain_pct, 2),
            "ssim_fpga": calculate_statistics(ssim_fpga),
            "ssim_bicubic": calculate_statistics(ssim_bicubic),
            "ssim_gain": calculate_statistics(ssim_gain),
            "ssim_gain_positive_count": pos_ssim_gain_count,
            "ssim_gain_positive_percentage": round(pos_ssim_gain_pct, 2),
            "latency_ms": calculate_statistics(latency) if len(latency) > 0 else None,
            "throughput_fps": round(1000.0 / float(np.mean(latency)), 2) if len(latency) > 0 and np.mean(latency) > 0 else 0.0,
        }

    return analysis_result


def plot_psnr_gain_histogram(records: List[Dict[str, Any]], out_path: Path, scale: int = 2):
    """Plot publication-grade 300 DPI histogram of PSNR Gain."""
    psnr_fpga = np.array([float(r.get("psnr_fpga_db", r.get("psnr_model_db", 0.0))) for r in records])
    psnr_bicubic = np.array([float(r.get("psnr_bicubic_db", 0.0)) for r in records])
    psnr_gain = psnr_fpga - psnr_bicubic

    mean_gain = np.mean(psnr_gain)
    std_gain = np.std(psnr_gain)
    median_gain = np.median(psnr_gain)
    pos_count = int(np.sum(psnr_gain > 0))
    pos_pct = (pos_count / len(psnr_gain)) * 100.0

    fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=DPI_PUBLICATION)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("#ffffff")

    # Histogram bins
    bins = np.linspace(np.min(psnr_gain), np.max(psnr_gain), 55)
    counts, edges, patches_list = ax.hist(
        psnr_gain,
        bins=bins,
        density=True,
        edgecolor="#ffffff",
        linewidth=0.8,
        alpha=0.85,
    )

    # Color code bars based on positive vs negative gain
    for patch, edge in zip(patches_list, edges[:-1]):
        if edge >= 0:
            patch.set_facecolor(COLOR_GAIN_POS)
        else:
            patch.set_facecolor(COLOR_FPGA)

    # KDE line
    if len(psnr_gain) > 1 and np.var(psnr_gain) > 1e-6:
        kde = gaussian_kde(psnr_gain)
        x_grid = np.linspace(np.min(psnr_gain) - 0.2, np.max(psnr_gain) + 0.2, 300)
        ax.plot(x_grid, kde(x_grid), color="#0f172a", linewidth=2.0, label="Kernel Density (KDE)")

    # Reference vertical lines
    ax.axvline(0.0, color="#ef4444", linestyle="--", linewidth=1.8, label="Zero Gain Threshold (0.0 dB)")
    ax.axvline(mean_gain, color="#2563eb", linestyle="-", linewidth=2.0, label=f"Mean Gain: {mean_gain:+.4f} dB")
    ax.axvline(median_gain, color="#8b5cf6", linestyle=":", linewidth=1.8, label=f"Median Gain: {median_gain:+.4f} dB")

    # Styling
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.set_title(
        f"Distribution of PSNR Gain (Compact SRCNN RTL vs. Bicubic Baseline) - Scale {scale}x",
        fontsize=12.5,
        fontweight="bold",
        color=COLOR_TEXT,
        pad=14,
        fontfamily="DejaVu Sans",
    )
    ax.set_xlabel("PSNR Gain (dB) = PSNR_FPGA - PSNR_Bicubic", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Probability Density", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)

    # Statistical Callout Box
    stat_box_text = (
        f"Sample Size (N): {len(psnr_gain):,}\n"
        f"Mean Gain (μ): {mean_gain:+.4f} dB\n"
        f"Std Dev (σ): ±{std_gain:.4f} dB\n"
        f"Median: {median_gain:+.4f} dB\n"
        f"Min / Max: {np.min(psnr_gain):+.2f} / {np.max(psnr_gain):+.2f} dB\n"
        f"Gain > 0 dB: {pos_count:,} ({pos_pct:.2f}%)"
    )
    ax.text(
        0.03,
        0.95,
        stat_box_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.2),
        fontfamily="DejaVu Sans",
    )

    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI_PUBLICATION, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    logger.info(f"Saved 300 DPI Histogram: {out_path.name}")


def plot_ssim_comparison_histogram(records: List[Dict[str, Any]], out_path: Path, scale: int = 2):
    """Plot publication-grade 300 DPI comparative histogram of SSIM."""
    ssim_fpga = np.array([float(r.get("ssim_fpga", r.get("ssim_model", 0.0))) for r in records])
    ssim_bicubic = np.array([float(r.get("ssim_bicubic", 0.0)) for r in records])

    mean_fpga = np.mean(ssim_fpga)
    std_fpga = np.std(ssim_fpga)
    mean_bicubic = np.mean(ssim_bicubic)
    std_bicubic = np.std(ssim_bicubic)

    fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=DPI_PUBLICATION)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("#ffffff")

    bins = np.linspace(min(np.min(ssim_fpga), np.min(ssim_bicubic)), 1.0, 50)

    # Bicubic Histogram
    ax.hist(
        ssim_bicubic,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLOR_BICUBIC,
        edgecolor="#ffffff",
        linewidth=0.6,
        label=f"Bicubic Baseline (μ = {mean_bicubic:.4f} ± {std_bicubic:.4f})",
    )

    # FPGA Histogram
    ax.hist(
        ssim_fpga,
        bins=bins,
        density=True,
        alpha=0.65,
        color=COLOR_FPGA,
        edgecolor="#ffffff",
        linewidth=0.6,
        label=f"Compact SRCNN RTL (μ = {mean_fpga:.4f} ± {std_fpga:.4f})",
    )

    # Reference Mean Lines
    ax.axvline(mean_bicubic, color=COLOR_BICUBIC, linestyle="--", linewidth=1.8)
    ax.axvline(mean_fpga, color=COLOR_FPGA, linestyle="-", linewidth=2.0)

    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.set_title(
        f"Structural Similarity (SSIM) Distribution: Compact SRCNN RTL vs. Bicubic - Scale {scale}x",
        fontsize=12.5,
        fontweight="bold",
        color=COLOR_TEXT,
        pad=14,
        fontfamily="DejaVu Sans",
    )
    ax.set_xlabel("Structural Similarity Index (SSIM)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Probability Density", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)

    stat_box_text = (
        f"Evaluation Testset N = {len(records):,}\n"
        f"FPGA SSIM: {mean_fpga:.4f} ± {std_fpga:.4f}\n"
        f"Bicubic SSIM: {mean_bicubic:.4f} ± {std_bicubic:.4f}\n"
        f"Mean SSIM Delta: {mean_fpga - mean_bicubic:+.4f}"
    )
    ax.text(
        0.03,
        0.95,
        stat_box_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.2),
        fontfamily="DejaVu Sans",
    )

    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI_PUBLICATION, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    logger.info(f"Saved 300 DPI Histogram: {out_path.name}")


def plot_psnr_comparison_histogram(records: List[Dict[str, Any]], out_path: Path, scale: int = 2):
    """Plot publication-grade 300 DPI comparative histogram of PSNR."""
    psnr_fpga = np.array([float(r.get("psnr_fpga_db", r.get("psnr_model_db", 0.0))) for r in records])
    psnr_bicubic = np.array([float(r.get("psnr_bicubic_db", 0.0)) for r in records])

    mean_fpga = np.mean(psnr_fpga)
    std_fpga = np.std(psnr_fpga)
    mean_bicubic = np.mean(psnr_bicubic)
    std_bicubic = np.std(psnr_bicubic)

    fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=DPI_PUBLICATION)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("#ffffff")

    bins = np.linspace(min(np.min(psnr_fpga), np.min(psnr_bicubic)), max(np.max(psnr_fpga), np.max(psnr_bicubic)), 50)

    ax.hist(
        psnr_bicubic,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLOR_BICUBIC,
        edgecolor="#ffffff",
        linewidth=0.6,
        label=f"Bicubic Baseline (μ = {mean_bicubic:.2f} ± {std_bicubic:.2f} dB)",
    )
    ax.hist(
        psnr_fpga,
        bins=bins,
        density=True,
        alpha=0.65,
        color=COLOR_FPGA,
        edgecolor="#ffffff",
        linewidth=0.6,
        label=f"Compact SRCNN RTL (μ = {mean_fpga:.2f} ± {std_fpga:.2f} dB)",
    )

    ax.axvline(mean_bicubic, color=COLOR_BICUBIC, linestyle="--", linewidth=1.8)
    ax.axvline(mean_fpga, color=COLOR_FPGA, linestyle="-", linewidth=2.0)

    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.set_title(
        f"Peak Signal-to-Noise Ratio (PSNR) Distribution: Compact SRCNN RTL vs. Bicubic - Scale {scale}x",
        fontsize=12.5,
        fontweight="bold",
        color=COLOR_TEXT,
        pad=14,
        fontfamily="DejaVu Sans",
    )
    ax.set_xlabel("PSNR (dB)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Probability Density", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)

    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI_PUBLICATION, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    logger.info(f"Saved 300 DPI Histogram: {out_path.name}")


def print_statistical_summary(analysis: Dict[str, Any], scale: int):
    """Print structured technical report to console."""
    print("=" * 82)
    print(f"STATISTICAL BENCHMARK ANALYSIS REPORT: COMPACT SRCNN RTL (SCALE {scale}x)")
    print("=" * 82)
    for grp_key in ["overall", "sub_NIH", "sub_chest"]:
        data = analysis.get(grp_key)
        if not data:
            continue
        print(f"\n[GROUP: {grp_key.upper()} - Sample Size N = {data['sample_size']:,}]")
        print("-" * 82)
        print(f"  • FPGA PSNR        : {data['psnr_fpga_db']['mean']:.4f} ± {data['psnr_fpga_db']['std']:.4f} dB  (Median: {data['psnr_fpga_db']['median']:.4f} dB)")
        print(f"  • Bicubic PSNR     : {data['psnr_bicubic_db']['mean']:.4f} ± {data['psnr_bicubic_db']['std']:.4f} dB  (Median: {data['psnr_bicubic_db']['median']:.4f} dB)")
        print(f"  • PSNR Gain        : {data['psnr_gain_db']['mean']:+.4f} ± {data['psnr_gain_db']['std']:.4f} dB  (Range: [{data['psnr_gain_db']['min']:+.2f}, {data['psnr_gain_db']['max']:+.2f}] dB)")
        print(f"  • Images Gain > 0  : {data['psnr_gain_positive_count']:,} / {data['sample_size']:,} ({data['psnr_gain_positive_percentage']}%)")
        print(f"  • FPGA SSIM        : {data['ssim_fpga']['mean']:.4f} ± {data['ssim_fpga']['std']:.4f}  (Median: {data['ssim_fpga']['median']:.4f})")
        print(f"  • Bicubic SSIM     : {data['ssim_bicubic']['mean']:.4f} ± {data['ssim_bicubic']['std']:.4f}  (Median: {data['ssim_bicubic']['median']:.4f})")
        print(f"  • SSIM Gain        : {data['ssim_gain']['mean']:+.4f} ± {data['ssim_gain']['std']:.4f}")
        if data.get("latency_ms"):
            print(f"  • Inference Latency: {data['latency_ms']['mean']:.2f} ± {data['latency_ms']['std']:.2f} ms  | Throughput: {data['throughput_fps']:.1f} FPS")
    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Benchmark Analysis & 300 DPI Histogram Generator for PYNQ-Z2 / RTL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="core_project/benchmarks_reports/pynq_z2/compact_srcnn_rtl_2x_benchmark.json",
        help="Path to input benchmark JSON file (or 'all' to process 2x, 3x, 4x)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="core_project/benchmarks_reports/pynq_z2/figures_dpi300",
        help="Directory to save 300 DPI figures",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default="core_project/benchmarks_reports/pynq_z2/pynq_z2_statistical_analysis.json",
        help="Path to save statistical analysis JSON report",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deliverables figure folder root
    root_fig_dir = project_root / "figures_dpi300"
    root_fig_dir.mkdir(parents=True, exist_ok=True)

    json_out_path = Path(args.export_json)
    if not json_out_path.is_absolute():
        json_out_path = project_root / json_out_path
    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    scales = [2, 3, 4] if args.input.lower() == "all" else [2]
    all_scale_results = {}

    for scale in scales:
        if args.input.lower() == "all":
            json_file = project_root / f"core_project/benchmarks_reports/pynq_z2/compact_srcnn_rtl_{scale}x_benchmark.json"
        else:
            json_file = Path(args.input)
            if not json_file.is_absolute():
                json_file = project_root / json_file

        if not json_file.exists():
            logger.warning(f"File {json_file} does not exist, skipping.")
            continue

        _, records = parse_benchmark_json(json_file)
        analysis = analyze_records(records)
        all_scale_results[f"scale_{scale}x"] = analysis

        # Display terminal report
        print_statistical_summary(analysis, scale=scale)

        # Plot 300 DPI figures
        hist_psnr_gain_path = out_dir / f"hist_psnr_gain_scale{scale}x.png"
        plot_psnr_gain_histogram(records, hist_psnr_gain_path, scale=scale)

        hist_ssim_path = out_dir / f"hist_ssim_scale{scale}x.png"
        plot_ssim_comparison_histogram(records, hist_ssim_path, scale=scale)

        hist_psnr_comp_path = out_dir / f"hist_psnr_comparison_scale{scale}x.png"
        plot_psnr_comparison_histogram(records, hist_psnr_comp_path, scale=scale)

        # Also copy scale 2x primary deliverable names as specified in Request2.txt
        if scale == 2:
            primary_psnr_gain = out_dir / "hist_psnr_gain.png"
            primary_ssim = out_dir / "hist_ssim.png"
            import shutil
            shutil.copyfile(hist_psnr_gain_path, primary_psnr_gain)
            shutil.copyfile(hist_ssim_path, primary_ssim)
            # Copy to root figures_dpi300/
            shutil.copyfile(hist_psnr_gain_path, root_fig_dir / "hist_psnr_gain.png")
            shutil.copyfile(hist_ssim_path, root_fig_dir / "hist_ssim.png")
            logger.info(f"Exported primary figures to {root_fig_dir}")

    # Export statistical analysis JSON
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(all_scale_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved complete statistical analysis report to {json_out_path}")


if __name__ == "__main__":
    main()
