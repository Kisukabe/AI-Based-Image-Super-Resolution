#!/usr/bin/env python3
"""
scripts/run_generate_paper_figures.py — Tái tạo trọn bộ hình ảnh độ phân giải cao 300 DPI phục vụ bài báo IEEE:
  - Fig 7: Overlap-Tiling Boundary Ablation
  - Fig 8: Visual Comparison Matrix (2 hàng x 7 mô hình kèm chỉ số PSNR/SSIM/LPIPS)
  - Heatmaps: Bản đồ nhiệt sai số dư tuyệt đối ROI 128x128
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VISUAL_DIR = BASE_DIR / "core_project" / "benchmarks_reports" / "visual_analysis"
ENV = {**os.environ, "PYTHONPATH": str(BASE_DIR)}

def main():
    fig7_script = VISUAL_DIR / "generate_paper_fig7.py"
    fig8_script = VISUAL_DIR / "generate_paper_fig8.py"
    roi_script = VISUAL_DIR / "generate_roi_heatmaps.py"

    print("================================================================================")
    print("BẮT ĐẦU TẠO HỆ THỐNG HÌNH ẢNH BÀI BÁO KHOA HỌC IEEE (300 DPI)")
    print("================================================================================")

    # 1. Fig 7
    if fig7_script.exists():
        print("\n[1/3] Đang tạo Fig. 7: Overlap-Tiling Boundary Ablation...")
        subprocess.run([sys.executable, str(fig7_script)], cwd=str(BASE_DIR), env=ENV, check=True)
    else:
        print(f"Cảnh báo: Không tìm thấy {fig7_script}")

    # 2. Fig 8
    if fig8_script.exists():
        print("\n[2/3] Đang tạo Fig. 8: Ma trận so sánh định tính 2 hàng x 7 mô hình...")
        subprocess.run([sys.executable, str(fig8_script)], cwd=str(BASE_DIR), env=ENV, check=True)
    else:
        print(f"Cảnh báo: Không tìm thấy {fig8_script}")

    # 3. ROI Heatmaps
    if roi_script.exists():
        print("\n[3/3] Đang tạo ROI Heatmaps & Composite...")
        subprocess.run([sys.executable, str(roi_script)], cwd=str(BASE_DIR), env=ENV, check=True)
    else:
        print(f"Cảnh báo: Không tìm thấy {roi_script}")

    print("\n================================================================================")
    print("HOÀN THÀNH TOÀN BỘ HÌNH ẢNH BÀI BÁO (Lưu trữ tại figures/ và visual_analysis/)")
    print("================================================================================")

if __name__ == "__main__":
    main()
