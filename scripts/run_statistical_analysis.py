#!/usr/bin/env python3
"""
scripts/run_statistical_analysis.py — Phân tích thống kê 2.200 ảnh y tế từ PYNQ-Z2 & xuất bộ Histogram 300 DPI.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_SCRIPT = BASE_DIR / "core_project" / "benchmarks_reports" / "pynq_z2" / "analyze_kaggle_benchmark.py"

def main():
    if not ANALYSIS_SCRIPT.exists():
        print(f"Lỗi: Không tìm thấy script phân tích tại {ANALYSIS_SCRIPT}")
        sys.exit(1)

    cmd = [sys.executable, str(ANALYSIS_SCRIPT)] + sys.argv[1:]
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
