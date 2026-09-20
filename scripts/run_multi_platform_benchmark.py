#!/usr/bin/env python3
"""
scripts/run_multi_platform_benchmark.py — Đo đạc thực nghiệm hiệu năng đa nền tảng (CPU vs GPU vs FPGA).
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = BASE_DIR / "core_project" / "benchmarks_reports" / "multi_platform" / "benchmark_multi_platform.py"

def main():
    if not BENCHMARK_SCRIPT.exists():
        print(f"Lỗi: Không tìm thấy script benchmark tại {BENCHMARK_SCRIPT}")
        sys.exit(1)

    cmd = [sys.executable, str(BENCHMARK_SCRIPT)] + sys.argv[1:]
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
