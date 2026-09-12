#!/usr/bin/env python3
"""
Root wrapper for analyze_kaggle_benchmark.py
Project: AI-Based Medical Image Super-Resolution
Task 2.3: Statistical Analysis & 300 DPI Histogram Plotting
"""

import sys
from pathlib import Path

# Add script directory to sys.path and run main
SCRIPT_PATH = Path(__file__).resolve().parent / "core_project/benchmarks_reports/pynq_z2/analyze_kaggle_benchmark.py"

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
