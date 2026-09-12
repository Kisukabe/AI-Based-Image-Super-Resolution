#!/usr/bin/env python3
"""
Root wrapper for generating Fig. 8 Qualitative Visual Comparison (IEEE GTSD 2026).
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "core_project"))

from core_project.benchmarks_reports.visual_analysis.generate_paper_fig8 import main

if __name__ == "__main__":
    main()
