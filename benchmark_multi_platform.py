#!/usr/bin/env python3
"""
Root wrapper for benchmark_multi_platform.py
Executes the multi-platform benchmark from core_project.
"""
import os
import sys
from pathlib import Path

target_script = Path(__file__).resolve().parent / "core_project" / "benchmarks_reports" / "multi_platform" / "benchmark_multi_platform.py"

if not target_script.exists():
    print(f"Error: Target script not found at {target_script}", file=sys.stderr)
    sys.exit(1)

# Execute target script with forwarded arguments
import runpy
sys.argv[0] = str(target_script)
runpy.run_path(str(target_script), run_name="__main__")
