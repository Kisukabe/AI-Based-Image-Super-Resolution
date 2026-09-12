#!/usr/bin/env python3
"""
Root wrapper for generate_deliverables_package.py
Executes the final deliverables packager from core_project.
"""
import os
import sys
from pathlib import Path

target_script = Path(__file__).resolve().parent / "core_project" / "benchmarks_reports" / "deliverables_export" / "generate_deliverables_package.py"

if not target_script.exists():
    print(f"Error: Target script not found at {target_script}", file=sys.stderr)
    sys.exit(1)

import runpy
sys.argv[0] = str(target_script)
runpy.run_path(str(target_script), run_name="__main__")
