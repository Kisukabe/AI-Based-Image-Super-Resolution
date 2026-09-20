#!/usr/bin/env python3
"""
scripts/run_dv_verification.py — Chạy kiểm chứng vi mạch Bit-Accurate Design Verification Scoreboard.
Target: Xilinx Zynq-7020 (Compact SRCNN 1.649 tham số).
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DV_SCRIPT = BASE_DIR / "core_project" / "hardware_fpga" / "dv_verification" / "dv_scoreboard_check.py"

def main():
    if not DV_SCRIPT.exists():
        print(f"Lỗi: Không tìm thấy script kiểm chứng tại {DV_SCRIPT}")
        sys.exit(1)

    cmd = [sys.executable, str(DV_SCRIPT)] + sys.argv[1:]
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
