#!/usr/bin/env python3
"""
================================================================================
Merge Performance Comparison Reports Script (Software Suite)
Dự án: Siêu phân giải ảnh y tế AI-Based Image Super-Resolution
================================================================================
Mục đích:
  Kết hợp toàn bộ các biểu đồ so sánh hiệu năng phần mềm (phân biệt Compact SRCNN RTL
  và SRCNN Original) thành báo cáo PDF tổng hợp có bookmark điều hướng.
================================================================================
"""

from pathlib import Path
from render_software_comparison_charts import generate_all_software_charts_and_pdf

if __name__ == "__main__":
    generate_all_software_charts_and_pdf()
