#!/usr/bin/env bash
# build_pdf.sh — Shortcut 1 bước biên dịch Báo cáo Nhiệm vụ từ Markdown sang PDF

set -e

if [ "$1" == "watch" ]; then
    python3 scripts/build_report.py --watch
elif [ "$1" == "1" ] || [ "$1" == "2" ] || [ "$1" == "3" ]; then
    python3 scripts/build_report.py --page "$1"
elif [ -n "$1" ]; then
    echo "Cách dùng:"
    echo "  ./build_pdf.sh         # Biên dịch toàn bộ ra BAO_CAO_NHIEM_VU.pdf (3 trang)"
    echo "  ./build_pdf.sh 1       # Biên dịch & xem trước riêng Trang 1 (PDF + ảnh PNG)"
    echo "  ./build_pdf.sh 2       # Biên dịch & xem trước riêng Trang 2 (PDF + ảnh PNG)"
    echo "  ./build_pdf.sh 3       # Biên dịch & xem trước riêng Trang 3 (PDF + ảnh PNG)"
    echo "  ./build_pdf.sh watch   # Tự động cập nhật khi lưu file (Cmd+S)"
else
    python3 scripts/build_report.py
fi
