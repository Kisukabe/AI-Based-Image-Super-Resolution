#!/usr/bin/env bash
# build_pdf.sh — Shortcut 1 bước biên dịch Báo cáo Nhiệm vụ từ Typst sang PDF

set -e

TYPST_BIN="/opt/homebrew/bin/typst"
if ! command -v "$TYPST_BIN" &> /dev/null; then
    TYPST_BIN="typst"
fi

PAGES_DIR="requirements/report_pages"
OUTPUT_PDF="requirements/BAO_CAO_NHIEM_VU.pdf"
MAIN_TYP="requirements/BAO_CAO_NHIEM_VU.typ"

if [ "$1" == "watch" ]; then
    echo "Đang bật chế độ lắng nghe (Typst Live Watch)..."
    echo "Chỉnh sửa file trong requirements/report_pages/*.typ và nhấn Cmd+S để cập nhật tức thì."
    echo "Nhấn Ctrl+C để dừng."
    "$TYPST_BIN" watch "$MAIN_TYP" "$OUTPUT_PDF"
elif [ "$1" == "1" ] || [ "$1" == "2" ] || [ "$1" == "3" ]; then
    PAGE="$1"
    PAGE_TYP="$PAGES_DIR/trang_${PAGE}.typ"
    PAGE_PDF="$PAGES_DIR/trang_${PAGE}.pdf"
    PAGE_PNG="$PAGES_DIR/trang_${PAGE}.png"
    
    "$TYPST_BIN" compile "$PAGE_TYP" "$PAGE_PDF"
    "$TYPST_BIN" compile --format png --ppi 150 "$PAGE_TYP" "$PAGE_PNG"
    echo "[OK] Đã xuất trang $PAGE:"
    echo "     - PDF: $PAGE_PDF"
    echo "     - PNG (xem trước): $PAGE_PNG"
elif [ -n "$1" ]; then
    echo "Cách dùng:"
    echo "  ./build_pdf.sh         # Biên dịch toàn bộ ra BAO_CAO_NHIEM_VU.pdf (3 trang)"
    echo "  ./build_pdf.sh 1       # Biên dịch & xem trước riêng Trang 1 (PDF + ảnh PNG)"
    echo "  ./build_pdf.sh 2       # Biên dịch & xem trước riêng Trang 2 (PDF + ảnh PNG)"
    echo "  ./build_pdf.sh 3       # Biên dịch & xem trước riêng Trang 3 (PDF + ảnh PNG)"
    echo "  ./build_pdf.sh watch   # Tự động cập nhật tức thì khi bấm Cmd+S (<30ms)"
else
    "$TYPST_BIN" compile "$MAIN_TYP" "$OUTPUT_PDF"
    echo "[OK] Đã xuất bản thành công: $OUTPUT_PDF (3 trang)"
fi
