#!/usr/bin/env python3
"""
build_report.py — Hệ thống biên dịch Báo cáo Nhiệm vụ từ Markdown sang PDF chuẩn xuất bản.

Tính năng:
  1. Biên dịch toàn bộ: python scripts/build_report.py
     -> Tự động gộp 3 trang md, sinh requirements/BAO_CAO_NHIEM_VU.pdf (chuẩn 3 trang)
     -> Đồng bộ requirements/BAO_CAO_NHIEM_VU.md
  2. Biên dịch từng trang: python scripts/build_report.py --page 1 (hoặc 2, 3)
     -> Sinh requirements/report_pages/trang_X.pdf và file ảnh trang_X.png để xem trước
  3. Lắng nghe tự động (Watch mode): python scripts/build_report.py --watch
     -> Mỗi khi sửa file .md và bấm Cmd+S, tự động biên dịch lại trong 0.05s
"""

import sys
import os
import re
import time
import argparse
import subprocess
from pathlib import Path

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = BASE_DIR / "requirements" / "report_pages"
OUTPUT_PDF = BASE_DIR / "requirements" / "BAO_CAO_NHIEM_VU.pdf"
COMBINED_MD = BASE_DIR / "requirements" / "BAO_CAO_NHIEM_VU.md"
GENERATED_TYP = BASE_DIR / "requirements" / "BAO_CAO_NHIEM_VU.typ"

TYPST_HEADER = """#set page(
  paper: "a4",
  margin: (top: 1.6cm, bottom: 1.6cm, left: 1.8cm, right: 1.8cm),
  header: context {
    if counter(page).get().first() > 1 [
      #grid(
        columns: (1fr, auto),
        align(left)[#text(size: 8pt, fill: rgb("#718096"), font: "Arial")[Báo Cáo Nhiệm Vụ 1 & 2 | AI-Based Medical Image Super-Resolution]],
        align(right)[#text(size: 8pt, fill: rgb("#718096"), font: "Arial")[FPGA Xilinx Zynq-7020]]
      )
      #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e0"))
    ]
  },
  footer: context [
    #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e0"))
    #v(3pt)
    #grid(
      columns: (1fr, auto),
      align(left)[#text(size: 8pt, fill: rgb("#a0aec0"), font: "Arial")[Hệ thống Kiểm thử & Đo đạc Tự động Antigravity IDE]],
      align(right)[#text(size: 8.5pt, fill: rgb("#4a5568"), font: "Arial", weight: "medium")[Trang #counter(page).display("1 / 1", both: true)]]
    )
  ]
)

#set text(
  font: "Arial",
  size: 9.6pt,
  lang: "vi",
  fill: rgb("#2d3748")
)
#set par(justify: true, leading: 0.55em)

// Styling definitions
#let brand-color = rgb("#1a365d")
#let brand-accent = rgb("#2b6cb0")
#let border-color = rgb("#e2e8f0")
#let bg-zebra = rgb("#f7fafc")
#let badge-pass(body) = box(
  fill: rgb("#ebf8ff"),
  inset: (x: 4pt, y: 2pt),
  radius: 3pt,
  stroke: 0.5pt + rgb("#bee3f8"),
  text(size: 7.2pt, weight: "bold", fill: rgb("#2b6cb0"), body)
)
#set table(inset: (x: 5pt, y: 3.3pt))
"""

def parse_markdown_to_typst(md_text: str, is_first_page: bool = False) -> str:
    """Chuyển đổi cú pháp Markdown sang Typst chất lượng cao với định dạng bảng tối ưu."""
    lines = md_text.splitlines()
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Xử lý khối tiêu đề chính đầu tài liệu (Trang 1)
        if is_first_page and line.strip().startswith("# BÁO CÁO NHIỆM VỤ"):
            # Bỏ qua các dòng tiêu đề và phân cách
            i += 1
            subtitle = ""
            while i < n and (lines[i].strip().startswith("# Dự án:") or lines[i].strip() == "---" or lines[i].strip() == ""):
                if lines[i].strip().startswith("# Dự án:"):
                    subtitle = lines[i].strip()[len("# Dự án:"):].strip()
                i += 1
            out.append(f"""#align(center)[
  #v(0.3cm)
  #text(size: 19pt, weight: "bold", fill: brand-color)[BÁO CÁO NHIỆM VỤ]
  #v(5pt)
  #text(size: 11pt, weight: "semibold", fill: brand-accent)[
    Dự án: {subtitle}
  ]
  #v(0.5cm)
  #line(length: 100%, stroke: 1.5pt + brand-accent)
  #v(0.3cm)
]""")
            continue

        # Tiêu đề PHẦN I / PHẦN II
        if line.startswith("# PHẦN"):
            part_title = line.strip("#").strip()
            out.append(f'\n#text(size: 13.5pt, weight: "bold", fill: brand-color)[{part_title}] \\')
            i += 1
            continue

        # Phụ đề chữ in hoa dưới Phần
        if line.startswith("## AI MODELING") or line.startswith("## BIT-ACCURATE"):
            sub_part = line.strip("#").strip()
            out.append(f'#text(size: 10.5pt, weight: "bold", fill: brand-accent)[{sub_part}] \\')
            i += 1
            continue

        # Ghi chú căn cứ
        if line.startswith("*(Căn cứ theo"):
            note = line.strip()
            out.append(f'#text(size: 8.5pt, style: "italic", fill: rgb("#718096"))[{note}]\n#v(0.15cm)')
            i += 1
            continue

        # Đường kẻ phân cách ngang
        if line.strip() == "---":
            # Trong Typst chúng ta dùng pagebreak giữa các trang
            i += 1
            continue

        # Heading cấp 3 (### 1. ... -> == 1. ...)
        if line.startswith("### "):
            h3_text = line[4:].strip()
            out.append(f"\n== {h3_text}")
            i += 1
            continue

        # Khối code công thức (```text ... ```)
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1  # Bỏ dòng đóng ```
            code_body = "\n".join(code_lines).strip()
            # Dùng raw string trong typst
            out.append(f"""#align(center)[
  #rect(fill: bg-zebra, stroke: 0.5pt + border-color, inset: 6pt, radius: 4pt)[
    #text(weight: "bold", size: 10pt)[`{code_body}`]
  ]
]""")
            continue

        # Bảng biểu Markdown (| ... |)
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                typst_table = convert_table(table_lines)
                out.append(typst_table)
            continue

        # Chuyển đổi bullet points và inline formatting
        formatted_line = convert_inline(line)
        out.append(formatted_line)
        i += 1

    return "\n".join(out)

def convert_inline(text: str) -> str:
    """Xử lý định dạng inline: bold, code, math symbols."""
    # Chuyển **text** thành *text* trong Typst (Typst dùng * cho bold)
    # Lưu ý không thay thế trong code span `...`
    parts = re.split(r"(`[^`]*`)", text)
    converted_parts = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            converted_parts.append(part)
        else:
            p = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", part)
            # Thay thế ký hiệu -> thành $->$ để hiển thị đẹp
            p = p.replace(" -> ", " $->$ ")
            converted_parts.append(p)
    return "".join(converted_parts)

def convert_table(table_lines: list[str]) -> str:
    """Chuyển bảng Markdown thành bảng Typst với tỉ lệ cột chuẩn xác."""
    header_row = [c.strip() for c in table_lines[0].strip("|").split("|")]
    data_rows = []
    for l in table_lines[2:]:  # Bỏ dòng separator thứ 2
        cells = [c.strip() for c in l.strip("|").split("|")]
        data_rows.append(cells)

    num_cols = len(header_row)

    # Nhận diện loại bảng dựa trên tiêu đề
    if "Mã" in header_row[0] and num_cols == 4:
        # Bảng sản phẩm bàn giao (Deliverables)
        cols_spec = "(38pt, 160pt, 205pt, 67pt)"
        align_spec = "(center + horizon, left + horizon, left + horizon, center + horizon)"
        fill_spec = "fill: (col, row) => if row == 0 { brand-color } else if calc.odd(row) { bg-zebra } else { none },"
        
        cells_typst = []
        for c in header_row:
            cells_typst.append(f"[*{convert_inline(c)}*]")
        
        for row in data_rows:
            for idx, c in enumerate(row):
                if "ĐÃ HOÀN THÀNH" in c:
                    cells_typst.append("[#badge-pass([ĐÃ HOÀN THÀNH])]")
                elif idx == 0:
                    cells_typst.append(f"[{c}]")
                else:
                    cells_typst.append(f"[{convert_inline(c)}]")
        
        return f"""#align(center)[
  #table(
    columns: {cols_spec},
    align: {align_spec},
    {fill_spec}
    stroke: (x, y) => if y == 0 {{ (bottom: 1.5pt + brand-accent) }} else {{ 0.5pt + border-color }},
    table.header({', '.join(cells_typst[:4])}),
    {', '.join(cells_typst[4:])}
  )
]"""

    elif "Nền tảng" in header_row[0] and num_cols == 6:
        # Bảng so sánh đa nền tảng (Benchmark)
        cols_spec = "(130pt, 80pt, 85pt, 55pt, 55pt, 65pt)"
        align_spec = "(left + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon)"
        fill_spec = "fill: (col, row) => if row == 0 { brand-color } else if row == 1 { rgb(\"#ebf8ff\") } else if calc.odd(row) { bg-zebra } else { none },"
        
        cells_typst = []
        for c in header_row:
            cells_typst.append(f"[*{convert_inline(c)}*]")
        
        for row in data_rows:
            for c in row:
                cells_typst.append(f"[{convert_inline(c)}]")
        
        return f"""#align(center)[
  #table(
    columns: {cols_spec},
    align: {align_spec},
    {fill_spec}
    stroke: (x, y) => if y == 0 {{ (bottom: 1.5pt + brand-accent) }} else {{ 0.5pt + border-color }},
    table.header({', '.join(cells_typst[:6])}),
    {', '.join(cells_typst[6:])}
  )
]"""

    else:
        # Bảng thông thường
        cols_spec = f"({', '.join(['1fr'] * num_cols)})"
        align_spec = "center + horizon"
        cells_typst = []
        for c in header_row:
            cells_typst.append(f"[*{convert_inline(c)}*]")
        for row in data_rows:
            for c in row:
                cells_typst.append(f"[{convert_inline(c)}]")
        return f"""#align(center)[
  #table(
    columns: {cols_spec},
    align: {align_spec},
    table.header({', '.join(cells_typst[:num_cols])}),
    {', '.join(cells_typst[num_cols:])}
  )
]"""

def compile_all():
    """Biên dịch cả 3 trang Markdown thành file PDF hoàn chỉnh và đồng bộ BAO_CAO_NHIEM_VU.md."""
    t0 = time.time()
    p1_file = PAGES_DIR / "trang_1.md"
    p2_file = PAGES_DIR / "trang_2.md"
    p3_file = PAGES_DIR / "trang_3.md"

    if not (p1_file.exists() and p2_file.exists() and p3_file.exists()):
        print("Lỗi: Không tìm thấy đủ 3 file trang trong requirements/report_pages/")
        sys.exit(1)

    t1_md = p1_file.read_text(encoding="utf-8")
    t2_md = p2_file.read_text(encoding="utf-8")
    t3_md = p3_file.read_text(encoding="utf-8")

    # 1. Đồng bộ sang BAO_CAO_NHIEM_VU.md
    combined = f"{t1_md.strip()}\n\n---\n\n{t2_md.strip()}\n\n---\n\n{t3_md.strip()}\n"
    COMBINED_MD.write_text(combined, encoding="utf-8")

    # 2. Chuyển sang Typst
    t1_typ = parse_markdown_to_typst(t1_md, is_first_page=True)
    t2_typ = parse_markdown_to_typst(t2_md, is_first_page=False)
    t3_typ = parse_markdown_to_typst(t3_md, is_first_page=False)

    full_typst = f"""{TYPST_HEADER}

// Trang 1
{t1_typ}

#pagebreak()

// Trang 2
{t2_typ}

#pagebreak()

// Trang 3
{t3_typ}
"""
    GENERATED_TYP.write_text(full_typst, encoding="utf-8")

    # 3. Biên dịch bằng Typst CLI
    cmd = ["/opt/homebrew/bin/typst", "compile", str(GENERATED_TYP), str(OUTPUT_PDF)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Lỗi biên dịch Typst:")
        print(res.stderr)
        sys.exit(1)

    elapsed = (time.time() - t0) * 1000
    print(f"[OK] Đã xuất bản thành công: {OUTPUT_PDF} ({elapsed:.1f} ms)")
    print(f"[OK] Đã đồng bộ file Markdown: {COMBINED_MD}")

def compile_single_page(page_num: int):
    """Biên dịch riêng 1 trang sang PDF và PNG để xem trước tức thì."""
    target_file = PAGES_DIR / f"trang_{page_num}.md"
    if not target_file.exists():
        print(f"Lỗi: Không tìm thấy {target_file}")
        sys.exit(1)

    md_content = target_file.read_text(encoding="utf-8")
    page_typst = parse_markdown_to_typst(md_content, is_first_page=(page_num == 1))

    temp_typ = PAGES_DIR / f"temp_trang_{page_num}.typ"
    out_pdf = PAGES_DIR / f"trang_{page_num}.pdf"
    out_png = PAGES_DIR / f"trang_{page_num}.png"

    full_content = f"{TYPST_HEADER}\n{page_typst}\n"
    temp_typ.write_text(full_content, encoding="utf-8")

    # Biên dịch PDF
    subprocess.run(["/opt/homebrew/bin/typst", "compile", str(temp_typ), str(out_pdf)], check=True)
    
    # Biên dịch PNG để xem trước trực tiếp trên IDE
    subprocess.run(["/opt/homebrew/bin/typst", "compile", "--format", "png", "--ppi", "150", str(temp_typ), str(out_png)], check=True)

    if temp_typ.exists():
        temp_typ.unlink()

    print(f"[OK] Đã xuất trang {page_num}:")
    print(f"     - PDF: {out_pdf}")
    print(f"     - PNG (xem trước): {out_png}")

def watch_mode():
    """Tự động theo dõi các file Markdown và biên dịch lại ngay khi lưu file."""
    print("Đang bật chế độ lắng nghe (Watch mode)...")
    print("Chỉnh sửa file trong requirements/report_pages/*.md và nhấn Cmd+S để cập nhật.")
    print("Nhấn Ctrl+C để dừng.")
    
    last_mtimes = {}
    compile_all()

    try:
        while True:
            changed = False
            for f in PAGES_DIR.glob("*.md"):
                mtime = f.stat().st_mtime
                if f not in last_mtimes or mtime > last_mtimes[f]:
                    last_mtimes[f] = mtime
                    changed = True
            if changed:
                print("\n[Phát hiện thay đổi] Đang cập nhật báo cáo...")
                compile_all()
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nĐã dừng chế độ lắng nghe.")

def main():
    parser = argparse.ArgumentParser(description="Biên dịch Báo cáo Nhiệm vụ từ Markdown sang PDF.")
    parser.add_argument("--page", type=int, choices=[1, 2, 3], help="Biên dịch riêng Trang 1, 2 hoặc 3 để xem trước")
    parser.add_argument("--watch", action="store_true", help="Chế độ tự động lắng nghe và cập nhật PDF khi lưu file Markdown")
    args = parser.parse_args()

    if args.watch:
        watch_mode()
    elif args.page:
        compile_single_page(args.page)
    else:
        compile_all()

if __name__ == "__main__":
    main()
