// style_config.typ — Cấu hình giao diện và quy chuẩn hiển thị cho Báo Cáo Nhiệm Vụ

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

#let report-style(body) = {
  set page(
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

  set text(
    font: "Arial",
    size: 9.6pt,
    lang: "vi",
    fill: rgb("#2d3748")
  )
  set par(justify: true, leading: 0.55em)
  set table(inset: (x: 5pt, y: 3.3pt))

  body
}
