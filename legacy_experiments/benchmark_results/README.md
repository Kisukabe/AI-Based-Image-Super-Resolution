# results/

Thư mục này chứa toàn bộ **kết quả benchmark JSON và CSV** tải về từ Kaggle sau mỗi lần chạy notebook.

---

## Cấu trúc

```
results/
├── software/          ← Kết quả từ notebooks/software/
│   └── srgan_2x_benchmark.json
│
└── hardware/          ← Kết quả từ notebooks/hardware/
    ├── bicubic_2x_benchmark.json
    ├── bicubic_3x_benchmark.json
    ├── bicubic_4x_benchmark.json
    ├── srcnn_benchmark.json
    └── swift_srgan_benchmark.json
```

---

## Quy ước đặt tên file

| Notebook | File JSON output | Đặt vào |
|---|---|---|
| `kaggle_srgan_2x_inference.ipynb` | `srgan_2x_benchmark.json` | `results/software/` |
| `kaggle_srgan_benchmark.ipynb` | `srgan_benchmark.json` | `results/software/` |
| `kaggle_bicubic_benchmark.ipynb` | `bicubic_{scale}x_benchmark.json` | `results/hardware/` |
| `kaggle_srcnn_benchmark.ipynb` | `srcnn_benchmark.json` | `results/hardware/` |
| `kaggle_swift_srgan_benchmark.ipynb` | `swift_srgan_benchmark.json` | `results/hardware/` |

---

## Workflow

```
1. Chạy notebook trên Kaggle
2. Tải file JSON/CSV từ /kaggle/working/ về máy
3. Đặt vào đúng thư mục results/software/ hoặc results/hardware/
4. Dùng benchmark_analysis_and_comparison.ipynb để so sánh tổng hợp
```

> Lưu ý: Không commit file JSON lớn (>10MB) vào Git — thêm vào .gitignore nếu cần.
