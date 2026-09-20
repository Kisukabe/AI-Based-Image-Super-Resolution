# BAO CAO TRANG THAI DU AN
# AI-Based Medical Image Super-Resolution — FPGA Acceleration on PYNQ-Z2
# Cap nhat: 2026-09-20

---

## 1. TOM TAT TONG QUAN

| Hang muc | So lieu |
| :--- | :---: |
| Tong so task trong Roadmap | 12 |
| Da hoan thanh | 12 |
| Chua hoan thanh | 0 |
| Tien do tong the | 100% |
| Commit hien tai | `19b2f1d` (HEAD = origin/main) |
| Nhanh | `main` (sach, khong co uncommitted changes) |

---

## 2. TRANG THAI CHI TIET TUNG TASK

### NHIEM VU 1 — AI Modeling, Data Degradation & Software Baseline

| STT | Ma task | Mo ta | Trang thai | File chinh |
| :---: | :---: | :--- | :---: | :--- |
| 1 | T1.1 | Xay dung 2 kien truc PyTorch Float32 (SRCNN_Original 8.129 params, Compact_SRCNN 1.649 params) + Huan luyen Loss = MSE + 0.1*Sobel + Do so lieu PSNR/SSIM | [HOAN THANH] | `core_project/ai_software/models/models_srcnn.py` |
| 2 | T1.2 | Pipeline suy thoai vat ly (blur + downsample + noise) tren NIH Chest X-ray, sinh 338 cap anh | [HOAN THANH] | `generate_degraded_dataset.py` |
| 3 | T1.3 | Do thi hoi tu Loss (300 DPI, 4200x1500 px) | [HOAN THANH] | `core_project/ai_software/training/loss_convergence_dpi300.png` |
| 4 | T1.4 | Script danh gia Baseline: PSNR, SSIM, Bicubic — xuat 4-sheet Excel + CSV + JSON | [HOAN THANH] | `core_project/ai_software/evaluation/evaluate_pytorch_baseline.py` |
| 5 | T1.5 | Bao cao so sanh 8 mo hinh sieu phan giai (song ngu VN/EN, 2 PDF, 12 bieu do 300 DPI) | [HOAN THANH] | `performance_comparison_report.pdf`, `performance_comparison_report_en.pdf` |

### NHIEM VU 2 — Bit-Accurate DV, Benchmark Parser & Statistical Analysis

| STT | Ma task | Mo ta | Trang thai | File chinh |
| :---: | :---: | :--- | :---: | :--- |
| 6 | T2.1 | Phan tich thong ke 2.200 anh PYNQ-Z2: Mean+-Std PSNR/SSIM/Gain, Histogram 300 DPI | [HOAN THANH] | `analyze_kaggle_benchmark.py`, `pynq_z2_statistical_analysis.json` |
| 7 | T2.2 | DV Scoreboard bit-exact: 4/4 ALL_PASS, so hoc RTL 100% chinh xac tung pixel | [HOAN THANH] | `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` |
| 8 | T2.3 | Notebook Kaggle Compact SRCNN RTL (13 cell, 3 scale 2x/3x/4x, 38 truong xuat) | [HOAN THANH] | `legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb` |
| 9 | T2.4 | Do dac da nen tang CPU vs GPU vs FPGA (100 lan lap, sau 10 warm-up, Mean+-Std) | [HOAN THANH] | `benchmark_multi_platform.py`, `multi_platform_benchmark.json` |
| 10 | T2.5 | ROI Zoom-in 4x, Error Heatmap, Fig.7 Overlap-Tiling Ablation, Fig.8 Visual Comparison (300 DPI) | [HOAN THANH] | `generate_paper_fig7.py`, `generate_paper_fig8.py` |
| 11 | T2.6 | Dong goi ban giao: Excel 6 Sheet, figures_dpi300/ (29 file), thu muc IEEE GTSD 2026 | [HOAN THANH] | `hardware_benchmark_statistics.xlsx`, `figures_dpi300/` |

---

## 3. CHECKLIST SAN PHAM BAN GIAO (13 san pham)

### A. Nhiem vu 1

| Ma | San pham | File | Trang thai |
| :---: | :--- | :--- | :---: |
| D1.1 | Checkpoint Compact SRCNN Float32 (1.649 params, Val PSNR 24.33 dB) | `core_project/ai_software/checkpoints/compact_srcnn_float32.pth` | [DA CO] |
| D1.2 | Script sinh tap suy thoai vat ly (338 cap anh) | `generate_degraded_dataset.py` | [DA CO] |
| D1.3 | Script danh gia Baseline (PSNR, SSIM, Bicubic) | `core_project/ai_software/evaluation/evaluate_pytorch_baseline.py` | [DA CO] |
| D1.4 | Do thi hoi tu Loss 300 DPI (4200x1500 px) | `core_project/ai_software/training/loss_convergence_dpi300.png` | [DA CO] |
| D1.5 | Bang so lieu PSNR/SSIM doi chung (4-sheet Excel + CSV + JSON) | `core_project/benchmarks_reports/deliverables_export/baseline_psnr_ssim_comparison.xlsx` | [DA CO] |

### B. Nhiem vu 2

| Ma | San pham | File | Trang thai |
| :---: | :--- | :--- | :---: |
| D2.1 | Notebook Kaggle Compact SRCNN RTL (13 cell) | `legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb` | [DA CO] |
| D2.2 | Script phan tich JSON + Histogram 300 DPI | `analyze_kaggle_benchmark.py` | [DA CO] |
| D2.3 | DV Scoreboard 4/4 ALL_PASS bit-exact | `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` | [DA CO] |
| D2.4 | Excel thong ke + do dac da nen tang (6 Sheet) | `hardware_benchmark_statistics.xlsx` | [DA CO] |
| D2.5 | figures_dpi300/ 300 DPI (29 file: Histogram, ROI, Heatmap) | `figures_dpi300/` | [DA CO] |
| D2.6 | Bao cao 8 mo hinh song ngu VN/EN (2 PDF + 12 bieu do) | `performance_comparison_report.pdf`, `performance_comparison_report_en.pdf` | [DA CO] |
| D2.7 | Fig.7 Overlap-Tiling Ablation (300 DPI, 4769x1685 px) | `figures_dpi300/fig7_boundary_ablation.png` | [DA CO] |
| D2.8 | Fig.8 Visual Comparison 2x7 Matrix (300 DPI, 4710x1415 px) | `figures_dpi300/fig8_visual_comparison.png` | [DA CO] |

---

## 4. BAO CAO 2 FILE REQUIREMENTS

### 4.1. requirements.txt chinh (root) — SU DUNG CHO MOI THIET BI MOI

Duong dan: `requirements.txt`

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
openpyxl>=3.0.0
Pillow>=9.0.0
```

Ket luan: Day du, co rang buoc phien ban toi thieu ro rang.

Cai dat Mac/Linux:
```bash
pip install -r requirements.txt
```

Cai dat Windows (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4.2. requirements.txt legacy (survey_assets) — CHI DE THAM KHAO

Duong dan: `legacy_experiments/survey_assets/requirements.txt`

```
torch
torchvision
Pillow
numpy
pandas
tqdm
opencv-python
```

Phan tich so sanh:

| Goi | Root requirements.txt | Legacy requirements.txt |
| :--- | :---: | :---: |
| torch | >=1.12.0 | (khong rang buoc) |
| torchvision | >=0.13.0 | (khong rang buoc) |
| numpy | >=1.21.0 | (khong rang buoc) |
| opencv-python | >=4.5.0 | (khong rang buoc) |
| scipy | >=1.7.0 | THIEU |
| matplotlib | >=3.5.0 | THIEU |
| pandas | >=1.3.0 | (khong rang buoc) |
| openpyxl | >=3.0.0 | THIEU |
| Pillow | >=9.0.0 | (khong rang buoc) |
| tqdm | KHONG CO | Co (khong bat buoc) |

Khuyen nghi: KHONG dung file legacy de cai dat. Chi dung `requirements.txt` o root.
File legacy duoc giu nguyen de tra lai lich su giai doan khao sat.

---

## 5. KIEM TRA MO HINH (MODEL AUDIT)

File: `core_project/ai_software/models/models_srcnn.py` (251 dong, 9.669 byte)

| Class | Kien truc | So params | Vai tro | Trang thai |
| :--- | :--- | :---: | :--- | :---: |
| `SRCNN_Original` | 1->64->32->1 (k9,k1,k5) | 8.129 | Baseline so sanh (software) | [GIU LAI] |
| `Compact_SRCNN` | 1->16->8->1 (k9,k1,k5) | 1.649 | Mo hinh chinh khop 100% RTL FPGA | [GIU LAI] |

Ket luan: File models_srcnn.py chi chua dung 2 class + 1 ham tien ich get_model_summary().
Khong con model du thua. Yeu cau "chi giu SRCNN goc va Compact_SRCNN" DA THOA MAN.

---

## 6. HO TRO WINDOWS

| File | Muc dich | Commit |
| :--- | :--- | :--- |
| `run_benchmark_windows.bat` | 1-click batch script (CMD) | 73fb54c |
| `run_benchmark_windows.ps1` | PowerShell script | 73fb54c |
| `HUONG_DAN_CHAY_WINDOWS.md` | Huong dan cai dat va chay | 73fb54c |
| `MEASUREMENT_GUIDE_X86_CUDA.md` | Huong dan do dac x86/CUDA | 0102647 |

---

## 7. LICH SU GIT (15 COMMIT GAN NHAT)

```
19b2f1d  fix(git): update .gitignore — whitelist eval image, add .vs/
73fb54c  feat(windows): 1-click windows scripts, requirements.txt, setup guide
fd239dd  docs(roadmap): record completion of Fig 7 and Fig 8
cc97c7a  feat(visual): multi-model visual comparison generator, Fig 8 (300 DPI)
5d2b2ff  feat(visual): overlap-tiling ablation generator, Fig 7 (300 DPI)
a7260ae  feat(deliverables): master Excel workbook, 300 DPI figures (T2.6)
1796832  feat(visual): ROI zoom-in 4x, residual error heatmaps (T2.5)
0102647  feat(benchmark): multi-platform benchmark script, results (T2.4)
3a636fb  feat(benchmark): statistical analysis 2,200 images, histograms (T2.3)
3f35ea3  style(analysis): standardize white row backgrounds
9a30a8e  refactor(analysis): synchronize analysis notebook
39369fd  style(analysis): remove bold from SRCNN models in catalog
f490f19  fix(analysis): restore baseline label for bicubic
9ce2ebc  feat(analysis): generate English edition of performance report
d5e9689  refactor(analysis): move channel numbers to footnotes
```

---

## 8. HANG MUC CON LAI / VIEC OPTIONAL

> Theo TASK_ROADMAP.md: tat ca 12/12 task DA HOAN THANH.
> Git: HEAD = origin/main, working tree sach.

| Uu tien | Hang muc | Mo ta |
| :---: | :--- | :--- |
| THAP | Chay Notebook Kaggle tren GPU | Upload .ipynb len Kaggle, chay ~20 phut, tai file zip ket qua |
| THAP | Them tqdm vao root requirements.txt | Tien ich, khong bat buoc |
| THAP | Them scikit-image vao requirements.txt | Tinh SSIM bang scikit-image chinh xac hon |

---

*Tao tu dong boi Antigravity IDE — 2026-09-20T15:25:00+07:00*
