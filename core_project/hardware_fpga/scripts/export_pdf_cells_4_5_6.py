#!/usr/bin/env python3
"""
export_pdf_cells_4_5_6.py
Xuất báo cáo PDF chuẩn khoa học: SRCNN_vs_Swift_SRGAN_Benchmark_Analysis.pdf
- Khắc phục triệt để lỗi tràn chữ / dính chữ ở cột Tập Dữ Liệu và Mô Hình
- Phân bổ colWidths theo độ dài nội dung từng cột
- Đồng bộ kích thước 22.0 x 10.5 inch (300 DPI) cho tất cả các trang
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ─────────────────────────────────────────────────────────────
# 0. CẤU HÌNH FONT & MATPLOTLIB TOÀN CỤC (ĐỒNG BỘ CHUẨN)
# ─────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = False

FIG_WIDTH = 22.0
FIG_HEIGHT = 10.5
FIG_DPI = 300

# ─────────────────────────────────────────────────────────────
# 1. LOAD & CHUẨN HÓA DỮ LIỆU
# ─────────────────────────────────────────────────────────────
PATH_SRCNN = os.path.expanduser('~/Downloads/benchmark_checkpoint.json')
PATH_SRGAN = os.path.expanduser('~/Downloads/srgan_checkpoint.json')
OUTPUT_SPLIT_DIR = os.path.expanduser('~/Downloads/split_benchmark_results')
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

def load_and_standardize(json_path, model_label):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data.get('per_image_results', [])
    df = pd.DataFrame(records)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    df['model_name'] = model_label
    
    if 'psnr_fpga_db' in df.columns:
        df['psnr_model_db'] = df['psnr_fpga_db']
    if 'ssim_fpga' in df.columns:
        df['ssim_model'] = df['ssim_fpga']
    if 'msssim_fpga' in df.columns:
        df['msssim_model'] = df['msssim_fpga']
    
    if 'lpips_fpga' in df.columns:
        df['lpips_model'] = df['lpips_fpga']
    elif 'lpips_srgan' in df.columns:
        df['lpips_model'] = df['lpips_srgan']
    elif 'lpips' in df.columns:
        df['lpips_model'] = df['lpips']
        
    if 'niqe_fpga' in df.columns:
        df['niqe_model'] = df['niqe_fpga']
    if 'mse_fpga' in df.columns:
        df['mse_model'] = df['mse_fpga']
    if 'rmse_fpga' in df.columns:
        df['rmse_model'] = df['rmse_fpga']
        
    df['fps'] = 1000.0 / df['latency_ms']
    return df

df_srcnn = load_and_standardize(PATH_SRCNN, 'SRCNN RTL 2x (Q7)')
df_srgan = load_and_standardize(PATH_SRGAN, 'Swift-SRGAN 4x (Q7)')

# ─────────────────────────────────────────────────────────────
# 2. TRANG 1: BẢNG TỔNG HỢP HIỆU NĂNG TOÀN DIỆN (CELL 4)
# ─────────────────────────────────────────────────────────────
def calc_stats_row(df, ds_name, model_label):
    sub = df[df['dataset'] == ds_name] if ds_name != 'TOÀN BỘ (2200)' else df
    if sub.empty: return None
    return {
        'Tập Dữ Liệu':       ds_name,
        'Mô Hình':           model_label,
        'Số Ảnh':            len(sub),
        'PSNR Bicubic (dB)': round(sub['psnr_bicubic_db'].mean(), 3),
        'PSNR Model (dB)':   round(sub['psnr_model_db'].mean(), 3),
        'Δ PSNR Gain (dB)':  round(sub['psnr_gain_db'].mean(), 3),
        'SSIM Bicubic':      round(sub['ssim_bicubic'].mean(), 4),
        'SSIM Model':        round(sub['ssim_model'].mean(), 4),
        'MS-SSIM Bicubic':   round(sub['msssim_bicubic'].mean(), 4),
        'MS-SSIM Model':     round(sub['msssim_model'].mean(), 4),
        'Δ MS-SSIM Gain':    round(sub['msssim_gain'].mean(), 4),
        'LPIPS Bicubic (↓)': round(sub['lpips_bicubic'].mean(), 4),
        'LPIPS Model (↓)':   round(sub['lpips_model'].mean(), 4),
        'Δ LPIPS Gain (↑)':  round(sub['lpips_gain'].mean(), 4),
        'NIQE Bicubic (↓)':  round(sub['niqe_bicubic'].mean(), 2),
        'NIQE Model (↓)':    round(sub['niqe_model'].mean(), 2),
        'Δ NIQE Gain (↑)':   round(sub['niqe_gain'].mean(), 2),
        'MSE Bicubic':       round(sub['mse_bicubic'].mean(), 2),
        'MSE Model':         round(sub['mse_model'].mean(), 2),
        'Latency (ms)':      round(sub['latency_ms'].mean(), 1),
        'Tốc Độ (FPS)':      round(sub['fps'].mean(), 1)
    }

rows = []
for ds in ['sub_NIH', 'sub_chest', 'TOÀN BỘ (2200)']:
    r1 = calc_stats_row(df_srcnn, ds, 'SRCNN RTL 2x (Q7)')
    r2 = calc_stats_row(df_srgan, ds, 'Swift-SRGAN 4x (Q7)')
    if r1: rows.append(r1)
    if r2: rows.append(r2)

df_compare = pd.DataFrame(rows)

cols_show = [
    'Tập Dữ Liệu', 'Mô Hình', 'Số Ảnh',
    'PSNR Model (dB)', 'Δ PSNR Gain (dB)',
    'SSIM Model', 'MS-SSIM Model', 'Δ MS-SSIM Gain',
    'LPIPS Model (↓)', 'Δ LPIPS Gain (↑)',
    'NIQE Model (↓)', 'Δ NIQE Gain (↑)',
    'MSE Model', 'Latency (ms)', 'Tốc Độ (FPS)'
]

fig4 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
ax4 = fig4.add_subplot(111)
ax4.axis('off')

df_display = df_compare[cols_show].copy()
df_display['Số Ảnh']           = df_display['Số Ảnh'].map('{:,}'.format)
df_display['PSNR Model (dB)']  = df_display['PSNR Model (dB)'].map('{:.3f}'.format)
df_display['Δ PSNR Gain (dB)'] = df_display['Δ PSNR Gain (dB)'].map('{:+.3f}'.format)
df_display['SSIM Model']       = df_display['SSIM Model'].map('{:.4f}'.format)
df_display['MS-SSIM Model']    = df_display['MS-SSIM Model'].map('{:.4f}'.format)
df_display['Δ MS-SSIM Gain']   = df_display['Δ MS-SSIM Gain'].map('{:+.4f}'.format)
df_display['LPIPS Model (↓)']  = df_display['LPIPS Model (↓)'].map('{:.4f}'.format)
df_display['Δ LPIPS Gain (↑)'] = df_display['Δ LPIPS Gain (↑)'].map('{:+.4f}'.format)
df_display['NIQE Model (↓)']   = df_display['NIQE Model (↓)'].map('{:.2f}'.format)
df_display['Δ NIQE Gain (↑)']  = df_display['Δ NIQE Gain (↑)'].map('{:+.2f}'.format)
df_display['MSE Model']        = df_display['MSE Model'].map('{:.2f}'.format)
df_display['Latency (ms)']     = df_display['Latency (ms)'].map('{:.1f}'.format)
df_display['Tốc Độ (FPS)']     = df_display['Tốc Độ (FPS)'].map('{:.1f}'.format)

# Phân bổ độ rộng cột chuẩn xác (không bị đè chữ)
raw_widths1 = [0.10, 0.13, 0.045, 0.06, 0.065, 0.055, 0.06, 0.065, 0.065, 0.065, 0.06, 0.065, 0.045, 0.045, 0.045]
widths1 = [w / sum(raw_widths1) * 0.96 for w in raw_widths1]

table4 = ax4.table(
    cellText=df_display.values,
    colLabels=df_display.columns,
    colWidths=widths1,
    cellLoc='center',
    loc='center',
    bbox=[0.02, 0.15, 0.96, 0.65]
)
table4.auto_set_font_size(False)
table4.set_fontsize(9.5)

for (r, c), cell in table4.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1e3a8a')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
    else:
        cell.set_facecolor('#f8fafc' if r % 2 == 1 else '#ffffff')
        cell.set_edgecolor('#cbd5e1')

fig4.suptitle(
    'BẢNG TỔNG HỢP HIỆU NĂNG TOÀN DIỆN: BICUBIC BASELINE vs SRCNN RTL (Q7) vs SWIFT-SRGAN (Q7)',
    fontsize=15, fontweight='bold', color='#0f172a', y=0.88
)

# ─────────────────────────────────────────────────────────────
# 3. TRANG 2: BẢNG SỐ LIỆU THUẦN BICUBIC PHÓNG TO FULL-WIDTH (CELL 6B)
# ─────────────────────────────────────────────────────────────
def calc_bicubic_stats_row(df, ds_name, baseline_label, scale_label):
    sub = df[df['dataset'] == ds_name] if ds_name != 'TOÀN BỘ (2200)' else df
    if sub.empty: return None
    return {
        'Tập Dữ Liệu':          ds_name,
        'Thuật Toán Baseline':  baseline_label,
        'Hệ Số Phóng':          scale_label,
        'Số Ảnh':               len(sub),
        'PSNR Bicubic (dB)':    round(sub['psnr_bicubic_db'].mean(), 3),
        'SSIM Bicubic':         round(sub['ssim_bicubic'].mean(), 4),
        'MS-SSIM Bicubic':      round(sub['msssim_bicubic'].mean(), 4),
        'LPIPS Bicubic (↓)':    round(sub['lpips_bicubic'].mean(), 4),
        'NIQE Bicubic (↓)':     round(sub['niqe_bicubic'].mean(), 2),
        'MSE Bicubic':          round(sub['mse_bicubic'].mean(), 2),
        'RMSE Bicubic':         round(sub['rmse_bicubic'].mean() if 'rmse_bicubic' in sub.columns else np.sqrt(sub['mse_bicubic']).mean(), 2)
    }

bicubic_rows = []
for ds in ['sub_NIH', 'sub_chest', 'TOÀN BỘ (2200)']:
    r1 = calc_bicubic_stats_row(df_srcnn, ds, 'Bicubic 2x Baseline', '2×')
    r2 = calc_bicubic_stats_row(df_srgan, ds, 'Bicubic 4x Baseline', '4×')
    if r1: bicubic_rows.append(r1)
    if r2: bicubic_rows.append(r2)

df_bicubic_pure = pd.DataFrame(bicubic_rows)

cols_bicubic_show = [
    'Tập Dữ Liệu', 'Thuật Toán Baseline', 'Hệ Số Phóng', 'Số Ảnh',
    'PSNR Bicubic (dB)', 'SSIM Bicubic', 'MS-SSIM Bicubic',
    'LPIPS Bicubic (↓)', 'NIQE Bicubic (↓)', 'MSE Bicubic', 'RMSE Bicubic'
]

fig_bicubic_table = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
ax_bic = fig_bicubic_table.add_subplot(111)
ax_bic.axis('off')

df_bic_disp = df_bicubic_pure[cols_bicubic_show].copy()
df_bic_disp['Số Ảnh']            = df_bic_disp['Số Ảnh'].map('{:,}'.format)
df_bic_disp['PSNR Bicubic (dB)'] = df_bic_disp['PSNR Bicubic (dB)'].map('{:.3f}'.format)
df_bic_disp['SSIM Bicubic']      = df_bic_disp['SSIM Bicubic'].map('{:.4f}'.format)
df_bic_disp['MS-SSIM Bicubic']   = df_bic_disp['MS-SSIM Bicubic'].map('{:.4f}'.format)
df_bic_disp['LPIPS Bicubic (↓)'] = df_bic_disp['LPIPS Bicubic (↓)'].map('{:.4f}'.format)
df_bic_disp['NIQE Bicubic (↓)']  = df_bic_disp['NIQE Bicubic (↓)'].map('{:.2f}'.format)
df_bic_disp['MSE Bicubic']       = df_bic_disp['MSE Bicubic'].map('{:.2f}'.format)
df_bic_disp['RMSE Bicubic']      = df_bic_disp['RMSE Bicubic'].map('{:.2f}'.format)

# Phân bổ độ rộng cột chuẩn xác (không bị đè chữ)
raw_widths2 = [0.12, 0.16, 0.07, 0.06, 0.09, 0.08, 0.09, 0.09, 0.08, 0.06, 0.06]
widths2 = [w / sum(raw_widths2) * 0.96 for w in raw_widths2]

table_bic = ax_bic.table(
    cellText=df_bic_disp.values,
    colLabels=df_bic_disp.columns,
    colWidths=widths2,
    cellLoc='center',
    loc='center',
    bbox=[0.02, 0.15, 0.96, 0.65]
)
table_bic.auto_set_font_size(False)
table_bic.set_fontsize(11)

for (r, c), cell in table_bic.get_celld().items():
    if r == 0:
        cell.set_facecolor('#0f766e')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    else:
        cell.set_facecolor('#f0fdfa' if r % 2 == 1 else '#ffffff')
        cell.set_edgecolor('#cbd5e1')

fig_bicubic_table.suptitle(
    'BẢNG TỔNG HỢP SỐ LIỆU THUẦN: THUẬT TOÁN NỘI SUY NỀN BICUBIC (SCALE 2× VÀ SCALE 4×)',
    fontsize=15, fontweight='bold', color='#0f172a', y=0.88
)

# ─────────────────────────────────────────────────────────────
# 4. TRANG 3: BIỂU ĐỒ SO SÁNH METRICS GIỮA SRCNN VÀ SRGAN (CELL 5)
# ─────────────────────────────────────────────────────────────
categories = ['sub_NIH (1750 ảnh)', 'sub_chest (450 ảnh)', 'Toàn Bộ (2200 ảnh)']
cat_keys   = ['sub_NIH', 'sub_chest', 'TOÀN BỘ (2200)']
x = np.arange(len(categories))
w = 0.35

def get_vals(df, col):
    return [df[df['dataset']==k][col].mean() if k!='TOÀN BỘ (2200)' else df[col].mean() for k in cat_keys]

fig5, axes5 = plt.subplots(2, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)

metrics_info = [
    ('psnr_model_db', 'PSNR',          'PSNR (dB)',           '{:.2f} dB',  52.0,   0, 'upper right', axes5[0, 0]),
    ('ssim_model',    'SSIM',          'SSIM Index',          '{:.4f}',     1.28,   0, 'upper right', axes5[0, 1]),
    ('msssim_model',  'MS-SSIM',       'MS-SSIM Index',       '{:.4f}',     1.28,   0, 'upper right', axes5[0, 2]),
    ('lpips_model',   'LPIPS',         'LPIPS Score (↓)',     '{:.4f}',     0.52,   0, 'upper left',  axes5[0, 3]),
    ('niqe_model',    'NIQE',          'NIQE Score (↓)',      '{:.2f}',     20.0,   0, 'upper right', axes5[1, 0]),
    ('mse_model',     'MSE',           'MSE Loss (↓)',        '{:.2f}',     28.0,   0, 'upper left',  axes5[1, 1]),
    ('latency_ms',    'Độ Trễ Latency', 'Latency (ms) (↓)',   '{:.1f} ms',  85.0,   0, 'upper left',  axes5[1, 2]),
    ('fps',           'Tốc Độ FPS',    'Tốc độ (FPS) (↑)',    '{:.1f} FPS', 50.0,   0, 'upper right', axes5[1, 3])
]

for col, title, ylabel, fmt, y_top, y_bottom, leg_loc, ax in metrics_info:
    v_src = get_vals(df_srcnn, col)
    v_srg = get_vals(df_srgan, col)
    
    ax.bar(x - w/2, v_src, w, label='SRCNN RTL 2x (Q7)', color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
    ax.bar(x + w/2, v_srg, w, label='Swift-SRGAN 4x (Q7)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold', fontsize=8.5)
    ax.set_ylim(y_bottom, y_top)
    
    ax.legend(frameon=True, loc=leg_loc, fontsize=8.5, facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.3, ls=':')
    
    max_val = max(v_src + v_srg)
    offset = max_val * 0.025 if max_val > 0 else 0.5
    for i in range(len(x)):
        ax.text(x[i] - w/2, v_src[i] + offset, fmt.format(v_src[i]), ha='center', fontsize=8.5, fontweight='bold', color='#1a365d')
        ax.text(x[i] + w/2, v_srg[i] + offset, fmt.format(v_srg[i]), ha='center', fontsize=8.5, fontweight='bold', color='#44337a')

fig5.suptitle('BIỂU ĐỒ SO SÁNH METRICS GIỮA SRCNN VÀ SRGAN', fontsize=15, fontweight='bold', y=0.99)
fig5.tight_layout()

# ─────────────────────────────────────────────────────────────
# 5. TRANG 4: BIỂU ĐỒ CHUYÊN BIỆT ĐỐI CHỨNG BICUBIC (CELL 6)
# ─────────────────────────────────────────────────────────────
p_bic_2x    = get_vals(df_srcnn, 'psnr_bicubic_db')
p_src_2x    = get_vals(df_srcnn, 'psnr_model_db')
gain_p_src  = get_vals(df_srcnn, 'psnr_gain_db')
gain_ms_src = get_vals(df_srcnn, 'msssim_gain')
gain_lp_src = get_vals(df_srcnn, 'lpips_gain')
gain_nq_src = get_vals(df_srcnn, 'niqe_gain')

p_bic_4x    = get_vals(df_srgan, 'psnr_bicubic_db')
p_srg_4x    = get_vals(df_srgan, 'psnr_model_db')
gain_p_srg  = get_vals(df_srgan, 'psnr_gain_db')
gain_ms_srg = get_vals(df_srgan, 'msssim_gain')
gain_lp_srg = get_vals(df_srgan, 'lpips_gain')
gain_nq_srg = get_vals(df_srgan, 'niqe_gain')

fig6, axes6 = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)

# PANEL 1: Task Scale 2x
axes6[0, 0].bar(x - w/2, p_bic_2x, w, label='Bicubic 2x Baseline', color='#feb2b2', edgecolor='#c53030', lw=1)
axes6[0, 0].bar(x + w/2, p_src_2x, w, label='SRCNN RTL 2x (Q7)', color='#3182ce', edgecolor='#2c5282', lw=1)
axes6[0, 0].set_ylabel('PSNR (dB)', fontweight='bold')
axes6[0, 0].set_title('Task Scale 2x: Bicubic 2x vs SRCNN RTL 2x', fontsize=11, fontweight='bold', pad=8)
axes6[0, 0].set_xticks(x)
axes6[0, 0].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[0, 0].set_ylim(0, 52)
axes6[0, 0].legend(frameon=True, loc='upper right', fontsize=8.5, facecolor='white', framealpha=0.95)
axes6[0, 0].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[0, 0].text(x[i] - w/2, p_bic_2x[i] + 0.6, f'{p_bic_2x[i]:.2f}', ha='center', fontsize=8.5)
    axes6[0, 0].text(x[i] + w/2, p_src_2x[i] + 0.6, f'{p_src_2x[i]:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#1a365d')

# PANEL 2: Task Scale 4x
axes6[0, 1].bar(x - w/2, p_bic_4x, w, label='Bicubic 4x Baseline', color='#fed7aa', edgecolor='#ea580c', lw=1)
axes6[0, 1].bar(x + w/2, p_srg_4x, w, label='Swift-SRGAN 4x (Q7)', color='#805ad5', edgecolor='#44337a', lw=1)
axes6[0, 1].set_ylabel('PSNR (dB)', fontweight='bold')
axes6[0, 1].set_title('Task Scale 4x: Bicubic 4x vs Swift-SRGAN 4x', fontsize=11, fontweight='bold', pad=8)
axes6[0, 1].set_xticks(x)
axes6[0, 1].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[0, 1].set_ylim(0, 48)
axes6[0, 1].legend(frameon=True, loc='upper right', fontsize=8.5, facecolor='white', framealpha=0.95)
axes6[0, 1].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[0, 1].text(x[i] - w/2, p_bic_4x[i] + 0.6, f'{p_bic_4x[i]:.2f}', ha='center', fontsize=8.5)
    axes6[0, 1].text(x[i] + w/2, p_srg_4x[i] + 0.6, f'{p_srg_4x[i]:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#44337a')

# PANEL 3: Δ PSNR Gain
axes6[0, 2].bar(x - w/2, gain_p_src, w, label='SRCNN 2x Gain (dB)', color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
axes6[0, 2].bar(x + w/2, gain_p_srg, w, label='Swift-SRGAN 4x Gain (dB)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
axes6[0, 2].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0 dB)')
axes6[0, 2].set_ylabel('Δ PSNR Gain (dB) [Model - Bicubic]', fontweight='bold')
axes6[0, 2].set_title('Mức Tăng/Giảm PSNR (ΔPSNR Gain)', fontsize=11, fontweight='bold', pad=8)
axes6[0, 2].set_xticks(x)
axes6[0, 2].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[0, 2].set_ylim(-2.2, 0.9)
axes6[0, 2].legend(frameon=True, loc='upper left', fontsize=8, facecolor='white', framealpha=0.95)
axes6[0, 2].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[0, 2].text(x[i] - w/2, gain_p_src[i] + (0.06 if gain_p_src[i]>=0 else -0.16), f'{gain_p_src[i]:+.2f} dB', ha='center', fontsize=8.5, fontweight='bold')
    axes6[0, 2].text(x[i] + w/2, gain_p_srg[i] + (0.06 if gain_p_srg[i]>=0 else -0.16), f'{gain_p_srg[i]:+.2f} dB', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 4: Δ MS-SSIM Gain
axes6[1, 0].bar(x - w/2, gain_ms_src, w, label='SRCNN 2x (MS-SSIM Gain)', color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 0].bar(x + w/2, gain_ms_srg, w, label='Swift-SRGAN 4x (MS-SSIM Gain)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 0].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0)')
axes6[1, 0].set_ylabel('Δ MS-SSIM Gain [Model - Bicubic]', fontweight='bold')
axes6[1, 0].set_title('Mức Thay Đổi MS-SSIM (ΔMS-SSIM Gain)', fontsize=11, fontweight='bold', pad=8)
axes6[1, 0].set_xticks(x)
axes6[1, 0].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[1, 0].set_ylim(-0.0058, 0.0022)
axes6[1, 0].legend(frameon=True, loc='upper left', fontsize=8, facecolor='white', framealpha=0.95)
axes6[1, 0].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[1, 0].text(x[i] - w/2, gain_ms_src[i] + (0.00015 if gain_ms_src[i]>=0 else -0.00045), f'{gain_ms_src[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')
    axes6[1, 0].text(x[i] + w/2, gain_ms_srg[i] + (0.00015 if gain_ms_srg[i]>=0 else -0.00045), f'{gain_ms_srg[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 5: Độ cải thiện thị giác LPIPS Gain
axes6[1, 1].bar(x - w/2, gain_lp_src, w, label='SRCNN 2x (LPIPS Gain)', color='#38a169', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 1].bar(x + w/2, gain_lp_srg, w, label='Swift-SRGAN 4x (LPIPS Gain)', color='#d69e2e', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 1].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0)')
axes6[1, 1].set_ylabel('Δ LPIPS Gain', fontweight='bold')
axes6[1, 1].set_title('Mức Cải Thiện Thị Giác (ΔLPIPS Gain)', fontsize=11, fontweight='bold', pad=8)
axes6[1, 1].set_xticks(x)
axes6[1, 1].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[1, 1].set_ylim(-0.01, 0.072)
axes6[1, 1].legend(frameon=True, loc='upper left', fontsize=8, facecolor='white', framealpha=0.95)
axes6[1, 1].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[1, 1].text(x[i] - w/2, gain_lp_src[i] + 0.002, f'{gain_lp_src[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')
    axes6[1, 1].text(x[i] + w/2, gain_lp_srg[i] + 0.002, f'{gain_lp_srg[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 6: Độ cải thiện độ tự nhiên NIQE Gain
axes6[1, 2].bar(x - w/2, gain_nq_src, w, label='SRCNN 2x (NIQE Gain)', color='#e53e3e', alpha=0.85, edgecolor='black', lw=0.6)
axes6[1, 2].bar(x + w/2, gain_nq_srg, w, label='Swift-SRGAN 4x (NIQE Gain)', color='#319795', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 2].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0)')
axes6[1, 2].set_ylabel('Δ NIQE Gain', fontweight='bold')
axes6[1, 2].set_title('Mức Cải Thiện Tự Nhiên (ΔNIQE Gain)', fontsize=11, fontweight='bold', pad=8)
axes6[1, 2].set_xticks(x)
axes6[1, 2].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[1, 2].set_ylim(-10.5, 2.5)
axes6[1, 2].legend(frameon=True, loc='upper right', fontsize=8, facecolor='white', framealpha=0.95)
axes6[1, 2].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[1, 2].text(x[i] - w/2, gain_nq_src[i] - 0.7, f'{gain_nq_src[i]:+.2f}', ha='center', fontsize=8.5, fontweight='bold')
    axes6[1, 2].text(x[i] + w/2, gain_nq_srg[i] + 0.25, f'{gain_nq_srg[i]:+.2f}', ha='center', fontsize=8.5, fontweight='bold')

fig6.suptitle('BIỂU ĐỒ CHUYÊN BIỆT: ĐỐI CHỨNG VỚI THUẬT TOÁN NỀN BICUBIC', fontsize=15, fontweight='bold', y=0.99)
fig6.tight_layout()

# ─────────────────────────────────────────────────────────────
# 6. XUẤT RA FILE PDF CHÍNH
# ─────────────────────────────────────────────────────────────
NEW_FILENAME = 'SRCNN_vs_Swift_SRGAN_Benchmark_Analysis.pdf'
PDF_TARGET_1 = os.path.join('/Users/giabao/Desktop/GitHub/code hardware', NEW_FILENAME)
PDF_TARGET_2 = os.path.join(os.path.expanduser('~/Downloads'), NEW_FILENAME)

for target_path in [PDF_TARGET_1, PDF_TARGET_2]:
    with PdfPages(target_path) as pdf:
        pdf.savefig(fig4, bbox_inches='tight', facecolor='white')
        pdf.savefig(fig_bicubic_table, bbox_inches='tight', facecolor='white')
        pdf.savefig(fig5, bbox_inches='tight', facecolor='white')
        pdf.savefig(fig6, bbox_inches='tight', facecolor='white')
    print(f'✓ Đã xuất PDF thành công: {target_path}')

# Cập nhật ảnh PNG
chart_models_path = os.path.join(OUTPUT_SPLIT_DIR, 'so_sanh_2_mo_hinh_srcnn_vs_srgan.png')
fig5.savefig(chart_models_path, dpi=FIG_DPI, bbox_inches='tight')
chart_bicubic_path = os.path.join(OUTPUT_SPLIT_DIR, 'so_sanh_chuyen_biet_voi_bicubic.png')
fig6.savefig(chart_bicubic_path, dpi=FIG_DPI, bbox_inches='tight')

plt.close('all')
print('=' * 70)
print('🎉 HOÀN TẤT: Đã sửa triệt để lỗi đè chữ / dính chữ trên tất cả các bảng!')
print('=' * 70)
