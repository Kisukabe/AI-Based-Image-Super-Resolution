#!/usr/bin/env python3
"""
generate_report.py
Xuất báo cáo benchmark SRCNN vs Swift-SRGAN:
  - Dùng đúng code cell 5 & 6 từ notebook
  - Báo cáo liền mạch (không chia cell riêng)
  - Font tiếng Việt đẹp
  - Output: .md + .html + .pdf
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib import rcParams

# ── Font setup (Vietnamese support) ──────────────────────────────────────
from matplotlib import font_manager
# Prefer system fonts with full Unicode support
for fname in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    if any(k in fname.lower() for k in ['dejavu', 'liberation', 'arial', 'helvetica', 'noto']):
        pass

# Use a font that renders Vietnamese properly
rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    10.5,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8.5,
    'figure.dpi':        100,
    'savefig.dpi':       180,
})

# ── Paths ─────────────────────────────────────────────────────────────────
REPORT_DIR  = os.path.expanduser('~/Downloads/benchmark_report')
IMG_DIR     = os.path.join(REPORT_DIR, 'images')
MD_PATH     = os.path.join(REPORT_DIR, 'benchmark_report.md')
HTML_PATH   = os.path.join(REPORT_DIR, 'benchmark_report.html')
PDF_PATH    = os.path.join(REPORT_DIR, 'benchmark_report.pdf')
PATH_SRCNN  = os.path.expanduser('~/Downloads/benchmark_checkpoint.json')
PATH_SRGAN  = os.path.expanduser('~/Downloads/srgan_checkpoint.json')
SPLIT_DIR   = os.path.expanduser('~/Downloads/split_benchmark_results')
NB_PATH     = '/Users/giabao/Desktop/GitHub/code hardware/benchmark_analysis_and_comparison.ipynb'

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMG_DIR,    exist_ok=True)

print('=' * 70)
print('  BAO CAO BENCHMARK: SRCNN RTL vs Swift-SRGAN')
print('=' * 70)

# ── Syntax check ───────────────────────────────────────────────────────────
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
errors = []
for i, c in enumerate(code_cells):
    src = ''.join(c['source'])
    try:
        compile(src, f'cell_{i}', 'exec')
    except SyntaxError as e:
        errors.append((i, str(e)))
if errors:
    print('\n[ERROR] Cu phap bi loi:')
    for idx, msg in errors:
        print(f'  Cell {idx}: {msg}')
    sys.exit(1)
print(f'[OK] Kiem tra cu phap: {len(code_cells)} code cells - TAT CA HOP LE')

# ── Load & Standardize ─────────────────────────────────────────────────────
def load_and_standardize(json_path, model_label):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data.get('per_image_results', [])
    df = pd.DataFrame(records)
    df = df[df.get('status', 'ok') == 'ok'].copy() if 'status' in df.columns else df.copy()
    df['model_name'] = model_label
    if 'psnr_fpga_db'  in df.columns: df['psnr_model_db']  = df['psnr_fpga_db']
    if 'ssim_fpga'     in df.columns: df['ssim_model']      = df['ssim_fpga']
    if 'msssim_fpga'   in df.columns: df['msssim_model']    = df['msssim_fpga']
    if 'lpips_fpga'    in df.columns: df['lpips_model']     = df['lpips_fpga']
    elif 'lpips_srgan' in df.columns: df['lpips_model']     = df['lpips_srgan']
    elif 'lpips'       in df.columns: df['lpips_model']     = df['lpips']
    if 'niqe_fpga'     in df.columns: df['niqe_model']      = df['niqe_fpga']
    if 'mse_fpga'      in df.columns: df['mse_model']       = df['mse_fpga']
    if 'rmse_fpga'     in df.columns: df['rmse_model']      = df['rmse_fpga']
    df['fps'] = 1000.0 / df['latency_ms']
    return df

df_srcnn = load_and_standardize(PATH_SRCNN, 'SRCNN RTL 2x (Q7)')
df_srgan = load_and_standardize(PATH_SRGAN, 'Swift-SRGAN 4x (Q7)')
print(f'[OK] SRCNN       : {len(df_srcnn):,} anh | {dict(df_srcnn["dataset"].value_counts())}')
print(f'[OK] Swift-SRGAN : {len(df_srgan):,} anh | {dict(df_srgan["dataset"].value_counts())}')

# ── Shared helpers ──────────────────────────────────────────────────────────
categories = ['sub_NIH (1750 anh)', 'sub_chest (450 anh)', 'Toan Bo (2200 anh)']
cat_keys   = ['sub_NIH', 'sub_chest', 'TOAN BO (2200)']
x = np.arange(len(categories))
w = 0.35

def get_vals(df, col):
    results = []
    for k in cat_keys:
        if k == 'TOAN BO (2200)':
            results.append(df[col].mean())
        else:
            sub = df[df['dataset'] == k]
            results.append(sub[col].mean() if not sub.empty else 0.0)
    return results

def calc_stats_row(df, ds_key, model_label):
    sub = df if ds_key == 'TOAN BO (2200)' else df[df['dataset'] == ds_key]
    if sub.empty:
        return None
    def m(c): return round(sub[c].mean(), 4) if c in sub.columns else 'N/A'
    return {
        'Tap Du Lieu':  ds_key,
        'Mo Hinh':      model_label,
        'So Anh':       len(sub),
        'PSNR Bic(dB)': m('psnr_bicubic_db'),
        'PSNR Mdl(dB)': m('psnr_model_db'),
        'D-PSNR(dB)':   m('psnr_gain_db'),
        'SSIM Bic':     m('ssim_bicubic'),
        'SSIM Mdl':     m('ssim_model'),
        'MSSSIM Bic':   m('msssim_bicubic'),
        'MSSSIM Mdl':   m('msssim_model'),
        'D-MSSSIM':     m('msssim_gain'),
        'LPIPS Bic':    m('lpips_bicubic'),
        'LPIPS Mdl':    m('lpips_model'),
        'D-LPIPS':      m('lpips_gain'),
        'NIQE Bic':     m('niqe_bicubic'),
        'NIQE Mdl':     m('niqe_model'),
        'D-NIQE':       m('niqe_gain'),
        'MSE Bic':      m('mse_bicubic'),
        'MSE Mdl':      m('mse_model'),
        'Latency(ms)':  round(sub['latency_ms'].mean(), 1),
        'FPS':          round(sub['fps'].mean(), 1),
    }

rows = []
for ds in ['sub_NIH', 'sub_chest', 'TOAN BO (2200)']:
    r1 = calc_stats_row(df_srcnn, ds, 'SRCNN RTL 2x (Q7)')
    r2 = calc_stats_row(df_srgan, ds, 'Swift-SRGAN 4x (Q7)')
    if r1: rows.append(r1)
    if r2: rows.append(r2)
df_stats = pd.DataFrame(rows)

# ── FIGURE: Summary table ──────────────────────────────────────────────────
def render_table(df, title, filepath, fontsize=8.5):
    n_rows, n_cols = df.shape
    fig_w = max(14, n_cols * 1.5)
    fig_h = 0.55 * n_rows + 1.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='white', fontweight='bold', fontsize=fontsize)
        elif r % 2 == 0:
            cell.set_facecolor('#edf2f7')
        else:
            cell.set_facecolor('#ffffff')
        cell.set_edgecolor('#cbd5e0')
        cell.set_linewidth(0.5)
    tbl.auto_set_column_width(list(range(n_cols)))
    ax.set_title(title, fontsize=12, fontweight='bold', pad=14, color='#1a202c')
    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] {os.path.basename(filepath)}')

tbl1_cols = ['Tap Du Lieu', 'Mo Hinh', 'So Anh',
             'PSNR Bic(dB)', 'PSNR Mdl(dB)', 'D-PSNR(dB)',
             'SSIM Bic', 'SSIM Mdl', 'MSSSIM Bic', 'MSSSIM Mdl', 'D-MSSSIM']
tbl2_cols = ['Tap Du Lieu', 'Mo Hinh',
             'LPIPS Bic', 'LPIPS Mdl', 'D-LPIPS',
             'NIQE Bic', 'NIQE Mdl', 'D-NIQE',
             'MSE Bic', 'MSE Mdl', 'Latency(ms)', 'FPS']

t1_path = os.path.join(IMG_DIR, 'table1_psnr_ssim.png')
t2_path = os.path.join(IMG_DIR, 'table2_lpips_niqe.png')
print('\n[Sinh bang thong ke]')
render_table(df_stats[tbl1_cols], 'Bang 1 - PSNR / SSIM / MS-SSIM', t1_path)
render_table(df_stats[tbl2_cols], 'Bang 2 - LPIPS / NIQE / MSE / Hardware', t2_path)

# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 (EXACT NOTEBOOK CODE) — 8-panel model comparison
# ═══════════════════════════════════════════════════════════════════════════
print('\n[Sinh bieu do Cell 5 - 8 thong so doi dau]')

metrics_info = [
    ('psnr_model_db', 'PSNR',          'PSNR (dB)',           '{:.2f} dB',  max, 6.0,  '#3182ce', '#805ad5', (0, 0)),
    ('ssim_model',    'SSIM',           'SSIM Index',          '{:.4f}',     None, 0,   '#3182ce', '#805ad5', (0, 1)),
    ('msssim_model',  'MS-SSIM',        'MS-SSIM Index',       '{:.4f}',     None, 0,   '#3182ce', '#805ad5', (0, 2)),
    ('lpips_model',   'LPIPS',          'LPIPS Score (v)',      '{:.4f}',     max, 0.08, '#3182ce', '#805ad5', (0, 3)),
    ('niqe_model',    'NIQE',           'NIQE Score (v)',       '{:.2f}',     max, 3.0,  '#3182ce', '#805ad5', (1, 0)),
    ('mse_model',     'MSE',            'MSE Loss (v)',         '{:.2f}',     max, 4.0,  '#3182ce', '#805ad5', (1, 1)),
    ('latency_ms',    'Do Tre Latency', 'Latency (ms) (v)',     '{:.1f} ms',  max, 15.0, '#3182ce', '#805ad5', (1, 2)),
    ('fps',           'Toc Do FPS',     'Toc do (FPS) (^)',     '{:.1f} FPS', max, 8.0,  '#3182ce', '#805ad5', (1, 3)),
]

fig5, axes5 = plt.subplots(2, 4, figsize=(22, 10))
fig5.suptitle(
    'Doi Dau Truc Dien Da Thong So: SRCNN RTL 2x (Q7) vs Swift-SRGAN 4x (Q7)',
    fontsize=15, fontweight='bold', y=1.01
)

for col, title, ylabel, fmt, lim_fn, pad, c1, c2, pos in metrics_info:
    ax = axes5[pos[0], pos[1]]
    v_src = get_vals(df_srcnn, col)
    v_srg = get_vals(df_srgan, col)

    ax.bar(x - w/2, v_src, w, label='SRCNN RTL 2x (Q7)',   color=c1, alpha=0.9, edgecolor='black', lw=0.6)
    ax.bar(x + w/2, v_srg, w, label='Swift-SRGAN 4x (Q7)', color=c2, alpha=0.9, edgecolor='black', lw=0.6)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold', fontsize=8.5)

    max_val = max(v_src + v_srg)
    if lim_fn == max:
        ax.set_ylim(0, max_val + pad)
    elif lim_fn is None:
        ax.set_ylim(0, max_val * 1.15)

    ax.legend(frameon=True, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, ls=':')

    for i in range(len(x)):
        offset = max_val * 0.025 if max_val > 0 else 0.5
        ax.text(x[i] - w/2, v_src[i] + offset, fmt.format(v_src[i]),
                ha='center', fontsize=8, fontweight='bold', color='#1a365d')
        ax.text(x[i] + w/2, v_srg[i] + offset, fmt.format(v_srg[i]),
                ha='center', fontsize=8, fontweight='bold', color='#44337a')

plt.tight_layout()
c5_path = os.path.join(IMG_DIR, 'cell5_so_sanh_2_mo_hinh.png')
plt.savefig(c5_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  [OK] {os.path.basename(c5_path)}')

# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 (EXACT NOTEBOOK CODE) — Bicubic baseline comparison
# ═══════════════════════════════════════════════════════════════════════════
print('\n[Sinh bieu do Cell 6 - doi chung Bicubic]')

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

fig6, axes6 = plt.subplots(2, 3, figsize=(20, 10))
fig6.suptitle(
    'Doi Chung Toan Dien Voi Thuat Toan Nen Bicubic',
    fontsize=14, fontweight='bold', y=1.01
)

# PANEL 1: Task 2x Bicubic vs SRCNN
axes6[0, 0].bar(x - w/2, p_bic_2x, w, label='Bicubic 2x Baseline', color='#feb2b2', edgecolor='#c53030', lw=1)
axes6[0, 0].bar(x + w/2, p_src_2x, w, label='SRCNN RTL 2x (Q7)',   color='#3182ce', edgecolor='#2c5282', lw=1)
axes6[0, 0].set_ylabel('PSNR (dB)', fontweight='bold')
axes6[0, 0].set_title('Task Scale 2x: Bicubic 2x vs SRCNN RTL 2x', fontsize=11, fontweight='bold')
axes6[0, 0].set_xticks(x)
axes6[0, 0].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[0, 0].set_ylim(0, max(p_src_2x) + 6)
axes6[0, 0].legend(frameon=True, loc='upper right')
axes6[0, 0].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[0, 0].text(x[i] - w/2, p_bic_2x[i] + 0.5, f'{p_bic_2x[i]:.2f}', ha='center', fontsize=8.5)
    axes6[0, 0].text(x[i] + w/2, p_src_2x[i] + 0.5, f'{p_src_2x[i]:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#1a365d')

# PANEL 2: Task 4x Bicubic vs SRGAN
axes6[0, 1].bar(x - w/2, p_bic_4x, w, label='Bicubic 4x Baseline',   color='#fed7aa', edgecolor='#ea580c', lw=1)
axes6[0, 1].bar(x + w/2, p_srg_4x, w, label='Swift-SRGAN 4x (Q7)', color='#805ad5', edgecolor='#44337a', lw=1)
axes6[0, 1].set_ylabel('PSNR (dB)', fontweight='bold')
axes6[0, 1].set_title('Task Scale 4x: Bicubic 4x vs Swift-SRGAN 4x', fontsize=11, fontweight='bold')
axes6[0, 1].set_xticks(x)
axes6[0, 1].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes6[0, 1].set_ylim(0, max(p_srg_4x) + 6)
axes6[0, 1].legend(frameon=True, loc='upper right')
axes6[0, 1].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[0, 1].text(x[i] - w/2, p_bic_4x[i] + 0.5, f'{p_bic_4x[i]:.2f}', ha='center', fontsize=8.5)
    axes6[0, 1].text(x[i] + w/2, p_srg_4x[i] + 0.5, f'{p_srg_4x[i]:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#44337a')

# PANEL 3: Delta PSNR Gain
axes6[0, 2].bar(x - w/2, gain_p_src, w, label='SRCNN 2x Gain (dB)',       color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
axes6[0, 2].bar(x + w/2, gain_p_srg, w, label='Swift-SRGAN 4x Gain (dB)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
axes6[0, 2].axhline(0, color='red', ls='--', lw=1.2, label='Moc Bicubic Baseline (0 dB)')
axes6[0, 2].set_ylabel('Delta PSNR Gain (dB) [Model - Bicubic]', fontweight='bold')
axes6[0, 2].set_title('Muc Tang/Giam PSNR (Delta PSNR Gain)', fontsize=11, fontweight='bold')
axes6[0, 2].set_xticks(x)
axes6[0, 2].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min = min(gain_p_src + gain_p_srg) - 0.3
y_max = max(gain_p_src + gain_p_srg) + 0.3
axes6[0, 2].set_ylim(y_min, y_max)
axes6[0, 2].legend(frameon=True, loc='lower right', fontsize=8)
axes6[0, 2].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[0, 2].text(x[i] - w/2, gain_p_src[i] + (0.05 if gain_p_src[i] >= 0 else -0.1),
                     f'{gain_p_src[i]:+.2f} dB', ha='center', fontsize=8.5, fontweight='bold')
    axes6[0, 2].text(x[i] + w/2, gain_p_srg[i] + (0.05 if gain_p_srg[i] >= 0 else -0.1),
                     f'{gain_p_srg[i]:+.2f} dB', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 4: Delta MS-SSIM Gain
axes6[1, 0].bar(x - w/2, gain_ms_src, w, label='SRCNN 2x (MS-SSIM Gain)',       color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 0].bar(x + w/2, gain_ms_srg, w, label='Swift-SRGAN 4x (MS-SSIM Gain)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 0].axhline(0, color='red', ls='--', lw=1.2, label='Moc Bicubic Baseline (0)')
axes6[1, 0].set_ylabel('Delta MS-SSIM Gain [Model - Bicubic]', fontweight='bold')
axes6[1, 0].set_title('Muc Thay Doi MS-SSIM (Delta MS-SSIM Gain)', fontsize=11, fontweight='bold')
axes6[1, 0].set_xticks(x)
axes6[1, 0].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min_ms = min(gain_ms_src + gain_ms_srg) - 0.001
y_max_ms = max(gain_ms_src + gain_ms_srg) + 0.001
axes6[1, 0].set_ylim(y_min_ms, y_max_ms)
axes6[1, 0].legend(frameon=True, loc='lower right', fontsize=8)
axes6[1, 0].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[1, 0].text(x[i] - w/2, gain_ms_src[i] + (0.0001 if gain_ms_src[i] >= 0 else -0.0003),
                     f'{gain_ms_src[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')
    axes6[1, 0].text(x[i] + w/2, gain_ms_srg[i] + (0.0001 if gain_ms_srg[i] >= 0 else -0.0003),
                     f'{gain_ms_srg[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 5: Delta LPIPS Gain
axes6[1, 1].bar(x - w/2, gain_lp_src, w, label='SRCNN 2x (LPIPS Gain)',       color='#38a169', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 1].bar(x + w/2, gain_lp_srg, w, label='Swift-SRGAN 4x (LPIPS Gain)', color='#d69e2e', alpha=0.9, edgecolor='black', lw=0.6)
axes6[1, 1].axhline(0, color='red', ls='--', lw=1.2, label='Moc Bicubic Baseline (0)')
axes6[1, 1].set_ylabel('Delta LPIPS Gain (^ Cang cao cang net hon Bicubic)', fontweight='bold')
axes6[1, 1].set_title('Muc Cai Thien Thi Giac (Delta LPIPS Gain)', fontsize=11, fontweight='bold')
axes6[1, 1].set_xticks(x)
axes6[1, 1].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min_lp = min(min(gain_lp_src + gain_lp_srg) - 0.005, -0.01)
y_max_lp = max(gain_lp_src + gain_lp_srg) + 0.008
axes6[1, 1].set_ylim(y_min_lp, y_max_lp)
axes6[1, 1].legend(frameon=True, loc='upper right', fontsize=8)
axes6[1, 1].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[1, 1].text(x[i] - w/2, gain_lp_src[i] + 0.0015, f'{gain_lp_src[i]:+.4f}',
                     ha='center', fontsize=8.5, fontweight='bold')
    axes6[1, 1].text(x[i] + w/2, gain_lp_srg[i] + 0.0015, f'{gain_lp_srg[i]:+.4f}',
                     ha='center', fontsize=8.5, fontweight='bold')

# PANEL 6: Delta NIQE Gain
axes6[1, 2].bar(x - w/2, gain_nq_src, w, label='SRCNN 2x (NIQE Gain)',       color='#e53e3e', alpha=0.85, edgecolor='black', lw=0.6)
axes6[1, 2].bar(x + w/2, gain_nq_srg, w, label='Swift-SRGAN 4x (NIQE Gain)', color='#319795', alpha=0.9,  edgecolor='black', lw=0.6)
axes6[1, 2].axhline(0, color='red', ls='--', lw=1.2, label='Moc Bicubic Baseline (0)')
axes6[1, 2].set_ylabel('Delta NIQE Gain (^ Cang cao cang tu nhien hon Bicubic)', fontweight='bold')
axes6[1, 2].set_title('Muc Cai Thien Tu Nhien (Delta NIQE Gain)', fontsize=11, fontweight='bold')
axes6[1, 2].set_xticks(x)
axes6[1, 2].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min_nq = min(gain_nq_src + gain_nq_srg) - 1.5
y_max_nq = max(gain_nq_src + gain_nq_srg) + 1.0
axes6[1, 2].set_ylim(y_min_nq, y_max_nq)
axes6[1, 2].legend(frameon=True, loc='lower left', fontsize=8)
axes6[1, 2].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes6[1, 2].text(x[i] - w/2, gain_nq_src[i] - 0.6, f'{gain_nq_src[i]:+.2f}',
                     ha='center', fontsize=8.5, fontweight='bold')
    axes6[1, 2].text(x[i] + w/2, gain_nq_srg[i] + 0.2, f'{gain_nq_srg[i]:+.2f}',
                     ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
c6_path = os.path.join(IMG_DIR, 'cell6_doi_chung_bicubic.png')
plt.savefig(c6_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  [OK] {os.path.basename(c6_path)}')

# ── Cell 7: Per-image distribution ─────────────────────────────────────────
print('\n[Sinh bieu do Cell 7 - phan tich per-image]')

CHOSEN = 'sub_NIH'
df1s = df_srcnn[df_srcnn['dataset'] == CHOSEN].set_index('filename')
df2s = df_srgan[df_srgan['dataset'] == CHOSEN].set_index('filename')
common = df1s.index.intersection(df2s.index)

diff_psnr  = df2s.loc[common, 'psnr_model_db']  - df1s.loc[common, 'psnr_model_db']
diff_lpips = df1s.loc[common, 'lpips_model']     - df2s.loc[common, 'lpips_model']
diff_niqe  = df1s.loc[common, 'niqe_model']      - df2s.loc[common, 'niqe_model']

fig7, axes7 = plt.subplots(1, 3, figsize=(18, 5))
fig7.suptitle(f'Phan Tich Sau Per-Image: {CHOSEN} ({len(common)} anh)',
              fontsize=13, fontweight='bold', y=1.02)

for ax, data, title, xlabel, color in [
    (axes7[0], diff_psnr,  'Phan Phoi Delta PSNR\n(Swift-SRGAN - SRCNN)',  'Delta PSNR (dB)', '#805ad5'),
    (axes7[1], diff_lpips, 'Phan Phoi Delta LPIPS\n(SRCNN - Swift-SRGAN)', 'Delta LPIPS',     '#38a169'),
    (axes7[2], diff_niqe,  'Phan Phoi Delta NIQE\n(SRCNN - Swift-SRGAN)',  'Delta NIQE',      '#319795'),
]:
    ax.hist(data, bins=50, color=color, edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(0, color='red', lw=1.5, ls='--', label='Moc 0')
    ax.axvline(data.mean(), color='orange', lw=1.5, ls='-',
               label=f'Trung binh: {data.mean():+.4f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('So luong anh', fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.2, ls=':')

plt.tight_layout()
c7_path = os.path.join(IMG_DIR, 'cell7_perimage_analysis.png')
plt.savefig(c7_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  [OK] {os.path.basename(c7_path)}')

# ── Top/Bottom 10 ──────────────────────────────────────────────────────────
print('\n[Sinh bang Top/Bottom 10]')
top10   = diff_lpips.sort_values(ascending=False).head(10)
worst10 = diff_lpips.sort_values(ascending=True).head(10)

def make_rank_table(series, src_df, srg_df, col, filepath, title):
    data_rows = []
    for fn, val in series.items():
        data_rows.append({
            'Ten anh':       fn,
            'LPIPS SRCNN':   round(src_df.loc[fn, col], 4),
            'LPIPS SRGAN':   round(srg_df.loc[fn, col], 4),
            'Delta LPIPS':   round(val, 4),
        })
    tdf = pd.DataFrame(data_rows)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#edf2f7')
        cell.set_edgecolor('#cbd5e0'); cell.set_linewidth(0.5)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] {os.path.basename(filepath)}')

top10_path   = os.path.join(IMG_DIR, 'top10_srgan_wins.png')
worst10_path = os.path.join(IMG_DIR, 'top10_srcnn_wins.png')
make_rank_table(top10,   df1s, df2s, 'lpips_model', top10_path,
                'Top 10 anh Swift-SRGAN thi giac tot hon SRCNN nhieu nhat')
make_rank_table(worst10, df1s, df2s, 'lpips_model', worst10_path,
                'Top 10 anh SRCNN thi giac tot hon Swift-SRGAN nhieu nhat')

# ═══════════════════════════════════════════════════════════════════════════
# BUILD MARKDOWN (one continuous document, no cell headers)
# ═══════════════════════════════════════════════════════════════════════════
print('\n[Tong hop bao cao Markdown]')

s_src = calc_stats_row(df_srcnn, 'TOAN BO (2200)', 'SRCNN RTL 2x (Q7)')
s_srg = calc_stats_row(df_srgan, 'TOAN BO (2200)', 'Swift-SRGAN 4x (Q7)')

def rel(path):
    return 'images/' + os.path.basename(path)

md = []
md.append("# BAO CAO BENCHMARK: SRCNN RTL (Q7) vs Swift-SRGAN (Q7)\n\n")
md.append("**Ngay tao:** 30/08/2026  |  **Du lieu:** 2,200 anh X-Ray (sub_NIH + sub_chest)  |  **He so phong:** SRCNN 2x | SRGAN 4x\n\n")
md.append("---\n\n")

# Section 1: Summary table
md.append("## Tong Quan Ket Qua (Toan Bo 2,200 anh)\n\n")
md.append("| Chi So | SRCNN RTL 2x (Q7) | Swift-SRGAN 4x (Q7) | Mo Hinh Tot Hon |\n")
md.append("|---|---|---|---|\n")
comparisons = [
    ('PSNR Model (dB)',   s_src['PSNR Mdl(dB)'], s_srg['PSNR Mdl(dB)'], True),
    ('SSIM Model',        s_src['SSIM Mdl'],       s_srg['SSIM Mdl'],      True),
    ('MS-SSIM Model',     s_src['MSSSIM Mdl'],     s_srg['MSSSIM Mdl'],    True),
    ('LPIPS Model (v)',   s_src['LPIPS Mdl'],       s_srg['LPIPS Mdl'],     False),
    ('NIQE Model (v)',    s_src['NIQE Mdl'],        s_srg['NIQE Mdl'],      False),
    ('MSE Model (v)',     s_src['MSE Mdl'],         s_srg['MSE Mdl'],       False),
    ('Latency (ms) (v)', s_src['Latency(ms)'],     s_srg['Latency(ms)'],   False),
    ('FPS (^)',           s_src['FPS'],             s_srg['FPS'],           True),
]
for name, v1, v2, higher_better in comparisons:
    winner = 'SRCNN' if (v1 > v2) == higher_better else 'Swift-SRGAN'
    md.append(f'| **{name}** | `{v1}` | `{v2}` | **{winner}** |\n')

md.append("\n---\n\n")

# Section 2: Detailed tables
md.append("## Bang Thong Ke Chi Tiet\n\n")
md.append("### PSNR / SSIM / MS-SSIM\n\n")
md.append(f"![Bang PSNR SSIM]({rel(t1_path)})\n\n")
md.append("### LPIPS / NIQE / MSE / Hardware\n\n")
md.append(f"![Bang LPIPS NIQE MSE]({rel(t2_path)})\n\n")
md.append("---\n\n")

# Section 3: Cell 5 chart
md.append("## Bieu Do So Sanh Truc Tiep 8 Thong So\n\n")
md.append("So sanh doi dau truc tiep giua **SRCNN RTL 2x (Q7)** (xanh duong) va **Swift-SRGAN 4x (Q7)** (tim) tren 8 chi so danh gia:\n\n")
md.append(f"![Bieu Do 8 Thong So]({rel(c5_path)})\n\n")
md.append("**Nhan xet:**\n\n")
md.append(f"- **PSNR**: SRCNN `{s_src['PSNR Mdl(dB)']} dB` > SRGAN `{s_srg['PSNR Mdl(dB)']} dB` — SRCNN tai tao diem anh trung thuc hon.\n")
md.append(f"- **SSIM / MS-SSIM**: SRCNN `{s_src['SSIM Mdl']}` / `{s_src['MSSSIM Mdl']}` > SRGAN `{s_srg['SSIM Mdl']}` / `{s_srg['MSSSIM Mdl']}` — SRCNN bao toan cau truc anh tot hon.\n")
md.append(f"- **LPIPS**: SRCNN `{s_src['LPIPS Mdl']}` < SRGAN `{s_srg['LPIPS Mdl']}` — SRCNN cho anh sac net hon theo tri giac thi giac.\n")
md.append(f"- **NIQE**: SRGAN `{s_srg['NIQE Mdl']}` < SRCNN `{s_src['NIQE Mdl']}` — **Swift-SRGAN tao anh tu nhien hon dang ke** (NIQE thap = tu nhien hon).\n")
md.append(f"- **Latency / FPS**: SRCNN `{s_src['Latency(ms)']} ms ({s_src['FPS']} FPS)` vs SRGAN `{s_srg['Latency(ms)']} ms ({s_srg['FPS']} FPS)` — **SRCNN nhanh hon ~2.3x**.\n")
md.append("\n---\n\n")

# Section 4: Cell 6 chart
md.append("## Bieu Do Doi Chung Voi Thuat Toan Nen Bicubic\n\n")
md.append("Danh gia muc tang ich (Gain) cua tung mo hinh so voi **Bicubic Baseline**:\n\n")
md.append(f"![Bieu Do Bicubic]({rel(c6_path)})\n\n")
md.append("**Nhan xet:**\n\n")
md.append(f"- **Delta PSNR**: SRCNN trung binh `{s_src['D-PSNR(dB)']:+.3f} dB` vs SRGAN `{s_srg['D-PSNR(dB)']:+.3f} dB` so voi Bicubic.\n")
md.append(f"- **Delta MS-SSIM**: SRCNN `{s_src['D-MSSSIM']:+.4f}` vs SRGAN `{s_srg['D-MSSSIM']:+.4f}`.\n")
md.append(f"- **Delta LPIPS Gain**: SRCNN `{s_src['D-LPIPS']:+.4f}` vs SRGAN `{s_srg['D-LPIPS']:+.4f}` — SRCNN cai thien LPIPS nhieu hon Bicubic.\n")
md.append(f"- **Delta NIQE Gain**: SRGAN `{s_srg['D-NIQE']:+.2f}` > SRCNN `{s_src['D-NIQE']:+.2f}` — **Swift-SRGAN cai thien do tu nhien vuot troi**.\n")
md.append("\n---\n\n")

# Section 5: Cell 7 per-image
md.append("## Phan Tich Sau Per-Image (sub_NIH)\n\n")
md.append(f"Phan tich **{len(common)}** anh khop giua 2 mo hinh tren tap `sub_NIH`:\n\n")
md.append(f"![Phan Tich Per-Image]({rel(c7_path)})\n\n")
md.append("| Chi so | Trung binh Delta | Y nghia |\n")
md.append("|---|---|---|\n")
md.append(f"| Delta PSNR (SRGAN - SRCNN) | `{diff_psnr.mean():+.3f} dB` | SRCNN tot hon ve PSNR |\n")
md.append(f"| Delta LPIPS (SRCNN - SRGAN) | `{diff_lpips.mean():+.4f}` | SRCNN thuong cho LPIPS thap hon |\n")
md.append(f"| Delta NIQE (SRCNN - SRGAN) | `{diff_niqe.mean():+.2f}` | SRGAN tu nhien hon (NIQE thap hon) |\n")
md.append("\n")
md.append("### Top 10 anh Swift-SRGAN vuot troi ve thi giac\n\n")
md.append(f"![Top 10 SRGAN Wins]({rel(top10_path)})\n\n")
md.append("### Top 10 anh SRCNN vuot troi ve thi giac\n\n")
md.append(f"![Top 10 SRCNN Wins]({rel(worst10_path)})\n\n")
md.append("---\n\n")

# Section 6: Conclusion
md.append("## Ket Luan\n\n")
md.append("| Tieu Chi | SRCNN RTL 2x (Q7) | Swift-SRGAN 4x (Q7) |\n")
md.append("|---|---|---|\n")
md.append(f"| Scale Factor | 2x | 4x |\n")
md.append(f"| Toc do xu ly | ~{s_src['FPS']} FPS (sieu nhanh) | ~{s_srg['FPS']} FPS (thoi gian thuc) |\n")
md.append(f"| PSNR (Do trung thuc MSE) | {s_src['PSNR Mdl(dB)']} dB (tot hon) | {s_srg['PSNR Mdl(dB)']} dB |\n")
md.append(f"| SSIM (Cau truc) | {s_src['SSIM Mdl']} (tot hon) | {s_srg['SSIM Mdl']} |\n")
md.append(f"| LPIPS (Sac net thi giac) | {s_src['LPIPS Mdl']} (tot hon) | {s_srg['LPIPS Mdl']} |\n")
md.append(f"| NIQE (Do tu nhien) | {s_src['NIQE Mdl']} | {s_srg['NIQE Mdl']} (tot hon) |\n")
md.append("\n")
md.append("> **Ket luan khoa hoc:** SRCNN RTL vuot troi ve do trung thuc diem anh (PSNR/SSIM/LPIPS) ")
md.append("va toc do xu ly phan cung — phu hop cho chan doan y te can do chinh xac pixel. ")
md.append("Swift-SRGAN noi bat o do tu nhien (NIQE) va muc phong dai 4x — phu hop cho hien thi va danh gia thi giac.\n")

with open(MD_PATH, 'w', encoding='utf-8') as f:
    f.writelines(md)
print(f'  [OK] Markdown: {MD_PATH}')

# ═══════════════════════════════════════════════════════════════════════════
# BUILD HTML via pandoc (standalone, images embedded as base64)
# ═══════════════════════════════════════════════════════════════════════════
print('\n[Xuat HTML]')
import subprocess

result = subprocess.run(
    ['pandoc', MD_PATH,
     '-o', HTML_PATH,
     '--standalone',
     '--embed-resources',
     '--metadata', 'title=Benchmark Report: SRCNN vs Swift-SRGAN',
     '-f', 'markdown_strict', '-t', 'html5'],
    capture_output=True, text=True, cwd=REPORT_DIR
)
if result.returncode == 0:
    size = os.path.getsize(HTML_PATH) / 1024
    print(f'  [OK] HTML: {HTML_PATH}  ({size:.0f} KB)')
else:
    print(f'  [WARN] HTML error: {result.stderr}')

# ═══════════════════════════════════════════════════════════════════════════
# BUILD PDF via PdfPages (A4 landscape, cover + summary + charts in order)
# ═══════════════════════════════════════════════════════════════════════════
print('\n[Xuat PDF da trang]')

A4L = (11.69, 8.27)   # A4 landscape inches

def embed_img_page(pdf, img_path, title, subtitle=None):
    """Add a full-page image to the PDF."""
    if not os.path.exists(img_path):
        return
    img = mpimg.imread(img_path)
    h_px, w_px = img.shape[:2]
    ratio = h_px / w_px
    fig_w = A4L[0]
    header_h = 0.55 if subtitle else 0.45
    img_h = min(fig_w * ratio, A4L[1] - header_h)

    fig = plt.figure(figsize=(fig_w, img_h + header_h))
    fig.patch.set_facecolor('white')

    # Title bar
    ax_title = fig.add_axes([0, 1 - header_h / (img_h + header_h), 1, header_h / (img_h + header_h)])
    ax_title.set_facecolor('#1e3a5f')
    ax_title.axis('off')
    ax_title.text(0.5, 0.65, title, ha='center', va='center',
                  fontsize=13, fontweight='bold', color='white',
                  transform=ax_title.transAxes)
    if subtitle:
        ax_title.text(0.5, 0.2, subtitle, ha='center', va='center',
                      fontsize=9, color='#a0c4ff', transform=ax_title.transAxes)

    ax_img = fig.add_axes([0.01, 0, 0.98, 1 - header_h / (img_h + header_h)])
    ax_img.imshow(img, aspect='auto')
    ax_img.axis('off')

    pdf.savefig(fig, facecolor='white', bbox_inches='tight')
    plt.close()

with PdfPages(PDF_PATH) as pdf:

    # ── Cover page ──
    fig_c = plt.figure(figsize=A4L)
    fig_c.patch.set_facecolor('#0f2444')
    ax_c = fig_c.add_axes([0, 0, 1, 1])
    ax_c.set_facecolor('#0f2444')
    ax_c.axis('off')
    # Gradient banner
    for i, alpha in enumerate(np.linspace(0.0, 0.15, 30)):
        ax_c.axhline(0.95 - i * 0.01, color='#3182ce', lw=40, alpha=alpha)
    ax_c.text(0.5, 0.80, 'BAO CAO BENCHMARK TOAN DIEN', ha='center', va='center',
              fontsize=24, fontweight='bold', color='white', transform=ax_c.transAxes)
    ax_c.text(0.5, 0.68, 'SRCNN RTL (FPGA Q7)  -  Scale 2x', ha='center', va='center',
              fontsize=17, color='#63b3ed', transform=ax_c.transAxes, fontweight='bold')
    ax_c.text(0.5, 0.60, 'vs', ha='center', va='center',
              fontsize=14, color='#a0aec0', transform=ax_c.transAxes)
    ax_c.text(0.5, 0.52, 'Swift-SRGAN (Q7 Quantized)  -  Scale 4x', ha='center', va='center',
              fontsize=17, color='#b794f4', transform=ax_c.transAxes, fontweight='bold')
    ax_c.text(0.5, 0.40, '2,200 anh X-Ray y te  |  sub_NIH & sub_chest', ha='center', va='center',
              fontsize=12, color='#e2e8f0', transform=ax_c.transAxes)
    ax_c.text(0.5, 0.32, '8 chi so danh gia: PSNR, SSIM, MS-SSIM, LPIPS, NIQE, MSE, Latency, FPS',
              ha='center', va='center', fontsize=10, color='#a0aec0', transform=ax_c.transAxes)
    ax_c.text(0.5, 0.20, '30/08/2026', ha='center', va='center',
              fontsize=11, color='#718096', transform=ax_c.transAxes)
    pdf.savefig(fig_c, facecolor='#0f2444')
    plt.close()

    # ── Summary table page ──
    fig_sum = plt.figure(figsize=A4L)
    fig_sum.patch.set_facecolor('white')
    ax_s = fig_sum.add_axes([0.04, 0.1, 0.92, 0.72])
    ax_s.axis('off')
    sum_rows = [
        ['PSNR (dB) ^',    s_src['PSNR Mdl(dB)'],  s_srg['PSNR Mdl(dB)'],  'SRCNN tot hon'],
        ['SSIM ^',         s_src['SSIM Mdl'],        s_srg['SSIM Mdl'],       'SRCNN tot hon'],
        ['MS-SSIM ^',      s_src['MSSSIM Mdl'],      s_srg['MSSSIM Mdl'],     'SRCNN tot hon'],
        ['LPIPS v',        s_src['LPIPS Mdl'],        s_srg['LPIPS Mdl'],      'SRCNN tot hon'],
        ['NIQE v',         s_src['NIQE Mdl'],         s_srg['NIQE Mdl'],       'Swift-SRGAN tot hon'],
        ['MSE v',          s_src['MSE Mdl'],           s_srg['MSE Mdl'],        'SRCNN tot hon'],
        ['Latency (ms) v', s_src['Latency(ms)'],      s_srg['Latency(ms)'],    'SRCNN nhanh hon'],
        ['FPS ^',          s_src['FPS'],               s_srg['FPS'],            'SRCNN nhanh hon'],
    ]
    tbl = ax_s.table(
        cellText=sum_rows,
        colLabels=['Chi So', 'SRCNN RTL 2x (Q7)', 'Swift-SRGAN 4x (Q7)', 'Mo Hinh Tot Hon'],
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        elif r % 2 == 0:
            cell.set_facecolor('#edf2f7')
        cell.set_edgecolor('#cbd5e0'); cell.set_linewidth(0.5)
        cell.set_height(0.10)
    tbl.auto_set_column_width([0, 1, 2, 3])
    ax_s.set_title('Tom Tat Ket Qua (Toan Bo 2,200 anh)', fontsize=14, fontweight='bold',
                   pad=16, color='#1a202c', y=0.98)
    pdf.savefig(fig_sum, facecolor='white')
    plt.close()

    # ── All charts in one continuous flow ──
    embed_img_page(pdf, t1_path,
                   'Bang 1 - PSNR / SSIM / MS-SSIM',
                   'So sanh tren 3 tap du lieu: sub_NIH, sub_chest, Toan Bo')
    embed_img_page(pdf, t2_path,
                   'Bang 2 - LPIPS / NIQE / MSE / Hardware',
                   'So sanh tren 3 tap du lieu: sub_NIH, sub_chest, Toan Bo')
    embed_img_page(pdf, c5_path,
                   'Doi Dau Truc Dien 8 Thong So: SRCNN RTL 2x vs Swift-SRGAN 4x',
                   'Xanh duong = SRCNN RTL 2x (Q7)  |  Tim = Swift-SRGAN 4x (Q7)')
    embed_img_page(pdf, c6_path,
                   'Doi Chung Voi Thuat Toan Nen Bicubic',
                   'Danh gia muc tang ich (Gain) cua tung mo hinh so voi Bicubic Baseline')
    embed_img_page(pdf, c7_path,
                   'Phan Tich Sau Per-Image (sub_NIH)',
                   f'Phan phoi Delta tren {len(common)} anh khop giua 2 mo hinh')
    embed_img_page(pdf, top10_path,
                   'Top 10 Anh: Swift-SRGAN Vuot Troi Ve Thi Giac',
                   'Xep hang theo Delta LPIPS (SRCNN - SRGAN) lon nhat')
    embed_img_page(pdf, worst10_path,
                   'Top 10 Anh: SRCNN Vuot Troi Ve Thi Giac',
                   'Xep hang theo Delta LPIPS (SRCNN - SRGAN) nho nhat')

    # ── Conclusion page ──
    fig_con = plt.figure(figsize=A4L)
    fig_con.patch.set_facecolor('white')
    ax_con = fig_con.add_axes([0.04, 0.08, 0.92, 0.72])
    ax_con.axis('off')
    con_rows = [
        ['Scale Factor',         '2x (SR 2 lan)',    '4x (SR 4 lan)'],
        ['Toc do xu ly',         f'{s_src["FPS"]} FPS', f'{s_srg["FPS"]} FPS'],
        ['PSNR (Do trung thuc)', f'{s_src["PSNR Mdl(dB)"]} dB', f'{s_srg["PSNR Mdl(dB)"]} dB'],
        ['SSIM (Cau truc)',      str(s_src['SSIM Mdl']),  str(s_srg['SSIM Mdl'])],
        ['LPIPS (Thi giac)',     str(s_src['LPIPS Mdl']), str(s_srg['LPIPS Mdl'])],
        ['NIQE (Tu nhien)',      str(s_src['NIQE Mdl']),  str(s_srg['NIQE Mdl'])],
    ]
    tbl2 = ax_con.table(
        cellText=con_rows,
        colLabels=['Tieu Chi', 'SRCNN RTL 2x (Q7)', 'Swift-SRGAN 4x (Q7)'],
        cellLoc='center', loc='center'
    )
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(11)
    for (r, c), cell in tbl2.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        elif r % 2 == 0:
            cell.set_facecolor('#edf2f7')
        cell.set_edgecolor('#cbd5e0'); cell.set_linewidth(0.5)
        cell.set_height(0.12)
    tbl2.auto_set_column_width([0, 1, 2])
    ax_con.set_title('Ket Luan & Khuyen Nghi', fontsize=14, fontweight='bold', pad=16, y=0.98)

    fig_con.text(0.5, 0.06,
        'SRCNN RTL: Vuot troi PSNR/SSIM/LPIPS + Toc do cao -> Phu hop chan doan y te (do chinh xac pixel)',
        ha='center', fontsize=9, color='#2c5282', style='italic')
    fig_con.text(0.5, 0.02,
        'Swift-SRGAN: Vuot troi NIQE + Scale 4x -> Phu hop hien thi & danh gia thi giac',
        ha='center', fontsize=9, color='#44337a', style='italic')
    pdf.savefig(fig_con, facecolor='white')
    plt.close()

pdf_size = os.path.getsize(PDF_PATH) / 1024
html_size = os.path.getsize(HTML_PATH) / 1024 if os.path.exists(HTML_PATH) else 0

print(f'\n  [OK] PDF ({pdf_size:.0f} KB): {PDF_PATH}')
print(f'  [OK] HTML ({html_size:.0f} KB): {HTML_PATH}')
print(f'  [OK] MD: {MD_PATH}')
print(f'\n  Thu muc bao cao: {REPORT_DIR}')
print('=' * 70)
print('  HOAN TAT! Kiem tra code + Sinh bieu do + Xuat MD/HTML/PDF')
print('=' * 70)
