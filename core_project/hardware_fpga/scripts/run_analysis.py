import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('=' * 80)
print('🚀 BẮT ĐẦU PHÂN TÍCH & BÓC TÁCH BENCHMARK (SRCNN vs SWIFT-SRGAN vs BICUBIC)')
print('=' * 80)

# 1. Đường dẫn file JSON
PATH_SRCNN = os.path.expanduser('~/Downloads/benchmark_checkpoint.json')
PATH_SRGAN = os.path.expanduser('~/Downloads/srgan_checkpoint.json')

if not os.path.exists(PATH_SRCNN):
    found = glob.glob(os.path.expanduser('~/Downloads/*benchmark*.json')) + glob.glob(os.path.expanduser('~/Downloads/*srcnn*.json'))
    if found: PATH_SRCNN = found[0]

if not os.path.exists(PATH_SRGAN):
    found = glob.glob(os.path.expanduser('~/Downloads/*srgan*.json')) + glob.glob(os.path.expanduser('~/Downloads/*swift*.json'))
    if found: PATH_SRGAN = found[0]

print(f'📂 File SRCNN      : {PATH_SRCNN}')
print(f'📂 File Swift-SRGAN: {PATH_SRGAN}\n')

def load_and_standardize(json_path, model_label):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data.get('per_image_results', [])
    df = pd.DataFrame(records)
    df = df[df.get('status', 'ok') == 'ok'].copy()
    df['model_name'] = model_label
    
    # Đồng bộ hóa tên cột để thống nhất phân tích
    if 'psnr_fpga_db' in df.columns and 'psnr_model_db' not in df.columns:
        df['psnr_model_db'] = df['psnr_fpga_db']
    if 'ssim_fpga' in df.columns and 'ssim_model' not in df.columns:
        df['ssim_model'] = df['ssim_fpga']
    if 'msssim_fpga' in df.columns and 'msssim_model' not in df.columns:
        df['msssim_model'] = df['msssim_fpga']
    if 'lpips_fpga' in df.columns:
        df['lpips_model'] = df['lpips_fpga']
    elif 'lpips_srgan' in df.columns:
        df['lpips_model'] = df['lpips_srgan']
    elif 'lpips' in df.columns:
        df['lpips_model'] = df['lpips']
    if 'niqe_fpga' in df.columns and 'niqe_model' not in df.columns:
        df['niqe_model'] = df['niqe_fpga']
    if 'mse_fpga' in df.columns and 'mse_model' not in df.columns:
        df['mse_model'] = df['mse_fpga']
    if 'rmse_fpga' in df.columns and 'rmse_model' not in df.columns:
        df['rmse_model'] = df['rmse_fpga']
        
    df['fps'] = 1000.0 / df['latency_ms']
        
    return df, data.get('summary', {})

df_srcnn, sum_srcnn = load_and_standardize(PATH_SRCNN, 'SRCNN RTL (Q7)')
df_srgan, sum_srgan = load_and_standardize(PATH_SRGAN, 'Swift-SRGAN (Q7)')

print(f'✓ SRCNN      : {len(df_srcnn):,} ảnh | Tập dữ liệu: {dict(df_srcnn["dataset"].value_counts())}')
print(f'✓ Swift-SRGAN: {len(df_srgan):,} ảnh | Tập dữ liệu: {dict(df_srgan["dataset"].value_counts())}\n')

# 2. Bóc tách và xuất file
OUTPUT_SPLIT_DIR = os.path.expanduser('~/Downloads/split_benchmark_results')
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

def export_subsets(df, prefix):
    for ds in df['dataset'].unique():
        sub_df = df[df['dataset'] == ds].copy()
        json_out = os.path.join(OUTPUT_SPLIT_DIR, f'{prefix}_{ds}.json')
        csv_out  = os.path.join(OUTPUT_SPLIT_DIR, f'{prefix}_{ds}.csv')
        
        summary = {
            'dataset':            ds,
            'model':              sub_df['model_name'].iloc[0],
            'total_images':       len(sub_df),
            'avg_psnr_bicubic':   round(float(sub_df['psnr_bicubic_db'].mean()), 3),
            'avg_psnr_model':     round(float(sub_df['psnr_model_db'].mean()), 3),
            'avg_psnr_gain':      round(float(sub_df['psnr_gain_db'].mean()), 3),
            'avg_ssim_bicubic':   round(float(sub_df['ssim_bicubic'].mean()), 4),
            'avg_ssim_model':     round(float(sub_df['ssim_model'].mean()), 4),
            'avg_msssim_bicubic': round(float(sub_df['msssim_bicubic'].mean()), 4) if 'msssim_bicubic' in sub_df.columns else None,
            'avg_msssim_model':   round(float(sub_df['msssim_model'].mean()), 4) if 'msssim_model' in sub_df.columns else None,
            'avg_msssim_gain':    round(float(sub_df['msssim_gain'].mean()), 4) if 'msssim_gain' in sub_df.columns else None,
            'avg_lpips_bicubic':  round(float(sub_df['lpips_bicubic'].mean()), 4) if 'lpips_bicubic' in sub_df.columns else None,
            'avg_lpips_model':    round(float(sub_df['lpips_model'].mean()), 4) if 'lpips_model' in sub_df.columns else None,
            'avg_lpips_gain':     round(float(sub_df['lpips_gain'].mean()), 4) if 'lpips_gain' in sub_df.columns else None,
            'avg_niqe_bicubic':   round(float(sub_df['niqe_bicubic'].mean()), 4) if 'niqe_bicubic' in sub_df.columns else None,
            'avg_niqe_model':     round(float(sub_df['niqe_model'].mean()), 4) if 'niqe_model' in sub_df.columns else None,
            'avg_niqe_gain':      round(float(sub_df['niqe_gain'].mean()), 4) if 'niqe_gain' in sub_df.columns else None,
            'avg_mse_bicubic':    round(float(sub_df['mse_bicubic'].mean()), 3) if 'mse_bicubic' in sub_df.columns else None,
            'avg_mse_model':      round(float(sub_df['mse_model'].mean()), 3) if 'mse_model' in sub_df.columns else None,
            'avg_latency_ms':     round(float(sub_df['latency_ms'].mean()), 2),
            'fps':                round(float(sub_df['fps'].mean()), 1)
        }
        
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'per_image_results': sub_df.to_dict(orient='records')}, f, indent=2, ensure_ascii=False)
        sub_df.to_csv(csv_out, index=False)
        print(f'  • Đã xuất tập {ds:<10s} ({len(sub_df):>4d} ảnh) của [{prefix}] → {os.path.basename(json_out)}')

print('📂 Đang xuất các file phân tách theo tập...')
export_subsets(df_srcnn, 'srcnn')
export_subsets(df_srgan, 'swift_srgan')
print(f'✅ Toàn bộ file đã lưu tại: {OUTPUT_SPLIT_DIR}\n')

# 3. Bảng so sánh
def calc_stats_row(df, ds_name, model_label):
    sub = df[df['dataset'] == ds_name] if ds_name != 'TOÀN BỘ (2200)' else df
    if sub.empty: return None
    return {
        'Tập Dữ Liệu':       ds_name,
        'Mô Hình':           model_label,
        'Số Lượng Ảnh':      len(sub),
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
    r1 = calc_stats_row(df_srcnn, ds, 'SRCNN RTL (Q7)')
    r2 = calc_stats_row(df_srgan, ds, 'Swift-SRGAN (Q7)')
    if r1: rows.append(r1)
    if r2: rows.append(r2)

df_compare = pd.DataFrame(rows)
print('═' * 145)
print('             BẢNG SO SÁNH HIỆU NĂNG ĐỐI ĐẦU ĐẦY ĐỦ: SRCNN (Q7) vs SWIFT-SRGAN (Q7) vs BICUBIC')
print('═' * 145)
cols_show = [
    'Tập Dữ Liệu', 'Mô Hình', 'Số Lượng Ảnh',
    'PSNR Model (dB)', 'Δ PSNR Gain (dB)',
    'SSIM Model', 'MS-SSIM Model', 'Δ MS-SSIM Gain',
    'LPIPS Model (↓)', 'Δ LPIPS Gain (↑)',
    'NIQE Model (↓)', 'Δ NIQE Gain (↑)',
    'MSE Model', 'Latency (ms)', 'Tốc Độ (FPS)'
]
print(df_compare[cols_show].to_string(index=False))
print('═' * 145 + '\n')

categories = ['sub_NIH (1750 ảnh)', 'sub_chest (450 ảnh)', 'Toàn Bộ (2200 ảnh)']
cat_keys   = ['sub_NIH', 'sub_chest', 'TOÀN BỘ (2200)']
x = np.arange(len(categories))
w = 0.35

def get_vals(df, col):
    return [df[df['dataset']==k][col].mean() if k!='TOÀN BỘ (2200)' else df[col].mean() for k in cat_keys]

# ─────────────────────────────────────────────────────────────
# 4. CELL 5: Biểu đồ chuyên so sánh 2 mô hình (8 thông số)
# ─────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 4, figsize=(22, 10))

metrics_info = [
    ('psnr_model_db', '① PSNR (dB) — Càng cao càng tốt', 'PSNR (dB)', '{:.2f} dB', max, 6.0, '#3182ce', '#805ad5', axes1[0, 0]),
    ('ssim_model',    '② SSIM — Càng gần 1 càng tốt', 'SSIM Index', '{:.4f}', lambda v: 1.15, 0, '#3182ce', '#805ad5', axes1[0, 1]),
    ('msssim_model',  '③ MS-SSIM — Càng gần 1 càng tốt', 'MS-SSIM Index', '{:.4f}', lambda v: 1.05, 0, '#3182ce', '#805ad5', axes1[0, 2]),
    ('lpips_model',   '④ LPIPS — Càng THẤP càng sắc nét', 'LPIPS Score (↓)', '{:.4f}', max, 0.08, '#3182ce', '#805ad5', axes1[0, 3]),
    ('niqe_model',    '⑤ NIQE — Càng THẤP càng tự nhiên', 'NIQE Score (↓)', '{:.2f}', max, 3.0, '#3182ce', '#805ad5', axes1[1, 0]),
    ('mse_model',     '⑥ MSE — Càng THẤP sai số càng ít', 'MSE Loss (↓)', '{:.2f}', max, 4.0, '#3182ce', '#805ad5', axes1[1, 1]),
    ('latency_ms',    '⑦ Độ Trễ Latency — Càng thấp càng nhanh', 'Latency (ms) (↓)', '{:.1f} ms', max, 15.0, '#3182ce', '#805ad5', axes1[1, 2]),
    ('fps',           '⑧ Tốc Độ FPS — Càng cao càng mượt', 'Tốc độ (FPS) (↑)', '{:.1f} FPS', max, 8.0, '#3182ce', '#805ad5', axes1[1, 3])
]

for col, title, ylabel, fmt, lim_fn, pad, c1, c2, ax in metrics_info:
    v_src = get_vals(df_srcnn, col)
    v_srg = get_vals(df_srgan, col)
    
    ax.bar(x - w/2, v_src, w, label='SRCNN RTL 2x (Q7)', color=c1, alpha=0.9, edgecolor='black', lw=0.6)
    ax.bar(x + w/2, v_srg, w, label='Swift-SRGAN 4x (Q7)', color=c2, alpha=0.9, edgecolor='black', lw=0.6)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold', fontsize=8.5)
    
    max_val = max(v_src + v_srg)
    if callable(lim_fn):
        top_lim = lim_fn(v_src + v_srg) + pad if lim_fn == max else lim_fn(v_src + v_srg)
        ax.set_ylim(0, top_lim)
    
    ax.legend(frameon=True, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, ls=':')
    
    for i in range(len(x)):
        offset = max_val * 0.025 if max_val > 0 else 0.5
        ax.text(x[i] - w/2, v_src[i] + offset, fmt.format(v_src[i]), ha='center', fontsize=8, fontweight='bold', color='#1a365d')
        ax.text(x[i] + w/2, v_srg[i] + offset, fmt.format(v_srg[i]), ha='center', fontsize=8, fontweight='bold', color='#44337a')

plt.tight_layout()
chart_models_path = os.path.join(OUTPUT_SPLIT_DIR, 'so_sanh_2_mo_hinh_srcnn_vs_srgan.png')
plt.savefig(chart_models_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'✓ [CELL 5] Đã lưu biểu đồ so sánh 2 mô hình: {chart_models_path}')

# ─────────────────────────────────────────────────────────────
# 5. CELL 6: Biểu đồ chuyên đối chứng với Bicubic Baseline (6 thông số)
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

fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10.5))

# PANEL 1: Task 2x
axes2[0, 0].bar(x - w/2, p_bic_2x, w, label='Bicubic 2x Baseline', color='#feb2b2', edgecolor='#e53e3e', lw=1)
axes2[0, 0].bar(x + w/2, p_src_2x, w, label='SRCNN RTL 2x (Q7)', color='#3182ce', edgecolor='#1a365d', lw=1)
axes2[0, 0].set_ylabel('PSNR (dB)', fontweight='bold')
axes2[0, 0].set_title('① Task Scale 2x: Bicubic 2x vs SRCNN RTL 2x', fontsize=11, fontweight='bold')
axes2[0, 0].set_xticks(x)
axes2[0, 0].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes2[0, 0].set_ylim(0, max(p_src_2x) + 6)
axes2[0, 0].legend(frameon=True, loc='upper right')
axes2[0, 0].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes2[0, 0].text(x[i] - w/2, p_bic_2x[i] + 0.5, f'{p_bic_2x[i]:.2f}', ha='center', fontsize=8.5)
    axes2[0, 0].text(x[i] + w/2, p_src_2x[i] + 0.5, f'{p_src_2x[i]:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#1a365d')

# PANEL 2: Task 4x
axes2[0, 1].bar(x - w/2, p_bic_4x, w, label='Bicubic 4x Baseline', color='#fed7aa', edgecolor='#ea580c', lw=1)
axes2[0, 1].bar(x + w/2, p_srg_4x, w, label='Swift-SRGAN 4x (Q7)', color='#805ad5', edgecolor='#44337a', lw=1)
axes2[0, 1].set_ylabel('PSNR (dB)', fontweight='bold')
axes2[0, 1].set_title('② Task Scale 4x: Bicubic 4x vs Swift-SRGAN 4x', fontsize=11, fontweight='bold')
axes2[0, 1].set_xticks(x)
axes2[0, 1].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
axes2[0, 1].set_ylim(0, max(p_srg_4x) + 6)
axes2[0, 1].legend(frameon=True, loc='upper right')
axes2[0, 1].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes2[0, 1].text(x[i] - w/2, p_bic_4x[i] + 0.5, f'{p_bic_4x[i]:.2f}', ha='center', fontsize=8.5)
    axes2[0, 1].text(x[i] + w/2, p_srg_4x[i] + 0.5, f'{p_srg_4x[i]:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#44337a')

# PANEL 3: PSNR Gain
axes2[0, 2].bar(x - w/2, gain_p_src, w, label='SRCNN 2x Gain (dB)', color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
axes2[0, 2].bar(x + w/2, gain_p_srg, w, label='Swift-SRGAN 4x Gain (dB)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
axes2[0, 2].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0 dB)')
axes2[0, 2].set_ylabel('Δ PSNR Gain (dB) [Model - Bicubic]', fontweight='bold')
axes2[0, 2].set_title('③ Mức Tăng/Giảm PSNR (ΔPSNR Gain)', fontsize=11, fontweight='bold')
axes2[0, 2].set_xticks(x)
axes2[0, 2].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min = min(gain_p_src + gain_p_srg) - 0.3
y_max = max(gain_p_src + gain_p_srg) + 0.3
axes2[0, 2].set_ylim(y_min, y_max)
axes2[0, 2].legend(frameon=True, loc='lower right', fontsize=8)
axes2[0, 2].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes2[0, 2].text(x[i] - w/2, gain_p_src[i] + (0.05 if gain_p_src[i]>=0 else -0.1), f'{gain_p_src[i]:+.2f} dB', ha='center', fontsize=8.5, fontweight='bold')
    axes2[0, 2].text(x[i] + w/2, gain_p_srg[i] + (0.05 if gain_p_srg[i]>=0 else -0.1), f'{gain_p_srg[i]:+.2f} dB', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 4: MS-SSIM Gain
axes2[1, 0].bar(x - w/2, gain_ms_src, w, label='SRCNN 2x (MS-SSIM Gain)', color='#3182ce', alpha=0.9, edgecolor='black', lw=0.6)
axes2[1, 0].bar(x + w/2, gain_ms_srg, w, label='Swift-SRGAN 4x (MS-SSIM Gain)', color='#805ad5', alpha=0.9, edgecolor='black', lw=0.6)
axes2[1, 0].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0)')
axes2[1, 0].set_ylabel('Δ MS-SSIM Gain [Model - Bicubic]', fontweight='bold')
axes2[1, 0].set_title('④ Mức Thay Đổi MS-SSIM (ΔMS-SSIM Gain)', fontsize=11, fontweight='bold')
axes2[1, 0].set_xticks(x)
axes2[1, 0].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min_ms = min(gain_ms_src + gain_ms_srg) - 0.001
y_max_ms = max(gain_ms_src + gain_ms_srg) + 0.001
axes2[1, 0].set_ylim(y_min_ms, y_max_ms)
axes2[1, 0].legend(frameon=True, loc='lower right', fontsize=8)
axes2[1, 0].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes2[1, 0].text(x[i] - w/2, gain_ms_src[i] + (0.0001 if gain_ms_src[i]>=0 else -0.0003), f'{gain_ms_src[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')
    axes2[1, 0].text(x[i] + w/2, gain_ms_srg[i] + (0.0001 if gain_ms_srg[i]>=0 else -0.0003), f'{gain_ms_srg[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 5: LPIPS Gain
axes2[1, 1].bar(x - w/2, gain_lp_src, w, label='SRCNN 2x (LPIPS Gain)', color='#38a169', alpha=0.9, edgecolor='black', lw=0.6)
axes2[1, 1].bar(x + w/2, gain_lp_srg, w, label='Swift-SRGAN 4x (LPIPS Gain)', color='#d69e2e', alpha=0.9, edgecolor='black', lw=0.6)
axes2[1, 1].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0)')
axes2[1, 1].set_ylabel('Δ LPIPS Gain (↑ Càng cao càng nét hơn Bicubic)', fontweight='bold')
axes2[1, 1].set_title('⑤ Mức Cải Thiện Thị Giác (ΔLPIPS Gain)', fontsize=11, fontweight='bold')
axes2[1, 1].set_xticks(x)
axes2[1, 1].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min_lp = min(min(gain_lp_src + gain_lp_srg) - 0.005, -0.01)
y_max_lp = max(gain_lp_src + gain_lp_srg) + 0.008
axes2[1, 1].set_ylim(y_min_lp, y_max_lp)
axes2[1, 1].legend(frameon=True, loc='upper right', fontsize=8)
axes2[1, 1].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes2[1, 1].text(x[i] - w/2, gain_lp_src[i] + 0.0015, f'{gain_lp_src[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')
    axes2[1, 1].text(x[i] + w/2, gain_lp_srg[i] + 0.0015, f'{gain_lp_srg[i]:+.4f}', ha='center', fontsize=8.5, fontweight='bold')

# PANEL 6: NIQE Gain
axes2[1, 2].bar(x - w/2, gain_nq_src, w, label='SRCNN 2x (NIQE Gain)', color='#e53e3e', alpha=0.85, edgecolor='black', lw=0.6)
axes2[1, 2].bar(x + w/2, gain_nq_srg, w, label='Swift-SRGAN 4x (NIQE Gain)', color='#319795', alpha=0.9, edgecolor='black', lw=0.6)
axes2[1, 2].axhline(0, color='red', ls='--', lw=1.2, label='Mốc Bicubic Baseline (0)')
axes2[1, 2].set_ylabel('Δ NIQE Gain (↑ Càng cao càng tự nhiên hơn Bicubic)', fontweight='bold')
axes2[1, 2].set_title('⑥ Mức Cải Thiện Tự Nhiên (ΔNIQE Gain)', fontsize=11, fontweight='bold')
axes2[1, 2].set_xticks(x)
axes2[1, 2].set_xticklabels(categories, fontweight='bold', fontsize=8.5)
y_min_nq = min(gain_nq_src + gain_nq_srg) - 1.5
y_max_nq = max(gain_nq_src + gain_nq_srg) + 1.0
axes2[1, 2].set_ylim(y_min_nq, y_max_nq)
axes2[1, 2].legend(frameon=True, loc='lower left', fontsize=8)
axes2[1, 2].grid(True, alpha=0.3, ls=':')
for i in range(len(x)):
    axes2[1, 2].text(x[i] - w/2, gain_nq_src[i] - 0.6, f'{gain_nq_src[i]:+.2f}', ha='center', fontsize=8.5, fontweight='bold')
    axes2[1, 2].text(x[i] + w/2, gain_nq_srg[i] + 0.2, f'{gain_nq_srg[i]:+.2f}', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
chart_bicubic_path = os.path.join(OUTPUT_SPLIT_DIR, 'so_sanh_chuyen_biet_voi_bicubic.png')
plt.savefig(chart_bicubic_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'✓ [CELL 6] Đã lưu biểu đồ so sánh với Bicubic: {chart_bicubic_path}')

# 6. CELL 7: Phân tích sâu trên dataset cụ thể
CHOSEN_DATASET = 'sub_NIH'
df1_sub = df_srcnn[df_srcnn['dataset'] == CHOSEN_DATASET].set_index('filename')
df2_sub = df_srgan[df_srgan['dataset'] == CHOSEN_DATASET].set_index('filename')
common_files = df1_sub.index.intersection(df2_sub.index)

print(f'\n🔍 Phân tích chi tiết trên tập: 【 {CHOSEN_DATASET} 】 ({len(common_files)} ảnh trùng khớp)\n')

diff_psnr = df2_sub.loc[common_files, 'psnr_model_db'] - df1_sub.loc[common_files, 'psnr_model_db']
diff_lpips = df1_sub.loc[common_files, 'lpips_model'] - df2_sub.loc[common_files, 'lpips_model']
diff_niqe = df1_sub.loc[common_files, 'niqe_model'] - df2_sub.loc[common_files, 'niqe_model']

print(f'• Trung bình chênh lệch PSNR (SRGAN - SRCNN)  : {diff_psnr.mean():+.3f} dB')
print(f'• Trung bình mức cải thiện LPIPS (SRCNN - SRGAN): {diff_lpips.mean():+.4f} (Điểm LPIPS càng thấp càng sắc nét)')
print(f'• Trung bình mức cải thiện NIQE (SRCNN - SRGAN) : {diff_niqe.mean():+.2f} (Điểm NIQE càng thấp càng tự nhiên)')

top_visual = diff_lpips.sort_values(ascending=False).head(5)
print('\nTop 5 ảnh Swift-SRGAN cải thiện thị giác vượt trội nhất so với SRCNN:')
for fn, val in top_visual.items():
    print(f'  ► {fn:<25s}: SRCNN LPIPS={df1_sub.loc[fn, "lpips_model"]:.4f} | SRGAN LPIPS={df2_sub.loc[fn, "lpips_model"]:.4f} (Cải thiện: {val:+.4f})')

print('\n' + '=' * 80)
print('🎉 HOÀN TẤT TOÀN BỘ TIẾN TRÌNH PHÂN TÍCH!')
print('=' * 80)
