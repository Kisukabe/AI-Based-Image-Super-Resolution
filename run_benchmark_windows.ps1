# ==============================================================================
# AI-Based Medical Image Super-Resolution: Windows PowerShell Benchmark Tool
# Compatible with Windows 10/11 x86_64 (PowerShell 5.1 / PowerShell 7+)
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  HỆ THỐNG ĐO ĐẠC HIỆU NĂNG VÀ KIỂM TRA THÔNG SỐ SIÊU PHÂN GIẢI ẢNH Y TẾ" -ForegroundColor Green
Write-Host "  Architecture: Compact SRCNN (1 -> 16 -> 8 -> 1, 1,649 Parameters)" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python
$pythonExe = "python"
$hasWorkingPython = $false
try {
    $null = & python -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $hasWorkingPython = $true }
} catch {}

if (-not $hasWorkingPython) {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\envs\superres\python.exe",
        "$env:USERPROFILE\anaconda3\envs\superres\python.exe",
        "$env:LOCALAPPDATA\miniconda3\envs\superres\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe"
    )
    foreach ($cand in $candidates) {
        if (Test-Path $cand) {
            $pythonExe = $cand
            $hasWorkingPython = $true
            break
        }
    }
}

if (-not $hasWorkingPython) {
    Write-Host "[LỖI] Không tìm thấy Python khả dụng trong PATH hoặc Conda!" -ForegroundColor Red
    Write-Host "Vui lòng cài đặt Python 3.8+ từ https://www.python.org/downloads/ hoặc kích hoạt môi trường conda."
    Read-Host "Nhấn Enter để thoát..."
    exit 1
}

# Kiểm tra PyTorch và CUDA
Write-Host "[1/3] Kiểm tra thông tin môi trường..." -ForegroundColor Yellow
& $pythonExe -c @"
import sys, torch
print(f'Python Version : {sys.version.split()[0]}')
print(f'PyTorch Version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA Available : {cuda_avail}')
if cuda_avail:
    print(f'GPU Device Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version   : {torch.version.cuda}')
mkl_avail = hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available()
print(f'Intel MKL DNN  : {mkl_avail}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[THÔNG BÁO] Chưa cài đủ thư viện. Đang cài đặt từ requirements.txt..." -ForegroundColor Yellow
    & $pythonExe -m pip install -r requirements.txt
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "CHỌN CHẾ ĐỘ THỰC THI:" -ForegroundColor White
Write-Host "  [1] Đo toàn bộ thiết bị khả dụng (CPU MKL-DNN + NVIDIA GPU CUDA)"
Write-Host "  [2] Chỉ đo trên CPU (Intel/AMD x86 qua MKL-DNN)"
Write-Host "  [3] Chỉ đo trên GPU NVIDIA (CUDA Float32)"
Write-Host "  [4] Kiểm tra 100% Bit-Accurate Golden Model (Scoreboard Check)"
Write-Host "  [5] Tạo lại hình ảnh Fig 7 và Fig 8 (300 DPI)"
Write-Host "  [0] Thoát"
Write-Host "==============================================================================" -ForegroundColor Cyan

$choice = Read-Host "Nhập lựa chọn [1-5, mặc định 1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }
if ($choice -eq "0") { exit 0 }

$outDir = "./results_windows"

switch ($choice) {
    "1" {
        Write-Host "`n[THỰC THI] Đang chạy đo toàn bộ thiết bị (10 warm-up, 100 loops)..." -ForegroundColor Green
        & $pythonExe benchmark_multi_platform.py --device all --iterations 100 --warmup 10 --output-dir $outDir
    }
    "2" {
        Write-Host "`n[THỰC THI] Đang chạy đo CPU Intel/AMD..." -ForegroundColor Green
        & $pythonExe benchmark_multi_platform.py --device cpu --iterations 100 --warmup 10 --output-dir $outDir
    }
    "3" {
        Write-Host "`n[THỰC THI] Đang chạy đo GPU NVIDIA CUDA..." -ForegroundColor Green
        & $pythonExe benchmark_multi_platform.py --device cuda --iterations 100 --warmup 10 --output-dir $outDir
    }
    "4" {
        Write-Host "`n[THỰC THI] Đang kiểm tra Scoreboard Bit-Exact 100%..." -ForegroundColor Green
        & $pythonExe core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py
    }
    "5" {
        Write-Host "`n[THỰC THI] Đang tạo hình Fig 7 và Fig 8 chuẩn 300 DPI..." -ForegroundColor Green
        & $pythonExe generate_paper_fig7.py
        & $pythonExe generate_paper_fig8.py
    }
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "[HOÀN TẤT] Kết quả thực nghiệm đã được lưu tại thư mục: $outDir" -ForegroundColor Green
Write-Host "  - File báo cáo chi tiết JSON : $outDir/multi_platform_benchmark.json"
Write-Host "  - Bảng Markdown đối sánh     : $outDir/multi_platform_comparison_table.md"
Write-Host "  - Bảng số liệu thô CSV       : $outDir/multi_platform_comparison_table.csv"
Write-Host "==============================================================================" -ForegroundColor Cyan
Read-Host "Nhấn Enter để kết thúc..."
