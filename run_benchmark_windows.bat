@echo off
rem ==============================================================================
rem AI-Based Medical Image Super-Resolution: Windows Benchmark & Verification Tool
rem Compatible with Windows 10/11 x86_64 (Intel, AMD, NVIDIA CUDA)
rem ==============================================================================
chcp 65001 > nul
title AI Super-Resolution Multi-Platform Benchmark (Windows)

echo ==============================================================================
echo   HE THONG DO DAC HIEU NANG VA KIEM TRA THONG SO SIEU PHAN GIAI ANH Y TE
echo   Architecture: Compact SRCNN (1 -> 16 -> 8 -> 1, 1,649 Parameters)
echo ==============================================================================
echo.

rem Kiem tra Python
set "PYTHON_EXE=python"
python -c "import sys" >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\miniconda3\envs\superres\python.exe" (
        set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\superres\python.exe"
    ) else if exist "%USERPROFILE%\anaconda3\envs\superres\python.exe" (
        set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\superres\python.exe"
    ) else if exist "%LOCALAPPDATA%\miniconda3\envs\superres\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\miniconda3\envs\superres\python.exe"
    ) else if exist "%USERPROFILE%\miniconda3\python.exe" (
        set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
    ) else (
        echo [LOI] Khong tim thay Python kha dung trong he thong hoac Conda!
        echo Vui long cai dat Python 3.8+ tu https://www.python.org/downloads/
        echo Chu y: Nho tich chon "Add Python to PATH" khi cai dat hoac kich hoat moi truong conda.
        echo.
        pause
        exit /b 1
    )
)

echo [1/3] Kiem tra phien ban Python va PyTorch...
"%PYTHON_EXE%" -c "import sys; print(f'Python Version : {sys.version.split()[0]}')"
"%PYTHON_EXE%" -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA Available : {cuda_avail}')
if cuda_avail:
    print(f'GPU Device Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version   : {torch.version.cuda}')
mkl_avail = hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available()
print(f'Intel MKL DNN  : {mkl_avail}')
" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo [THONG BAO] Chua cai dat day du thu vien PyTorch / OpenCV / NumPy.
    echo Dang tu dong cai dat tu requirements.txt...
    echo.
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [LOI] Cai dat that bai. Vui long kiem tra ket noi mang.
        pause
        exit /b 1
    )
)

echo.
echo ==============================================================================
echo CHON CHE DO THUC THI KIEM TRA:
echo   [1] Do toan bo thiet bi kha dung tren may (CPU MKL-DNN + NVIDIA GPU CUDA)
echo   [2] Chi do tren CPU (Intel/AMD x86 qua MKL-DNN)
echo   [3] Chi do tren GPU NVIDIA (CUDA Float32)
echo   [4] Chay kiem tra 100%% Bit-Accurate Golden Model (DV Scoreboard Check)
echo   [5] Tao lai hinh anh thuc nghiem Fig 7 va Fig 8 (300 DPI)
echo   [0] Thoat
echo ==============================================================================
set /p choice="Nhap lua chon [1-5, mac dinh 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="0" exit /b 0

echo.
if "%choice%"=="1" (
    echo [THUC THI] Dang chay do toan bo thiet bi (10 warm-up, 100 loops)...
    "%PYTHON_EXE%" benchmark_multi_platform.py --device all --iterations 100 --warmup 10 --output-dir ./results_windows
)
if "%choice%"=="2" (
    echo [THUC THI] Dang chay do CPU Intel/AMD (10 warm-up, 100 loops)...
    "%PYTHON_EXE%" benchmark_multi_platform.py --device cpu --iterations 100 --warmup 10 --output-dir ./results_windows
)
if "%choice%"=="3" (
    echo [THUC THI] Dang chay do GPU NVIDIA CUDA (10 warm-up, 100 loops)...
    "%PYTHON_EXE%" benchmark_multi_platform.py --device cuda --iterations 100 --warmup 10 --output-dir ./results_windows
)
if "%choice%"=="4" (
    echo [THUC THI] Dang kiem tra Scoreboard Bit-Exact 100%%...
    "%PYTHON_EXE%" core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py
)
if "%choice%"=="5" (
    echo [THUC THI] Dang tao Fig 7 (Overlap-Tiling Ablation) va Fig 8 (Multi-Model Visual)...
    "%PYTHON_EXE%" generate_paper_fig7.py
    "%PYTHON_EXE%" generate_paper_fig8.py
)

echo.
echo ==============================================================================
echo [HOAN TAT] Ket qua thuc nghiem da duoc luu tru tai:
echo   - File bao cao chi tiet JSON: ./results_windows/multi_platform_benchmark.json
echo   - Bang Markdown so sanh     : ./results_windows/multi_platform_comparison_table.md
echo   - Bang so lieu tho CSV       : ./results_windows/multi_platform_comparison_table.csv
echo ==============================================================================
echo.
pause
