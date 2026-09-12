# Hướng Dẫn Thực Nghiệm Đo Đạc Hiệu Năng Trên Thiết Bị x86 (Intel/AMD) và NVIDIA GPU (CUDA)

Tài liệu chi tiết hướng dẫn đo đạc trên máy tính x86, máy trạm hoặc GPU đám mây (Google Colab / Kaggle) được lưu trữ tại:
[core_project/benchmarks_reports/multi_platform/MEASUREMENT_GUIDE_X86_CUDA.md](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/core_project/benchmarks_reports/multi_platform/MEASUREMENT_GUIDE_X86_CUDA.md)

### Lệnh chạy nhanh (Quick Run):
```bash
# Đo toàn bộ thiết bị khả dụng trên máy (10 warm-up, 100 loops)
python benchmark_multi_platform.py --device all --iterations 100 --warmup 10

# Chỉ đo GPU NVIDIA (CUDA)
python benchmark_multi_platform.py --device cuda --iterations 100 --warmup 10

# Chỉ đo CPU x86 (MKL-DNN)
python benchmark_multi_platform.py --device cpu --iterations 100 --warmup 10
```
