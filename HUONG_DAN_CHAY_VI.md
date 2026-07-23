# Hướng dẫn chạy dự án UWB Passive Tracking bằng Python

## 1. Môi trường

Khuyến nghị:

- Python 3.10-3.12;
- RAM từ 8 GB;
- quick run: CPU là đủ;
- full benchmark: GPU NVIDIA từ 6-8 GB VRAM giúp giảm thời gian huấn luyện.

## 2. Cài đặt

### Windows PowerShell

```powershell
cd UWB_Passive_Tracking_Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Linux/macOS

```bash
cd UWB_Passive_Tracking_Python
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## 3. Kiểm tra code

```bash
pytest
```

Kết quả chuẩn của bản giao hiện tại: **10 tests passed**.

## 4. Chạy nhanh end-to-end

```bash
uwb-track quick-run --config configs/quick.yaml
```

Hoặc Windows:

```powershell
.\RUN_QUICK.bat
```

Kết quả được ghi vào `results/quick/`. Đây chỉ là smoke test trên dữ liệu tổng hợp, một case và một seed.

## 5. Tái lập mã MATLAB chính thức

### Bước 1 - Chuẩn bị dữ liệu

Cần ba file:

- `Bg_CIR_VAR.mat`;
- `Dyn_CIR_VAR.mat`;
- `AnchorPos.mat`.

### Bước 2 - Chuyển sang định dạng chuẩn

```bash
python scripts/convert_original_matlab_data.py \
  --background path/to/Bg_CIR_VAR.mat \
  --dynamic path/to/Dyn_CIR_VAR.mat \
  --anchors path/to/AnchorPos.mat \
  --output data/uwb_original_standard.mat
```

### Bước 3 - Chạy protocol repository-parity

```bash
uwb-track reproduce-paper --config configs/paper_reproduction_original.yaml
```

Protocol này giữ các lựa chọn của repository MATLAB: 50 epoch, Adam, batch 10%, sample split 85/15, delay-index formula gốc và Particle Filter 200 particle resample mỗi bước.

## 6. Chạy thí nghiệm chính

```bash
uwb-track full --config configs/full.yaml
```

Mặc định:

- 3 case;
- 5 seed;
- LoS, NLoS-1, NLoS-2, outlier và dropout;
- bốn phương pháp;
- mean/std/95% CI và paired Wilcoxon.

## 7. Ablation và robustness

```bash
uwb-track ablation --config configs/ablation.yaml
uwb-track robustness-sweep --config configs/robustness_sweep.yaml
```

## 8. File kết quả

Mỗi thư mục experiment có:

- `results_raw.csv`: từng case/seed/scenario/method;
- `results_summary.csv`: mean, std và 95% CI;
- `statistical_tests.csv`: so sánh cặp;
- `checkpoints/`: trọng số model;
- `figures/`: quỹ đạo, CDF và summary;
- `resolved_config.json`: cấu hình thực tế đã chạy.

## 9. Cách diễn giải đúng

- `results/quick` chứng minh code chạy được, không phải bằng chứng khoa học cuối cùng.
- Kết quả synthetic không được dùng để tuyên bố độ chính xác trên DWM1000 thực.
- Chỉ kết luận tái lập số liệu bài báo sau khi chạy dữ liệu gốc và kiểm tra intermediate arrays.
- Khi so sánh phương pháp đề xuất, dùng protocol timestamp-validation và corrected indexing trong `full.yaml` để giảm leakage và sửa lệch một bin của deployment formula gốc.
