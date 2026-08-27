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

Kết quả chuẩn của bản giao hiện tại: **61 tests passed**.

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

### Bước 1 - Tự động tải và chuyển dữ liệu

```bash
python scripts/fetch_original_data.py
```

Script tự tải repository `CLongLi/UWB-Radar-Pedestrian-Tracking` từ GitHub, tải
`Dyn_CIR_VAR.mat` từ Google Drive do chính README của repository gốc cung cấp, sau đó tạo
`data/uwb_original_standard.mat`. Bộ chuyển đổi xử lý đúng `AnchorPos` 4x3 -> XY, tên biến
`Dyn_var_CIRxx`/`Bg_var_CIRxx` và dùng `abs(CIR phức)` giống MATLAB gốc.

Nếu đã có ba file MATLAB, vẫn có thể chạy `scripts/convert_original_matlab_data.py` thủ công.

### Bước 2 - Chạy protocol repository-parity

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

## 10. Huấn luyện + Lottery Ticket + export INT8 cho ESP32

Dữ liệu demo:

```bash
./RUN_ESP32_TRAIN.sh
```

Dữ liệu chính thức, tự tải nếu chưa có:

```bash
./RUN_ESP32_OFFICIAL.sh
```

Pipeline ESP32 hiện **bắt buộc export model từ structured Lottery Ticket Hypothesis**. Supernet
được train trước, các channel/neuron quan trọng được chọn từ supernet đã train nhưng trọng số sống
sót được rewind về initialization. Nhiều kiến trúc compact được thử từ nhỏ đến lớn. Model đầu tiên
đạt giới hạn suy giảm ToF MAE trong `lth_search` sẽ được export. Random compact model chỉ là control
để báo cáo khoa học và không được phép thay thế LTH checkpoint.

Nếu không ticket nào đạt accuracy guard trong cấu hình official, pipeline dừng trước export thay vì
xuất một model nhẹ nhưng chất lượng thấp.
