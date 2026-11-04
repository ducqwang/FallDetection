# Fall Detection from Surveillance Video — Nhóm 05

Repo: https://github.com/ducqwang/FallDetection

Hệ thống phát hiện té ngã tự động trong video giám sát (Object Detection + Tracking + Action/Event Recognition).
Môn: Một số vấn đề chọn lọc về Thị giác máy tính — MAT3563 (2026–2027).

- **Thành viên:** Hoàng Minh Quang (23001917), Nguyễn Đức Quang (23001918), Hà Minh Quang (23001916).
- **Dữ liệu (công khai):** UR Fall Detection (URFD) + Le2i Fall Detection.
- **Hạ tầng:** Kaggle Notebooks (GPU) + Kaggle Datasets làm input.
- **Kế hoạch chi tiết:** xem [`Ke_Hoach_Chi_Tiet_GK_CK_Nhom5.docx`](Ke_Hoach_Chi_Tiet_GK_CK_Nhom5.docx).

## Ba mô hình so sánh (bài toán chính)
| # | Mô hình | Ý tưởng |
|---|---------|---------|
| M1 | Rule-based (hình học) | Tỷ lệ khung bao + vận tốc/gia tốc tâm — không huấn luyện |
| M2 | Pose + LSTM/GRU | Keypoint (YOLO11-pose) → chuỗi thời gian |
| M3 | CNN + GRU | Backbone CNN đóng băng trên vùng người → GRU |

Pipeline dùng chung (không tính là 1 trong 3 mô hình): **YOLO11 + ByteTrack**.

## Cấu trúc thư mục
```
FallDetection/
├── configs/default.yaml        # tham số tập trung
├── src/
│   ├── utils.py                # seed, I/O, logging
│   ├── data_splits.py          # chia tập theo video + cross-dataset
│   ├── preprocessing.py        # đọc video, trích frame, augmentation
│   ├── detection_tracking.py   # YOLO11 + ByteTrack
│   ├── features.py             # đặc trưng hình học/pose/CNN + PCA/t-SNE
│   ├── evaluate.py             # độ đo mức sự kiện (cửa sổ dung sai T)
│   ├── models/
│   │   ├── model1_rule.py
│   │   ├── model2_pose_lstm.py
│   │   └── model3_cnn_gru.py
│   └── app.py                  # Web Demo (Streamlit)
└── notebooks/                  # 01_eda ... 06_evaluation (chạy trên Kaggle)
```

## TV1 — cách chạy lại

Chạy trên Kaggle theo đúng thứ tự. Chi tiết và trạng thái ở [`docs/HANDOFF_TV1.md`](docs/HANDOFF_TV1.md); định dạng file ở [`docs/SCHEMA.md`](docs/SCHEMA.md); giao thức đánh giá ở [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md).

| Bước | Notebook | Input | Máy | Đầu ra |
|---|---|---|---|---|
| 1 | `notebooks/00_prepare_data.ipynb` | URFD ảnh, URFD nhãn, Le2i | Internet | Kaggle Dataset `nhom05-falldet-processed` |
| 2 | `notebooks/01_eda.ipynb` | `nhom05-falldet-processed` | Internet | `reports/figures/` |
| 3 | `notebooks/02_m1_baseline.ipynb` | `nhom05-falldet-processed` | GPU, Internet | `reports/results/` |

Lệnh tương ứng:

```bash
python -m src.data.prepare_data --config configs/data.yaml --out-root $OUT --workers 4
python -m src.data.split --meta $OUT/metadata.csv --out $OUT/splits.json --seed 42
python -m src.eda.plots --meta $OUT/metadata.csv --gt-bbox $OUT/annotations/le2i_gt_bbox.csv --out-dir reports/figures
python scripts/make_tracks_tmp.py --meta $OUT/metadata.csv --out tracks.parquet   # tạm thời, thay bằng tracks của TV2
python scripts/tune_m1.py --tracks tracks.parquet --meta $OUT/metadata.csv --dataset urfd --out reports/results/m1_tuned_urfd.yaml
python -m src.models.m1_rule --tracks tracks.parquet --meta $OUT/metadata.csv --tuned reports/results/m1_tuned_urfd.yaml --out predictions.csv
python scripts/evaluate.py --pred predictions.csv --meta $OUT/metadata.csv --split test --dataset urfd --tuned reports/results/m1_tuned_urfd.yaml
```

Sau khi chạy, commit `metadata.csv`, `splits.json`, `DATA_CARD.md` vào `dataset_index/`, cùng `reports/figures/*.png` và `reports/results/*`. Không commit video, trọng số hay `tracks.parquet`.

Chạy test: `python -m pytest tests -q` (phần end-to-end cần ffmpeg và ffprobe).

**Cross-dataset** không cần file chia riêng. Protocol A: train trên URFD, test toàn bộ Le2i (127 video có nhãn). Protocol B: train trên Le2i, test toàn bộ URFD (70 video). Dùng `--split all` trong `scripts/evaluate.py`.

## Chạy nhanh (local)
```bash
pip install -r requirements.txt
# chỉnh đường dẫn data trong configs/default.yaml
python -m src.data_splits --config configs/default.yaml
```

## Chạy trên Kaggle
1. Tạo/thêm Kaggle Dataset cho URFD và Le2i → **Add Input**.
2. Mở notebook trong `notebooks/`, bật **GPU**.
3. `pip install -r requirements.txt` (hoặc cài gói thiếu).
4. Lưu checkpoint vào `/kaggle/working`; đặt notebook **Public** để nộp link.

> Quota Kaggle (tham khảo): ~30 GPU-h/tuần, 12h/session, reset thứ Bảy (UTC). Cố định seed để tái lập.
