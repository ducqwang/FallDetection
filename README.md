# Fall Detection from Surveillance Video — Nhóm 05

Repo: https://github.com/ducqwang/FallDetection

Hệ thống phát hiện té ngã tự động trong video giám sát (Object Detection + Tracking + Action/Event Recognition).
Môn: Một số vấn đề chọn lọc về Thị giác máy tính — MAT3563 (2026–2027).

- **Thành viên:** Hoàng Minh Quang (23001917), Nguyễn Đức Quang (23001918), Hà Minh Quang (23001916).
- **Dữ liệu (công khai):** UR Fall Detection (URFD) + Le2i Fall Detection.
- **Hạ tầng:** Kaggle Notebooks (GPU) + Kaggle Datasets làm input.
- **Kế hoạch chi tiết (bản chốt):** xem [`Ke_Hoach_Fall_Detection_Phan_Cong_Moi.docx`](Ke_Hoach_Fall_Detection_Phan_Cong_Moi.docx).

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
├── configs/                    # data.yaml, m1.yaml, default.yaml
├── src/
│   ├── data/                   # HẠ TẦNG CHUNG: prepare_data, loader, split, labels, common...
│   ├── eda/                    # biểu đồ EDA + PCA/t-SNE
│   ├── eval/                   # event_metrics, frame_metrics (độ đo dùng chung)
│   ├── models/
│   │   ├── m1_rule.py          # M1 (TV1) → predictions.csv
│   │   ├── m3_cnn_gru.py       # M3 (TV3) TÍCH HỢP → predictions.csv  ← dùng chung loader/labels/eval
│   │   ├── model2_pose_lstm.py, model3_cnn_gru.py, model1_rule.py  # bản notebook cũ (độc lập, giữ lại)
│   │   └── ...
│   ├── pipeline.py, features.py, detection_tracking.py  # hạ tầng cho Web Demo (app.py)
│   └── app.py                  # Web Demo (Streamlit)
├── scripts/                    # make_tracks, check_tracks_quality, tune_m1, evaluate, summarize_results
└── notebooks/                  # tv1_01_eda, 02_m1_baseline, tv3_05_m3 ... (chạy trên Kaggle)
```

> Lưu ý: hiện tồn tại song song **bản hạ tầng chung** (`src/data`, `src/eval`, `models/m1_rule.py`, `models/m3_cnn_gru.py`) và **bản notebook độc lập cũ** (`pipeline.py`, `features.py`, `model*_*.py`). Code mới nên dùng hạ tầng chung; bản cũ còn giữ để Web Demo chạy.

## TV1 — cách chạy lại

Chạy trên Kaggle theo đúng thứ tự. Chi tiết và trạng thái ở [`docs/HANDOFF_TV1.md`](docs/HANDOFF_TV1.md); định dạng file ở [`docs/SCHEMA.md`](docs/SCHEMA.md); giao thức đánh giá ở [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md).

| Bước | Notebook | Input | Máy | Đầu ra |
|---|---|---|---|---|
| 1 | `notebooks/00_prepare_data.ipynb` | URFD ảnh, URFD nhãn, Le2i | Internet | Kaggle Dataset `nhom05-falldet-processed` |
| 2 | `notebooks/tv1_01_eda.ipynb` | `nhom05-falldet-processed` | Internet | `reports/figures/` |
| 3 | `notebooks/02_m1_baseline.ipynb` | `nhom05-falldet-processed` | GPU, Internet | `reports/results/` |

Lệnh tương ứng:

```bash
python -m src.data.prepare_data --config configs/data.yaml --out-root $OUT --workers 4
python -m src.data.split --meta $OUT/metadata.csv --out $OUT/splits.json --seed 42
python -m src.eda.plots --meta $OUT/metadata.csv --gt-bbox $OUT/annotations/le2i_gt_bbox.csv --out-dir reports/figures --results-dir reports/results --data-card DATA_CARD.md
python scripts/make_tracks.py --root $OUT --out-dir tracks   # tracks chính thức TV1 (GPU), ghi tracks/tracks.parquet
python scripts/check_tracks_quality.py --tracks tracks/tracks.parquet --root $OUT
python scripts/tune_m1.py --tracks tracks/tracks.parquet --meta $OUT/metadata.csv --dataset urfd --out reports/results/m1_tuned_urfd.yaml
python -m src.models.m1_rule --tracks tracks/tracks.parquet --meta $OUT/metadata.csv --tuned reports/results/m1_tuned_urfd.yaml --out predictions.csv
python scripts/evaluate.py --pred predictions.csv --meta $OUT/metadata.csv --split test --dataset urfd --tuned reports/results/m1_tuned_urfd.yaml
```

Sau khi chạy, commit `metadata.csv`, `splits.json`, `DATA_CARD.md` vào `dataset_index/`, cùng `reports/figures/*.png` và `reports/results/*`. Không commit video, trọng số hay `tracks.parquet`.

Chạy test: `python -m pytest tests -q` (phần end-to-end cần ffmpeg và ffprobe).

**Cross-dataset** không cần file chia riêng. Protocol A: train trên URFD, test toàn bộ Le2i (130 video có nhãn). Protocol B: train trên Le2i, test toàn bộ URFD (70 video). Dùng `--split all` trong `scripts/evaluate.py`.

## TV3 — M3 (CNN+GRU) & Web Demo

M3 dùng **chung** data/nhãn/tracks/đánh giá với M1, M2 (không tự index, không tự chạy YOLO, không tự đánh giá). Notebook: [`notebooks/tv3_05_m3.ipynb`](notebooks/tv3_05_m3.ipynb) (clone repo + Add Input `nhom05-falldet-processed` + GPU). CLI:

```bash
# 1) tracks dùng chung với M1 (hoặc Add Input Kaggle Dataset nhom05-falldet-tracks)
python scripts/make_tracks.py --root $ROOT --out-dir m3/tracks   # ghi m3/tracks/tracks.parquet
# 2) huấn luyện (backbone đóng băng, chỉ train GRU)
python -m src.models.m3_cnn_gru train --tracks m3/tracks/tracks.parquet --meta $ROOT/metadata.csv \
    --root $ROOT --dataset le2i --gt-bbox $ROOT/annotations/le2i_gt_bbox.csv --ckpt m3/m3_best.pt
# 3) dự đoán → predictions.csv (schema chung) rồi đánh giá bằng script chung
python -m src.models.m3_cnn_gru predict --tracks m3/tracks/tracks.parquet --meta $ROOT/metadata.csv \
    --root $ROOT --ckpt m3/m3_best.pt --out m3/m3_pred.csv
python scripts/evaluate.py --pred m3/m3_pred.csv --meta $ROOT/metadata.csv --split test --dataset le2i --T 0.5 1.0 2.0 --tag m3
```

**Web Demo:** `streamlit run src/app.py` (chọn M1/M2/M3, thanh độ nhạy, xuất log `.json` + MOT `.txt`). M2/M3 cần checkpoint đã huấn luyện ở `outputs/`.

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
