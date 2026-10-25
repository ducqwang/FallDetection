# Fall Detection from Surveillance Video — Nhóm 05

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
