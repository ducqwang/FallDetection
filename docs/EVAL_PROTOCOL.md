# EVAL PROTOCOL — giao thức đánh giá chung cho M1, M2, M3

Mã nguồn: `src/eval/event_metrics.py`, `src/eval/frame_metrics.py`, `scripts/evaluate.py`. Unit test: `tests/test_event_metrics.py`.

## 1. Từ điểm đến cảnh báo

1. Tại mỗi frame lấy **điểm lớn nhất trên mọi track**. Frame không có dự đoán được điểm 0.
2. Các frame liên tiếp có `fall_score ≥ thr` gộp thành một cảnh báo.
3. Hai cảnh báo cách nhau **dưới** `merge_gap` giây (mặc định 1.0 s) thì gộp làm một. Khoảng cách tính từ frame cuối cảnh báo trước đến frame đầu cảnh báo sau; đúng bằng `merge_gap` thì không gộp.
4. Thời điểm cảnh báo là **frame đầu của đoạn**, đổi sang giây bằng fps của video.

## 2. So khớp với sự kiện

Cửa sổ dung sai mặc định `T = 1.0 s`. Cửa sổ của một sự kiện là `[fall_start − T, fall_end + T]`, tính cả hai mép.

| Trường hợp | Kết quả |
|---|---|
| Cảnh báo **sớm nhất** trong cửa sổ | TP. Mỗi sự kiện tối đa 1 TP |
| Cảnh báo khác cũng trong cửa sổ đó | **Bỏ qua**, không tính TP cũng không tính FP |
| Cảnh báo ngoài mọi cửa sổ | FP |
| Cảnh báo trong video ADL | FP |
| Sự kiện không có cảnh báo nào trong cửa sổ | FN |

Lựa chọn "bỏ qua" nghĩa là hệ thống báo lặp lại ngay trong lúc ngã không bị phạt; số cảnh báo bị bỏ qua được ghi ở `ignored_alarms`.

## 3. Độ đo mức sự kiện

Cộng dồn TP, FP, FN trên toàn tập rồi mới tính (micro).

- **Precision** = TP / (TP + FP), **Recall** = TP / (TP + FN), **F1** là trung bình điều hòa. Mẫu bằng 0 thì độ đo bằng 0.
- **FAR** (`far_per_hour`) = số FP / tổng số giờ video, tính trên cả video fall lẫn ADL. FP của video fall là các cảnh báo ngoài cửa sổ dung sai.
- `far_nonfall_per_hour` là số phụ: cùng số FP nhưng mẫu chỉ gồm thời gian ngoài cửa sổ dung sai. Clip URFD ngắn nên hai số có thể khác nhau nhiều.
- **Latency** = thời điểm cảnh báo TP − thời điểm `fall_start` (giây). Báo cả trung bình và trung vị, **giữ giá trị âm** (báo trước `fall_start`).
- Video không có dòng dự đoán tính là không có cảnh báo. Video `has_event_label = 0` không được đánh giá.

## 4. Độ đo mức frame (phụ)

Accuracy (tại `thr`), ROC-AUC, PR-AUC (average precision).

- Frame té ngã là frame trong `[fall_start, fall_end]`, tính cả hai đầu. Mọi frame ADL là bình thường.
- URFD có `urfd_frame_labels.csv`: với ADL chỉ tính trên các frame **có nhãn**, bỏ frame NaN. Nhãn 0/1 của ADL vẫn tính là bình thường.
- Thiếu một lớp thì ROC-AUC và PR-AUC là `null`.

## 5. Hạn chế của nhãn

- **URFD:** nhãn 0 của mọi chuỗi Fall dài đúng 30 frame, nên khoảng sự kiện là `[fall_start, fall_start + 30 frame]`. Đây là **quy ước của bộ dữ liệu, không phải thời điểm đo được**. `fall_end` không phải thời điểm kết thúc đo thật, vì vậy **latency luôn tính từ `fall_start`, không tính từ `fall_end`**.
- **URFD:** 8/30 chuỗi Fall chỉ còn dưới 15 frame sau khi nằm (fall-13, 15, 16, 17, 21, 25, 27, 28), nên mô hình đòi bất động dài sẽ bỏ sót các chuỗi này.
- **Le2i:** khoảng ngã dài 7 đến 45 frame (trung vị 23.5). Người thường đã nằm xuống trước `fall_end`, nên cửa sổ bao trọn cú ngã. Latency vẫn tính từ `fall_start`.
- **Le2i:** tập test chỉ có khoảng 5 ADL, kết quả within-dataset nhiễu. Ưu tiên báo cáo cross-dataset trên toàn bộ 127 video có nhãn.
- Precision của URFD phụ thuộc mạnh vào 16 ADL negative khó, nên luôn báo cáo riêng nhóm này.

## 6. ADL negative khó

Trên video `adl_hard_neg = 1`, `scripts/evaluate.py` báo riêng: số video, số video có FP, số FP, FAR. Không có sự kiện nên không có Recall.

## 7. Cách chạy

```bash
python scripts/evaluate.py --pred predictions.csv --meta metadata.csv --split test --dataset urfd --thr 0.7
# Ablation dung sai
python scripts/evaluate.py --pred predictions.csv --meta metadata.csv --split test --dataset urfd --thr 0.7 --T 0.5 1.0 2.0
# Cross-dataset: toàn bộ video có nhãn của một bộ
python scripts/evaluate.py --pred predictions.csv --meta metadata.csv --split all --dataset le2i --thr 0.7
```

Kết quả in thành bảng và lưu JSON vào `reports/results/`. Ngưỡng `thr` chỉ được chọn trên tập val.
