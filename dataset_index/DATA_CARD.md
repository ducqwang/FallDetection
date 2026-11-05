# DATA CARD — nhom05-falldet-processed

## Nguồn dữ liệu

| Dataset | URL | Version |
|---|---|---|
| urfd_images | https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset | chưa điền trong configs/data.yaml |
| urfd_labels | https://www.kaggle.com/datasets/minhquang2701/urfall-labels | chưa điền trong configs/data.yaml |
| le2i | https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia | chưa điền trong configs/data.yaml |

## Số lượng thực tế

| Dataset | Scene | Phần | fps | Kích thước | Lớp | Số video |
|---|---|---|---|---|---|---|
| le2i | coffee_room | 01 | 25 | 320×240 | fall | 48 |
| le2i | coffee_room | 02 | 25 | 320×240 | adl | 8 |
| le2i | coffee_room | 02 | 25 | 320×240 | fall | 14 |
| le2i | home | 01 | 24 | 320×180 | fall | 19 |
| le2i | home | 01 | 24 | 320×240 | fall | 11 |
| le2i | home | 02 | 24 | 320×180 | fall | 6 |
| le2i | home | 02 | 24 | 320×240 | adl | 23 |
| le2i | home | 02 | 24 | 320×240 | fall | 1 |
| urfd | urfd |  | 30 | 640×480 | adl | 40 |
| urfd | urfd |  | 30 | 640×480 | fall | 30 |

Tổng 200 video. `adl_hard_neg = 1`: 16 chuỗi ADL của URFD (cúi, bò, ngồi/nằm giường, nằm sofa; không phải té ngã).

## Kiểm tra bắt buộc

- URFD Fall khớp hoàn toàn: 30/30
- URFD ADL CSV_THIEU_FRAME: 18 chuỗi: adl-01, adl-02, adl-06, adl-08, adl-13, adl-15, adl-16, adl-17, adl-18, adl-19, adl-20, adl-21, adl-30, adl-36, adl-37, adl-38, adl-39, adl-40
- URFD dòng CSV không có ảnh: 0
- URFD chuỗi giữ lại: 70, không loại chuỗi nào
- Le2i: 130 video (99 fall, 31 adl)
- Le2i has_event_label = 1: 130/130
- Le2i nguồn nhãn: {'header_mid': 3, 'header_top': 127}
- Le2i fps: {'coffee_room 25 fps': 70, 'home 24 fps': 60}
- Le2i kích thước: {'320×180': 25, '320×240': 105}

## Chỗ lệch giữa ảnh và CSV của URFD

Fall khớp hoàn toàn: 30. ADL khớp: 22. ADL `CSV_THIEU_FRAME`: 18. Không chuỗi nào bị loại, không cắt clip. Nguyên nhân chưa xác nhận.

| Chuỗi | Số ảnh | Dòng CSV | Thiếu đầu | Thiếu giữa | Thiếu cuối |
|---|---|---|---|---|---|
| adl-01 | 150 | 144 | 5 | 1 | 0 |
| adl-02 | 180 | 150 | 30 | 0 | 0 |
| adl-06 | 230 | 216 | 13 | 1 | 0 |
| adl-08 | 180 | 171 | 9 | 0 | 0 |
| adl-13 | 265 | 252 | 13 | 0 | 0 |
| adl-15 | 275 | 263 | 12 | 0 | 0 |
| adl-16 | 240 | 224 | 16 | 0 | 0 |
| adl-17 | 230 | 178 | 4 | 1 | 47 |
| adl-18 | 265 | 237 | 28 | 0 | 0 |
| adl-19 | 250 | 228 | 22 | 0 | 0 |
| adl-20 | 270 | 247 | 23 | 0 | 0 |
| adl-21 | 280 | 263 | 17 | 0 | 0 |
| adl-30 | 400 | 369 | 31 | 0 | 0 |
| adl-36 | 340 | 323 | 17 | 0 | 0 |
| adl-37 | 350 | 328 | 22 | 0 | 0 |
| adl-38 | 345 | 327 | 18 | 0 | 0 |
| adl-39 | 270 | 252 | 18 | 0 | 0 |
| adl-40 | 330 | 286 | 44 | 0 | 0 |

Fall còn dưới 15 frame sau khi nằm (8 chuỗi, số frame trong ngoặc): fall-13 (12), fall-15 (14), fall-16 (9), fall-17 (7), fall-21 (5), fall-25 (11), fall-27 (12), fall-28 (10).

## Phạm vi Le2i

Chỉ dùng Home và Coffee room. **Office và Lecture room bị loại hẳn** (không chuyển mp4, không có trong
`metadata.csv`, `splits.json` và EDA) vì hai scene này không có file annotation: không có khoảng té ngã,
không có bbox, và cũng không dùng làm mẫu ADL được vì không biết video nào có té ngã.
Nguồn tham khảo: arXiv 1812.01923 ("Due to loss of annotation files, we only use the recordings
from home and coffee room").

README của Le2i ghi "320x240 25FPS" và mô tả bbox là chiều cao, chiều rộng, tọa độ tâm. Dữ liệu thực tế
khác: fps và kích thước ở bảng dưới; bbox là góc `x1,y1,x2,y2`.

## Chỗ lệch và lưu ý của Le2i

Các video sau có hai dòng khoảng ngã nằm giữa file thay vì ở đầu file (`label_source = header_mid`). Chúng được dùng như mọi video fall khác, `fall_start_frame` và `fall_end_frame` trừ 1 về 0-based:

| Video | Frame bắt đầu ngã (gốc) | Frame kết thúc ngã (gốc) | Số dòng của cặp |
|---|---|---|---|
| le2i_coffee_room01_026 | 197 | 227 | 219-220 |
| le2i_coffee_room02_050 | 1816 | 1852 | 1894-1895 |
| le2i_coffee_room02_052 | 87 | 113 | 156-157 |

- `aux_code` trong `le2i_gt_bbox.csv` là cột 2 gốc của dòng bbox, số nguyên, chưa rõ nghĩa, không dùng ở đâu.
- Dòng bbox đã đọc: 42417; bỏ dòng toàn 0: 71; bỏ hộp không còn diện tích sau khi clip: 92.
- bbox vượt khung là hợp lệ (người bị cắt ở mép), đã clip vào khung hình.
- `adl_hard_neg = 0` cho toàn bộ Le2i ở giữa kỳ: danh sách video Le2i negative khó chưa có.
- Le2i chia theo (nhãn, scene), nên cùng người và cùng cảnh quay có thể nằm ở cả train lẫn test. Kết quả trong cùng bộ Le2i có thể lạc quan. Tập test Le2i rất nhỏ (ước tính khoảng 15 fall và 5 ADL), nên kết quả chính là cross-dataset (`--split all`).

## Quy ước

- `video_id` không dấu cách, không ngoặc, ví dụ `urfd_fall-01_cam0`, `le2i_home02_031`.
- `frame_idx` bắt đầu từ 0. Frame gốc của URFD và Le2i đánh số từ 1 nên đã trừ 1.
- bbox `x1, y1, x2, y2` theo pixel tuyệt đối trên video gốc.
- Thời gian `t = frame_idx / fps` với fps của từng video. Không giả định fps chung.
- Chia tập theo video, seed 42.
- **URFD: nhãn 0 dài đúng 30 frame (1.0 s) ở mọi chuỗi Fall, nên `fall_end = fall_start + 30`.
  Đây là quy ước của bộ dữ liệu, không phải thời điểm đo được.**
- ADL của URFD chỉ dùng nhãn mức video. Nhãn 1 của ADL không tạo sự kiện ngã.
- Frame không có trong CSV của URFD là chưa có nhãn (NaN), không tự điền -1.

## Giấy phép và trích dẫn

- **URFD:** CC BY-NC. Trích dẫn: B. Kwolek, M. Kępski (2014), *Human fall detection on embedded
  platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine.
- **Le2i:** chỉ dùng cho học thuật. Trích dẫn: I. Charfi, J. Miteran, J. Dubois, M. Atri, R. Tourki,
  *Optimised spatio-temporal descriptors for real-time fall detection: comparison of SVM and Adaboost
  based classification*, Journal of Electronic Imaging, 22(4), 2013.
