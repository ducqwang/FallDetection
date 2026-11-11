# SCHEMA — định dạng file dùng chung của Nhóm 05

Tài liệu này để TV1, TV2, TV3 chốt định dạng. Các hằng số tên cột nằm ở `src/data/common.py`.

## Quy ước chung

| Mục | Quy ước |
|---|---|
| `video_id` | Không dấu cách, không ngoặc: `urfd_fall-01_cam0`, `urfd_adl-01_cam0`, `le2i_home02_031`, `le2i_coffee_room01_012` |
| `frame_idx` | Bắt đầu từ 0. Ảnh URFD số `NNN` và dòng CSV `frame = N` ứng với `frame_idx = N − 1`. Le2i gốc đánh số từ 1 nên cũng trừ 1 |
| bbox | `x1, y1, x2, y2`, pixel tuyệt đối trên video gốc |
| Thời gian | `t = frame_idx / fps`, fps của từng video: URFD 30, Le2i Home 24, Le2i Coffee room 25. Không giả định fps chung |
| Seed | Mặc định 42, cố định cho `random`, `numpy`, `torch` |

## metadata.csv

Một dòng mỗi video: 70 dòng URFD, 130 dòng Le2i (chỉ Home và Coffee room; 99 fall, 31 ADL).

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `video_id` | chuỗi | Khóa chính |
| `dataset` | `urfd` \| `le2i` | Nguồn |
| `scene` | chuỗi | `urfd`, `home`, `coffee_room` |
| `scene_part` | chuỗi | Le2i: `01`, `02` (giữ số 0 đầu, đọc bằng `dtype=str`). URFD: trống |
| `camera` | chuỗi | URFD: `cam0`. Le2i: trống |
| `label` | `fall` \| `adl` | Nhãn mức video |
| `fall_start_frame` | int, 0-based | URFD: frame đầu có nhãn 0. Le2i: frame bắt đầu ngã trong annotation (dòng 1, hoặc dòng đầu của cặp giữa file) trừ 1. Trống với ADL |
| `fall_end_frame` | int, 0-based | URFD: frame đầu có nhãn 1. Le2i: frame kết thúc ngã trong annotation (dòng 2, hoặc dòng sau của cặp giữa file) trừ 1. Trống với ADL |
| `fps` | số | fps của video |
| `n_frames` | int | Số frame thật của mp4 |
| `width`, `height` | int | Kích thước gốc, pixel. Không resize |
| `duration_s` | số | `n_frames / fps`, giây |
| `has_event_label` | 0 \| 1 | 1 = dùng được cho đánh giá sự kiện và có trong `splits.json` |
| `label_first_frame`, `label_last_frame` | int, 0-based | Chỉ URFD: frame đầu/cuối có trong CSV nhãn. Le2i trống |
| `label_coverage` | số | Chỉ URFD: số frame có nhãn / `n_frames`. Le2i trống |
| `adl_hard_neg` | 0 \| 1 | 1 = ADL của URFD có nhãn frame 0 hoặc 1 (negative khó). Mọi dòng khác là 0 |
| `path` | chuỗi | Đường dẫn mp4, tương đối so với thư mục dữ liệu |
| `label_source` | `header_top` \| `header_mid` \| `urfd_csv` | Nguồn nhãn: hai dòng đầu file annotation Le2i; cặp dòng nằm giữa hoặc cuối file annotation Le2i (3 video); CSV nhãn URFD |

Đọc bằng `src.data.loader.load_metadata()` để giữ đúng kiểu cột.

## annotations/le2i_gt_bbox.csv

`video_id, frame_idx, x1, y1, x2, y2, aux_code`

`aux_code` là cột 2 gốc của dòng bbox, số nguyên, chưa rõ nghĩa, không dùng ở đâu. Đã bỏ dòng toàn 0 (không có người) và đã clip vào `[0,w]×[0,h]`. Frame không có dòng nghĩa là không có người, không điền bbox.

## annotations/urfd_frame_labels.csv

`video_id, frame_idx, label`

`label` ∈ {-1, 0, 1}: chưa nằm, đang ngã, đã nằm. Chỉ gồm frame CÓ trong CSV gốc; frame thiếu là chưa có nhãn, không phải -1. Nhãn 1 của ADL không tạo sự kiện ngã.

## Gán nhãn cửa sổ

`src.data.labels.window_label(video_row, start_frame, end_frame, min_overlap=0.5)`, frame 0-based, tính cả hai đầu cửa sổ.

Cửa sổ là dương khi số frame chồng lên sự kiện `[fall_start_frame, fall_end_frame]` lớn hơn hoặc bằng `min_overlap × min(độ dài cửa sổ, độ dài sự kiện)` và lớn hơn 0. Dùng `min` vì cú ngã có thể ngắn hơn cửa sổ nhiều (Le2i ngắn nhất 7 frame). Video ADL luôn trả về 0; video chưa xác định lớp thì báo lỗi.

## splits.json

```json
{"urfd": {"train": ["..."], "val": ["..."], "test": ["..."]},
 "le2i": {"train": ["..."], "val": ["..."], "test": ["..."]},
 "seed": 42, "ratio": [0.7, 0.15, 0.15]}
```

Cross-dataset không có file chia riêng, xem README.

## tracks.parquet (TV1 tạo bằng `scripts/make_tracks.py`)

`video_id, frame_idx, track_id, x1, y1, x2, y2, det_conf`

Kiểu: `video_id` chuỗi; `frame_idx`, `track_id` int64; `x1, y1, x2, y2, det_conf` float64. `frame_idx` 0-based, là số thứ tự frame đọc được; hộp pixel xyxy, đã clip vào khung, `x2 > x1`, `y2 > y1`.

Mỗi `(video_id, frame_idx, track_id)` xuất hiện tối đa một lần. `scripts/make_tracks.py` sinh bản chính thức của TV1 (kèm `tracks_meta.json`); TV2 có thể thay bằng bản của mình nếu qua `src.data.tracks.validate_tracks(tracks, meta)` không vi phạm, khi đó phải chạy lại M1. Video không có người không có dòng nào (hợp lệ), được liệt kê ở `videos_without_tracks` trong `tracks_meta.json`.

## keypoints.parquet (TV2 tạo)

`video_id, frame_idx, track_id, kp{0..16}_{x,y,c}`

## predictions.csv (M1, M2, M3 đều xuất)

`video_id, frame_idx, track_id, fall_score`

`fall_score` ∈ [0, 1]. Mỗi frame có thể có nhiều dòng, mỗi track một dòng. Video hoặc frame không có dòng được tính điểm 0.
