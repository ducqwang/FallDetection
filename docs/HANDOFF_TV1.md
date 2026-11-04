# HANDOFF TV1
Nhánh: tv1/gk-data-eda-m1  |  Commit cuối: d5248c2  |  Cập nhật: 2026-11-04 23:58

> Agent không chạy pipeline trên dataset: theo quyết định của TV1, TV1 tự chạy trên Kaggle. Mọi output cần dataset được liệt kê ở mục "Việc còn tồn đọng".

## Trạng thái từng Việc
| Việc | Trạng thái | Đã chạy trên dữ liệu thật? | Ghi chú |
|---|---|---|---|
| 0. Notebook kiểm tra | xong | chưa | Đã thêm 2 notebook, giữ nguyên source, xóa output. Agent không chạy lại; bản gốc đã được TV1 chạy trên Kaggle |
| 1. prepare_data | một phần | chỉ dữ liệu giả | Code xong. 2 CSV nhãn URFD là thật, ảnh và video giả. Chưa có output thật |
| 2. split | một phần | chỉ dữ liệu giả | Code và test xong. Chưa có `splits.json` thật |
| 3. loader, labels, augment | một phần | chỉ dữ liệu giả | Code và test xong. Chưa có hình minh họa augmentation thật |
| 4. EDA | một phần | chỉ dữ liệu giả | Code xong. Chưa có 5 hình thật trong `reports/figures/` |
| 5. M1 | một phần | chỉ dữ liệu giả | Code xong. Ngưỡng chưa tune. `make_tracks_tmp.py` đã chạy với YOLO thật trên 1 ảnh mẫu, không phải dataset |
| 6. Bộ đánh giá | một phần | chỉ dữ liệu giả | Code và unit test xong. Chưa có bảng kết quả thật |

## Bản đồ file
| File | Làm gì | Hàm/CLI chính | Đầu vào | Đầu ra |
|---|---|---|---|---|
| `configs/data.yaml` | Đường dẫn, nguồn dataset, số liệu mong đợi | khối `paths`, `dataset_sources`, `expect` | | |
| `configs/m1.yaml` | Mọi ngưỡng và lưới tune của M1 | `signals`, `rules`, `weights`, `alarm`, `grid` | | |
| `src/data/prepare_data.py` | CLI Việc 1 | `--config --input-root --out-root --workers --skip-existing` | `/kaggle/input` | mp4, csv, DATA_CARD |
| `src/data/urfd.py` | Đọc URFD, ghép nhãn theo số frame | `read_labels`, `scan_sequences`, `match_labels` | ảnh, 2 CSV | dict chuỗi |
| `src/data/le2i.py` | Đọc Le2i, annotation, bbox | `scan_videos`, `parse_annotation`, `clean_boxes(video_id, boxes, width, height, n_frames)` | avi, txt | dict video, bảng bbox |
| `src/data/media.py` | ffmpeg, ffprobe | `png_sequence_to_mp4`, `avi_to_mp4`, `probe_video`, `count_frames` | ảnh, video | mp4, thông số |
| `src/data/checks.py` | So với số liệu đã xác nhận | `check_urfd`, `check_le2i` | dữ liệu đã quét | danh sách chỗ lệch |
| `src/data/metadata.py` | Dựng `metadata.csv` | `urfd_row`, `le2i_row`, `metadata_frame` | dict video | DataFrame |
| `src/data/data_card.py` | Sinh DATA_CARD | `build_data_card` | số liệu đo | Markdown |
| `src/data/common.py` | Tên cột, `video_id`, `DataCheckError` | `urfd_video_id`, `le2i_video_id`, `sequence_id` | | |
| `src/data/split.py` | CLI Việc 2 | `make_splits(meta, seed, ratio)`; `--meta --out --seed` | `metadata.csv` | `splits.json` |
| `src/data/loader.py` | Hàm đọc dùng chung | `load_metadata`, `get_split(dataset, part)`, `read_frames(video_id, start, end)`, `load_tracks(path)`, `load_frame_labels(video_id)` | thư mục dữ liệu | DataFrame, ndarray, Series |
| `src/data/labels.py` | Nhãn cửa sổ, track chính | `window_label(row, start, end, min_overlap=0.5)`, `select_main_track(tracks, row, gt=None)` | metadata, tracks | 0/1, track_id |
| `src/data/augment.py` | Tăng cường khi train | `augment_clip(clip, config, seed)`, `save_augment_preview(clip, out)` | clip T×H×W×3 | clip, PNG |
| `src/data/env_info.py` | Ô đầu notebook | `setup_notebook(config, seed)` | yaml | in version |
| `src/eda/window_features.py` | Đặc trưng theo cửa sổ | `build_window_features(...)` | bbox, metadata | X, y |
| `src/eda/plots.py` | CLI Việc 4 | `plot_*`, `plot_embedding(X, y, source, out, perplexity, seed)`; `--meta --gt-bbox --tracks --out-dir` | metadata, bbox | 5 PNG |
| `src/models/m1_features.py` | Tín hiệu theo track | `track_signals(track, fps, smooth_s, reference_s)` | track | DataFrame |
| `src/models/m1_rule.py` | CLI M1 | `predict`, `score_signals`; `--tracks \| --gt-bbox --meta --out --tuned` | tracks | `predictions.csv` |
| `src/eval/event_metrics.py` | Độ đo sự kiện | `find_alarms`, `match_alarms`, `evaluate(pred, meta_rows, thr, tolerance_s, merge_gap_s, frame_labels)` | dự đoán, metadata | dict |
| `src/eval/frame_metrics.py` | Độ đo frame | `frame_level_metrics(records, thr)` | điểm theo frame | dict |
| `scripts/evaluate.py` | CLI đánh giá | `--pred --meta --split --dataset --thr --tuned --T --tag` | csv | bảng, JSON |
| `scripts/tune_m1.py` | Grid search chỉ trên val | `--tracks \| --gt-bbox --meta --dataset --out` | tracks | yaml, csv |
| `scripts/make_tracks_tmp.py` | TẠM THỜI: YOLO11 + ByteTrack | `--meta --out --dataset --model --conf --imgsz` | mp4 | `tracks.parquet` |
| `scripts/summarize_results.py` | Gom bảng kết quả | `--results-dir --out` | JSON | csv |
| `tests/` | Test TV1, `fake_kaggle.py` sinh dữ liệu giả | `python -m pytest tests -q` | | |
| `notebooks/00_prepare_data`, `01_eda`, `02_m1_baseline` | Chạy trên Kaggle | | | |
| `docs/SCHEMA.md`, `docs/EVAL_PROTOCOL.md` | Định dạng file, giao thức đánh giá | | | |

## Cách chạy (lệnh y nguyên, theo thứ tự, trên Kaggle)
1. Điền `version` của 3 dataset vào `configs/data.yaml`, push lên nhánh.
2. `notebooks/00_prepare_data.ipynb`: Add Input 3 dataset gốc, bật Internet, không cần GPU, Run All. Lệnh chính:
   ```
   python -m src.data.prepare_data --config configs/data.yaml --out-root /kaggle/working/nhom05-falldet-processed --workers 4
   python -m src.data.split --meta /kaggle/working/nhom05-falldet-processed/metadata.csv --out /kaggle/working/nhom05-falldet-processed/splits.json --seed 42
   ```
3. Tạo Kaggle Dataset `nhom05-falldet-processed` từ thư mục trên.
4. `notebooks/01_eda.ipynb`: Add Input dataset vừa tạo, bật Internet, không cần GPU, Run All.
5. `notebooks/02_m1_baseline.ipynb`: Add Input dataset vừa tạo, bật GPU và Internet, Run All.
6. Tải về và commit lên nhánh: `metadata.csv`, `splits.json`, `DATA_CARD.md` vào `dataset_index/`; `reports/figures/*.png`; `reports/results/*`. Không dùng thư mục `data/` vì đã bị `.gitignore` chặn.
7. Mở Pull Request vào `main` theo checklist mục 7 của prompt.

| Bước | Internet | GPU | Thời gian (ước tính, chưa đo) | Ghi ra /kaggle/working (ước tính, chưa đo) |
|---|---|---|---|---|
| 00_prepare_data | có | không | 15–30 phút | 1–3 GB |
| 01_eda | có | không | 5–10 phút | dưới 50 MB |
| 02_m1_baseline | có | có | 30–60 phút | dưới 200 MB |

## Số liệu thật đã kiểm tra
Đo trên máy local, ảnh và video giả, **2 CSV nhãn URFD là thật**. Lệnh: `python tests/fake_kaggle.py <thư mục> --urfd-falls urfall-cam0-falls.csv --urfd-adls urfall-cam0-adls.csv`, sau đó `python -m src.data.prepare_data --config <thư mục>/data.yaml --workers 6`.
- URFD: 70 chuỗi (30 fall, 40 ADL); trạng thái `OK` 52, `CSV_THIEU_FRAME` 18; `adl_hard_neg = 1`: 16; 11544 dòng nhãn; `fall-16` có `fall_start_frame = 16`, `fall_end_frame = 46`; `adl-17` có 52 frame NaN.
- Tổng số ảnh 11936 khớp output notebook `check_urfd_structure_v2`, nhưng số ảnh được dựng lại từ CSV và bảng trong notebook đó, không phải đếm trên ảnh thật.
- Le2i: chưa đo trên video thật. Số 130/96/31/3 trong log chạy thử là của dữ liệu giả.
- Chất lượng M1 (P, R, F1, FAR, latency): chưa đo.

- YOLO11 + ByteTrack (không phải dataset): `scripts/make_tracks_tmp.py` trên video 20 frame ghép từ ảnh mẫu `bus.jpg` của ultralytics 8.4.202: 80 dòng, 4 track, frame 0..19, không trùng dòng, đúng schema; `src.models.m1_rule` đọc được file đó.

Test: `python -m pytest tests -q`: 96 pass, 0 fail (phần lớn dữ liệu giả; 1 file end-to-end cần ffmpeg). `python tests/smoke_test.py`: 18/18.

## Chỗ khác với prompt này
- Toàn bộ chạy trên Kaggle, có output thật → agent không chạy trên dataset → TV1 quyết định tự chạy sau.
- Tạo HANDOFF từ Việc 1 và cập nhật sau mỗi Việc → chỉ tạo sau khi xong code → agent làm thiếu, TV1 phát hiện.
- Chỉ sửa file của mình → đã sửa `.gitignore` chung → dòng `data/` chặn cả `src/data/`, phải thêm `!src/data/`, `!reports/figures/*.png`, `*.parquet`.
- `requirements.txt` chốt cứng → đã làm, nhưng notebook Kaggle chỉ cài `ultralytics` và `lap` → cài cả file sẽ thay torch có sẵn của Kaggle.
- Đặt `notebooks/01_eda.ipynb` → file này đã có sẵn của TV3 và đã được thay bằng notebook mới theo đúng tên prompt yêu cầu; bản cũ còn trong lịch sử git.
- Việc 0 không sửa logic → giữ nguyên source, chỉ xóa output → notebook Le2i nặng 6 MB do ảnh ghép của người thật.
- FAR = FP / tổng giờ video → giữ đúng, thêm `far_nonfall_per_hour` → mục 2.2 nói FAR tính trên đoạn không-ngã, có thể hiểu theo cả hai cách.
- Thêm ngoài danh sách: `--split all`, `--tuned`, `--T` nhiều giá trị, `scripts/summarize_results.py`, tách `src/data/{common,media,urfd,le2i,metadata,checks,data_card,env_info}.py`, `src/models/m1_features.py`, `src/eval/frame_metrics.py` → để giữ mỗi file dưới 250 dòng và chạy cross-dataset.

## Giả định và việc bạn tự quyết
- `label` để trống với 3 video ODD: schema chỉ có `fall|adl`, điền giá trị nào cũng là đoán.
- URFD: `scene = urfd`, `scene_part` trống. Le2i: `camera` trống, vì dữ liệu không có thông tin camera.
- `adl_hard_neg = 0` ở mọi video Le2i: chỉ có 1 ví dụ sofa được nêu, chưa có danh sách.
- `path` trong metadata là đường dẫn tương đối, để dataset chuyển sang `/kaggle/input` vẫn dùng được.
- `select_main_track` Le2i: mẫu số là số frame có GT, để track rất ngắn không thắng.
- bbox không còn diện tích sau khi clip thì bỏ và đếm trong DATA_CARD.
- Chiều cao chuẩn hóa vận tốc là chiều cao lớn nhất trong 1.5 s gần nhất, vì bbox thấp đi khi ngã.
- M1 chỉ dùng quá khứ, nên điểm tính được khi video đang chạy.
- Thành phần bất động của M1: số giây bất động liên tục từ khi tư thế đổi, chia cho `immobile_s`, tối đa bằng 1; chỉ cộng điểm, không bắt buộc.
- Lưu file nhỏ của dữ liệu vào `dataset_index/`, vì thư mục tên `data/` bị `.gitignore` chặn.
- Trọng số M1 cố định 0.30/0.10/0.45/0.15, không nằm trong lưới tune, để lưới nhỏ và dễ giải thích.
- Giữ `conf = 0.25` và `yolo11n.pt` theo `configs/default.yaml`.
- Le2i chia theo (nhãn, scene), không tách thêm phần 01/02.

## Việc còn tồn đọng / rủi ro đã biết
Output cần dataset, TV1 tự chạy (theo checklist mục 7):
- `metadata.csv`, `annotations/*.csv`, `splits.json`, `DATA_CARD.md`, `videos/*.mp4` thật.
- 5 biểu đồ EDA và `augment_demo.png` trong `reports/figures/`.
- `tracks.parquet`, `predictions.csv`, bảng Precision/Recall/F1/FAR/latency và kết quả ADL negative khó trong `reports/results/`.
- Pull Request vào `main`: mở sau khi có các output trên.

Rủi ro đã biết:
- Toàn bộ pipeline chưa chạy trên dữ liệu thật, có thể lộ lỗi mới. Số liệu lệch thì `prepare_data` dừng và in từng chỗ lệch.
- Ngưỡng mặc định trong `configs/m1.yaml` là giá trị khởi đầu, chưa kiểm chứng trên dữ liệu thật. Chất lượng M1 chưa đo.
- Le2i thật có audio mp3. Khi thử, ffmpeg tạo AVI mp3 có packet rỗng làm mp4 dư 1 frame. Notebook kiểm tra đo số frame khớp 190/190 nên dự kiến không gặp; nếu gặp, assert dừng kèm tên video.
- `window_label` với cửa sổ dài hơn 14 frame không bao giờ dương ở khoảng ngã Le2i ngắn nhất (7 frame).
- M1 khó thấy người ngã thẳng về phía camera vì w/h ít đổi.
- Không có test tracker reset giữa 2 video trong `make_tracks_tmp.py`; code dùng `persist=False`.
- Thay đổi chưa commit ở `src/trainer.py` (chỉ khoảng trắng) không phải của nhánh này, được để nguyên.

## Câu hỏi cho TV1
1. FAR báo cáo chính dùng mẫu số nào: tổng giờ video (`far_per_hour`) hay giờ không-ngã (`far_nonfall_per_hour`)?
2. Hai module cũ của TV3 trái định nghĩa mới: `src/evaluate.py` cắt latency âm về 0, `src/datasets.py` coi cả chuỗi URFD là một sự kiện. Có cần TV3 chuyển sang `src/eval/event_metrics.py` không?
3. Có đánh dấu `adl_hard_neg = 1` cho video sofa của Le2i không? Nếu có cần danh sách.
4. Version của 3 dataset Kaggle là bao nhiêu?
