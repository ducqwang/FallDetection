# HANDOFF TV1
Nhánh: tv1/make-tracks (từ `main` 50068ca)  |  Cập nhật: 2026-11-11

> Agent không chạy pipeline trên dataset: theo quyết định của TV1, TV1 tự chạy trên Kaggle. Output EDA và M1 còn thiếu được liệt kê ở mục "Việc còn tồn đọng". PR vào `main`: https://github.com/ducqwang/FallDetection/pull/1

## Trạng thái từng Việc
| Việc | Trạng thái | Đã chạy trên dữ liệu thật? | Ghi chú |
|---|---|---|---|
| 0. Notebook kiểm tra | xong | chưa | Đã thêm 2 notebook, giữ nguyên source, xóa output. Agent không chạy lại; bản gốc đã được TV1 chạy trên Kaggle |
| 1. prepare_data | xong | có | TV1 chạy trên Kaggle sau khi sửa lỗi `parse_annotation` với dòng `197`. `metadata.csv` và `DATA_CARD.md` thật đã ở `dataset_index/` |
| 2. split | xong | có | `splits.json` thật đã ở `dataset_index/`, seed 42, tỉ lệ 70/15/15 |
| 3. loader, labels, augment | một phần | chỉ dữ liệu giả | Code và test xong. Chưa có hình minh họa augmentation thật |
| 4. EDA | một phần | chỉ dữ liệu giả | Code xong, đã thêm bảng kích thước, `color_by` và đo độ tách. Chưa có hình thật trong `reports/figures/` |
| 5. M1 | một phần | chỉ dữ liệu giả | Code xong. Ngưỡng chưa tune. Tracks chính thức `scripts/make_tracks.py` mới test bằng model giả; bản tạm trước đó đã chạy với YOLO thật trên 1 ảnh mẫu, không phải dataset |
| 6. Bộ đánh giá | một phần | chỉ dữ liệu giả | Code và unit test xong. Chưa có bảng kết quả thật |

## Bản đồ file
| File | Làm gì | Hàm/CLI chính | Đầu vào | Đầu ra |
|---|---|---|---|---|
| `configs/data.yaml` | Đường dẫn, nguồn dataset, số liệu mong đợi | khối `paths`, `dataset_sources`, `expect` | | |
| `configs/m1.yaml` | Mọi ngưỡng và lưới tune của M1 | `signals`, `rules`, `weights`, `alarm`, `grid` | | |
| `src/data/prepare_data.py` | CLI Việc 1 | `--config --input-root --out-root --workers --skip-existing` | `/kaggle/input` | mp4, csv, DATA_CARD |
| `src/data/urfd.py` | Đọc URFD, ghép nhãn theo số frame | `read_labels`, `scan_sequences`, `match_labels` | ảnh, 2 CSV | dict chuỗi |
| `src/data/le2i.py` | Đọc Le2i, annotation, bbox | `scan_videos`, `parse_annotation(path)` (header_top, header_mid), `clean_boxes(video_id, boxes, width, height, n_frames)` | avi, txt | dict video, bảng bbox |
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
| `src/eda/plots.py` | CLI Việc 4 | `plot_*`, `plot_scene_size_table`, `export_scene_size_table`, `plot_embedding_pair`; `--meta --gt-bbox --tracks --out-dir --results-dir --data-card --size-warn` | metadata, bbox | 7 PNG (1–5, 5b, 6), CSV bảng kích thước, `size_separation.json`, khối trong DATA_CARD |
| `src/eda/embedding.py` | PCA/t-SNE | `plot_embedding(X, y, source, out, perplexity, seed, color_by='label'\|'size', sizes)`, `plot_embedding_pair` | X, y, sizes | PNG |
| `src/eda/size_check.py` | Đo độ tách theo kích thước, ghi DATA_CARD | `measure_size_separation(X, y, sizes)`, `update_data_card(path, meta, separation)` | X, y, sizes | dict, Markdown |
| `src/eda/style.py` | Bảng màu, lưu hình dùng chung | | | |
| `src/data/frame_size.py` | Bảng scene × kích thước × lớp, khối cảnh báo | `scene_size_table(meta)`, `size_block(meta, separation)` | metadata | DataFrame, Markdown |
| `src/models/m1_features.py` | Tín hiệu theo track | `split_at_long_gaps(track, fps, max_gap_s)`, `track_signals(track, fps, smooth_s, reference_s)` | track | DataFrame |
| `src/models/m1_rule.py` | CLI M1 | `predict`, `score_signals`, `posture_score`; `--tracks \| --gt-bbox --meta --out --tuned` | tracks | `predictions.csv` |
| `src/eval/event_metrics.py` | Độ đo sự kiện | `find_alarms`, `match_alarms`, `evaluate(pred, meta_rows, thr, tolerance_s, merge_gap_s, frame_labels, reference_meta)` | dự đoán, metadata | dict |
| `src/eval/frame_metrics.py` | Độ đo frame | `frame_level_metrics(records, thr)` | điểm theo frame | dict |
| `scripts/evaluate.py` | CLI đánh giá, cảnh báo video không có dự đoán (`n_videos_without_predictions` trong JSON) | `--pred --meta --split --dataset --thr --tuned --T --tag` | csv | bảng, JSON |
| `scripts/tune_m1.py` | Grid search chỉ trên val | `--tracks \| --gt-bbox --meta --dataset --out` | tracks | yaml, csv |
| `scripts/make_tracks.py` | Tracks chính thức TV1: YOLO11 + ByteTrack, shard từng video, resume, sidecar | `--meta --root --out-dir --dataset --model --conf --imgsz --iou --device --tracker --limit --resume/--no-resume` | mp4 | `tracks.parquet`, `tracks_meta.json`, `shards/` |
| `src/data/track_runner.py` | Lõi của `make_tracks.py` | `track_video`, `process_video(model, row, root, options)`, `reset_trackers`, `write_shard`, `read_shard` | model, 1 dòng metadata | bảng tracks, thống kê |
| `src/data/tracks.py` | Kiểm tra tracks dùng chung | `validate_tracks(tracks, meta)` → `TrackCheck(issues, empty_videos)`, `clip_to_frame`, `cast_tracks` | tracks, metadata | danh sách vi phạm |
| `scripts/check_tracks_quality.py`, `src/eval/track_quality.py` | Chất lượng detector | `--tracks --root --meta --gt-bbox --out-dir`; `le2i_track_quality`, `urfd_track_quality`, `summarize` | tracks, GT | 5 CSV trong `quality/` |
| `scripts/summarize_results.py` | Gom bảng kết quả | `--results-dir --out` | JSON | csv |
| `tests/` | Test TV1, `fake_kaggle.py` sinh dữ liệu giả | `python -m pytest tests -q` | | |
| `notebooks/00_prepare_data`, `tv1_01_eda`, `02_m1_baseline` | Chạy trên Kaggle | | | |
| `docs/SCHEMA.md`, `docs/EVAL_PROTOCOL.md` | Định dạng file, giao thức đánh giá | | | |

## Cách chạy (lệnh y nguyên, theo thứ tự, trên Kaggle)
Bước 1 đến 3 đã xong, còn bước 4 và 5, rồi commit output ở bước 6.

1. Điền `version` của 3 dataset vào `configs/data.yaml`, push lên nhánh.
2. `notebooks/00_prepare_data.ipynb`: Add Input 3 dataset gốc, bật Internet, không cần GPU, Run All. Lệnh chính:
   ```
   python -m src.data.prepare_data --config configs/data.yaml --out-root /kaggle/working/nhom05-falldet-processed --workers 4
   python -m src.data.split --meta /kaggle/working/nhom05-falldet-processed/metadata.csv --out /kaggle/working/nhom05-falldet-processed/splits.json --seed 42
   ```
3. Tạo Kaggle Dataset `nhom05-falldet-processed` từ thư mục trên.
4. `notebooks/tv1_01_eda.ipynb`: Add Input dataset vừa tạo, bật Internet, không cần GPU, Run All.
5. `notebooks/02_m1_baseline.ipynb`: Add Input dataset vừa tạo, bật GPU và Internet, Run All.
6. Tải về và commit lên nhánh: `metadata.csv`, `splits.json`, `DATA_CARD.md` vào `dataset_index/`; `reports/figures/*.png`; `reports/results/*`. Không dùng thư mục `data/` vì đã bị `.gitignore` chặn.
7. Pull Request vào `main`: đã mở (PR #1). Commit output lên nhánh này thì PR tự cập nhật.

| Bước | Internet | GPU | Thời gian (ước tính, chưa đo) | Ghi ra /kaggle/working (ước tính, chưa đo) |
|---|---|---|---|---|
| 00_prepare_data | có | không | 15–30 phút | 1–3 GB |
| tv1_01_eda | có | không | 5–10 phút | dưới 50 MB |
| 02_m1_baseline | có | có | 30–60 phút | dưới 200 MB |

### Quy trình M1
Đúng thứ tự, mỗi bộ dữ liệu dùng cùng một file tune cho cả 3 bước:
1. (a) `scripts/tune_m1.py --tracks|--gt-bbox ... --meta ... --dataset urfd|le2i --out <yaml>`: tune trên val của từng bộ → 1 file yaml tuned mỗi bộ.
2. (b) `python -m src.models.m1_rule --tracks|--gt-bbox ... --meta ... --tuned <yaml> --out predictions.csv`.
3. (c) `scripts/evaluate.py --pred ... --meta ... --split test --dataset <bộ đó> --tuned <yaml>`, chạy MỘT lần; rồi `--split all --dataset <bộ kia>` cho cross-dataset.

Lệnh đầy đủ ở `docs/EVAL_PROTOCOL.md` mục 7; `notebooks/02_m1_baseline.ipynb` theo đúng thứ tự này.

**Cảnh báo: không trộn luật mặc định với `thr` đã tune.** `evaluate.py --tuned` chỉ lấy `alarm.thr`. Các luật đã tune (`ar_upright`, `ar_lying`, `vy_full`, `window_s`) phải có mặt lúc sinh predictions (`m1_rule --tuned`). Bỏ `--tuned` ở `m1_rule` thì lệnh vẫn chạy, không báo lỗi, nhưng kết quả sai.

## Tracks chính thức (TV1, GK)
`scripts/make_tracks.py` là **tracks chính thức của TV1 cho giữa kỳ**; M1, M2, M3 dùng chung một file. TV2 có thể thay bằng bản của họ nếu đúng schema (`docs/SCHEMA.md`) và `src.data.tracks.validate_tracks` không báo vi phạm; khi thay thì **phải chạy lại M1** (tune → predict → evaluate) và ghi lại nguồn tracks trong báo cáo.

Chạy trên Kaggle (`notebooks/02_m1_baseline.ipynb` mục 1, bật GPU + Internet):
```
python scripts/make_tracks.py --root $ROOT --out-dir /kaggle/working/tracks --limit 3      # chạy thử, kiểm tra track_id video 2 bắt đầu từ 1
python scripts/make_tracks.py --root $ROOT --out-dir /kaggle/working/tracks --dataset all --model yolo11m.pt
python scripts/check_tracks_quality.py --tracks /kaggle/working/tracks/tracks.parquet --root $ROOT
```
`$ROOT` là thư mục `nhom05-falldet-processed` (trong notebook: `data_root()`). Không truyền `--out-dir` thì script ghi vào `/kaggle/working/tracks` (Kaggle) hoặc `reports/tracks` (máy khác) và in đường dẫn.
Kaggle ngắt: chạy lại đúng lệnh; `--resume` (mặc định) bỏ qua video đã có shard hợp lệ trong `shards/`. `--no-resume` để chạy lại hết.

| Bước | Internet | GPU | Thời gian (ước tính, chưa đo) | Dung lượng (ước tính, chưa đo) |
|---|---|---|---|---|
| `make_tracks.py`, 200 video, 54 002 frame | có (tải `yolo11m.pt` ~40 MB) | có | 25–45 phút trên T4/P100; CPU vài giờ | dưới 50 MB |
| `check_tracks_quality.py` | không | không | dưới 1 phút | dưới 1 MB |

Đầu ra trong `--out-dir`: `tracks.parquet`, `tracks_meta.json` (tham số, `weights_sha256`, version ultralytics/torch/opencv, nội dung tracker yaml, git commit, số dòng bỏ theo lý do, cảnh báo số frame, video không có người, thời gian), `quality/*.csv`, `shards/`. Lưu cả thư mục thành Kaggle Dataset private `nhom05-falldet-tracks`, chia sẻ cho nhóm (cách làm ở notebook mục 1c).

Cách dùng:
- M1: `--tracks <dir>/tracks.parquet` cho `scripts/tune_m1.py`, `src.models.m1_rule` theo mục "Quy trình M1". So `quality/le2i_track_quality_by_scene.csv` với dòng `m1_gt` (M1 chạy trên GT bbox): hit_rate thấp ở scene nào thì lỗi M1 ở scene đó có thể do detector.
- M2 (pose, TV2): lấy hộp người và `track_id` từ file này, chọn track chính bằng `src.data.labels.select_main_track`.
- M3: `python -m src.models.m3_cnn_gru ... --tracks <dir>/tracks.parquet` (README mục TV3).
- Mọi bản tracks (kể cả của TV2) kiểm tra bằng `validate_tracks(pd.read_parquet(path), load_metadata())`.

## Số liệu thật đã kiểm tra
Từ `dataset_index/` do TV1 chạy trên Kaggle:
- Tổng 200 video: URFD 70 (30 fall, 40 ADL), Le2i 130 (99 fall, 31 ADL). Tổng số frame URFD 11936.
- Le2i: `header_top` 127, `header_mid` 3; 42417 dòng bbox đã đọc, bỏ 71 dòng toàn 0, bỏ 92 hộp không còn diện tích sau khi clip.
- Split (fall/ADL): URFD train 21/28, val 4/6, test 5/6; Le2i train 69/22, val 15/4, test 15/5.

Đo trước đó trên máy local, ảnh và video giả, **2 CSV nhãn URFD là thật**. Lệnh: `python tests/fake_kaggle.py <thư mục> --urfd-falls urfall-cam0-falls.csv --urfd-adls urfall-cam0-adls.csv`, sau đó `python -m src.data.prepare_data --config <thư mục>/data.yaml --workers 6`.
- URFD: 70 chuỗi (30 fall, 40 ADL); trạng thái `OK` 52, `CSV_THIEU_FRAME` 18; `adl_hard_neg = 1`: 16; 11544 dòng nhãn; `fall-16` có `fall_start_frame = 16`, `fall_end_frame = 46`; `adl-17` có 52 frame NaN.
- Tổng số ảnh 11936 khớp output notebook `check_urfd_structure_v2`, nhưng số ảnh được dựng lại từ CSV và bảng trong notebook đó, không phải đếm trên ảnh thật.
- Lỗi thật đầu tiên trên Kaggle (TV1 báo, đã sửa): `parse_annotation` dừng ở `dòng bbox không đủ 6 cột: '197'`, file Coffee_room_01 video (26).
- Chất lượng M1 (P, R, F1, FAR, latency): chưa đo.

- Dữ liệu tổng hợp, không phải dataset (đo bằng đoạn lệnh tạm trước khi sửa; tính chất tương ứng được kiểm tra trong `tests/test_m1.py`): `aspect` đứng 0.353, nằm 2.833; tại frame vừa nằm `posture_score` = 1.0. Khoảng trống 10 frame: `vy` lớn nhất 0.1765, bằng lúc không có khoảng trống; nếu tính theo số dòng sẽ là 2.75.
- YOLO11 + ByteTrack (không phải dataset): bản tạm trước đây của `scripts/make_tracks.py` trên video 20 frame ghép từ ảnh mẫu `bus.jpg` của ultralytics 8.4.202: 80 dòng, 4 track, frame 0..19, không trùng dòng, đúng schema; `src.models.m1_rule` đọc được file đó.

Test: `python3 -m pytest tests -q`: 196 pass, 9 skip, 0 fail (phần lớn dữ liệu giả; tracks: `tests/test_make_tracks.py` dùng model giả, `tests/test_tracks_validate.py`, `tests/test_track_quality.py`). Cả 9 test skip ở `tests/test_pipeline_fake.py` vì máy chạy không có ffmpeg, chúng chưa chạy lại từ khi có ffmpeg đạt 145 pass. `python3 tests/smoke_test.py`: 18/18.

## Chỗ khác với prompt này
- Prompt phiên bản 5 có 3 video chưa xác định, 127 video có nhãn, schema không có `label_source`, `aux_code` → đổi theo chỉ đạo mới của TV1: 130 video, thêm 2 cột → `docs/PROMPT_TV1_GiuaKy.md` không có trong repo, bản trên máy agent vẫn là phiên bản 5, nên agent làm theo tóm tắt 6 mục của TV1.
- Toàn bộ chạy trên Kaggle, có output thật → agent không chạy trên dataset → TV1 quyết định tự chạy sau.
- Tạo HANDOFF từ Việc 1 và cập nhật sau mỗi Việc → chỉ tạo sau khi xong code → agent làm thiếu, TV1 phát hiện.
- Chỉ sửa file của mình → đã sửa `.gitignore` chung → dòng `data/` chặn cả `src/data/`, phải thêm `!src/data/`, `!reports/figures/*.png`, `*.parquet`.
- `requirements.txt` chốt cứng → đã làm, nhưng notebook Kaggle chỉ cài `ultralytics` và `lap` → cài cả file sẽ thay torch có sẵn của Kaggle.
- Đặt EDA ở `notebooks/01_eda.ipynb` → đặt ở `notebooks/tv1_01_eda.ipynb` → `01_eda.ipynb` là file có sẵn của TV3, đã khôi phục nguyên bản từ `main`.
- Việc 0 không sửa logic → giữ nguyên source, chỉ xóa output → notebook Le2i nặng 6 MB do ảnh ghép của người thật.
- FAR = FP / tổng giờ video (Việc 6) → FAR chính là `far_nonfall_per_hour`, `far_per_hour` là số phụ → TV1 đã chốt.
- `window_label` chia theo độ dài cửa sổ (Việc 3) → chia theo min(độ dài cửa sổ, độ dài sự kiện) → TV1 đã chốt, tránh cửa sổ dài không bao giờ dương.
- Gọi `check_predictions` ở script và trong `evaluate()`, nhưng chỉ một chỗ → chỉ gọi trong `evaluate()`; script truyền toàn bộ metadata qua `reference_meta` → nếu lọc trước thì `video_id` lạ sẽ không bị bắt.
- Thêm ngoài danh sách: `--split all`, `--tuned`, `--T` nhiều giá trị, `reference_meta` trong `evaluate()`, `scripts/summarize_results.py`, tách `src/data/{common,media,urfd,le2i,metadata,checks,data_card,env_info}.py`, `src/models/m1_features.py`, `src/eval/frame_metrics.py` → để giữ mỗi file dưới 250 dòng và chạy cross-dataset.

## Giả định và việc bạn tự quyết
- URFD: `scene = urfd`, `scene_part` trống. Le2i: `camera` trống, vì dữ liệu không có thông tin camera.
- `path` trong metadata là đường dẫn tương đối, để dataset chuyển sang `/kaggle/input` vẫn dùng được.
- `select_main_track` Le2i: mẫu số là số frame có GT, để track rất ngắn không thắng.
- bbox không còn diện tích sau khi clip thì bỏ và đếm trong DATA_CARD.
- Chiều cao chuẩn hóa vận tốc là chiều cao lớn nhất trong 1.5 s gần nhất, vì bbox thấp đi khi ngã.
- M1 chỉ dùng quá khứ, nên điểm tính được khi video đang chạy.
- Thành phần bất động của M1: số giây bất động liên tục từ khi tư thế đổi, chia cho `immobile_s`, tối đa bằng 1; chỉ cộng điểm, không bắt buộc.
- Khoảng trống track dài hơn `max_gap_s = 0.5` giây: tách đoạn, reset trạng thái, không xuất điểm trong khoảng trống; ngắn hơn: nội suy tuyến tính.
- Kiểm tra `n_bbox_rows`, `n_bbox_zero_rows`, `fall_len_frames` chỉ áp dụng cho video `header_top`, vì các số 39970, 71 và 7/23.5/45 được đo trên nhóm này; số đo cho 130 video nằm trong `DATA_CARD.md`.
- Lưu file nhỏ của dữ liệu vào `dataset_index/`, vì thư mục tên `data/` bị `.gitignore` chặn.
- Trọng số M1 cố định 0.30/0.10/0.45/0.15, không nằm trong lưới tune, để lưới nhỏ và dễ giải thích.
- Tracks chính thức dùng `yolo11m.pt`, `conf = 0.25`, `imgsz = 640`, `iou = 0.7` (mặc định NMS của ultralytics), `bytetrack.yaml` theo chỉ đạo TV1; `configs/default.yaml` (demo của TV3) vẫn ghi `yolo11n.pt`.
- `make_tracks.py`: video lệch đúng 1 frame so với metadata là cảnh báo (frame dư bị bỏ, đếm `n_out_of_range`); lệch hơn 1 là lỗi, không ghi `tracks.parquet`, thoát mã 1. Tracks vi phạm `validate_tracks` thì thoát mã 2.
- `--resume` chỉ dùng lại shard khi cùng `--model`, `--conf`, `--iou`, `--imgsz`, `--tracker`; khác thì chạy lại video đó.

## Việc còn tồn đọng / rủi ro đã biết
Output cần dataset, TV1 tự chạy (theo checklist mục 7):
- 5 biểu đồ EDA và `augment_demo.png` trong `reports/figures/`.
- `tracks.parquet` (chạy `make_tracks.py` trên Kaggle), báo cáo chất lượng detector, `predictions.csv`, bảng Precision/Recall/F1/FAR/latency và kết quả ADL negative khó trong `reports/results/`.
- Đã có: `metadata.csv`, `splits.json`, `DATA_CARD.md` trong `dataset_index/`; PR #1 mở trước khi có các output trên, nên phải commit bổ sung hoặc mở PR sau.

Rủi ro đã biết:
- `prepare_data` và `split` đã chạy thật; EDA, tracking và M1 chưa từng chạy trên dữ liệu thật, có thể lộ lỗi mới. Khi chạy lại, số liệu lệch thì `prepare_data` dừng và in từng chỗ lệch; annotation hoặc video lỗi được in đủ một lần.
- Ngưỡng mặc định trong `configs/m1.yaml` là giá trị khởi đầu, chưa kiểm chứng trên dữ liệu thật. Chất lượng M1 chưa đo.
- Le2i thật có audio mp3. Khi thử, ffmpeg tạo AVI mp3 có packet rỗng làm mp4 dư 1 frame. `prepare_data` đã chạy xong trên dữ liệu thật nên chưa gặp lỗi này. `avi_to_mp4` giữ nguyên tham số của prompt, không thêm cờ trước khi có lỗi thật.
- Nếu mp4 lệch số frame khi chạy lại: `_verify_mp4` trong `src/data/prepare_data.py` in tên video, số frame nguồn, số frame mp4, kèm gợi ý `thử -fps_mode passthrough (hoặc -vsync 0 với ffmpeg cũ)` cho Le2i, rồi `convert_videos` in mọi video lệch và dừng. Test: `test_le2i_frame_mismatch_names_video_counts_and_hint`.
- Khi tune M1, để ý frame người mới vào khung: bbox lớn dần và bị cắt mép có thể làm vận tốc dọc tăng giả, gây FP. Chưa xử lý vì chưa thấy trên dữ liệu thật. Nếu thấy, thêm quy tắc bỏ qua vài frame đầu của mỗi track.
- M1 khó thấy người ngã thẳng về phía camera vì w/h ít đổi.
- Tracker reset giữa 2 video: `make_tracks.py` dùng `persist=False` và gọi `reset()` mọi tracker còn trong `model.predictor` sau mỗi video. Đã test bằng model giả (`test_tracker_resets_between_videos`), **chưa kiểm chứng với ultralytics thật**: kiểm tra trên Kaggle bằng `--limit 3`, video thứ 2 phải có `track_id` bắt đầu từ 1.

## Quyết định của TV1 đã chốt
1. FAR báo cáo chính là `far_nonfall_per_hour`; `far_per_hour` là số phụ. Ghi ở `docs/EVAL_PROTOCOL.md`.
2. Hai module cũ của TV3 khác định nghĩa mới: TV1 không sửa, xem mục "Việc cho TV3".
3. `adl_hard_neg = 0` cho toàn bộ Le2i ở giữa kỳ; danh sách video Le2i negative khó chưa có. Ghi ở `DATA_CARD.md`.
4. Cảnh báo tính tại frame đầu của đoạn; đoạn bắt đầu trước cửa sổ dung sai bị tính cả FP lẫn FN. Ghi ở `docs/EVAL_PROTOCOL.md`, kèm ví dụ số.
5. Le2i chia theo (nhãn, scene) nên kết quả trong cùng bộ có thể lạc quan; kết quả chính là cross-dataset (`--split all`). Ghi ở `DATA_CARD.md` và `docs/EVAL_PROTOCOL.md`.
6. `mp4 frame mismatch` là số đếm thật từ `convert_videos`; có video lệch thì đã raise trước dòng log.

7. Le2i dùng đủ 130 video (99 fall, 31 ADL), `has_event_label = 1` cho tất cả, không còn khái niệm video chưa xác định. 3 video có cặp khoảng ngã giữa file có `label_source = header_mid`: `le2i_coffee_room01_026`, `le2i_coffee_room02_050`, `le2i_coffee_room02_052`.

## Rà đặc trưng theo toạ độ pixel (kích thước khung có thể thành đường tắt)
Le2i có 25 video 320×180, tất cả đều là fall; mọi ADL đều 320×240 (xem khối "Kích thước khung hình" trong `DATA_CARD.md`). Đã rà mọi hàm đặc trưng trong `src/`.

| Đặc trưng (file) | Trước | Sau | Ghi chú |
|---|---|---|---|
| `aspect` = w/h (`m1_features.py`, `window_features.py`) | tỉ lệ | không đổi | Đã là tỉ lệ. Chỉ bất biến nếu kích thước khác nhau do co đều; nếu 320×180 là co không đều hoặc cắt thì aspect lệch. **Chưa kiểm chứng**, xem cách đo độ tách bên dưới |
| `vy`, `ay`, `speed` (`m1_features.py`) | chiều cao người/giây | không đổi | Đã chia cho chiều cao chuẩn (lớn nhất trong 1.5 s), không phụ thuộc độ phân giải; `test_resolution_invariant` kiểm tra |
| `aspect_*`, `vy_*`, `ay_max`, `speed_mean`, `still_ratio` (`window_features.py`) | thống kê của các tín hiệu trên | không đổi | Không có toạ độ x, y hay w, h tuyệt đối |
| ngưỡng `still_speed`, `vy_*`, `ay_*` (`configs/m1.yaml`) | chiều cao người/giây | không đổi | |
| `normalize_keypoints` (`features.py`) | chia cho w, h của bbox | không đổi | Đã chuẩn hóa theo bbox |
| `bbox_geometric_features` (`features.py`, TV3) | `vy`, `ay` tính bằng **pixel/frame** (`np.diff(cy)`) | **chưa sửa** | File của TV3. `ar` là tỉ lệ; `stillness` là tỉ lệ so với trung bình; chỉ `vy`, `ay` phụ thuộc kích thước. `datasets.py` dùng hàm này cho M2/M3 |
| `RuleThresholds.vy_fall = 8.0` (`models/model1_rule.py`, TV3) | ngưỡng **pixel/frame** | **chưa sửa** | File của TV3, M1 cũ. Cùng ngưỡng 8 px/frame ứng với tỉ lệ khác nhau trên khung 180 và 240 dòng; `m1_rule.py` mới đã thay bằng chiều cao người/giây |

TV1 không sửa file của TV3, nên 2 dòng "chưa sửa" chuyển cho TV3 (xem "Việc cho TV3"). Mọi đặc trưng của TV1 đã là tỉ lệ hoặc chuẩn hóa theo chiều cao người, nên không có đặc trưng nào bị đổi định nghĩa.

Cách đo trên dữ liệu thật: `python -m src.eda.plots ...` ghi `reports/results/size_separation.json` (silhouette theo kích thước, khoảng cách centroid, silhouette riêng trong lớp fall, silhouette theo lớp để so sánh) và tự thêm cảnh báo vào `DATA_CARD.md` nếu silhouette ≥ 0.25 (`--size-warn` đổi ngưỡng). Ngưỡng 0.25 là quy ước chung của silhouette cho "gần như chưa có cấu trúc nhóm", không phải đo từ dữ liệu này. Đo chỉ trên cửa sổ của video Le2i vì chỉ Le2i có 2 kích thước.

## Hạn chế dữ liệu đã biết
- **Le2i bbox trùng:** `le2i_gt_bbox.csv` có 350 dòng trùng `(video_id, frame_idx)` (`le2i_coffee_room01_035` chiếm 471 dòng trong các nhóm trùng, 46 video khác mỗi video ≤ 10 dòng). Xử lý ở chỗ đọc bằng hàm chung `src/data/common.drop_duplicate_frames` (giữ dòng đầu theo thứ tự file), dùng trong `m1_rule.gt_boxes_as_tracks` (cả `tune_m1 --gt-bbox`), `src/eda/plots.py` và `labels.select_main_track` (M3 dùng qua hàm này). File do `prepare_data` sinh ra giữ nguyên.
- **Le2i scene trùng với lớp:** Coffee room 01 (48) và Home 01 (30) toàn video fall; ADL chỉ có ở Coffee room 02 (8 ADL, 14 fall) và Home 02 (23 ADL, 7 fall). Khuyến nghị báo kết quả Le2i theo scene, không chỉ con số gộp.
- **URFD độ dài clip khác theo lớp:** trung vị fall 3.1 s, ADL 7.8 s. Đánh giá ở mức sự kiện/cửa sổ; không dùng độ dài clip làm đặc trưng.

### Le2i GT có frame ghi trùng
Lần chạy EDA đầu tiên trên Kaggle dừng với `Track có frame trùng` vì `annotations/le2i_gt_bbox.csv` có 350 dòng trùng `(video_id, frame_idx)` (627 dòng thuộc 47 video; `le2i_coffee_room01_035` có 471 dòng trong nhóm trùng; trong 231 nhóm frame, hộp khác nhau). `aux_code` chưa rõ nghĩa nên không biết hộp nào đúng.
Cách xử lý: `src/data/common.drop_duplicate_frames` giữ dòng đầu của mỗi frame và in `CẢNH BÁO | Le2i GT: bỏ N dòng bbox trùng`. Áp dụng ở `plots.py`, `m1_rule.gt_boxes_as_tracks` (cũng dùng bởi `tune_m1`) và `labels.select_main_track`. `contiguous_track` vẫn raise nếu còn trùng. `prepare_data` chưa đổi nên `le2i_gt_bbox.csv` vẫn còn dòng trùng; cần TV1 xác nhận hộp nào đúng trước khi bỏ ở nguồn.
Bảng kích thước và khối DATA_CARD giờ được ghi trước bước đặc trưng cửa sổ, nên không mất khi bước sau lỗi.

## Việc cho TV3
TV3 tự chuyển các chỗ sau sang `src/eval/event_metrics.py`; TV1 không sửa.

| File, dòng (trên `main`) | Code hiện tại | Khác định nghĩa mới |
|---|---|---|
| `src/evaluate.py:41` | `latencies.append(max(0.0, (a - s) / fps))` | Cắt latency âm về 0; định nghĩa mới giữ giá trị âm |
| `src/features.py:23-25` (`bbox_geometric_features`) | `vy = np.diff(cy)`, `ay = np.diff(vy)`, `speed` theo pixel/frame | Phụ thuộc kích thước khung; chia cho chiều cao người (xem `m1_features.track_signals`) |
| `src/models/model1_rule.py:14` | `vy_fall: float = 8.0` px/frame | Đổi sang chiều cao người/frame hoặc dùng `m1_rule.py` |
| `src/datasets.py:43` | `return [(0, max(1, n_frames - 1))]` | Coi cả chuỗi URFD là một sự kiện; định nghĩa mới dùng `fall_start_frame`, `fall_end_frame` trong `metadata.csv` |

## Câu hỏi cho TV1
1. Có thể đưa bản prompt mới nhất vào repo tại `docs/PROMPT_TV1_GiuaKy.md` không, để agent đối chiếu nguyên văn?
