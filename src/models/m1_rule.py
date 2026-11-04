"""Việc 5: M1 rule-based. Không huấn luyện, chỉ tổ hợp các điều kiện hình học.

Đầu vào: tracks.parquet (hoặc le2i_gt_bbox.csv để lấy kết quả cận trên), metadata.csv, configs/m1.yaml.
Đầu ra: predictions.csv (video_id, frame_idx, track_id, fall_score).
fall_score ∈ [0, 1] là tổng có trọng số của: vận tốc dọc, gia tốc dọc, đổi tư thế, bất động ngắn.
Ví dụ: python -m src.models.m1_rule --tracks tracks.parquet --meta metadata.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.common import METADATA_DTYPES, PREDICTION_COLUMNS, TRACK_COLUMNS
from .m1_features import seconds_to_frames, track_signals

GT_TRACK_ID = 1
GT_DET_CONF = 1.0
WEIGHT_NAMES = ("speed", "accel", "posture", "still")
WEIGHT_SUM_TOLERANCE = 1e-6


def load_config(path: str | Path, tuned_path: str | Path | None = None) -> dict:
    """Đọc configs/m1.yaml; tuned_path (kết quả tune_m1) ghi đè rules và alarm."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if tuned_path:
        tuned = yaml.safe_load(Path(tuned_path).read_text(encoding="utf-8"))
        config["rules"].update(tuned["rules"])
        config["alarm"].update(tuned["alarm"])
    if abs(sum(config["weights"][name] for name in WEIGHT_NAMES) - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Tổng trọng số phải bằng 1: {config['weights']}")
    return config


def ramp(values: pd.Series, low: float, high: float) -> pd.Series:
    """Đưa giá trị về [0, 1]: dưới low là 0, từ high trở lên là 1, ở giữa tuyến tính.

    Điểm liên tục thay vì đúng/sai để vẽ được đường cong PR/ROC.
    """
    return ((values - low) / (high - low)).clip(0.0, 1.0)


def score_signals(signals: pd.DataFrame, fps: float, rules: dict, weights: dict) -> pd.Series:
    """Tính fall_score theo frame cho 1 track; mọi điều kiện chỉ nhìn về quá khứ.

    Index là frame_idx 0-based; fps: frame/giây; mọi tham số *_s trong rules tính bằng giây.

    speed: vận tốc dọc lớn nhất trong window_s giây gần nhất.
    accel: gia tốc dọc lớn nhất trong cùng khoảng.
    posture: đang nằm VÀ trong window_s giây gần nhất đã từng đứng.
    still: đang nằm VÀ tỷ lệ frame ít chuyển động trong immobile_s giây gần nhất.
    """
    window = seconds_to_frames(rules["window_s"], fps) + 1
    speed_score = ramp(signals["vy"].rolling(window, min_periods=1).max(), rules["vy_min"], rules["vy_full"])
    accel_score = ramp(signals["ay"].rolling(window, min_periods=1).max(), rules["ay_min"], rules["ay_full"])
    lying = ramp(signals["aspect"], rules["ar_upright"], rules["ar_lying"])
    upright_before = 1.0 - ramp(signals["aspect"].rolling(window, min_periods=1).min(),
                                rules["ar_upright"], rules["ar_lying"])
    posture_score = lying * upright_before
    calm = (signals["speed"] < rules["still_speed"]).astype(float)
    still_score = lying * calm.rolling(seconds_to_frames(rules["immobile_s"], fps), min_periods=1).mean()
    return (weights["speed"] * speed_score + weights["accel"] * accel_score
            + weights["posture"] * posture_score + weights["still"] * still_score)


def gt_boxes_as_tracks(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    """Đổi Le2i GT bbox (frame_idx 0-based, pixel) thành tracks đúng schema: mỗi video 1 track, det_conf = 1."""
    return gt_boxes.assign(track_id=GT_TRACK_ID, det_conf=GT_DET_CONF)[TRACK_COLUMNS]


def signals_by_track(tracks: pd.DataFrame, meta: pd.DataFrame, config: dict) -> list[dict]:
    """Tính sẵn tín hiệu từng track để tune nhanh.

    Trả về list dict: video_id, track_id, fps, signals. video_id lạ thì raise.
    """
    fps_of = meta.set_index("video_id")["fps"].to_dict()
    unknown = sorted(set(tracks["video_id"]) - set(fps_of))
    if unknown:
        raise ValueError(f"tracks có video_id không có trong metadata: {unknown[:5]}")
    items = []
    params = config["signals"]
    for (video_id, track_id), track in tracks.groupby(["video_id", "track_id"], sort=True):
        fps = float(fps_of[video_id])
        items.append({"video_id": video_id, "track_id": int(track_id), "fps": fps,
                      "signals": track_signals(track, fps, params["smooth_s"], params["reference_s"])})
    return items


def predictions_from_signals(items: list[dict], rules: dict, weights: dict) -> pd.DataFrame:
    """Chấm điểm các track đã tính tín hiệu, trả về bảng predictions.csv (frame_idx 0-based, fall_score ∈ [0, 1])."""
    tables = []
    for item in items:
        signals = item["signals"]
        too_short = len(signals) < seconds_to_frames(rules["min_track_s"], item["fps"])
        score = pd.Series(0.0, index=signals.index) if too_short else score_signals(signals, item["fps"], rules, weights)
        tables.append(pd.DataFrame({"video_id": item["video_id"], "frame_idx": signals.index,
                                    "track_id": item["track_id"], "fall_score": np.clip(score.to_numpy(), 0.0, 1.0)}))
    if not tables:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(tables, ignore_index=True)[PREDICTION_COLUMNS]


def predict(tracks: pd.DataFrame, meta: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Chạy M1 trên toàn bộ tracks (frame_idx 0-based). Trả về bảng predictions."""
    return predictions_from_signals(signals_by_track(tracks, meta, config), config["rules"], config["weights"])


def main() -> None:
    """Đọc tracks hoặc GT bbox, chấm điểm và ghi predictions.csv."""
    parser = argparse.ArgumentParser(description="Chạy M1 rule-based, xuất predictions.csv.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracks", help="tracks.parquet theo schema")
    source.add_argument("--gt-bbox", help="le2i_gt_bbox.csv: chạy trên GT để lấy cận trên")
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--out", required=True, help="predictions.csv sẽ ghi")
    parser.add_argument("--config", default="configs/m1.yaml", help="file cấu hình M1")
    parser.add_argument("--tuned", help="yaml kết quả của scripts/tune_m1.py")
    args = parser.parse_args()
    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES)
    tracks = gt_boxes_as_tracks(pd.read_csv(args.gt_bbox)) if args.gt_bbox else pd.read_parquet(args.tracks)
    predictions = predict(tracks, meta, load_config(args.config, args.tuned))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    print(f"TÓM TẮT | video có track: {predictions['video_id'].nunique()} | track: {tracks[['video_id', 'track_id']].drop_duplicates().shape[0]} "
          f"| dòng: {len(predictions)} | bỏ qua: 0 | Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
