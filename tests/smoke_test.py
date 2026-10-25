"""Smoke test các thành phần KHÔNG cần torch/cv2/ultralytics.

Chạy: python tests/smoke_test.py  (từ gốc repo)
Kiểm tra: seed, chia tập theo video (chống rò rỉ), đặc trưng hình học, rule-based,
độ đo mức sự kiện, gộp dataset, bảng kết quả, parse annotation Le2i, PCA/t-SNE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils import set_seed
from src.data_splits import VideoItem, split_by_group, parse_le2i_annotation
from src.detection_tracking import Track
from src.features import bbox_geometric_features, reduce_and_plot
from src.datasets import make_gt_falls, geometric_windows, _tracks_by_id
from src.models.model1_rule import RuleThresholds, predict_sequence
from src.evaluate import merge_alarms, evaluate_events, aggregate_over_dataset, results_table

PASS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name}")


def synth_fall_track(tid=1, n=40, fall_at=20):
    """Chuỗi Track mô phỏng: đứng (h>w) rồi ngã (w>h) + tâm rơi nhanh."""
    seq = []
    for f in range(n):
        if f < fall_at:
            x, y, w, h = 100, 100, 40, 100           # đứng
        else:
            drop = min(f - fall_at, 5) * 12
            x, y, w, h = 100, 100 + drop, 100, 40     # nằm, tâm rơi
        seq.append(Track(f, tid, x, y, w, h, 0.9))
    return seq


def main():
    set_seed(42)

    # 1) chia tập theo video — không rò rỉ giữa 2 camera cùng chuỗi
    items = []
    for i in range(1, 31):
        base = f"fall-{i:02d}"
        items.append(VideoItem(vid=f"{base}-cam0", dataset="urfd", label=1, path="x", subject=base))
        items.append(VideoItem(vid=f"{base}-cam1", dataset="urfd", label=1, path="x", subject=base))
    sp = split_by_group(items, (0.7, 0.15, 0.15), key="subject", seed=42)
    allv = sp["train"] + sp["val"] + sp["test"]
    check("split: phủ hết video", len(allv) == len(items))
    check("split: không trùng giữa các tập", len(set(allv)) == len(allv))
    # hai camera của cùng subject phải cùng một tập (chống rò rỉ)
    def tset(v):
        return next(s for s in ("train", "val", "test") if v in sp[s])
    leak = any(tset(f"fall-{i:02d}-cam0") != tset(f"fall-{i:02d}-cam1") for i in range(1, 31))
    check("split: cam0 & cam1 cùng tập (chống rò rỉ)", not leak)

    # 2) đặc trưng hình học
    feat = bbox_geometric_features(synth_fall_track())
    check("geometric features: shape (5,)", feat.shape == (5,))
    check("geometric features: aspect ratio tăng khi ngã", feat[0] > 1.0)

    # 3) rule-based + merge + event metric
    seq = synth_fall_track(n=40, fall_at=20)
    flags = predict_sequence(seq, RuleThresholds())
    alarms = merge_alarms(flags)
    check("rule-based: có phát cảnh báo", len(alarms) >= 1)
    gt = [(20, 25)]
    m = evaluate_events(alarms, gt, video_seconds=40 / 30, fps=30, tolerance_sec=1.0)
    check("event metric: TP=1", m.tp == 1)
    check("event metric: recall=1.0", abs(m.recall - 1.0) < 1e-6)

    # không ngã -> không TP
    adl = [Track(f, 1, 100, 100, 40, 100, 0.9) for f in range(40)]
    m0 = evaluate_events(merge_alarms(predict_sequence(adl, RuleThresholds())), [], 40 / 30, 30)
    check("event metric: ADL không FN/TP", m0.tp == 0 and m0.fn == 0)

    # 4) geometric_windows
    X, y = geometric_windows(seq, gt, seq_len=16, stride=8)
    check("geometric_windows: X,y khớp", len(X) == len(y) and X.shape[1] == 5)
    check("geometric_windows: có nhãn dương", int(y.sum()) >= 1)

    # 5) make_gt_falls (dict item kiểu splits['_items'])
    g_le2i = make_gt_falls({"dataset": "le2i", "label": 1, "fall_start": 10, "fall_end": 20}, 100)
    g_urfd = make_gt_falls({"dataset": "urfd", "label": 1, "fall_start": None, "fall_end": None}, 50)
    check("make_gt_falls: Le2i dùng mốc annotation", g_le2i == [(10, 20)])
    check("make_gt_falls: URFD coi cả chuỗi là 1 sự kiện", g_urfd == [(0, 49)])

    # 6) aggregate + results_table
    agg = aggregate_over_dataset([m, m0], video_seconds_total=80 / 30)
    check("aggregate: TP cộng dồn", agg.tp == 1)
    df = results_table([{"model": "M1", "protocol": "within_urfd", "f1": agg.f1,
                         "precision": agg.precision, "recall": agg.recall}])
    check("results_table: tạo DataFrame", df.shape[0] == 1)

    # 7) parse annotation Le2i (định dạng: 2 dòng mốc + bbox)
    ann = Path("tests/_tmp_ann.txt")
    ann.write_text("15\n40\n1,0,10,20,60,120\n2,0,12,22,62,122\n")
    fs, fe, bb = parse_le2i_annotation(ann)
    ann.unlink()
    check("parse Le2i: fall_start/end đúng", fs == 15 and fe == 40)
    check("parse Le2i: đọc được bbox", len(bb) == 2)

    # 8) PCA/t-SNE (sklearn + matplotlib)
    rng = np.random.default_rng(0)
    Xf = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(3, 1, (30, 8))])
    yf = np.array([0] * 30 + [1] * 30)
    out = reduce_and_plot(Xf, yf, method="pca", out_png="outputs/_smoke_pca.png")
    check("reduce_and_plot: xuất ảnh PCA", Path(out).exists())

    print(f"\n{sum(PASS)}/{len(PASS)} kiểm tra PASS")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
