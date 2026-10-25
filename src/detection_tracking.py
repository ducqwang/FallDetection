"""Pipeline dùng chung: YOLO11 (phát hiện người) + ByteTrack (gán ID).

Dùng tracker tích hợp của Ultralytics (model.track(..., tracker="bytetrack.yaml")).
Xuất kết quả chuẩn MOT để tính IDF1 và cho các mô hình phân loại dùng lại.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float


class DetectorTracker:
    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.25, person_class: int = 0):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf = conf
        self.person_class = person_class

    def run(self, source, tracker: str = "bytetrack.yaml"):
        """Chạy detection + tracking trên 1 video, trả về list[Track].

        source: đường dẫn video (Le2i) hoặc list ảnh/thư mục (URFD).
        """
        tracks: list[Track] = []
        results = self.model.track(
            source=source, persist=True, tracker=tracker,
            classes=[self.person_class], conf=self.conf, stream=True, verbose=False,
        )
        for fi, r in enumerate(results):
            if r.boxes is None or r.boxes.id is None:
                continue
            for box, tid, cf in zip(r.boxes.xywh.cpu().numpy(),
                                    r.boxes.id.int().cpu().numpy(),
                                    r.boxes.conf.cpu().numpy()):
                x, y, w, h = box  # tâm + kích thước
                tracks.append(Track(fi, int(tid), float(x - w / 2), float(y - h / 2),
                                    float(w), float(h), float(cf)))
        return tracks


def write_mot(tracks: list[Track], out_path: str | Path) -> None:
    """Ghi file chuẩn MOT: frame,id,x,y,w,h,conf,-1,-1,-1 (phục vụ TrackEval)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tracks:
            f.write(f"{t.frame + 1},{t.track_id},{t.x:.2f},{t.y:.2f},"
                    f"{t.w:.2f},{t.h:.2f},{t.conf:.3f},-1,-1,-1\n")
