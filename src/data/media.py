"""Làm việc với video bằng ffmpeg/ffprobe, luôn chạy trong tiến trình riêng.

Không dùng cv2.VideoCapture vì đọc .avi của Le2i trên Kaggle làm chết kernel (mục 2.2).
Đầu vào: ảnh PNG hoặc .avi/.mp4. Đầu ra: mp4 H.264 hoặc thông số video.
Ví dụ: avi_to_mp4(Path("video (1).avi"), Path("out.mp4")).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

X264_ARGS = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"]


def require_ffmpeg() -> None:
    """Báo lỗi ngay nếu máy không có ffmpeg hoặc ffprobe."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FileNotFoundError(f"Thiếu công cụ trong PATH: {missing}")


def run_command(cmd: list[str], context: str) -> str:
    """Chạy lệnh ngoài, trả stdout; lỗi thì raise kèm tên video và stderr."""
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{context}: lệnh lỗi {cmd[0]} (mã {done.returncode}): {done.stderr[-500:]}")
    return done.stdout


def _parse_rate(text: str) -> float:
    """Đổi chuỗi '25/1' thành 25.0; '0/0' thành 0.0."""
    num, _, den = text.partition("/")
    if not den or int(den) == 0:
        return float(num) if not den else 0.0
    return float(Fraction(int(num), int(den)))


def probe_video(path: Path) -> dict:
    """Đọc thông số luồng video đầu tiên, không giải mã.

    Trả về dict: fps (frame/giây), width, height (pixel).
    """
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=avg_frame_rate,r_frame_rate,width,height", "-of", "json", str(path)]
    streams = json.loads(run_command(cmd, str(path))).get("streams", [])
    if not streams:
        raise RuntimeError(f"{path}: không có luồng video")
    stream = streams[0]
    fps = _parse_rate(stream["avg_frame_rate"]) or _parse_rate(stream["r_frame_rate"])
    return {"fps": round(fps, 3), "width": int(stream["width"]), "height": int(stream["height"])}


def count_frames(path: Path) -> int:
    """Đếm số frame bằng cách giải mã thật (ffprobe -count_frames)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)]
    text = run_command(cmd, str(path)).strip().rstrip(",")
    if not text.isdigit():
        raise RuntimeError(f"{path}: không đếm được frame, ffprobe trả về {text!r}")
    return int(text)


def png_sequence_to_mp4(frames: list[Path], out_path: Path, fps: int) -> None:
    """Ghép PNG theo đúng thứ tự danh sách `frames` thành mp4; fps là frame/giây.

    Tạo symlink đánh số liên tục trong thư mục tạm để ffmpeg đọc đúng thứ tự,
    không phụ thuộc cách đệm số 0 trong tên file gốc.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for order, frame in enumerate(frames, start=1):
            (Path(tmp) / f"{order:06d}.png").symlink_to(frame.resolve())
        cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-start_number", "1",
               "-i", str(Path(tmp) / "%06d.png"), *X264_ARGS, str(out_path)]
        run_command(cmd, str(out_path))


def avi_to_mp4(src: Path, out_path: Path) -> None:
    """Chuyển .avi sang mp4, bỏ audio; không ép -r và không resize."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", *X264_ARGS, str(out_path)]
    run_command(cmd, str(src))
