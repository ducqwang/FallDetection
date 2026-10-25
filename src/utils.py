"""Tiện ích chung: seed tái lập, nạp config, I/O cơ bản."""
from __future__ import annotations
import os
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả tái lập được (yêu cầu nộp bài của thầy)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str | Path) -> dict:
    """Nạp file YAML cấu hình."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def on_kaggle() -> bool:
    return os.path.exists("/kaggle/input")
