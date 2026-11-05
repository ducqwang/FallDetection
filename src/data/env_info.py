"""Ô đầu tiên của mỗi notebook: cố định seed, in version thư viện và version dataset.

Đầu vào: seed và configs/data.yaml. Đầu ra: in ra màn hình.
Ví dụ: from src.data.env_info import setup_notebook; setup_notebook("configs/data.yaml").
"""
from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path

import yaml

from ..utils import set_seed

PACKAGES = ("torch", "ultralytics")


def _package_version(name: str) -> str:
    """Version đã cài của 1 gói, hoặc 'chưa cài'."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "chưa cài"


def _opencv_version() -> str:
    """Version OpenCV đang import được; Kaggle có thể dùng bản contrib hoặc headless."""
    try:
        import cv2
    except ImportError:
        return "chưa cài"
    return cv2.__version__


def setup_notebook(config_path: str | Path, seed: int = 42) -> None:
    """Cố định seed (random, numpy, torch) rồi in version môi trường và dataset."""
    set_seed(seed)
    print(f"seed = {seed} | Python {platform.python_version()}")
    for package in PACKAGES:
        print(f"{package}: {_package_version(package)}")
    print(f"opencv: {_opencv_version()}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for name, source in config["dataset_sources"].items():
        print(f"dataset {name}: {source['url']} | version: {source.get('version') or 'chưa điền'}")
