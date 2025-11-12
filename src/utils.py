import csv
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_METRIC_FIELDS = [
    "timestamp",
    "run_id",
    "architecture",
    "activation",
    "optimizer",
    "sequence_length",
    "gradient_clipping",
    "accuracy",
    "f1_macro",
    "loss",
    "epoch_time_seconds",
    "num_epochs",
    "best_epoch",
    "notes",
]


def ensure_dir(path: os.PathLike) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def set_global_seed(seed: int = 42) -> None:
    """Fix Python, NumPy, and Torch RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device when available (and allowed), otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_logging(run_id: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a console-only logger."""
    logger = logging.getLogger(run_id)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs in notebooks / reruns.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger


class EpochTimer:
    """Context manager to measure elapsed time for training epochs."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "EpochTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time


def compute_metrics(
    y_true: Iterable[int],
    y_pred_probs: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Return accuracy and macro F1 for predictions."""
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics


def format_run_id(config: Dict[str, str]) -> str:
    """Generate a deterministic identifier from experiment configuration."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    parts = [
        config.get("architecture", "arch"),
        config.get("activation", "act"),
        config.get("optimizer", "opt"),
        f"seq{config.get('sequence_length', 'NA')}",
        "clip" if config.get("gradient_clipping", False) else "noclip",
    ]
    suffix = config.get("suffix")
    if suffix:
        parts.append(str(suffix))
    parts.append(timestamp)
    return "_".join(parts)


def save_config(run_dir: os.PathLike, config: Dict) -> None:
    """Persist run configuration as JSON."""
    ensure_dir(run_dir)
    config_path = Path(run_dir) / "config.json"
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)


def append_metrics_row(
    csv_path: os.PathLike,
    row: Dict[str, Optional[float]],
    fieldnames: Optional[Iterable[str]] = None,
) -> None:
    """Append experiment metrics to CSV, creating header when file empty."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    if fieldnames is None:
        fieldnames = DEFAULT_METRIC_FIELDS
    else:
        fieldnames = list(fieldnames)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def save_history(run_dir: os.PathLike, history: Dict[str, Iterable[float]]) -> None:
    """Persist training history as JSON for downstream plotting."""
    ensure_dir(run_dir)
    history_path = Path(run_dir) / "history.json"
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


@contextmanager
def temporary_logging_level(logger: logging.Logger, level: int):
    """Context manager to temporarily change logging level."""
    original_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)

