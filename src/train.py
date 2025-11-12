import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from .models import build_model
from .preprocess import get_dataloaders
from .utils import (
    EpochTimer,
    append_metrics_row,
    compute_metrics,
    configure_logging,
    ensure_dir,
    format_run_id,
    get_device,
    save_config,
    save_history,
    set_global_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"

VALID_ARCHITECTURES = {"rnn", "lstm", "bilstm"}
VALID_ACTIVATIONS = {"relu", "tanh", "sigmoid"}
VALID_OPTIMIZERS = {"adam", "sgd", "rmsprop"}
VALID_SEQUENCE_LENGTHS = {25, 50, 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recurrent models for IMDB sentiment classification using JSON configuration files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration JSON. Defaults to configs/default.json.",
    )
    parser.add_argument(
        "--prefer_gpu",
        action="store_true",
        help="Attempt to use GPU regardless of config setting when available.",
    )
    return parser.parse_args()


def build_optimizer(name: str, parameters, lr: float) -> Optimizer:
    if name == "adam":
        return optim.Adam(parameters, lr=lr)
    if name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optimizer = None,
    clip_value: float = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        probs = model(inputs)
        loss = loss_fn(probs, labels)

        if is_train:
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    mean_loss = total_loss / len(dataloader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(mean_loss)
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs = model(inputs)
            loss = loss_fn(probs, labels)
            total_loss += loss.item() * inputs.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    mean_loss = total_loss / len(dataloader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(mean_loss)
    return metrics


def resolve_path(path_value: Any, default_relative: str) -> Path:
    if path_value is None:
        return PROJECT_ROOT / default_relative
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_run_configs(config_path: Path) -> List[Dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as fp:
        config_data = json.load(fp)

    defaults = config_data.get("defaults", {})
    experiments = config_data.get("experiments")

    if experiments:
        configs = []
        for exp in experiments:
            merged = deepcopy(defaults)
            merged.update(exp)
            configs.append(merged)
    else:
        if not defaults:
            raise ValueError("Configuration file must contain 'defaults' or 'experiments'.")
        configs = [defaults]

    validated = []
    for cfg in configs:
        validate_run_config(cfg)
        validated.append(cfg)
    return validated


def validate_run_config(config: Dict[str, Any]) -> None:
    missing = {"architecture", "activation", "optimizer", "sequence_length"} - set(config.keys())
    if missing:
        raise ValueError(f"Run configuration missing required keys: {missing}")

    if config["architecture"] not in VALID_ARCHITECTURES:
        raise ValueError(f"Invalid architecture '{config['architecture']}'. Expected {VALID_ARCHITECTURES}.")
    if config["activation"] not in VALID_ACTIVATIONS:
        raise ValueError(f"Invalid activation '{config['activation']}'. Expected {VALID_ACTIVATIONS}.")
    if config["optimizer"] not in VALID_OPTIMIZERS:
        raise ValueError(f"Invalid optimizer '{config['optimizer']}'. Expected {VALID_OPTIMIZERS}.")
    if config["sequence_length"] not in VALID_SEQUENCE_LENGTHS:
        raise ValueError(
            f"Invalid sequence length '{config['sequence_length']}'. Expected {sorted(VALID_SEQUENCE_LENGTHS)}."
        )


def normalise_config_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(config)
    config["data_dir"] = str(resolve_path(config.get("data_dir"), "data"))
    config["cache_dir"] = str(resolve_path(config.get("cache_dir"), "data/processed"))
    config["results_dir"] = str(resolve_path(config.get("results_dir"), "results"))
    return config


def train_single_run(run_config: Dict[str, Any], prefer_gpu: bool = False) -> None:
    run_config = normalise_config_paths(run_config)

    seed = int(run_config.get("seed", 42))
    set_global_seed(seed)

    prefer_gpu_flag = prefer_gpu or run_config.get("prefer_gpu", False)
    device = get_device(prefer_gpu_flag)

    data_dir = Path(run_config["data_dir"])
    cache_dir = Path(run_config["cache_dir"])
    results_dir = Path(run_config["results_dir"])
    runs_dir = results_dir / "runs"

    ensure_dir(results_dir)
    ensure_dir(runs_dir)

    run_id = format_run_id(
        {
            "architecture": run_config["architecture"],
            "activation": run_config["activation"],
            "optimizer": run_config["optimizer"],
            "sequence_length": run_config["sequence_length"],
            "gradient_clipping": run_config.get("gradient_clipping", False),
            "suffix": run_config.get("name") or run_config.get("run_name"),
        }
    )
    run_output_dir = runs_dir / run_id
    ensure_dir(run_output_dir)

    logger = configure_logging(run_id)
    logger.info("Starting run %s on device %s", run_id, device)
    if run_config.get("name"):
        logger.info("Experiment tag: %s", run_config["name"])

    merged_config = deepcopy(run_config)
    merged_config.update({"run_id": run_id, "device": str(device)})
    save_config(run_output_dir, merged_config)

    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        seq_length=run_config["sequence_length"],
        batch_size=int(run_config.get("batch_size", 32)),
        data_dir=data_dir,
        cache_dir=cache_dir,
        num_workers=int(run_config.get("num_workers", 0)),
        pin_memory=bool(run_config.get("pin_memory", False)),
        seed=seed,
    )

    model = build_model(
        architecture=run_config["architecture"],
        activation=run_config["activation"],
        vocab_size=vocab_size,
        embedding_dim=int(run_config.get("embedding_dim", 100)),
        hidden_size=int(run_config.get("hidden_size", 64)),
        dropout=float(run_config.get("dropout", 0.4)),
        num_layers=int(run_config.get("num_layers", 2)),
    )
    model.to(device)
    logger.info(
        "Model initialised | architecture=%s activation=%s vocab=%d seq_length=%d",
        run_config["architecture"],
        run_config["activation"],
        vocab_size,
        run_config["sequence_length"],
    )

    optimizer = build_optimizer(run_config["optimizer"], model.parameters(), lr=float(run_config.get("learning_rate", 1e-3)))
    loss_fn = nn.BCELoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "epoch_time": [],
    }

    best_val_f1 = -math.inf
    best_epoch = -1
    best_state = None
    clip_value = float(run_config.get("clip_value", 1.0)) if run_config.get("gradient_clipping", False) else None

    num_epochs = int(run_config.get("num_epochs", 10))

    for epoch in range(1, num_epochs + 1):
        with EpochTimer() as timer:
            train_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                clip_value=clip_value,
            )
        val_metrics = evaluate_model(model, val_loader, device, loss_fn)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1_macro"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["epoch_time"].append(timer.elapsed if timer.elapsed is not None else 0.0)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f train_f1=%.4f val_f1=%.4f epoch_time=%.2fs",
            epoch,
            num_epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            train_metrics["f1_macro"],
            val_metrics["f1_macro"],
            history["epoch_time"][-1],
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = model.state_dict()
            if run_config.get("save_checkpoints"):
                torch.save(best_state, run_output_dir / "best_model.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device, loss_fn)
    logger.info(
        "Test metrics | loss=%.4f accuracy=%.4f f1=%.4f",
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
    )

    save_history(run_output_dir, history)
    metrics_row = {
        "timestamp": run_id.split("_")[-1],
        "run_id": run_id,
        "architecture": run_config["architecture"],
        "activation": run_config["activation"],
        "optimizer": run_config["optimizer"],
        "sequence_length": run_config["sequence_length"],
        "gradient_clipping": int(bool(run_config.get("gradient_clipping", False))),
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "loss": test_metrics["loss"],
        "epoch_time_seconds": float(np.mean(history["epoch_time"])),
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "notes": run_config.get("notes") or run_config.get("name") or "",
    }
    append_metrics_row(results_dir / "metrics.csv", metrics_row)

    if run_config.get("save_checkpoints") and best_state is not None:
        torch.save(best_state, run_output_dir / "best_model.pt")

    logger.info("Run complete. Metrics saved to %s", results_dir / "metrics.csv")


def main():
    args = parse_args()
    config_path = args.config or DEFAULT_CONFIG_PATH
    run_configs = load_run_configs(config_path)

    for idx, run_cfg in enumerate(run_configs, start=1):
        print(f"[Train] Starting run {idx}/{len(run_configs)}")
        train_single_run(run_cfg, prefer_gpu=args.prefer_gpu)


if __name__ == "__main__":
    main()

