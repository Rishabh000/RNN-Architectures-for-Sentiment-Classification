import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ensure_dir


plt.switch_backend("agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained runs and produce summary plots.")
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--metrics_csv", type=Path, help="Path to metrics.csv file. Defaults to <results_dir>/metrics.csv")
    parser.add_argument("--plots_dir", type=Path, help="Directory to store generated plots. Defaults to <results_dir>/plots")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top runs (by F1) to report.")
    parser.add_argument("--bottom_k", type=int, default=1, help="Number of worst runs (by F1) to report.")
    parser.add_argument("--hardware", type=str, default="CPU-only", help="Hardware description for reporting.")
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Run training first.")
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError("Metrics CSV is empty. Ensure training runs have been logged.")
    return df


def summarise_runs(df: pd.DataFrame, top_k: int, bottom_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(by="f1_macro", ascending=False)
    top_runs = df_sorted.head(top_k).copy()
    bottom_runs = df_sorted.tail(bottom_k).copy()
    return top_runs, bottom_runs


def plot_sequence_length_performance(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    agg = (
        df.groupby("sequence_length")
        .agg({"accuracy": "mean", "f1_macro": "mean"})
        .reset_index()
        .sort_values("sequence_length")
    )
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=agg, x="sequence_length", y="accuracy", marker="o", label="Accuracy")
    sns.lineplot(data=agg, x="sequence_length", y="f1_macro", marker="o", label="F1 (macro)")
    plt.title("Accuracy and F1 vs. Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_history(results_dir: Path, run_id: str) -> Dict[str, np.ndarray]:
    history_path = results_dir / "runs" / run_id / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found for run {run_id}. Expected {history_path}")
    with open(history_path, "r", encoding="utf-8") as fp:
        history = json.load(fp)
    return history


def plot_loss_curves(
    results_dir: Path,
    best_run_id: str,
    worst_run_id: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    best_history = load_history(results_dir, best_run_id)
    worst_history = load_history(results_dir, worst_run_id)

    epochs = list(range(1, len(best_history["train_loss"]) + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, best_history["train_loss"], label=f"Best Train ({best_run_id})", linestyle="-")
    plt.plot(epochs, best_history["val_loss"], label=f"Best Val ({best_run_id})", linestyle="--")

    epochs_worst = list(range(1, len(worst_history["train_loss"]) + 1))
    plt.plot(epochs_worst, worst_history["train_loss"], label=f"Worst Train ({worst_run_id})", linestyle="-", color="tab:orange")
    plt.plot(epochs_worst, worst_history["val_loss"], label=f"Worst Val ({worst_run_id})", linestyle="--", color="tab:orange")
    plt.title("Training vs Validation Loss (Best vs Worst)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()

    results_dir = args.results_dir
    metrics_path = args.metrics_csv or (results_dir / "metrics.csv")
    plots_dir = args.plots_dir or (results_dir / "plots")
    ensure_dir(plots_dir)

    metrics_df = load_metrics(metrics_path)
    top_runs, bottom_runs = summarise_runs(metrics_df, args.top_k, args.bottom_k)

    summary_table_path = results_dir / "metrics_summary.csv"
    top_runs.to_csv(summary_table_path, index=False)

    sequence_plot_path = plots_dir / "accuracy_f1_vs_sequence_length.png"
    plot_sequence_length_performance(metrics_df, sequence_plot_path)

    # Plot best vs worst loss curves using first entries from top/bottom sets.
    try:
        best_run_id = top_runs.iloc[0]["run_id"]
        worst_run_id = bottom_runs.iloc[-1]["run_id"]
        loss_plot_path = plots_dir / "loss_curves_best_vs_worst.png"
        plot_loss_curves(results_dir, best_run_id, worst_run_id, loss_plot_path)
    except (IndexError, FileNotFoundError) as err:
        print(f"Skipping loss curves plot: {err}")

    print("\n=== Top Runs (by F1) ===")
    print(top_runs[["run_id", "architecture", "activation", "optimizer", "sequence_length", "gradient_clipping", "accuracy", "f1_macro", "epoch_time_seconds"]])

    print("\n=== Bottom Runs (by F1) ===")
    print(bottom_runs[["run_id", "architecture", "activation", "optimizer", "sequence_length", "gradient_clipping", "accuracy", "f1_macro", "epoch_time_seconds"]])


if __name__ == "__main__":
    main()

