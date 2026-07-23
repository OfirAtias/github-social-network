# -*- coding: utf-8 -*-
"""Repeated leakage-safe node classification with course-covered features."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from community_analysis import (
    detect_louvain_communities,
    summarize_communities,
)
from data_loader import load_dataset
from modeling import (
    run_repeated_experiments,
    run_repeated_threshold_tuning,
)


SEEDS = (42, 7, 21, 84, 123)
BASE_CLASSIFICATION_MODELS = (
    "Majority Baseline",
    "Neighbor Majority",
    "Logistic Regression - Graph Features Only",
    "Logistic Regression - Graph + Community Features",
)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable mean/std row per model."""
    summary = results_df.groupby("model", as_index=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "model" if column == ("model", "") else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary.sort_values("roc_auc_mean").reset_index(drop=True)


def summarize_thresholds(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation-only threshold selection across seeds."""
    metrics = [
        "selected_threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
        "specificity",
    ]
    summary = results_df.groupby("evaluation_type", as_index=False)[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "evaluation_type"
        if column == ("evaluation_type", "")
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary


def _plot_models(summary_df: pd.DataFrame, output_path: Path) -> None:
    order = list(BASE_CLASSIFICATION_MODELS)
    plot_df = summary_df.set_index("model").reindex(order).reset_index()
    labels = [
        "Majority",
        "Neighbor\nMajority",
        "Graph\nFeatures",
        "Graph +\nCommunity",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, metric, title in (
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ):
        means = plot_df[f"{metric}_mean"]
        stds = plot_df[f"{metric}_std"]
        bars = ax.bar(
            labels,
            means,
            yerr=stds,
            capsize=4,
            color=["#9A9A9A", "#6B86A8", "#315C8D", "#4F8A5B"],
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylim(0.35, 1.02)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Leakage-safe node classification\n"
        "Mean and standard deviation across five stratified splits"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_thresholds(summary_df: pd.DataFrame, output_path: Path) -> None:
    order = ["default_threshold_0.5", "tuned_threshold_validation_f1"]
    plot_df = summary_df.set_index("evaluation_type").reindex(order)
    metrics = ["precision_mean", "recall_mean", "f1_mean"]
    labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for offset, policy, color, name in (
        (-width / 2, order[0], "#7F8C8D", "Threshold 0.5"),
        (width / 2, order[1], "#315C8D", "Validation-tuned threshold"),
    ):
        ax.bar(
            x + offset,
            [plot_df.loc[policy, metric] for metric in metrics],
            width,
            color=color,
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Mean test score")
    ax.set_title("Threshold selection uses validation data only")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    G, _, labels_df, labels_dict = load_dataset(data_dir)
    results_df = run_repeated_experiments(
        G,
        labels_df,
        n_runs=len(SEEDS),
        test_size=0.30,
        random_states=SEEDS,
    )
    unexpected = set(results_df["model"]) - set(BASE_CLASSIFICATION_MODELS)
    missing = set(BASE_CLASSIFICATION_MODELS) - set(results_df["model"])
    if unexpected or missing:
        raise AssertionError(
            f"Model mismatch: unexpected={sorted(unexpected)}, missing={sorted(missing)}"
        )
    summary_df = summarize_results(results_df)

    partition, _ = detect_louvain_communities(G, random_state=42)
    community_df = summarize_communities(G, labels_dict, partition)
    threshold_df = run_repeated_threshold_tuning(
        G,
        labels_df,
        partition,
        community_df,
        random_states=SEEDS,
        test_size=0.30,
        validation_size=0.20,
        threshold_metric="f1",
    )
    threshold_summary_df = summarize_thresholds(threshold_df)

    results_df.to_csv(
        tables_dir / "classification_results.csv", index=False
    )
    summary_df.to_csv(
        tables_dir / "classification_summary.csv", index=False
    )
    threshold_df.to_csv(
        tables_dir / "threshold_tuning_results.csv", index=False
    )
    threshold_summary_df.to_csv(
        tables_dir / "threshold_tuning_summary.csv", index=False
    )

    _plot_models(
        summary_df, figures_dir / "classification_comparison.png"
    )
    _plot_thresholds(
        threshold_summary_df, figures_dir / "threshold_tradeoff.png"
    )
    print(summary_df.to_string(index=False))
    print(threshold_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
