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
    add_community_features,
    detect_louvain_communities,
    summarize_communities,
)
from data_loader import load_dataset
from modeling import (
    evaluate_by_degree_group,
    evaluate_errors_by_community_purity,
    extract_graph_neighbor_features,
    prepare_train_test_split,
    run_repeated_experiments,
    run_repeated_threshold_tuning,
    train_logistic_regression_graph_features,
)
from visualization import plot_community_purity_error_rates


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


def _plot_degree_groups(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot held-out performance across ordered node-degree groups."""
    labels = ["0-2", "3-5", "6-10", "11-30", "31+"]
    x = np.arange(len(results_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        x - width / 2,
        results_df["balanced_accuracy"],
        width,
        color="#315C8D",
        label="Balanced Accuracy",
    )
    ax.bar(
        x + width / 2,
        results_df["f1"],
        width,
        color="#4F8A5B",
        label="F1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Held-out score")
    ax.set_title("Classification performance by node degree")
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

    # Diagnose the strongest handcrafted model on one prespecified split.
    # The split is not selected from the results, and all label-derived
    # neighborhood/community features use training labels only.
    train_nodes, test_nodes, y_train, y_test = prepare_train_test_split(
        labels_df,
        test_size=0.30,
        random_state=SEEDS[0],
    )
    nodes_for_features = list(train_nodes) + list(test_nodes)
    graph_features = extract_graph_neighbor_features(
        G,
        labels_dict,
        train_nodes,
        nodes_for_features,
    ).drop_duplicates(subset=["node_id"])
    community_features = add_community_features(
        G,
        labels_dict,
        train_nodes,
        nodes_for_features,
        partition,
        community_df,
    ).drop_duplicates(subset=["node_id"])
    combined_features = graph_features.merge(
        community_features,
        on="node_id",
        how="inner",
    ).set_index("node_id")
    X_train = combined_features.reindex(train_nodes).reset_index()
    X_test = combined_features.reindex(test_nodes).reset_index()
    _, y_pred, _ = train_logistic_regression_graph_features(
        X_train,
        y_train,
        X_test,
    )
    degree_group_df = evaluate_by_degree_group(
        G,
        test_nodes,
        y_test,
        y_pred,
    )
    purity_error_df = evaluate_errors_by_community_purity(
        test_nodes,
        y_test,
        y_pred,
        partition,
        community_df,
    )

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
    degree_group_df.to_csv(
        tables_dir / "classification_by_degree_group.csv", index=False
    )
    purity_error_df.to_csv(
        tables_dir / "classification_by_community_purity.csv", index=False
    )

    _plot_models(
        summary_df, figures_dir / "classification_comparison.png"
    )
    _plot_thresholds(
        threshold_summary_df, figures_dir / "threshold_tradeoff.png"
    )
    _plot_degree_groups(
        degree_group_df,
        figures_dir / "classification_by_degree_group.png",
    )
    plot_community_purity_error_rates(
        purity_error_df,
        figures_dir,
    )
    print(summary_df.to_string(index=False))
    print(threshold_summary_df.to_string(index=False))
    print(degree_group_df.to_string(index=False))
    print(purity_error_df.to_string(index=False))


if __name__ == "__main__":
    main()
