"""
Ablation experiment: how much is the network worth?

Compares three feature families under the exact same protocol used in
run_repeated_experiments (same seeds, same stratified 70/30 splits,
same metrics):

1. Content Only   - MUSAE node features (starred repos / employer / location),
                    no network information at all.
2. Network Only   - the existing Graph + Community feature model.
3. Hybrid         - network features and content features combined.

Outputs:
    outputs/tables/ablation_model_results.csv
    outputs/tables/ablation_model_results_summary.csv
    outputs/figures/ablation_f1_auc_comparison.png
"""
from pathlib import Path
from typing import Sequence

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
from content_features import (
    build_content_matrix,
    load_content_features,
    train_logistic_regression_content_only,
    train_logistic_regression_hybrid,
)
from data_loader import load_dataset
from modeling import (
    _merge_node_feature_frames,
    evaluate_predictions,
    extract_graph_neighbor_features,
    prepare_train_test_split,
    train_logistic_regression_graph_features,
)


MODEL_CONTENT_ONLY = "Logistic Regression - Content Features Only"
MODEL_NETWORK_ONLY = "Logistic Regression - Network Only (Graph + Community)"
MODEL_HYBRID = "Logistic Regression - Hybrid (Network + Content)"


def run_ablation_experiments(
    data_dir: Path,
    test_size: float = 0.3,
    random_states: Sequence[int] = (42, 7, 21, 84, 123),
) -> pd.DataFrame:
    """Run the three ablation models across repeated stratified splits."""
    G, _, labels_df, labels_dict = load_dataset(data_dir)

    features_dict, n_features = load_content_features(data_dir / "musae_git_features.json")
    print(f"Loaded content features: {len(features_dict)} nodes, {n_features} feature dims")

    # Louvain depends only on graph structure, so we compute it once (same as modeling).
    partition, modularity = detect_louvain_communities(G, random_state=42)
    community_summary_df = summarize_communities(G, labels_dict, partition)
    print(f"Louvain: {len(set(partition.values()))} communities, modularity={modularity:.4f}")

    results_rows = []

    for run_idx, random_state in enumerate(random_states, start=1):
        print(f"[run {run_idx}/{len(random_states)}] random_state={random_state}")

        train_nodes, test_nodes, y_train, y_test = prepare_train_test_split(
            labels_df=labels_df,
            test_size=test_size,
            random_state=int(random_state),
        )
        nodes_for_features = list(train_nodes) + list(test_nodes)
        train_labels_dict = {
            int(node): int(labels_dict[int(node)]) for node in train_nodes
        }

        # --- Content Only ---
        X_train_content = build_content_matrix(features_dict, n_features, train_nodes)
        X_test_content = build_content_matrix(features_dict, n_features, test_nodes)

        _, y_pred_content, y_score_content = train_logistic_regression_content_only(
            X_train=X_train_content,
            y_train=y_train,
            X_test=X_test_content,
        )
        metrics_content = evaluate_predictions(y_test, y_pred_content, y_score_content)

        # --- Network Only (Graph + Community, identical to the existing model) ---
        graph_features_df = extract_graph_neighbor_features(
            G=G,
            labels_dict=train_labels_dict,
            train_nodes=train_nodes,
            nodes=nodes_for_features,
        ).drop_duplicates(subset=["node_id"])

        community_features_df = add_community_features(
            G=G,
            labels_dict=train_labels_dict,
            train_nodes=train_nodes,
            nodes=nodes_for_features,
            partition=partition,
        ).drop_duplicates(subset=["node_id"])

        combined_features_df = _merge_node_feature_frames(graph_features_df, community_features_df)
        combined_features_indexed = combined_features_df.set_index("node_id")

        X_train_network = combined_features_indexed.reindex(train_nodes).reset_index()
        X_test_network = combined_features_indexed.reindex(test_nodes).reset_index()

        _, y_pred_network, y_score_network = train_logistic_regression_graph_features(
            X_train=X_train_network,
            y_train=y_train,
            X_test=X_test_network,
        )
        metrics_network = evaluate_predictions(y_test, y_pred_network, y_score_network)

        # --- Hybrid (Network + Content) ---
        _, y_pred_hybrid, y_score_hybrid = train_logistic_regression_hybrid(
            X_train_network=X_train_network,
            X_train_content=X_train_content,
            y_train=y_train,
            X_test_network=X_test_network,
            X_test_content=X_test_content,
        )
        metrics_hybrid = evaluate_predictions(y_test, y_pred_hybrid, y_score_hybrid)

        for model_name, metrics in (
            (MODEL_CONTENT_ONLY, metrics_content),
            (MODEL_NETWORK_ONLY, metrics_network),
            (MODEL_HYBRID, metrics_hybrid),
        ):
            results_rows.append(
                {
                    "run": run_idx,
                    "random_state": int(random_state),
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
            print(
                f"    {model_name}: F1={metrics['f1']:.4f} "
                f"MacroF1={metrics['macro_f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f}"
            )

    return pd.DataFrame(results_rows)


def summarize_ablation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated ablation results per model (mean/std)."""
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
    ]
    summary_df = results_df.groupby("model", as_index=False)[metric_columns].agg(["mean", "std"])
    summary_df.columns = [
        "model" if col == ("model", "") else f"{col[0]}_{col[1]}" for col in summary_df.columns
    ]
    return summary_df


def plot_ablation_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of F1 / Macro F1 / ROC-AUC per ablation model with std error bars."""
    model_order = [MODEL_CONTENT_ONLY, MODEL_NETWORK_ONLY, MODEL_HYBRID]
    short_names = ["Content Only", "Network Only", "Hybrid"]
    metrics = [("f1", "F1 (ML class)"), ("macro_f1", "Macro F1"), ("roc_auc", "ROC-AUC")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5), sharey=False)
    colors = ["#d1884f", "#4f7bd1", "#5aa469"]

    for ax, (metric, title) in zip(axes, metrics):
        means = [results_df.loc[results_df["model"] == m, metric].mean() for m in model_order]
        stds = [results_df.loc[results_df["model"] == m, metric].std() for m in model_order]

        bars = ax.bar(short_names, means, yerr=stds, capsize=5, color=colors)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation: What Is the Network Worth? (mean over 5 splits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df = run_ablation_experiments(data_dir)

    results_path = tables_dir / "ablation_model_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    summary_df = summarize_ablation_results(results_df)
    summary_path = tables_dir / "ablation_model_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    figure_path = figures_dir / "ablation_f1_auc_comparison.png"
    plot_ablation_comparison(results_df, figure_path)
    print(f"Saved: {figure_path}")

    print("\n=== Summary (mean over splits) ===")
    display_cols = ["model", "f1_mean", "macro_f1_mean", "roc_auc_mean", "recall_mean", "precision_mean"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))


if __name__ == "__main__":
    main()
