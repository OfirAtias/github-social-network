# -*- coding: utf-8 -*-
"""
DeepWalk experiment: learned structural embeddings vs handcrafted features.

Pipeline:
1. Generate random walks over the full graph and train DeepWalk embeddings
   (unsupervised, structure-only - computed once, like Louvain).
2. Evaluate under the exact protocol of run_repeated_experiments
   (same seeds, same stratified 70/30 splits, same metrics):
     - Logistic Regression - DeepWalk Embeddings Only
     - Logistic Regression - DeepWalk + Graph + Community Features
3. Visualize the embedding space with t-SNE colored by label.

Outputs:
    outputs/embeddings/deepwalk_128.npz
    outputs/tables/deepwalk_model_results.csv
    outputs/tables/deepwalk_model_results_summary.csv
    outputs/figures/deepwalk_tsne.png
    outputs/figures/deepwalk_comparison.png
"""
import time
from pathlib import Path
from typing import Sequence

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from community_analysis import (
    add_community_features,
    detect_louvain_communities,
    summarize_communities,
)
from data_loader import load_dataset
from embeddings import embeddings_to_frame, generate_random_walks, train_deepwalk
from modeling import (
    _merge_node_feature_frames,
    evaluate_predictions,
    extract_graph_neighbor_features,
    prepare_train_test_split,
    train_logistic_regression_graph_features,
)


MODEL_DEEPWALK_ONLY = "Logistic Regression - DeepWalk Embeddings Only"
MODEL_DEEPWALK_FULL = "Logistic Regression - DeepWalk + Graph + Community Features"

NUM_WALKS = 10
WALK_LENGTH = 80
DIMENSIONS = 128
WINDOW = 10
EPOCHS = 5
TSNE_SAMPLE = 5000


def main(random_states: Sequence[int] = (42, 7, 21, 84, 123)) -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    embeddings_dir = PROJECT_DIR / "outputs" / "embeddings"
    for directory in (tables_dir, figures_dir, embeddings_dir):
        directory.mkdir(parents=True, exist_ok=True)

    G, _, labels_df, labels_dict = load_dataset(data_dir)
    all_nodes = labels_df["id"].to_numpy(dtype=np.int64)
    all_labels = labels_df["label"].to_numpy(dtype=np.int64)

    # --- 1) Walks + Word2Vec (structure-only, computed once like Louvain) ---
    t0 = time.time()
    print(f"Generating walks: {NUM_WALKS} walks/node, length {WALK_LENGTH} ...")
    walks = generate_random_walks(G, num_walks=NUM_WALKS, walk_length=WALK_LENGTH, seed=42)
    print(f"  {len(walks):,} walks in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"Training DeepWalk (dim={DIMENSIONS}, window={WINDOW}, epochs={EPOCHS}) ...")
    w2v_model = train_deepwalk(
        walks, dimensions=DIMENSIONS, window=WINDOW, epochs=EPOCHS, seed=42
    )
    print(f"  trained in {time.time() - t0:.1f}s")

    embedding_df = embeddings_to_frame(w2v_model, list(all_nodes))
    embedding_matrix = embedding_df.drop(columns=["node_id"]).to_numpy(dtype=np.float64)
    np.savez_compressed(
        embeddings_dir / "deepwalk_128.npz",
        nodes=all_nodes,
        embeddings=embedding_matrix,
    )
    print(f"Saved embeddings: {embeddings_dir / 'deepwalk_128.npz'}")

    # --- 2) Repeated evaluation, same protocol as run_repeated_experiments ---
    partition, _ = detect_louvain_communities(G, random_state=42)
    community_summary_df = summarize_communities(G, labels_dict, partition)
    embedding_indexed = embedding_df.set_index("node_id")

    results_rows = []
    for run_idx, random_state in enumerate(random_states, start=1):
        print(f"[run {run_idx}/{len(random_states)}] random_state={random_state}")

        train_nodes, test_nodes, y_train, y_test = prepare_train_test_split(
            labels_df=labels_df,
            random_state=int(random_state),
        )
        nodes_for_features = list(train_nodes) + list(test_nodes)

        # DeepWalk only
        X_train_emb = embedding_indexed.reindex(train_nodes).reset_index()
        X_test_emb = embedding_indexed.reindex(test_nodes).reset_index()

        _, y_pred_emb, y_score_emb = train_logistic_regression_graph_features(
            X_train=X_train_emb,
            y_train=y_train,
            X_test=X_test_emb,
        )
        metrics_emb = evaluate_predictions(y_test, y_pred_emb, y_score_emb)

        # DeepWalk + Graph + Community
        graph_features_df = extract_graph_neighbor_features(
            G=G,
            labels_dict=labels_dict,
            train_nodes=train_nodes,
            nodes=nodes_for_features,
        ).drop_duplicates(subset=["node_id"])

        community_features_df = add_community_features(
            G=G,
            labels_dict=labels_dict,
            train_nodes=train_nodes,
            nodes=nodes_for_features,
            partition=partition,
            community_summary_df=community_summary_df,
        ).drop_duplicates(subset=["node_id"])

        network_features_df = _merge_node_feature_frames(graph_features_df, community_features_df)
        full_features_df = _merge_node_feature_frames(network_features_df, embedding_df)
        full_features_indexed = full_features_df.set_index("node_id")

        X_train_full = full_features_indexed.reindex(train_nodes).reset_index()
        X_test_full = full_features_indexed.reindex(test_nodes).reset_index()

        _, y_pred_full, y_score_full = train_logistic_regression_graph_features(
            X_train=X_train_full,
            y_train=y_train,
            X_test=X_test_full,
        )
        metrics_full = evaluate_predictions(y_test, y_pred_full, y_score_full)

        for model_name, metrics in (
            (MODEL_DEEPWALK_ONLY, metrics_emb),
            (MODEL_DEEPWALK_FULL, metrics_full),
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

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(tables_dir / "deepwalk_model_results.csv", index=False)

    metric_columns = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1", "roc_auc"]
    summary_df = results_df.groupby("model", as_index=False)[metric_columns].agg(["mean", "std"])
    summary_df.columns = [
        "model" if col == ("model", "") else f"{col[0]}_{col[1]}" for col in summary_df.columns
    ]
    summary_df.to_csv(tables_dir / "deepwalk_model_results_summary.csv", index=False)
    print(f"Saved: {tables_dir / 'deepwalk_model_results.csv'}")

    # --- 3) t-SNE visualization (stratified sample) ---
    print("Computing t-SNE ...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_nodes), size=min(TSNE_SAMPLE, len(all_nodes)), replace=False)
    tsne_coords = TSNE(
        n_components=2, learning_rate="auto", init="pca", random_state=42
    ).fit_transform(embedding_matrix[sample_idx])

    fig, ax = plt.subplots(figsize=(9, 8))
    sample_labels = all_labels[sample_idx]
    for label_value, name, color in ((0, "Web", "#4f7bd1"), (1, "ML", "#e07b39")):
        mask = sample_labels == label_value
        ax.scatter(
            tsne_coords[mask, 0], tsne_coords[mask, 1],
            s=6, alpha=0.5, c=color, label=name, linewidths=0,
        )
    ax.legend(markerscale=3)
    ax.set_title(
        f"t-SNE of DeepWalk Embeddings ({TSNE_SAMPLE:,} sampled developers)\n"
        "Structure-only embeddings - labels were never used in training"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figures_dir / "deepwalk_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'deepwalk_tsne.png'}")

    # --- 4) Comparison figure vs existing handcrafted network model ---
    existing_df = pd.read_csv(tables_dir / "repeated_model_results_with_communities.csv")
    handcrafted = existing_df[existing_df["model"] == "Logistic Regression - Graph + Community Features"]

    labels_bars = ["Handcrafted\n(Graph + Community)", "DeepWalk\nOnly", "DeepWalk +\nHandcrafted"]
    frames = [handcrafted, results_df[results_df["model"] == MODEL_DEEPWALK_ONLY],
              results_df[results_df["model"] == MODEL_DEEPWALK_FULL]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["macro_f1", "roc_auc"], ["Macro F1", "ROC-AUC"]):
        means = [frame[metric].mean() for frame in frames]
        stds = [frame[metric].std() for frame in frames]
        bars = ax.bar(labels_bars, means, yerr=stds, capsize=5,
                      color=["#4f7bd1", "#8a5fc2", "#5aa469"])
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.008, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Handcrafted vs Learned Network Representations (mean over 5 splits)")
    fig.tight_layout()
    fig.savefig(figures_dir / "deepwalk_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figures_dir / 'deepwalk_comparison.png'}")

    print("\n=== Summary (mean over splits) ===")
    display_cols = ["model", "f1_mean", "macro_f1_mean", "roc_auc_mean"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))


if __name__ == "__main__":
    main()
