"""DeepWalk and Node2Vec classification under one fair repeated protocol."""

from pathlib import Path
import sys
import time

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
from embeddings import (
    embeddings_to_frame,
    generate_node2vec_walks,
    generate_random_walks,
    train_deepwalk,
    train_node2vec,
)
from modeling import (
    _merge_node_feature_frames,
    evaluate_predictions,
    extract_graph_neighbor_features,
    prepare_train_test_split,
    train_logistic_regression_graph_features,
)


SEEDS = (42, 7, 21, 84, 123)
NUM_WALKS = 3
WALK_LENGTH = 40
DIMENSIONS = 64
WINDOW = 5
EPOCHS = 5
NODE2VEC_P = 1.0
NODE2VEC_Q = 2.0
TSNE_SAMPLE = 4_000

EMBEDDING_MODELS = (
    "Logistic Regression - DeepWalk",
    "Logistic Regression - DeepWalk + Graph + Community",
    "Logistic Regression - Node2Vec",
    "Logistic Regression - Node2Vec + Graph + Community",
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


def _learn_embeddings(
    G,
    all_nodes: np.ndarray,
    embeddings_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Train both methods with identical walk and Word2Vec budgets."""
    frames: dict[str, pd.DataFrame] = {}

    started = time.time()
    deepwalk_walks = generate_random_walks(
        G, num_walks=NUM_WALKS, walk_length=WALK_LENGTH, seed=42
    )
    deepwalk_model = train_deepwalk(
        deepwalk_walks,
        dimensions=DIMENSIONS,
        window=WINDOW,
        epochs=EPOCHS,
        seed=42,
    )
    frames["DeepWalk"] = embeddings_to_frame(
        deepwalk_model, list(all_nodes)
    )
    np.savez_compressed(
        embeddings_dir / "deepwalk_64.npz",
        nodes=all_nodes,
        embeddings=frames["DeepWalk"].drop(columns=["node_id"]).to_numpy(),
    )
    del deepwalk_walks, deepwalk_model
    print(f"DeepWalk trained in {time.time() - started:.1f}s")

    started = time.time()
    node2vec_walks = generate_node2vec_walks(
        G,
        num_walks=NUM_WALKS,
        walk_length=WALK_LENGTH,
        p=NODE2VEC_P,
        q=NODE2VEC_Q,
        seed=42,
    )
    node2vec_model = train_node2vec(
        node2vec_walks,
        dimensions=DIMENSIONS,
        window=WINDOW,
        epochs=EPOCHS,
        seed=42,
    )
    frames["Node2Vec"] = embeddings_to_frame(
        node2vec_model, list(all_nodes)
    )
    np.savez_compressed(
        embeddings_dir / "node2vec_64.npz",
        nodes=all_nodes,
        embeddings=frames["Node2Vec"].drop(columns=["node_id"]).to_numpy(),
    )
    del node2vec_walks, node2vec_model
    print(f"Node2Vec trained in {time.time() - started:.1f}s")
    return frames


def _evaluate(
    G,
    labels_df: pd.DataFrame,
    labels_dict: dict[int, int],
    embedding_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    partition, _ = detect_louvain_communities(G, random_state=42)
    community_df = summarize_communities(G, labels_dict, partition)
    indexed_embeddings = {
        name: frame.set_index("node_id")
        for name, frame in embedding_frames.items()
    }
    rows: list[dict] = []

    for run_idx, seed in enumerate(SEEDS, start=1):
        train_nodes, test_nodes, y_train, y_test = prepare_train_test_split(
            labels_df, test_size=0.30, random_state=int(seed)
        )
        feature_nodes = list(train_nodes) + list(test_nodes)
        train_labels = {
            int(node): int(labels_dict[int(node)]) for node in train_nodes
        }
        graph_features = extract_graph_neighbor_features(
            G, train_labels, train_nodes, feature_nodes
        ).drop_duplicates("node_id")
        community_features = add_community_features(
            G,
            train_labels,
            train_nodes,
            feature_nodes,
            partition,
        ).drop_duplicates("node_id")
        handcrafted = _merge_node_feature_frames(
            graph_features, community_features
        )

        for method in ("DeepWalk", "Node2Vec"):
            embedding_df = embedding_frames[method]
            embedding_indexed = indexed_embeddings[method]
            X_train_embedding = embedding_indexed.reindex(
                train_nodes
            ).reset_index()
            X_test_embedding = embedding_indexed.reindex(
                test_nodes
            ).reset_index()
            _, pred_embedding, score_embedding = (
                train_logistic_regression_graph_features(
                    X_train_embedding, y_train, X_test_embedding
                )
            )
            embedding_metrics = evaluate_predictions(
                y_test, pred_embedding, score_embedding
            )

            combined = _merge_node_feature_frames(
                handcrafted, embedding_df
            ).set_index("node_id")
            X_train_combined = combined.reindex(train_nodes).reset_index()
            X_test_combined = combined.reindex(test_nodes).reset_index()
            _, pred_combined, score_combined = (
                train_logistic_regression_graph_features(
                    X_train_combined, y_train, X_test_combined
                )
            )
            combined_metrics = evaluate_predictions(
                y_test, pred_combined, score_combined
            )

            for model_name, metrics in (
                (f"Logistic Regression - {method}", embedding_metrics),
                (
                    f"Logistic Regression - {method} + Graph + Community",
                    combined_metrics,
                ),
            ):
                rows.append(
                    {
                        "run": int(run_idx),
                        "random_state": int(seed),
                        "model": model_name,
                        **{metric: float(metrics[metric]) for metric in METRICS},
                    }
                )
    return pd.DataFrame(rows)


def _summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby("model", as_index=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "model" if column == ("model", "") else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary.sort_values("roc_auc_mean").reset_index(drop=True)


def _plot_representation_comparison(
    embedding_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    sources = [
        (
            "Handcrafted\nnetwork",
            ablation_summary[
                ablation_summary["model"].str.contains(
                    "Network Only", regex=False
                )
            ].iloc[0],
        ),
        (
            "Content",
            ablation_summary[
                ablation_summary["model"].str.contains(
                    "Content Features Only", regex=False
                )
            ].iloc[0],
        ),
        (
            "DeepWalk",
            embedding_summary[
                embedding_summary["model"]
                == "Logistic Regression - DeepWalk"
            ].iloc[0],
        ),
        (
            "Node2Vec",
            embedding_summary[
                embedding_summary["model"]
                == "Logistic Regression - Node2Vec"
            ].iloc[0],
        ),
        (
            "Hybrid\nnetwork + content",
            ablation_summary[
                ablation_summary["model"].str.contains(
                    "Hybrid", regex=False
                )
            ].iloc[0],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    colors = ["#315C8D", "#D57837", "#6E5AA3", "#4F8A5B", "#2A7F78"]
    for ax, metric, title in (
        (axes[0], "macro_f1", "Macro F1"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ):
        values = [float(row[f"{metric}_mean"]) for _, row in sources]
        errors = [float(row[f"{metric}_std"]) for _, row in sources]
        bars = ax.bar(
            [label for label, _ in sources],
            values,
            yerr=errors,
            capsize=4,
            color=colors,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.010,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylim(0.65, 0.93)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Manual, learned and content representations\n"
        "Same five splits and Logistic Regression evaluation"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_tsne(
    embedding_frames: dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    output_path: Path,
) -> None:
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        len(labels_df), size=min(TSNE_SAMPLE, len(labels_df)), replace=False
    )
    labels = labels_df["label"].to_numpy(dtype=np.int64)[sample_idx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, method in zip(axes, ("DeepWalk", "Node2Vec")):
        matrix = embedding_frames[method].drop(
            columns=["node_id"]
        ).to_numpy(dtype=np.float64)
        coordinates = TSNE(
            n_components=2,
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(matrix[sample_idx])
        for label, field, color in (
            (0, "Web", "#315C8D"),
            (1, "Machine Learning", "#D57837"),
        ):
            mask = labels == label
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=6,
                alpha=0.45,
                color=color,
                linewidths=0,
                label=field,
            )
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(markerscale=3, loc="best")
    fig.suptitle(
        f"t-SNE view of unsupervised graph embeddings ({len(sample_idx):,} developers)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    embeddings_dir = PROJECT_DIR / "outputs" / "embeddings"
    for directory in (tables_dir, figures_dir, embeddings_dir):
        directory.mkdir(parents=True, exist_ok=True)

    G, _, labels_df, labels_dict = load_dataset(data_dir)
    all_nodes = labels_df["id"].to_numpy(dtype=np.int64)
    frames = _learn_embeddings(G, all_nodes, embeddings_dir)
    results_df = _evaluate(G, labels_df, labels_dict, frames)
    if set(results_df["model"]) != set(EMBEDDING_MODELS):
        raise AssertionError("Embedding model names do not match the declared set.")
    summary_df = _summarize(results_df)
    results_df.to_csv(
        tables_dir / "embedding_classification_results.csv", index=False
    )
    summary_df.to_csv(
        tables_dir / "embedding_classification_summary.csv", index=False
    )

    ablation_summary = pd.read_csv(
        tables_dir / "ablation_model_results_summary.csv"
    )
    _plot_representation_comparison(
        summary_df,
        ablation_summary,
        figures_dir / "representation_comparison.png",
    )
    _plot_tsne(
        frames, labels_df, figures_dir / "embedding_tsne.png"
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
