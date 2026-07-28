"""
Extension only: this exploratory subfield analysis is excluded because its
central methods are outside the selected course-only scope.

Content-based sub-field discovery.

The binary Web/ML target hides the fact that developers span many
sub-specializations. This script clusters the MUSAE content features
(starred repos / employer / location) to discover data-driven "sub-fields",
then measures how well the network's Louvain communities align with them.

Questions answered:
1. Do content clusters split the binary labels into finer sub-fields?
2. Do network communities align more with content clusters than with the
   binary Web/ML label? (NMI / ARI)

Outputs:
    outputs/tables/subfield_cluster_summary.csv
    outputs/tables/subfield_alignment_scores.csv
    outputs/figures/subfield_ml_share.png
    outputs/figures/subfield_community_overlap.png
"""
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

EXTENSIONS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXTENSIONS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(EXTENSIONS_DIR))

from community_analysis import detect_louvain_communities
from content_features import build_content_matrix, load_content_features
from data_loader import load_dataset


SVD_COMPONENTS = 64
K_CANDIDATES = range(4, 11)
SILHOUETTE_SAMPLE = 5000
RANDOM_STATE = 42


def choose_k(embedding: np.ndarray) -> tuple[int, pd.DataFrame]:
    """Pick the number of clusters by silhouette score over a candidate range."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(embedding.shape[0], size=min(SILHOUETTE_SAMPLE, embedding.shape[0]), replace=False)

    rows = []
    for k in K_CANDIDATES:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)
        score = silhouette_score(embedding[sample_idx], cluster_labels[sample_idx])
        rows.append({"k": k, "silhouette": float(score)})
        print(f"  k={k}: silhouette={score:.4f}")

    scores_df = pd.DataFrame(rows)
    best_k = int(scores_df.loc[scores_df["silhouette"].idxmax(), "k"])
    return best_k, scores_df


def main() -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    G, _, labels_df, labels_dict = load_dataset(data_dir)
    features_dict, n_features = load_content_features(data_dir / "musae_git_features.json")

    nodes = labels_df["id"].to_numpy(dtype=np.int64)
    labels = labels_df["label"].to_numpy(dtype=np.int64)

    # --- Content embedding: sparse binary -> dense low-rank representation ---
    content_matrix = build_content_matrix(features_dict, n_features, nodes)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    embedding = svd.fit_transform(content_matrix)
    embedding = normalize(embedding)
    print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    # --- Cluster into sub-fields ---
    print("Choosing k by silhouette:")
    best_k, k_scores_df = choose_k(embedding)
    print(f"Selected k={best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(embedding)

    # --- Per-cluster label composition ---
    cluster_df = pd.DataFrame({"node_id": nodes, "cluster": clusters, "label": labels})
    summary_df = (
        cluster_df.groupby("cluster")
        .agg(size=("node_id", "count"), ml_count=("label", "sum"))
        .reset_index()
    )
    summary_df["ml_share"] = summary_df["ml_count"] / summary_df["size"]
    summary_df = summary_df.sort_values("ml_share", ascending=False).reset_index(drop=True)
    summary_df.to_csv(tables_dir / "subfield_cluster_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # --- Alignment: clusters vs labels vs Louvain communities ---
    partition, _ = detect_louvain_communities(G, random_state=RANDOM_STATE)
    communities = np.asarray([partition.get(int(node), -1) for node in nodes])

    alignment_rows = [
        {
            "comparison": "content_clusters_vs_binary_labels",
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
            "ari": float(adjusted_rand_score(labels, clusters)),
        },
        {
            "comparison": "louvain_communities_vs_binary_labels",
            "nmi": float(normalized_mutual_info_score(labels, communities)),
            "ari": float(adjusted_rand_score(labels, communities)),
        },
        {
            "comparison": "louvain_communities_vs_content_clusters",
            "nmi": float(normalized_mutual_info_score(clusters, communities)),
            "ari": float(adjusted_rand_score(clusters, communities)),
        },
    ]
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(tables_dir / "subfield_alignment_scores.csv", index=False)
    print(alignment_df.to_string(index=False))

    # --- Figure 1: ML share per sub-field cluster ---
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(summary_df))
    colors = plt.cm.coolwarm(summary_df["ml_share"].to_numpy())
    ax1.bar(x, summary_df["ml_share"], color=colors)
    ax1.axhline(labels.mean(), color="black", linestyle="--", linewidth=1,
                label=f"Global ML share ({labels.mean():.2f})")
    for xi, (share, size) in enumerate(zip(summary_df["ml_share"], summary_df["size"])):
        ax1.text(xi, share + 0.015, f"n={size:,}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in summary_df["cluster"]])
    ax1.set_xlabel("Content cluster (sorted by ML share)")
    ax1.set_ylabel("ML share")
    ax1.set_title(f"Content-Based Sub-Fields (k={best_k}): ML Share per Cluster")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "subfield_ml_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: overlap heatmap, content clusters x largest communities ---
    top_communities = (
        pd.Series(communities).value_counts().head(10).index.to_numpy()
    )
    overlap = np.zeros((best_k, len(top_communities)))
    for ci in range(best_k):
        in_cluster = communities[clusters == ci]
        for cj, comm in enumerate(top_communities):
            overlap[ci, cj] = np.mean(in_cluster == comm) if len(in_cluster) else 0.0

    fig, ax2 = plt.subplots(figsize=(10, 6))
    im = ax2.imshow(overlap, cmap="viridis", aspect="auto")
    ax2.set_xticks(np.arange(len(top_communities)))
    ax2.set_xticklabels([f"Comm {c}" for c in top_communities], rotation=45, ha="right")
    ax2.set_yticks(np.arange(best_k))
    ax2.set_yticklabels([f"Cluster {c}" for c in range(best_k)])
    ax2.set_xlabel("Louvain community (top 10 by size)")
    ax2.set_ylabel("Content cluster")
    ax2.set_title("Where Do Content Sub-Fields Live in the Network?\n(row = share of cluster members in each community)")
    fig.colorbar(im, ax=ax2, label="Share of cluster")
    fig.tight_layout()
    fig.savefig(figures_dir / "subfield_community_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    k_scores_df.to_csv(tables_dir / "subfield_k_selection.csv", index=False)
    print("Saved tables and figures.")


if __name__ == "__main__":
    main()
