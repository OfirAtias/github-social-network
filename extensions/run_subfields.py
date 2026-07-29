"""Reproducible content-subfield clustering on the MUSAE feature matrix.

Sparse binary content is reduced with TruncatedSVD, L2-normalized, and then
clustered with K-means.  Candidate cluster counts are evaluated without using
the supplied binary target: silhouette uses a fixed sample, while stability is
the pairwise adjusted Rand index across five independent fixed seeds.
"""

from __future__ import annotations

import itertools
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
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

from community_analysis import detect_louvain_communities
from content_features import build_content_matrix, load_content_features
from data_loader import load_dataset


SVD_COMPONENTS = 64
K_CANDIDATES = tuple(range(3, 11))
SILHOUETTE_SAMPLE = 5_000
SEEDS = (42, 7, 21, 84, 123)
KMEANS_N_INIT = 10
PROJECTION_SAMPLE = 8_000


def evaluate_cluster_counts(
    embedding: np.ndarray,
) -> tuple[int, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    """Evaluate silhouette and cross-seed stability for every candidate k."""
    rng = np.random.default_rng(SEEDS[0])
    sample_idx = np.sort(
        rng.choice(
            embedding.shape[0],
            size=min(SILHOUETTE_SAMPLE, embedding.shape[0]),
            replace=False,
        )
    )
    diagnostic_rows: list[dict] = []
    pair_rows: list[dict] = []
    reference_assignments: dict[int, np.ndarray] = {}

    for k in K_CANDIDATES:
        assignments: dict[int, np.ndarray] = {}
        for seed in SEEDS:
            assignments[seed] = KMeans(
                n_clusters=k,
                n_init=KMEANS_N_INIT,
                max_iter=300,
                algorithm="lloyd",
                random_state=seed,
            ).fit_predict(embedding)
        reference = assignments[SEEDS[0]]
        reference_assignments[k] = reference
        silhouette = silhouette_score(
            embedding[sample_idx], reference[sample_idx], metric="euclidean"
        )
        pair_values = []
        for first, second in itertools.combinations(SEEDS, 2):
            stability = adjusted_rand_score(assignments[first], assignments[second])
            pair_values.append(float(stability))
            pair_rows.append(
                {"k": k, "seed_a": first, "seed_b": second, "adjusted_rand": stability}
            )
        cluster_sizes = np.bincount(reference, minlength=k)
        diagnostic_rows.append(
            {
                "k": k,
                "silhouette_fixed_sample": float(silhouette),
                "stability_ari_mean": float(np.mean(pair_values)),
                "stability_ari_std": float(np.std(pair_values, ddof=1)),
                "smallest_cluster": int(cluster_sizes.min()),
                "largest_cluster": int(cluster_sizes.max()),
            }
        )
        print(
            f"  k={k}: silhouette={silhouette:.4f}, "
            f"stability ARI={np.mean(pair_values):.4f} +/- {np.std(pair_values, ddof=1):.4f}"
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    pairs = pd.DataFrame(pair_rows)
    # Selection is target-free and predeclared: maximize reference silhouette.
    best_k = int(
        diagnostics.sort_values(
            ["silhouette_fixed_sample", "stability_ari_mean", "k"],
            ascending=[False, False, True],
        ).iloc[0]["k"]
    )
    return best_k, diagnostics, pairs, reference_assignments


def _plot_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        diagnostics["k"], diagnostics["silhouette_fixed_sample"], marker="o", color="#4E79A7"
    )
    axes[0].set_title("Separation: silhouette on fixed 5,000-node sample")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette")
    axes[1].errorbar(
        diagnostics["k"],
        diagnostics["stability_ari_mean"],
        yerr=diagnostics["stability_ari_std"],
        marker="o",
        capsize=4,
        color="#59A14F",
    )
    axes[1].set_title("Stability across five K-means seeds")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Pairwise adjusted Rand index")
    for axis in axes:
        axis.set_xticks(diagnostics["k"])
        axis.grid(alpha=0.25)
    fig.suptitle("Content-subfield cluster-count diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_summary(summary: pd.DataFrame, global_ml_share: float, output_path: Path) -> None:
    ordered = summary.sort_values("cluster")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(ordered["cluster"].astype(str), ordered["size"], color="#4E79A7")
    axes[0].set_title("Selected-clustering sizes")
    axes[0].set_xlabel("Anonymous content cluster")
    axes[0].set_ylabel("Developers")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(ordered["cluster"].astype(str), ordered["ml_share"], color="#F28E2B")
    axes[1].axhline(
        global_ml_share,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Global ML share ({global_ml_share:.3f})",
    )
    axes[1].set_title("Post-hoc binary-label composition")
    axes[1].set_xlabel("Anonymous content cluster")
    axes[1].set_ylabel("ML-label share")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Selected content clustering (feature indices have no semantic dictionary)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> dict[str, Path]:
    """Run the complete subfield experiment and save tables/figures."""
    started = time.perf_counter()
    data_dir = PROJECT_DIR / "data"
    output_root = PROJECT_DIR / "outputs" / "extensions"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    for directory in (tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    graph, _, labels_df, labels_dict = load_dataset(data_dir)
    nodes = labels_df["id"].to_numpy(dtype=np.int64)
    binary_labels = labels_df["label"].to_numpy(dtype=np.int64)
    features_dict, n_features = load_content_features(
        data_dir / "musae_git_features.json"
    )
    content = build_content_matrix(features_dict, n_features, nodes)

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEEDS[0])
    embedding = normalize(svd.fit_transform(content)).astype(np.float32)
    explained_variance = float(svd.explained_variance_ratio_.sum())
    print(
        f"Content matrix {content.shape} with {content.nnz:,} nonzeros; "
        f"{SVD_COMPONENTS}-component variance={explained_variance:.4f}"
    )
    best_k, diagnostics, stability_pairs, assignments = evaluate_cluster_counts(embedding)
    clusters = assignments[best_k]
    print(f"Selected k={best_k} by target-free silhouette maximization.")

    assignments_df = pd.DataFrame(
        {"node_id": nodes, "content_cluster": clusters, "binary_label": binary_labels}
    )
    projection_rng = np.random.default_rng(SEEDS[0])
    projection_idx = np.sort(
        projection_rng.choice(
            len(nodes), size=min(PROJECTION_SAMPLE, len(nodes)), replace=False
        )
    )
    projection_df = pd.DataFrame(
        {
            "node_id": nodes[projection_idx],
            "svd_component_1": embedding[projection_idx, 0],
            "svd_component_2": embedding[projection_idx, 1],
            "content_cluster": clusters[projection_idx],
        }
    )
    cluster_summary = (
        assignments_df.groupby("content_cluster", as_index=False)
        .agg(size=("node_id", "size"), ml_count=("binary_label", "sum"))
        .rename(columns={"content_cluster": "cluster"})
    )
    cluster_summary["share_of_nodes"] = cluster_summary["size"] / len(assignments_df)
    cluster_summary["ml_share"] = cluster_summary["ml_count"] / cluster_summary["size"]

    partition, modularity = detect_louvain_communities(graph, random_state=SEEDS[0])
    communities = np.asarray([partition[int(node)] for node in nodes], dtype=np.int64)
    alignment = pd.DataFrame(
        [
            {
                "comparison": "content_clusters_vs_binary_labels",
                "nmi": normalized_mutual_info_score(binary_labels, clusters),
                "ari": adjusted_rand_score(binary_labels, clusters),
            },
            {
                "comparison": "content_clusters_vs_louvain_communities",
                "nmi": normalized_mutual_info_score(communities, clusters),
                "ari": adjusted_rand_score(communities, clusters),
            },
            {
                "comparison": "louvain_communities_vs_binary_labels",
                "nmi": normalized_mutual_info_score(binary_labels, communities),
                "ari": adjusted_rand_score(binary_labels, communities),
            },
        ]
    )

    paths = {
        "diagnostics": tables_dir / "subfield_k_diagnostics.csv",
        "stability": tables_dir / "subfield_stability_pairs.csv",
        "sizes": tables_dir / "subfield_cluster_sizes.csv",
        "assignments": tables_dir / "subfield_assignments.csv",
        "alignment": tables_dir / "subfield_alignment.csv",
        "projection": tables_dir / "subfield_projection_sample.csv",
        "diagnostics_figure": figures_dir / "subfield_k_diagnostics.png",
        "sizes_figure": figures_dir / "subfield_cluster_sizes.png",
    }
    diagnostics.to_csv(paths["diagnostics"], index=False)
    stability_pairs.to_csv(paths["stability"], index=False)
    cluster_summary.to_csv(paths["sizes"], index=False)
    assignments_df.to_csv(paths["assignments"], index=False)
    alignment.to_csv(paths["alignment"], index=False)
    projection_df.to_csv(paths["projection"], index=False)
    _plot_diagnostics(diagnostics, paths["diagnostics_figure"])
    _plot_cluster_summary(cluster_summary, float(binary_labels.mean()), paths["sizes_figure"])

    total_seconds = time.perf_counter() - started
    metadata = {
        "experiment": "content_subfield_clustering",
        "nodes": int(content.shape[0]),
        "raw_content_dimensions": int(content.shape[1]),
        "nonzero_entries": int(content.nnz),
        "svd_components": SVD_COMPONENTS,
        "svd_explained_variance": explained_variance,
        "l2_normalized": True,
        "candidate_k": list(K_CANDIDATES),
        "selection_rule": "maximum silhouette on fixed seed-42 sample; stability breaks ties",
        "silhouette_sample": min(SILHOUETTE_SAMPLE, embedding.shape[0]),
        "stability_seeds": list(SEEDS),
        "kmeans_n_init": KMEANS_N_INIT,
        "selected_k": best_k,
        "projection_sample": int(len(projection_df)),
        "louvain_modularity": float(modularity),
        "uses_binary_labels_for_selection": False,
        "total_seconds": total_seconds,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    metadata_path = tables_dir / "subfield_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = metadata_path
    print(f"Subfield experiment completed in {total_seconds:.1f}s")
    print(cluster_summary.to_string(index=False))
    print(alignment.to_string(index=False))
    return paths


if __name__ == "__main__":
    main()
