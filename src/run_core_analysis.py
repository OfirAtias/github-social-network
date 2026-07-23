# -*- coding: utf-8 -*-
"""Core descriptive, centrality and Louvain analyses for the final project."""

from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
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
from graph_analysis import (
    compute_basic_stats,
    compute_edge_type_counts,
    compute_homophily,
    compute_label_distribution,
)
from visualization import (
    plot_degree_distribution,
    plot_edge_type_distribution,
    plot_label_distribution,
    plot_sampled_network,
)


SEED = 42
TOP_K = 10
OVERLAP_K = 20
BETWEENNESS_SAMPLES = 100


def _data_quality_table(
    G: nx.Graph, edges_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    canonical = pd.DataFrame(
        {
            "u": edges_df[["id_1", "id_2"]].min(axis=1),
            "v": edges_df[["id_1", "id_2"]].max(axis=1),
        }
    )
    checks = [
        ("raw_edge_rows", int(len(edges_df)), True),
        ("graph_nodes", int(G.number_of_nodes()), True),
        ("graph_edges", int(G.number_of_edges()), True),
        ("self_loops", int(nx.number_of_selfloops(G)), nx.number_of_selfloops(G) == 0),
        (
            "duplicate_undirected_edges",
            int(canonical.duplicated().sum()),
            not canonical.duplicated().any(),
        ),
        (
            "duplicate_label_ids",
            int(labels_df["id"].duplicated().sum()),
            not labels_df["id"].duplicated().any(),
        ),
        (
            "missing_labels",
            int(labels_df["label"].isna().sum()),
            not labels_df["label"].isna().any(),
        ),
        (
            "all_graph_nodes_labeled",
            int(set(G.nodes()) <= set(labels_df["id"])),
            set(G.nodes()) <= set(labels_df["id"]),
        ),
        (
            "connected_components",
            int(nx.number_connected_components(G)),
            nx.is_connected(G),
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "passed"])


def _centrality_tables(
    G: nx.Graph, labels_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Computing centralities ...")
    degree = nx.degree_centrality(G)
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(
        G, k=BETWEENNESS_SAMPLES, seed=SEED, normalized=True
    )
    clustering = nx.clustering(G)
    try:
        hits_hub, _ = nx.hits(G, max_iter=800)
    except nx.PowerIterationFailedConvergence:
        hits_hub = {int(node): float("nan") for node in G}

    metrics = {
        "Degree Centrality": degree,
        "PageRank": pagerank,
        f"Betweenness Centrality (approx., k={BETWEENNESS_SAMPLES})": betweenness,
        "Clustering Coefficient": clustering,
        "HITS Hub": hits_hub,
    }
    metadata = labels_df.set_index("id")
    top_rows: list[dict] = []
    top_sets: dict[str, set[int]] = {}
    for metric_name, scores in metrics.items():
        ranked = sorted(
            ((int(node), float(score)) for node, score in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_sets[metric_name] = {node for node, _ in ranked[:OVERLAP_K]}
        for rank, (node, score) in enumerate(ranked[:TOP_K], start=1):
            label = int(metadata.at[node, "label"])
            top_rows.append(
                {
                    "metric": metric_name,
                    "rank": rank,
                    "node_id": node,
                    "developer_name": str(metadata.at[node, "name"])
                    if "name" in metadata.columns
                    else str(node),
                    "field": "Machine Learning" if label == 1 else "Web",
                    "score": score,
                    "degree": int(G.degree(node)),
                    "pagerank": float(pagerank[node]),
                    "betweenness_approx": float(betweenness[node]),
                    "clustering_coefficient": float(clustering[node]),
                }
            )

    overlap_rows = []
    for metric_a, metric_b in combinations(top_sets, 2):
        set_a, set_b = top_sets[metric_a], top_sets[metric_b]
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        overlap_rows.append(
            {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "top_k": OVERLAP_K,
                "intersection_count": intersection,
                "jaccard_overlap": intersection / union if union else 0.0,
            }
        )
    return pd.DataFrame(top_rows), pd.DataFrame(overlap_rows)


def _plot_communities(
    community_df: pd.DataFrame, figures_dir: Path
) -> None:
    top = community_df.head(12).sort_values("size")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["community_id"].astype(str),
        top["web_count"],
        label="Web",
        color="#315C8D",
    )
    ax.barh(
        top["community_id"].astype(str),
        top["ml_count"],
        left=top["web_count"],
        label="Machine Learning",
        color="#D57837",
    )
    ax.set_xlabel("Developers")
    ax.set_ylabel("Louvain community ID")
    ax.set_title("Twelve largest Louvain communities: size and field composition")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "community_size_and_composition.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sizes = community_df["size"].sort_values(ascending=False).to_numpy()
    ax.plot(np.arange(1, len(sizes) + 1), sizes, marker="o", color="#315C8D")
    ax.set_yscale("log")
    ax.set_xlabel("Community rank by size")
    ax.set_ylabel("Number of developers (log scale)")
    ax.set_title("Louvain community-size distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "community_size_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    G, edges_df, labels_df, labels_dict = load_dataset(data_dir)
    quality_df = _data_quality_table(G, edges_df, labels_df)
    if not quality_df["passed"].all():
        raise AssertionError(
            quality_df.loc[~quality_df["passed"]].to_string(index=False)
        )

    basic_stats = compute_basic_stats(G, labels_df)
    homophily = compute_homophily(G, labels_dict)
    basic_df = pd.DataFrame([{**basic_stats, **homophily}])
    label_df = compute_label_distribution(labels_df)
    edge_counts = compute_edge_type_counts(G, labels_dict)
    edge_df = pd.DataFrame(
        [
            {"edge_type": key, "count": int(value)}
            for key, value in edge_counts.items()
        ]
    )

    partition, modularity = detect_louvain_communities(
        G, random_state=SEED
    )
    community_df = summarize_communities(G, labels_dict, partition)
    community_metrics_df = pd.DataFrame(
        [
            {
                "number_of_communities": int(community_df.shape[0]),
                "modularity": float(modularity),
                "largest_community_size": int(community_df["size"].max()),
                "median_community_size": float(community_df["size"].median()),
                "size_weighted_mean_purity": float(
                    np.average(community_df["purity"], weights=community_df["size"])
                ),
            }
        ]
    )

    centrality_df, overlap_df = _centrality_tables(G, labels_df)

    tables = {
        "data_quality_checks.csv": quality_df,
        "network_basic_stats.csv": basic_df,
        "label_distribution.csv": label_df,
        "edge_type_counts.csv": edge_df,
        "community_summary.csv": community_df,
        "community_metrics.csv": community_metrics_df,
        "centrality_top_developers.csv": centrality_df,
        "centrality_top_overlap.csv": overlap_df,
    }
    for filename, frame in tables.items():
        frame.to_csv(tables_dir / filename, index=False)

    plot_label_distribution(label_df, figures_dir)
    plot_degree_distribution(G, figures_dir)
    plot_edge_type_distribution(edge_counts, figures_dir)
    plot_sampled_network(
        G,
        labels_dict,
        partition,
        community_df,
        figures_dir,
        sample_size=350,
        random_state=SEED,
    )
    _plot_communities(community_df, figures_dir)

    print("Core analysis completed.")
    print(basic_df.to_string(index=False))
    print(community_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
