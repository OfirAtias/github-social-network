from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _prepare_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_label_distribution(label_distribution_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot and save label distribution (Web vs ML)."""
    output_path = _prepare_output_dir(output_dir) / "label_distribution.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=label_distribution_df, x="class_name", y="count", hue="class_name", legend=False, palette="Set2", ax=ax)
    ax.set_title("GitHub Developers by Label", fontsize=13, weight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=10)

    for idx, row in label_distribution_df.reset_index(drop=True).iterrows():
        ax.text(idx, row["count"], f"{row['count']:,}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_degree_distribution(G: nx.Graph, output_dir: str | Path) -> Path:
    """Plot and save degree distribution histogram."""
    output_path = _prepare_output_dir(output_dir) / "degree_distribution.png"
    degrees = [degree for _, degree in G.degree()]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(degrees, bins=60, color="#2ca02c", ax=ax)
    ax.set_title("Degree Distribution", fontsize=13, weight="bold")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of Nodes")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_edge_type_distribution(edge_type_counts: Dict[str, int], output_dir: str | Path) -> Path:
    """Plot and save distribution of edge types by node labels."""
    output_path = _prepare_output_dir(output_dir) / "edge_type_distribution.png"

    label_map = {
        "web_web": "Web-Web",
        "ml_ml": "ML-ML",
        "web_ml": "Web-ML",
        "missing_label": "Missing Label",
    }

    plot_df = pd.DataFrame(
        {
            "edge_type": [label_map[key] for key in label_map],
            "count": [edge_type_counts.get(key, 0) for key in label_map],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="edge_type", y="count", hue="edge_type", legend=False, palette="Blues", ax=ax)
    ax.set_title("Edge Type Distribution", fontsize=13, weight="bold")
    ax.set_xlabel("Edge Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=10)

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, row["count"], f"{int(row['count']):,}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_sampled_network(
    G: nx.Graph,
    labels_dict: Dict[int, int],
    partition: Dict[int, int],
    community_summary_df: pd.DataFrame,
    output_dir: str | Path,
    sample_size: int = 350,
    random_state: int = 42,
) -> Path:
    """Plot a readable sampled developer network colored by label."""
    output_path = _prepare_output_dir(output_dir) / "sampled_network_by_label.png"
    rng = np.random.default_rng(random_state)

    top_communities = (
        community_summary_df.sort_values("size", ascending=False)["community_id"]
        .astype(int)
        .head(4)
        .tolist()
    )
    nodes_by_community: Dict[int, list[int]] = {community_id: [] for community_id in top_communities}
    for node, community_id in partition.items():
        community_id_int = int(community_id)
        if community_id_int in nodes_by_community and int(node) in labels_dict:
            nodes_by_community[community_id_int].append(int(node))

    sampled_nodes: list[int] = []
    per_community = max(1, sample_size // max(1, len(top_communities)))
    for community_id, nodes in nodes_by_community.items():
        ranked_nodes = sorted(nodes, key=lambda node: G.degree(node), reverse=True)
        head = ranked_nodes[: min(len(ranked_nodes), per_community * 2)]
        if len(head) > per_community:
            selected = rng.choice(head, size=per_community, replace=False).tolist()
        else:
            selected = head
        sampled_nodes.extend(int(node) for node in selected)

    sampled_nodes = list(dict.fromkeys(sampled_nodes))[:sample_size]
    subgraph = G.subgraph(sampled_nodes).copy()

    label_colors = {0: "#4C78A8", 1: "#F58518"}
    node_colors = [label_colors.get(int(labels_dict.get(node, 0)), "#999999") for node in subgraph.nodes()]
    node_sizes = [35 + min(120, G.degree(node) * 2) for node in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(subgraph, seed=random_state, k=0.22, iterations=80)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.12, width=0.5, edge_color="#555555")
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.2,
        edgecolors="#222222",
        alpha=0.9,
    )

    ax.set_title("Sampled GitHub Developer Network Colored by Label", fontsize=13, weight="bold")
    ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=label_colors[0], label="Web", markersize=8),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=label_colors[1],
            label="Machine Learning",
            markersize=8,
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_community_purity_error_rates(error_by_purity_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot model error rate by community-purity bucket."""
    output_path = _prepare_output_dir(output_dir) / "community_purity_error_analysis.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=error_by_purity_df, x="purity_group", y="error_rate", color="#D55E00", ax=ax)
    ax.set_title("Error Rate by Community Purity", fontsize=13, weight="bold")
    ax.set_xlabel("Community Purity Group")
    ax.set_ylabel("Error Rate")
    ax.tick_params(axis="x", rotation=12)

    for idx, row in error_by_purity_df.reset_index(drop=True).iterrows():
        if pd.isna(row["error_rate"]):
            continue
        ax.text(idx, row["error_rate"], f"{row['error_rate']:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_error_type_counts(error_types_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot counts of correct predictions and directional error types."""
    output_path = _prepare_output_dir(output_dir) / "prediction_error_types.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=error_types_df, x="error_type", y="count", hue="error_type", legend=False, palette="Set2", ax=ax)
    ax.set_title("Prediction Outcomes by Type", fontsize=13, weight="bold")
    ax.set_xlabel("Prediction Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=12)

    for idx, row in error_types_df.reset_index(drop=True).iterrows():
        ax.text(idx, row["count"], f"{int(row['count']):,}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path
