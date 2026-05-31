# -*- coding: utf-8 -*-
from typing import Any, Dict, Iterable

import networkx as nx
import pandas as pd


def compute_basic_stats(G: nx.Graph, labels_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute core graph statistics for initial analysis."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G) if num_nodes > 1 else 0.0
    average_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0

    num_components = nx.number_connected_components(G) if num_nodes > 0 else 0
    largest_component_size = len(max(nx.connected_components(G), key=len)) if num_nodes > 0 else 0
    largest_component_ratio = (largest_component_size / num_nodes) if num_nodes > 0 else 0.0

    labeled_nodes_total = int(labels_df["id"].nunique())
    labeled_nodes_in_graph = int(labels_df["id"].isin(G.nodes()).sum())
    labeled_node_ratio = (labeled_nodes_in_graph / num_nodes) if num_nodes > 0 else 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "average_degree": average_degree,
        "num_connected_components": num_components,
        "largest_component_size": largest_component_size,
        "largest_component_ratio": largest_component_ratio,
        "labeled_nodes_total": labeled_nodes_total,
        "labeled_nodes_in_graph": labeled_nodes_in_graph,
        "labeled_node_ratio": labeled_node_ratio,
    }


def compute_label_distribution(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Return label counts and ratios in a compact table."""
    counts = labels_df["label"].value_counts().sort_index()
    distribution_df = pd.DataFrame(
        {
            "label": counts.index.astype(int),
            "count": counts.values.astype(int),
        }
    )
    distribution_df["class_name"] = distribution_df["label"].map({0: "Web", 1: "Machine Learning"})
    distribution_df["ratio"] = distribution_df["count"] / distribution_df["count"].sum()

    return distribution_df


def compute_edge_type_counts(G: nx.Graph, labels_dict: Dict[int, int]) -> Dict[str, int]:
    """
    Count edge types by endpoint labels:
    - web_web: both endpoints label 0
    - ml_ml: both endpoints label 1
    - web_ml: mixed labels
    - missing_label: at least one endpoint has no label
    """
    counts = {"web_web": 0, "ml_ml": 0, "web_ml": 0, "missing_label": 0}

    for u, v in G.edges():
        label_u = labels_dict.get(u)
        label_v = labels_dict.get(v)

        if label_u is None or label_v is None:
            counts["missing_label"] += 1
            continue

        if label_u == 0 and label_v == 0:
            counts["web_web"] += 1
        elif label_u == 1 and label_v == 1:
            counts["ml_ml"] += 1
        else:
            counts["web_ml"] += 1

    return counts


def compute_homophily(G: nx.Graph, labels_dict: Dict[int, int]) -> Dict[str, float]:
    """Compute homophily based on the share of same-label edges among labeled edges."""
    edge_type_counts = compute_edge_type_counts(G, labels_dict)
    same_label_edges = edge_type_counts["web_web"] + edge_type_counts["ml_ml"]
    labeled_edges = same_label_edges + edge_type_counts["web_ml"]

    if labeled_edges == 0:
        homophily_rate = float("nan")
    else:
        homophily_rate = same_label_edges / labeled_edges

    return {
        "homophily_rate": homophily_rate,
        "same_label_edges": same_label_edges,
        "different_label_edges": edge_type_counts["web_ml"],
        "labeled_edges": labeled_edges,
    }


def compute_label_assortativity(
    G: nx.Graph,
    labels_dict: Dict[int, int],
    nodes: Iterable[int] | None = None,
) -> float:
    """
    Compute label assortativity on a labeled subgraph.

    When nodes is provided, only those nodes and labels are used. This is useful
    for train-only diagnostics that avoid test-label leakage.
    """
    if nodes is None:
        candidate_nodes = [int(node) for node in G.nodes() if int(node) in labels_dict]
    else:
        candidate_nodes = [int(node) for node in nodes if int(node) in G and int(node) in labels_dict]

    if len(candidate_nodes) < 2:
        return float("nan")

    subgraph = G.subgraph(candidate_nodes).copy()
    if subgraph.number_of_edges() == 0:
        return float("nan")

    nx.set_node_attributes(subgraph, {node: int(labels_dict[node]) for node in candidate_nodes}, "label")
    return float(nx.attribute_assortativity_coefficient(subgraph, "label"))
