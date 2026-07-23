# -*- coding: utf-8 -*-
"""Leakage-safe reconstruction of randomly hidden edges.

Each split protects a spanning tree, removes test positives only from the
remaining edges, and samples true non-edges with the same endpoint-label
composition as the hidden positives.  Consequently the train graph stays
connected and same-field/cross-field comparisons use identical sampling rules.
"""

from collections import Counter
import math
import random
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


Pair = Tuple[int, int]
EDGE_TYPES = ("Web-Web", "ML-ML", "Web-ML")


def canonical_pair(u: int, v: int) -> Pair:
    """Return an undirected pair in stable order."""
    return (int(min(u, v)), int(max(u, v)))


def edge_label_type(u: int, v: int, labels_dict: Mapping[int, int]) -> str:
    """Categorize a pair by endpoint labels."""
    label_u = labels_dict.get(int(u))
    label_v = labels_dict.get(int(v))
    if label_u == 0 and label_v == 0:
        return "Web-Web"
    if label_u == 1 and label_v == 1:
        return "ML-ML"
    if {label_u, label_v} == {0, 1}:
        return "Web-ML"
    raise ValueError(f"Missing or invalid labels for pair ({u}, {v})")


def edge_field_group(u: int, v: int, labels_dict: Mapping[int, int]) -> str:
    """Collapse endpoint labels to the research comparison."""
    return (
        "same-field"
        if labels_dict.get(int(u)) == labels_dict.get(int(v))
        else "cross-field"
    )


def _protected_spanning_tree_edges(G: nx.Graph, seed: int) -> set[Pair]:
    """Build one reproducible spanning tree and return its protected edges."""
    if G.number_of_nodes() == 0 or not nx.is_connected(G):
        raise ValueError("Connectivity-preserving splitting requires a connected graph.")
    rng = random.Random(seed)
    root = rng.choice(sorted(int(node) for node in G.nodes()))
    protected = {canonical_pair(u, v) for u, v in nx.bfs_edges(G, root)}
    if len(protected) != G.number_of_nodes() - 1:
        raise AssertionError("The protected edge set is not a spanning tree.")
    return protected


def _sample_negative_pairs_by_type(
    G: nx.Graph,
    labels_dict: Mapping[int, int],
    counts_by_type: Mapping[str, int],
    rng: random.Random,
) -> List[Pair]:
    """Sample true non-edges matching positive counts for every edge type."""
    web_nodes = sorted(int(node) for node in G if labels_dict.get(int(node)) == 0)
    ml_nodes = sorted(int(node) for node in G if labels_dict.get(int(node)) == 1)
    if len(web_nodes) < 2 or len(ml_nodes) < 2:
        raise ValueError("Both label groups need at least two nodes.")

    pools = {"Web-Web": web_nodes, "ML-ML": ml_nodes}
    negatives: List[Pair] = []
    seen: set[Pair] = set()

    for edge_type in EDGE_TYPES:
        target = int(counts_by_type.get(edge_type, 0))
        accepted = 0
        attempts = 0
        max_attempts = max(10_000, target * 200)
        while accepted < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Could not sample {target:,} unique {edge_type} non-edges."
                )

            if edge_type == "Web-ML":
                u = web_nodes[rng.randrange(len(web_nodes))]
                v = ml_nodes[rng.randrange(len(ml_nodes))]
            else:
                pool = pools[edge_type]
                u = pool[rng.randrange(len(pool))]
                v = pool[rng.randrange(len(pool))]
                if u == v:
                    continue

            pair = canonical_pair(u, v)
            if pair in seen or G.has_edge(*pair):
                continue
            seen.add(pair)
            negatives.append(pair)
            accepted += 1

    rng.shuffle(negatives)
    return negatives


def sample_link_prediction_split(
    G: nx.Graph,
    labels_dict: Mapping[int, int],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[nx.Graph, List[Pair], List[Pair], Dict[str, int | float | bool]]:
    """Create a connected train graph and label-stratified test pairs.

    The protected spanning tree makes disconnection impossible.  Negative
    examples are sampled against the full graph, so every negative is a real
    non-edge and the positive/negative count matches within Web-Web, ML-ML and
    Web-ML strata.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1).")

    rng = random.Random(seed)
    protected = _protected_spanning_tree_edges(G, seed)
    removable = [
        canonical_pair(u, v)
        for u, v in G.edges()
        if canonical_pair(u, v) not in protected
    ]
    rng.shuffle(removable)

    requested_n_test = int(round(G.number_of_edges() * test_fraction))
    n_test = min(requested_n_test, len(removable))
    if n_test <= 0:
        raise ValueError("No non-tree edges are available for testing.")
    positive_pairs = removable[:n_test]

    G_train = G.copy()
    G_train.remove_edges_from(positive_pairs)
    assert nx.is_connected(G_train)
    isolated_nodes = list(nx.isolates(G_train))
    if isolated_nodes:
        raise AssertionError(f"Split created {len(isolated_nodes)} isolated nodes.")

    positive_type_counts = Counter(
        edge_label_type(u, v, labels_dict) for u, v in positive_pairs
    )
    negative_pairs = _sample_negative_pairs_by_type(
        G=G,
        labels_dict=labels_dict,
        counts_by_type=positive_type_counts,
        rng=rng,
    )
    negative_type_counts = Counter(
        edge_label_type(u, v, labels_dict) for u, v in negative_pairs
    )
    if positive_type_counts != negative_type_counts:
        raise AssertionError("Negative sampling did not preserve edge-type counts.")

    diagnostics: Dict[str, int | float | bool] = {
        "seed": int(seed),
        "requested_test_fraction": float(test_fraction),
        "actual_test_fraction": float(n_test / G.number_of_edges()),
        "protected_tree_edges": int(len(protected)),
        "hidden_positive_edges": int(n_test),
        "negative_pairs": int(len(negative_pairs)),
        "train_nodes": int(G_train.number_of_nodes()),
        "train_edges": int(G_train.number_of_edges()),
        "train_connected": bool(nx.is_connected(G_train)),
        "isolated_nodes": int(len(isolated_nodes)),
    }
    for edge_type in EDGE_TYPES:
        key = edge_type.lower().replace("-", "_")
        diagnostics[f"positive_{key}"] = int(positive_type_counts.get(edge_type, 0))
        diagnostics[f"negative_{key}"] = int(negative_type_counts.get(edge_type, 0))

    return G_train, positive_pairs, negative_pairs, diagnostics


def score_heuristics(
    G_train: nx.Graph,
    pairs: Sequence[Pair],
    partition: Mapping[int, int],
) -> pd.DataFrame:
    """Score pairs with course-covered structural heuristics."""
    neighbor_sets: Dict[int, set[int]] = {}

    def neighbors_of(node: int) -> set[int]:
        if node not in neighbor_sets:
            neighbor_sets[node] = (
                set(int(n) for n in G_train.neighbors(node))
                if node in G_train
                else set()
            )
        return neighbor_sets[node]

    rows = []
    for u, v in pairs:
        nbrs_u = neighbors_of(u)
        nbrs_v = neighbors_of(v)
        common = nbrs_u & nbrs_v
        union_size = len(nbrs_u | nbrs_v)
        adamic_adar = sum(
            1.0 / math.log(len(neighbors_of(w)))
            for w in common
            if len(neighbors_of(w)) > 1
        )
        rows.append(
            {
                "u": int(u),
                "v": int(v),
                "common_neighbors": int(len(common)),
                "jaccard": float(len(common) / union_size if union_size else 0.0),
                "adamic_adar": float(adamic_adar),
                "preferential_attachment": int(len(nbrs_u) * len(nbrs_v)),
                "same_community": int(
                    partition.get(int(u), -1) == partition.get(int(v), -2)
                ),
            }
        )
    return pd.DataFrame(rows)


def score_embedding_dot(w2v_model, pairs: Sequence[Pair]) -> List[float]:
    """Score a pair by the dot product of train-graph embeddings."""
    scores = []
    for u, v in pairs:
        key_u, key_v = str(int(u)), str(int(v))
        if key_u in w2v_model.wv and key_v in w2v_model.wv:
            scores.append(
                float(np.dot(w2v_model.wv[key_u], w2v_model.wv[key_v]))
            )
        else:
            scores.append(0.0)
    return scores


def auc_and_ap(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float]:
    """Return ROC-AUC and Average Precision for a non-degenerate stratum."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )
