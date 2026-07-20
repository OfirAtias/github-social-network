# -*- coding: utf-8 -*-
"""
Link prediction over the mutual-follow network.

Protocol (per split):
1. Hide a random fraction of edges as positive test pairs.
2. Sample an equal number of non-adjacent pairs as negative test pairs.
3. Score all test pairs with classical neighborhood heuristics computed on
   the TRAIN graph only (the hidden edges are removed before any computation,
   so no information about them can leak into the scores).

The label-stratified analysis (same-label vs cross-label edges) connects
link prediction back to the homophily finding: if same-label edges are more
predictable, the same force that enables node classification also shapes
edge formation.
"""
import math
import random
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


Pair = Tuple[int, int]


def sample_link_prediction_split(
    G: nx.Graph,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[nx.Graph, List[Pair], List[Pair]]:
    """
    Hide a random fraction of edges and sample matched negative pairs.

    Returns:
        G_train: graph with the positive test edges removed
        positive_pairs: hidden edges
        negative_pairs: uniformly sampled non-adjacent node pairs (checked
                        against the FULL graph, so they are true non-edges)
    """
    rng = random.Random(seed)

    edges = list(G.edges())
    rng.shuffle(edges)
    n_test = int(len(edges) * test_fraction)
    positive_pairs = [(int(u), int(v)) for u, v in edges[:n_test]]

    G_train = G.copy()
    G_train.remove_edges_from(positive_pairs)

    nodes = list(G.nodes())
    negative_pairs: List[Pair] = []
    seen = set()
    while len(negative_pairs) < n_test:
        u = nodes[rng.randrange(len(nodes))]
        v = nodes[rng.randrange(len(nodes))]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen or G.has_edge(u, v):
            continue
        seen.add(key)
        negative_pairs.append((int(u), int(v)))

    return G_train, positive_pairs, negative_pairs


def score_heuristics(
    G_train: nx.Graph,
    pairs: Sequence[Pair],
    partition: Dict[int, int],
) -> pd.DataFrame:
    """
    Score node pairs with classical heuristics on the train graph.

    Heuristics: Common Neighbors, Jaccard, Adamic-Adar, Preferential
    Attachment, and a same-community indicator (Louvain on the train graph).
    """
    neighbor_sets: Dict[int, set] = {}

    def neighbors_of(node: int) -> set:
        if node not in neighbor_sets:
            neighbor_sets[node] = set(G_train.neighbors(node)) if node in G_train else set()
        return neighbor_sets[node]

    rows = []
    for u, v in pairs:
        nbrs_u = neighbors_of(u)
        nbrs_v = neighbors_of(v)
        common = nbrs_u & nbrs_v
        union_size = len(nbrs_u | nbrs_v)

        adamic_adar = 0.0
        for w in common:
            degree_w = len(neighbors_of(w))
            if degree_w > 1:
                adamic_adar += 1.0 / math.log(degree_w)

        rows.append(
            {
                "u": u,
                "v": v,
                "common_neighbors": len(common),
                "jaccard": len(common) / union_size if union_size else 0.0,
                "adamic_adar": adamic_adar,
                "preferential_attachment": len(nbrs_u) * len(nbrs_v),
                "same_community": int(partition.get(u, -1) == partition.get(v, -2)),
            }
        )

    return pd.DataFrame(rows)


def score_embedding_dot(w2v_model, pairs: Sequence[Pair]) -> List[float]:
    """
    Score pairs by the dot product of their DeepWalk embeddings.

    The embeddings must be trained on the TRAIN graph only (after removing
    the hidden test edges), so the score contains no leaked information.
    Nodes unseen during training (isolated in the train graph) score 0.
    """
    scores = []
    for u, v in pairs:
        key_u, key_v = str(int(u)), str(int(v))
        if key_u in w2v_model.wv and key_v in w2v_model.wv:
            scores.append(float(np.dot(w2v_model.wv[key_u], w2v_model.wv[key_v])))
        else:
            scores.append(0.0)
    return scores


def edge_label_type(u: int, v: int, labels_dict: Dict[int, int]) -> str:
    """Categorize a pair by its endpoint labels: Web-Web, ML-ML or Web-ML."""
    label_u = labels_dict.get(int(u))
    label_v = labels_dict.get(int(v))
    if label_u == 0 and label_v == 0:
        return "Web-Web"
    if label_u == 1 and label_v == 1:
        return "ML-ML"
    return "Web-ML"


def auc_and_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """ROC-AUC and Average Precision, NaN-safe for degenerate strata."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
