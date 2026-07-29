"""Clamped label propagation with explicit hidden-label handling."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import networkx as nx
import numpy as np
from scipy import sparse


def build_transition_matrix(
    graph: nx.Graph, node_order: Sequence[int]
) -> sparse.csr_matrix:
    """Build the row-normalized adjacency used by propagation."""
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=list(node_order), dtype=np.float64, format="csr"
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]
    return (sparse.diags(inv_degree) @ adjacency).tocsr()


def propagation_checkpoints(
    transition: sparse.csr_matrix,
    visible_labels: np.ndarray,
    labeled_idx: np.ndarray,
    *,
    alpha: float,
    checkpoints: Sequence[int],
) -> dict[int, np.ndarray]:
    """Return positive-class scores at requested propagation iterations.

    Labels outside ``labeled_idx`` must be ``-1``.  Labeled rows are clamped
    after every update; unlabeled rows are weakly pulled toward the prior
    estimated from visible labels.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    requested = sorted({int(value) for value in checkpoints})
    if not requested or requested[0] <= 0:
        raise ValueError("checkpoints must contain positive iteration counts.")

    labels = np.asarray(visible_labels, dtype=np.int64)
    labeled = np.asarray(labeled_idx, dtype=np.int64)
    allowed = np.zeros(labels.shape[0], dtype=bool)
    allowed[labeled] = True
    if not np.all(np.isin(labels[labeled], (0, 1))):
        raise ValueError("Every labeled index must expose a binary label.")
    if np.any(labels[~allowed] != -1):
        raise ValueError("Labels outside labeled_idx must be hidden with value -1.")
    if transition.shape != (labels.size, labels.size):
        raise ValueError("transition and visible_labels have incompatible shapes.")

    positive_prior = float(labels[labeled].mean()) if labeled.size else 0.5
    prior = np.array((1.0 - positive_prior, positive_prior), dtype=np.float64)
    scores = np.tile(prior, (labels.size, 1))
    scores[labeled] = np.eye(2, dtype=np.float64)[labels[labeled]]

    results: dict[int, np.ndarray] = {}
    for iteration in range(1, requested[-1] + 1):
        scores = float(alpha) * (transition @ scores) + (1.0 - float(alpha)) * prior
        scores[labeled] = np.eye(2, dtype=np.float64)[labels[labeled]]
        if iteration in requested:
            results[iteration] = scores[:, 1].copy()
    return results


def label_propagation_classifier(
    graph: nx.Graph,
    labels_dict: Dict[int, int],
    train_nodes: Iterable[int],
    test_nodes: Iterable[int],
    default_label: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper; ``tol`` is retained but fixed iterations are used."""
    del tol
    node_order = [int(node) for node in graph.nodes()]
    index_of = {node: idx for idx, node in enumerate(node_order)}
    train = np.asarray([index_of[int(node)] for node in train_nodes], dtype=np.int64)
    test = np.asarray([index_of[int(node)] for node in test_nodes], dtype=np.int64)
    visible = np.full(len(node_order), -1, dtype=np.int64)
    visible[train] = np.asarray(
        [int(labels_dict[int(node_order[idx])]) for idx in train], dtype=np.int64
    )
    transition = build_transition_matrix(graph, node_order)
    scores = propagation_checkpoints(
        transition,
        visible,
        train,
        alpha=alpha,
        checkpoints=(int(max_iter),),
    )[int(max_iter)][test]
    predictions = np.where(
        scores > 0.5, 1, np.where(scores < 0.5, 0, int(default_label))
    )
    return predictions.astype(np.int64), scores.astype(np.float64)
