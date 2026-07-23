"""Optional label-propagation extension, excluded from the final project."""

from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np


def label_propagation_classifier(
    G: nx.Graph,
    labels_dict: Dict[int, int],
    train_nodes: Iterable[int],
    test_nodes: Iterable[int],
    default_label: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate clamped train labels over a row-normalized adjacency matrix."""
    train_set = {int(node) for node in train_nodes}
    test_list = [int(node) for node in test_nodes]
    train_label_dict = {
        node: int(labels_dict[node]) for node in train_set if node in labels_dict
    }
    train_ml_prior = (
        float(np.mean(list(train_label_dict.values()))) if train_label_dict else 0.5
    )

    nodelist = list(G.nodes())
    node_to_idx = {int(node): idx for idx, node in enumerate(nodelist)}
    adjacency = nx.to_scipy_sparse_array(
        G, nodelist=nodelist, dtype=float, format="csr"
    )
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    transition = adjacency.multiply(1.0 / row_sums[:, None]).tocsr()

    scores = np.full(len(nodelist), train_ml_prior, dtype=np.float64)
    fixed_mask = np.zeros(len(nodelist), dtype=bool)
    for node, label in train_label_dict.items():
        idx = node_to_idx.get(node)
        if idx is not None:
            scores[idx] = float(label)
            fixed_mask[idx] = True

    for _ in range(max_iter):
        new_scores = np.asarray(transition @ scores).ravel()
        new_scores[fixed_mask] = scores[fixed_mask]
        if float(np.max(np.abs(new_scores - scores))) < tol:
            scores = new_scores
            break
        scores = new_scores

    y_score = np.asarray(
        [scores[node_to_idx[node]] if node in node_to_idx else train_ml_prior
         for node in test_list],
        dtype=np.float64,
    )
    y_pred = np.where(
        y_score > 0.5, 1, np.where(y_score < 0.5, 0, int(default_label))
    ).astype(np.int64)
    return y_pred, y_score
