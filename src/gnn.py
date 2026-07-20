# -*- coding: utf-8 -*-
"""
Graph Convolutional Network (Kipf & Welling, ICLR 2017) in plain PyTorch.

Each GCN layer propagates node features over the symmetrically-normalized
adjacency with self-loops:  H' = ReLU( A_hat @ H @ W ),
A_hat = D^-1/2 (A + I) D^-1/2.

Implemented with a sparse adjacency tensor, so the full 37.7K-node graph
trains comfortably on CPU. No labels are used to build A_hat or the inputs;
supervision enters only through the cross-entropy loss on train nodes
(transductive node classification, matching the project protocol).
"""
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


def build_normalized_adjacency(G: nx.Graph, node_order: List[int]) -> torch.Tensor:
    """Build A_hat = D^-1/2 (A + I) D^-1/2 as a torch sparse tensor."""
    index_of: Dict[int, int] = {int(node): idx for idx, node in enumerate(node_order)}
    n = len(node_order)

    rows, cols = [], []
    for u, v in G.edges():
        iu, iv = index_of[int(u)], index_of[int(v)]
        rows.extend((iu, iv))
        cols.extend((iv, iu))
    rows.extend(range(n))  # self-loops
    cols.extend(range(n))

    data = np.ones(len(rows), dtype=np.float32)
    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    normalized = adjacency.multiply(inv_sqrt_degree[:, None]).multiply(inv_sqrt_degree[None, :])
    normalized = normalized.tocoo()

    indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
    values = torch.tensor(normalized.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


class GCN(nn.Module):
    """Two-layer GCN for binary node classification."""

    def __init__(self, in_features: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, a_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(a_hat, x)
        h = F.relu(self.linear1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(a_hat, h)
        return self.linear2(h)


def build_mean_adjacency(G: nx.Graph, node_order: List[int]) -> torch.Tensor:
    """
    Build the row-normalized adjacency D^-1 A (no self-loops) used by
    GraphSAGE mean aggregation: (D^-1 A) @ H = mean of neighbor features.
    """
    index_of: Dict[int, int] = {int(node): idx for idx, node in enumerate(node_order)}
    n = len(node_order)

    rows, cols = [], []
    for u, v in G.edges():
        iu, iv = index_of[int(u)], index_of[int(v)]
        rows.extend((iu, iv))
        cols.extend((iv, iu))

    data = np.ones(len(rows), dtype=np.float32)
    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = 1.0 / np.maximum(degree, 1.0)
    normalized = adjacency.multiply(inv_degree[:, None]).tocoo()

    indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
    values = torch.tensor(normalized.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


class GraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE with mean aggregation (Hamilton et al., 2017).

    Each layer concatenates the node's own representation with the mean of
    its neighbors' representations, then applies a linear transformation:
        h_v' = ReLU( W @ [h_v ; mean_{u in N(v)} h_u] )
    """

    def __init__(self, in_features: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_features * 2, hidden)
        self.linear2 = nn.Linear(hidden * 2, n_classes)
        self.dropout = dropout

    def forward(self, a_mean: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        neighbor_mean = torch.sparse.mm(a_mean, x)
        h = F.relu(self.linear1(torch.cat([x, neighbor_mean], dim=1)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        neighbor_mean = torch.sparse.mm(a_mean, h)
        return self.linear2(torch.cat([h, neighbor_mean], dim=1))


def train_gcn(
    a_hat: torch.Tensor,
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 42,
    architecture: str = "gcn",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full-batch transductive training with class weights and best-epoch
    selection on the validation macro loss. Returns (y_pred, y_score) for
    the test indices, where y_score is the ML-class probability.

    architecture: "gcn" (pass the symmetric-normalized adjacency) or
    "graphsage" (pass the row-normalized mean adjacency).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    train_t = torch.tensor(train_idx, dtype=torch.long)
    val_t = torch.tensor(val_idx, dtype=torch.long)

    class_counts = np.bincount(labels[train_idx], minlength=2).astype(np.float64)
    class_weights = torch.tensor(
        class_counts.sum() / (2.0 * np.maximum(class_counts, 1.0)), dtype=torch.float32
    )

    if architecture == "graphsage":
        model = GraphSAGE(in_features=x.shape[1], hidden=hidden)
    else:
        model = GCN(in_features=x.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(a_hat, x)
        loss = F.cross_entropy(logits[train_t], y[train_t], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(a_hat, x)
            val_loss = F.cross_entropy(logits[val_t], y[val_t], weight=class_weights).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(a_hat, x), dim=1).numpy()

    y_score = probabilities[test_idx, 1]
    y_pred = (y_score >= 0.5).astype(np.int64)
    return y_pred, y_score.astype(np.float64)
