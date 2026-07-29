"""Leakage-safe full-batch GCN and GraphSAGE implementations.

Both models are evaluated transductively: the complete graph and label-free
node covariates are available during training.  Supervision is restricted to
explicitly visible labels.  Hidden labels must be encoded as ``-1`` and are
rejected if they appear outside the declared training/validation indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import f1_score, roc_auc_score


def _scipy_to_torch(matrix: sparse.spmatrix) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced float32 tensor."""
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def build_normalized_adjacency(
    graph: nx.Graph, node_order: Sequence[int]
) -> torch.Tensor:
    """Return ``D^-1/2 (A + I) D^-1/2`` for a GCN."""
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=list(node_order), dtype=np.float32, format="csr"
    )
    adjacency = adjacency + sparse.eye(adjacency.shape[0], dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    normalized = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    return _scipy_to_torch(normalized)


def build_mean_adjacency(
    graph: nx.Graph, node_order: Sequence[int]
) -> torch.Tensor:
    """Return row-normalized ``D^-1 A`` for GraphSAGE mean aggregation."""
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=list(node_order), dtype=np.float32, format="csr"
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = np.power(np.maximum(degree, 1.0), -1.0)
    normalized = sparse.diags(inv_degree) @ adjacency
    return _scipy_to_torch(normalized)


class GCN(nn.Module):
    """Two-layer full-batch graph convolutional network."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        n_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, n_classes)
        self.dropout = float(dropout)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.sparse.mm(adjacency, features)
        hidden = F.relu(self.linear1(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = torch.sparse.mm(adjacency, hidden)
        return self.linear2(hidden)


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE with full-neighborhood mean aggregation."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        n_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features * 2, hidden)
        self.linear2 = nn.Linear(hidden * 2, n_classes)
        self.dropout = float(dropout)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        neighbor_mean = torch.sparse.mm(adjacency, features)
        hidden = F.relu(self.linear1(torch.cat((features, neighbor_mean), dim=1)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        neighbor_mean = torch.sparse.mm(adjacency, hidden)
        return self.linear2(torch.cat((hidden, neighbor_mean), dim=1))


@dataclass
class GNNFitResult:
    """Predictions and model-selection diagnostics from one fit."""

    probabilities: np.ndarray
    best_epoch: int
    epochs_ran: int
    best_validation_loss: float
    history: list[dict[str, float]]


def _validate_visible_labels(
    visible_labels: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray | None,
) -> None:
    """Fail closed if a hidden node label was accidentally exposed."""
    labels = np.asarray(visible_labels, dtype=np.int64)
    train = np.asarray(train_idx, dtype=np.int64)
    validation = (
        np.asarray(validation_idx, dtype=np.int64)
        if validation_idx is not None
        else np.empty(0, dtype=np.int64)
    )
    allowed = np.zeros(labels.shape[0], dtype=bool)
    allowed[train] = True
    allowed[validation] = True
    if not np.all(np.isin(labels[allowed], (0, 1))):
        raise ValueError("Every declared train/validation node must have a visible binary label.")
    if np.any(labels[~allowed] != -1):
        raise ValueError("Labels outside train/validation must be hidden with value -1.")
    if np.intersect1d(train, validation).size:
        raise ValueError("Training and validation indices overlap.")


def fit_gnn(
    adjacency: torch.Tensor,
    features: np.ndarray,
    visible_labels: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray | None = None,
    *,
    architecture: str = "gcn",
    hidden: int = 64,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 30,
    fixed_epochs: int | None = None,
    seed: int = 42,
) -> GNNFitResult:
    """Fit one model using only explicitly visible labels.

    When ``validation_idx`` is supplied, validation loss selects the stopping
    epoch with patience-based early stopping.  For the final outer-training
    refit, pass ``validation_idx=None`` and the selected ``fixed_epochs``.
    """
    if architecture not in {"gcn", "graphsage"}:
        raise ValueError("architecture must be 'gcn' or 'graphsage'.")
    if fixed_epochs is not None and validation_idx is not None:
        raise ValueError("fixed_epochs is only for a validation-free final refit.")

    labels = np.asarray(visible_labels, dtype=np.int64)
    train = np.asarray(train_idx, dtype=np.int64)
    validation = (
        np.asarray(validation_idx, dtype=np.int64)
        if validation_idx is not None
        else None
    )
    _validate_visible_labels(labels, train, validation)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)

    x_tensor = torch.as_tensor(features, dtype=torch.float32)
    y_tensor = torch.as_tensor(labels, dtype=torch.long)
    train_tensor = torch.as_tensor(train, dtype=torch.long)
    validation_tensor = (
        torch.as_tensor(validation, dtype=torch.long)
        if validation is not None
        else None
    )

    model_class = GCN if architecture == "gcn" else GraphSAGE
    model = model_class(
        in_features=x_tensor.shape[1],
        hidden=int(hidden),
        dropout=float(dropout),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    counts = np.bincount(labels[train], minlength=2).astype(np.float64)
    class_weights = torch.as_tensor(
        counts.sum() / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32
    )

    epochs_limit = int(fixed_epochs if fixed_epochs is not None else max_epochs)
    best_epoch = epochs_limit
    best_validation_loss = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_ran = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, epochs_limit + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(adjacency, x_tensor)
        loss = F.cross_entropy(
            logits[train_tensor], y_tensor[train_tensor], weight=class_weights
        )
        loss.backward()
        optimizer.step()
        epochs_ran = epoch

        if validation_tensor is None:
            history_rows.append(
                {
                    "epoch": float(epoch),
                    "training_loss": float(loss.item()),
                    "validation_loss": float("nan"),
                    "validation_macro_f1": float("nan"),
                    "validation_roc_auc": float("nan"),
                }
            )
            continue

        model.eval()
        with torch.no_grad():
            validation_logits = model(adjacency, x_tensor)[validation_tensor]
            validation_loss = F.cross_entropy(
                validation_logits,
                y_tensor[validation_tensor],
                weight=class_weights,
            ).item()
            validation_probabilities = F.softmax(validation_logits, dim=1)[:, 1].cpu().numpy()
        validation_truth = labels[validation]
        validation_predictions = (validation_probabilities >= 0.5).astype(np.int64)
        validation_macro_f1 = f1_score(
            validation_truth, validation_predictions, average="macro", zero_division=0
        )
        try:
            validation_roc_auc = roc_auc_score(validation_truth, validation_probabilities)
        except ValueError:
            validation_roc_auc = float("nan")
        history_rows.append(
            {
                "epoch": float(epoch),
                "training_loss": float(loss.item()),
                "validation_loss": float(validation_loss),
                "validation_macro_f1": float(validation_macro_f1),
                "validation_roc_auc": float(validation_roc_auc),
            }
        )
        if validation_loss < best_validation_loss - 1e-7 or best_state is None:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(adjacency, x_tensor), dim=1).cpu().numpy()

    return GNNFitResult(
        probabilities=probabilities.astype(np.float64),
        best_epoch=int(best_epoch),
        epochs_ran=int(epochs_ran),
        best_validation_loss=float(best_validation_loss),
        history=history_rows,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper that still hides labels outside train/val."""
    visible = np.full(len(labels), -1, dtype=np.int64)
    visible[train_idx] = np.asarray(labels, dtype=np.int64)[train_idx]
    visible[val_idx] = np.asarray(labels, dtype=np.int64)[val_idx]
    result = fit_gnn(
        a_hat,
        features,
        visible,
        train_idx,
        val_idx,
        architecture=architecture,
        hidden=hidden,
        learning_rate=lr,
        weight_decay=weight_decay,
        max_epochs=epochs,
        patience=epochs,
        seed=seed,
    )
    scores = result.probabilities[np.asarray(test_idx, dtype=np.int64), 1]
    return (scores >= 0.5).astype(np.int64), scores
