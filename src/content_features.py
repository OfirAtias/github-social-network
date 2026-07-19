# -*- coding: utf-8 -*-
"""
Content-feature loading and ablation models (features-only / hybrid).

The MUSAE features file (musae_git_features.json) maps each node id to a list
of sparse binary feature ids derived from starred repositories, location,
employer and e-mail. These features are content-based (not network-based),
so comparing them against the network-only models quantifies how much the
network structure itself contributes to prediction.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from modeling import _prepare_feature_matrix


def load_content_features(path: str | Path) -> Tuple[Dict[int, List[int]], int]:
    """
    Load the sparse binary content features.

    Returns:
        features_dict: node_id -> list of active feature ids
        n_features: total feature dimension (max feature id + 1)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing content features file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    features_dict = {int(node): [int(fid) for fid in fids] for node, fids in raw.items()}
    n_features = max((max(fids) for fids in features_dict.values() if fids), default=-1) + 1

    return features_dict, n_features


def build_content_matrix(
    features_dict: Dict[int, List[int]],
    n_features: int,
    nodes: Iterable[int],
) -> sparse.csr_matrix:
    """Build a sparse binary matrix whose rows are aligned to the given node order."""
    nodes_list = [int(node) for node in nodes]

    row_indices = []
    col_indices = []
    for row_idx, node in enumerate(nodes_list):
        for fid in features_dict.get(node, []):
            row_indices.append(row_idx)
            col_indices.append(fid)

    data = np.ones(len(row_indices), dtype=np.float64)
    matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(nodes_list), n_features),
    )
    return matrix


def train_logistic_regression_content_only(
    X_train: sparse.csr_matrix,
    y_train: Iterable[int],
    X_test: sparse.csr_matrix,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Train Logistic Regression on content features only (no network information)."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, np.asarray(list(y_train), dtype=np.int64))

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return model, y_pred.astype(np.int64), y_score.astype(np.float64)


def train_logistic_regression_hybrid(
    X_train_network: pd.DataFrame,
    X_train_content: sparse.csr_matrix,
    y_train: Iterable[int],
    X_test_network: pd.DataFrame,
    X_test_content: sparse.csr_matrix,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """
    Train Logistic Regression on network features (scaled dense) combined with
    content features (raw sparse binary), stacked side by side.
    """
    x_train_dense = _prepare_feature_matrix(X_train_network)
    x_test_dense = _prepare_feature_matrix(X_test_network, fit_columns=x_train_dense.columns)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_dense.to_numpy(dtype=np.float64))
    x_test_scaled = scaler.transform(x_test_dense.to_numpy(dtype=np.float64))

    x_train_combined = sparse.hstack(
        [sparse.csr_matrix(x_train_scaled), X_train_content], format="csr"
    )
    x_test_combined = sparse.hstack(
        [sparse.csr_matrix(x_test_scaled), X_test_content], format="csr"
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train_combined, np.asarray(list(y_train), dtype=np.int64))

    y_pred = model.predict(x_test_combined)
    y_score = model.predict_proba(x_test_combined)[:, 1]

    return model, y_pred.astype(np.int64), y_score.astype(np.float64)
