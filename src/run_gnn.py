# -*- coding: utf-8 -*-
"""
GNN experiment (L7): does message passing beat everything else?

Two GCN variants, both network-only, evaluated under the standard project
protocol (same seeds, stratified 70/30 splits, same metrics):

1. GCN - Structural Input : input features are cheap structure-only node
   descriptors (log-degree, clustering, core number, log-triangles, PageRank).
2. GCN - DeepWalk Input   : input features are the 128-d DeepWalk embeddings
   (requires run_deepwalk.py to have been run first).

Within each split, train nodes are further divided 90/10 into
train-inner/validation for best-epoch selection - test labels are never
touched during training.

Outputs:
    outputs/tables/gnn_model_results.csv
    outputs/tables/gnn_model_results_summary.csv
    outputs/figures/gnn_network_only_ladder.png
"""
import time
from pathlib import Path
from typing import Sequence

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from gnn import build_normalized_adjacency, train_gcn
from modeling import evaluate_predictions, prepare_train_test_split


MODEL_GCN_STRUCT = "GCN - Structural Input"
MODEL_GCN_DEEPWALK = "GCN - DeepWalk Input"


def structural_input_features(G: nx.Graph, node_order: list[int]) -> np.ndarray:
    """Cheap structure-only node features (no labels, no content)."""
    print("Computing structural input features ...")
    clustering = nx.clustering(G)
    core = nx.core_number(G)
    triangles = nx.triangles(G)
    pagerank = nx.pagerank(G)

    rows = []
    for node in node_order:
        rows.append(
            [
                np.log1p(G.degree(node)),
                clustering.get(node, 0.0),
                float(core.get(node, 0)),
                np.log1p(triangles.get(node, 0)),
                pagerank.get(node, 0.0),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def main(random_states: Sequence[int] = (42, 7, 21, 84, 123)) -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    embeddings_path = PROJECT_DIR / "outputs" / "embeddings" / "deepwalk_128.npz"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    G, _, labels_df, _ = load_dataset(data_dir)

    node_order = sorted(int(n) for n in labels_df["id"])
    index_of = {node: idx for idx, node in enumerate(node_order)}
    labels_arr = (
        labels_df.set_index("id").loc[node_order, "label"].to_numpy(dtype=np.int64)
    )

    print("Building normalized adjacency ...")
    a_hat = build_normalized_adjacency(G, node_order)

    # --- Input variant 1: structural features ---
    x_struct = StandardScaler().fit_transform(structural_input_features(G, node_order))

    # --- Input variant 2: DeepWalk embeddings ---
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"{embeddings_path} not found - run src/run_deepwalk.py first."
        )
    npz = np.load(embeddings_path)
    emb_nodes = npz["nodes"].astype(np.int64)
    emb_matrix = npz["embeddings"].astype(np.float64)
    reorder = np.array([index_of[int(n)] for n in emb_nodes])
    x_deepwalk = np.zeros_like(emb_matrix)
    x_deepwalk[reorder] = emb_matrix
    x_deepwalk = StandardScaler().fit_transform(x_deepwalk)

    results_rows = []
    for run_idx, random_state in enumerate(random_states, start=1):
        print(f"[run {run_idx}/{len(random_states)}] random_state={random_state}")

        train_nodes, test_nodes, y_train, _ = prepare_train_test_split(
            labels_df=labels_df, random_state=int(random_state)
        )
        train_inner_nodes, val_nodes = train_test_split(
            train_nodes, test_size=0.1, random_state=int(random_state), stratify=y_train
        )

        train_idx = np.array([index_of[int(n)] for n in train_inner_nodes])
        val_idx = np.array([index_of[int(n)] for n in val_nodes])
        test_idx = np.array([index_of[int(n)] for n in test_nodes])
        y_test = labels_arr[test_idx]

        for model_name, features in (
            (MODEL_GCN_STRUCT, x_struct),
            (MODEL_GCN_DEEPWALK, x_deepwalk),
        ):
            t0 = time.time()
            y_pred, y_score = train_gcn(
                a_hat=a_hat,
                features=features,
                labels=labels_arr,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=int(random_state),
            )
            metrics = evaluate_predictions(y_test, y_pred, y_score)
            results_rows.append(
                {
                    "run": run_idx,
                    "random_state": int(random_state),
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
            print(
                f"    {model_name}: F1={metrics['f1']:.4f} "
                f"MacroF1={metrics['macro_f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
                f"({time.time() - t0:.0f}s)"
            )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(tables_dir / "gnn_model_results.csv", index=False)

    metric_columns = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1", "roc_auc"]
    summary_df = results_df.groupby("model", as_index=False)[metric_columns].agg(["mean", "std"])
    summary_df.columns = [
        "model" if col == ("model", "") else f"{col[0]}_{col[1]}" for col in summary_df.columns
    ]
    summary_df.to_csv(tables_dir / "gnn_model_results_summary.csv", index=False)

    # --- Network-only ladder figure ---
    handcrafted_df = pd.read_csv(tables_dir / "repeated_model_results_with_communities.csv")
    deepwalk_df = pd.read_csv(tables_dir / "deepwalk_model_results.csv")

    ladder = [
        ("Handcrafted\n(Graph+Community)",
         handcrafted_df[handcrafted_df["model"] == "Logistic Regression - Graph + Community Features"]),
        ("DeepWalk\n+ LogReg",
         deepwalk_df[deepwalk_df["model"] == "Logistic Regression - DeepWalk Embeddings Only"]),
        ("GCN\n(structural input)", results_df[results_df["model"] == MODEL_GCN_STRUCT]),
        ("GCN\n(DeepWalk input)", results_df[results_df["model"] == MODEL_GCN_DEEPWALK]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ["macro_f1", "roc_auc"], ["Macro F1", "ROC-AUC"]):
        names = [name for name, _ in ladder]
        means = [frame[metric].mean() for _, frame in ladder]
        stds = [frame[metric].std() for _, frame in ladder]
        bars = ax.bar(names, means, yerr=stds, capsize=5,
                      color=["#4f7bd1", "#8a5fc2", "#5aa469", "#2c7a2c"])
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.008, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Network-Only Ladder: Handcrafted -> Learned -> Message Passing (mean over 5 splits)")
    fig.tight_layout()
    fig.savefig(figures_dir / "gnn_network_only_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Summary (mean over splits) ===")
    display_cols = ["model", "f1_mean", "macro_f1_mean", "roc_auc_mean"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))
    print("Saved tables and figure.")


if __name__ == "__main__":
    main()
