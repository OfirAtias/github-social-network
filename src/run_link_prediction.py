# -*- coding: utf-8 -*-
"""
Link prediction experiment: can we predict mutual-follow edges, and are
same-field edges easier to predict than cross-field edges?

Per split (5 seeds, matching the project protocol):
1. Hide 10% of edges, sample matched negatives.
2. Louvain is recomputed on the train graph (no leakage from hidden edges).
3. Score with Common Neighbors / Jaccard / Adamic-Adar / Preferential
   Attachment / Same-Community; evaluate ROC-AUC and Average Precision.
4. Stratify results by endpoint labels (Web-Web / ML-ML / Web-ML).

Outputs:
    outputs/tables/link_prediction_results.csv
    outputs/tables/link_prediction_results_summary.csv
    outputs/tables/link_prediction_by_edge_type.csv
    outputs/figures/link_prediction_auc.png
    outputs/figures/link_prediction_by_edge_type.png
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

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from community_analysis import detect_louvain_communities
from data_loader import load_dataset
from embeddings import generate_random_walks, train_deepwalk
from link_prediction import (
    auc_and_ap,
    edge_label_type,
    sample_link_prediction_split,
    score_embedding_dot,
    score_heuristics,
)


HEURISTICS = [
    ("common_neighbors", "Common Neighbors"),
    ("jaccard", "Jaccard"),
    ("adamic_adar", "Adamic-Adar"),
    ("preferential_attachment", "Pref. Attachment"),
    ("same_community", "Same Community"),
    ("deepwalk_dot", "DeepWalk (dot)"),
]

EDGE_TYPES = ["Web-Web", "ML-ML", "Web-ML"]


def main(random_states: Sequence[int] = (42, 7, 21, 84, 123)) -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    G, _, _, labels_dict = load_dataset(data_dir)

    overall_rows = []
    stratified_rows = []

    for run_idx, seed in enumerate(random_states, start=1):
        t0 = time.time()
        print(f"[run {run_idx}/{len(random_states)}] seed={seed}")

        G_train, positive_pairs, negative_pairs = sample_link_prediction_split(
            G, test_fraction=0.1, seed=int(seed)
        )
        print(
            f"  hidden {len(positive_pairs):,} edges; train graph: "
            f"{G_train.number_of_edges():,} edges, "
            f"{nx.number_connected_components(G_train)} components"
        )

        # Louvain on the train graph only - hidden edges must not affect it.
        partition, _ = detect_louvain_communities(G_train, random_state=42)

        # DeepWalk retrained per split on the train graph only, so the
        # embedding-based score never sees the hidden edges.
        print("  training DeepWalk on the train graph ...")
        walks = generate_random_walks(G_train, num_walks=10, walk_length=80, seed=int(seed))
        w2v_model = train_deepwalk(walks, dimensions=128, window=10, epochs=5, seed=int(seed))

        scores_pos = score_heuristics(G_train, positive_pairs, partition)
        scores_neg = score_heuristics(G_train, negative_pairs, partition)
        scores_pos["deepwalk_dot"] = score_embedding_dot(w2v_model, positive_pairs)
        scores_neg["deepwalk_dot"] = score_embedding_dot(w2v_model, negative_pairs)
        scores_pos["is_edge"] = 1
        scores_neg["is_edge"] = 0
        scores_df = pd.concat([scores_pos, scores_neg], ignore_index=True)
        scores_df["edge_type"] = [
            edge_label_type(u, v, labels_dict) for u, v in zip(scores_df["u"], scores_df["v"])
        ]

        y_true = scores_df["is_edge"].to_numpy()
        for column, heuristic_name in HEURISTICS:
            auc, ap = auc_and_ap(y_true, scores_df[column].to_numpy(dtype=np.float64))
            overall_rows.append(
                {
                    "run": run_idx,
                    "random_state": int(seed),
                    "heuristic": heuristic_name,
                    "roc_auc": auc,
                    "average_precision": ap,
                }
            )

            for edge_type in EDGE_TYPES:
                subset = scores_df[scores_df["edge_type"] == edge_type]
                auc_t, ap_t = auc_and_ap(
                    subset["is_edge"].to_numpy(),
                    subset[column].to_numpy(dtype=np.float64),
                )
                stratified_rows.append(
                    {
                        "run": run_idx,
                        "random_state": int(seed),
                        "heuristic": heuristic_name,
                        "edge_type": edge_type,
                        "n_positive": int(subset["is_edge"].sum()),
                        "n_negative": int((1 - subset["is_edge"]).sum()),
                        "roc_auc": auc_t,
                        "average_precision": ap_t,
                    }
                )

        print(f"  done in {time.time() - t0:.1f}s")

    overall_df = pd.DataFrame(overall_rows)
    stratified_df = pd.DataFrame(stratified_rows)

    overall_df.to_csv(tables_dir / "link_prediction_results.csv", index=False)
    stratified_df.to_csv(tables_dir / "link_prediction_by_edge_type.csv", index=False)

    summary_df = (
        overall_df.groupby("heuristic", as_index=False)[["roc_auc", "average_precision"]]
        .agg(["mean", "std"])
    )
    summary_df.columns = [
        "heuristic" if col == ("heuristic", "") else f"{col[0]}_{col[1]}"
        for col in summary_df.columns
    ]
    summary_df.to_csv(tables_dir / "link_prediction_results_summary.csv", index=False)

    # --- Figure 1: overall AUC per heuristic ---
    heuristic_names = [name for _, name in HEURISTICS]
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [overall_df.loc[overall_df["heuristic"] == h, "roc_auc"].mean() for h in heuristic_names]
    stds = [overall_df.loc[overall_df["heuristic"] == h, "roc_auc"].std() for h in heuristic_names]
    bars = ax.bar(heuristic_names, means, yerr=stds, capsize=5, color="#4f7bd1")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.01, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Link Prediction: ROC-AUC per Heuristic (mean over 5 splits)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "link_prediction_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: AUC per heuristic by edge type ---
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(heuristic_names))
    width = 0.25
    colors = {"Web-Web": "#4f7bd1", "ML-ML": "#e07b39", "Web-ML": "#8a8a8a"}
    for offset, edge_type in zip((-width, 0, width), EDGE_TYPES):
        means_t = [
            stratified_df[
                (stratified_df["heuristic"] == h) & (stratified_df["edge_type"] == edge_type)
            ]["roc_auc"].mean()
            for h in heuristic_names
        ]
        ax.bar(x + offset, means_t, width, label=edge_type, color=colors[edge_type])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(heuristic_names)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Are Same-Field Edges Easier to Predict?\n(ROC-AUC by endpoint label type, mean over 5 splits)")
    ax.legend(title="Edge type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "link_prediction_by_edge_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Overall summary (mean over splits) ===")
    print(summary_df.to_string(index=False))
    print("\n=== By edge type (ROC-AUC mean) ===")
    pivot = stratified_df.pivot_table(index="heuristic", columns="edge_type", values="roc_auc", aggfunc="mean")
    print(pivot.to_string())
    print("\nSaved tables and figures.")


if __name__ == "__main__":
    main()
