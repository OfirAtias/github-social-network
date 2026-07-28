"""Connected, repeated edge-reconstruction experiment.

The task reconstructs randomly hidden edges in a static snapshot; it does not
forecast future link formation.  Every split protects a spanning tree, matches
negative examples by endpoint-label type, and recomputes Louvain, DeepWalk and
Node2Vec on the train graph only.
"""

from pathlib import Path
import sys
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from community_analysis import detect_louvain_communities
from data_loader import load_dataset
from embeddings import (
    generate_node2vec_walks,
    generate_random_walks,
    train_deepwalk,
    train_node2vec,
)
from link_prediction import (
    EDGE_TYPES,
    auc_and_ap,
    edge_field_group,
    edge_label_type,
    sample_link_prediction_split,
    score_embedding_dot,
    score_heuristics,
)


SEEDS = (42, 7, 21, 84, 123)
TEST_FRACTION = 0.10
LINK_NUM_WALKS = 2
LINK_WALK_LENGTH = 20
LINK_DIMENSIONS = 32
LINK_WINDOW = 5
LINK_EPOCHS = 3
NODE2VEC_P = 1.0
NODE2VEC_Q = 2.0

HEURISTICS = [
    ("common_neighbors", "Common Neighbors"),
    ("jaccard", "Jaccard"),
    ("adamic_adar", "Adamic-Adar"),
    ("preferential_attachment", "Preferential Attachment"),
    ("same_community", "Same Community"),
    ("deepwalk_dot", "DeepWalk"),
    ("node2vec_dot", "Node2Vec"),
]


def _summarize(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Aggregate mean and sample standard deviation over seeds."""
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            n_positive_mean=("n_positive", "mean"),
            n_negative_mean=("n_negative", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return summary


def _plot_overall(summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = [name for _, name in HEURISTICS]
    plot_df = summary_df.set_index("heuristic").reindex(ordered).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(plot_df))
    bars = ax.bar(
        x,
        plot_df["roc_auc_mean"],
        yerr=plot_df["roc_auc_std"],
        capsize=4,
        color="#315C8D",
    )
    for bar, value in zip(bars, plot_df["roc_auc_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(
        "Reconstruction of randomly hidden edges\n"
        "Mean and standard deviation across five connected splits"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_same_cross(summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = [name for _, name in HEURISTICS]
    x = np.arange(len(ordered))
    width = 0.36
    colors = {"same-field": "#315C8D", "cross-field": "#D57837"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
    for ax, metric, label in (
        (axes[0], "roc_auc_mean", "ROC-AUC"),
        (axes[1], "average_precision_mean", "Average Precision"),
    ):
        for offset, group in ((-width / 2, "same-field"), (width / 2, "cross-field")):
            values = (
                summary_df[summary_df["edge_group"] == group]
                .set_index("heuristic")
                .reindex(ordered)[metric]
                .to_numpy()
            )
            ax.bar(
                x + offset,
                values,
                width,
                label=group,
                color=colors[group],
            )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_ylim(0.45, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=24, ha="right")
    axes[0].set_ylabel("Score")
    axes[1].legend(title="Hidden-edge group", loc="lower right")
    fig.suptitle(
        "Direct same-field versus cross-field reconstruction\n"
        "Matched positive and negative counts within every endpoint-label stratum"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(random_states: Sequence[int] = SEEDS) -> None:
    data_dir = PROJECT_DIR / "data"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    G, _, _, labels_dict = load_dataset(data_dir)
    overall_rows: list[dict] = []
    type_rows: list[dict] = []
    group_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for run_idx, seed in enumerate(random_states, start=1):
        started = time.time()
        print(f"[edge split {run_idx}/{len(random_states)}] seed={seed}")
        G_train, positive_pairs, negative_pairs, diagnostics = (
            sample_link_prediction_split(
                G,
                labels_dict=labels_dict,
                test_fraction=TEST_FRACTION,
                seed=int(seed),
            )
        )
        diagnostics["run"] = int(run_idx)
        diagnostic_rows.append(diagnostics)
        print(
            f"  hidden={len(positive_pairs):,}; train_connected="
            f"{diagnostics['train_connected']}; isolated={diagnostics['isolated_nodes']}"
        )

        partition, _ = detect_louvain_communities(
            G_train, random_state=int(seed)
        )

        deepwalk_walks = generate_random_walks(
            G_train,
            num_walks=LINK_NUM_WALKS,
            walk_length=LINK_WALK_LENGTH,
            seed=int(seed),
        )
        deepwalk_model = train_deepwalk(
            deepwalk_walks,
            dimensions=LINK_DIMENSIONS,
            window=LINK_WINDOW,
            epochs=LINK_EPOCHS,
            seed=int(seed),
        )
        del deepwalk_walks

        node2vec_walks = generate_node2vec_walks(
            G_train,
            num_walks=LINK_NUM_WALKS,
            walk_length=LINK_WALK_LENGTH,
            p=NODE2VEC_P,
            q=NODE2VEC_Q,
            seed=int(seed),
        )
        node2vec_model = train_node2vec(
            node2vec_walks,
            dimensions=LINK_DIMENSIONS,
            window=LINK_WINDOW,
            epochs=LINK_EPOCHS,
            seed=int(seed),
        )
        del node2vec_walks

        positive_scores = score_heuristics(
            G_train, positive_pairs, partition
        )
        negative_scores = score_heuristics(
            G_train, negative_pairs, partition
        )
        positive_scores["deepwalk_dot"] = score_embedding_dot(
            deepwalk_model, positive_pairs
        )
        negative_scores["deepwalk_dot"] = score_embedding_dot(
            deepwalk_model, negative_pairs
        )
        positive_scores["node2vec_dot"] = score_embedding_dot(
            node2vec_model, positive_pairs
        )
        negative_scores["node2vec_dot"] = score_embedding_dot(
            node2vec_model, negative_pairs
        )
        positive_scores["is_edge"] = 1
        negative_scores["is_edge"] = 0
        scores_df = pd.concat(
            [positive_scores, negative_scores], ignore_index=True
        )
        scores_df["edge_type"] = [
            edge_label_type(u, v, labels_dict)
            for u, v in zip(scores_df["u"], scores_df["v"])
        ]
        scores_df["edge_group"] = [
            edge_field_group(u, v, labels_dict)
            for u, v in zip(scores_df["u"], scores_df["v"])
        ]

        for score_column, heuristic_name in HEURISTICS:
            auc, ap = auc_and_ap(
                scores_df["is_edge"].to_numpy(),
                scores_df[score_column].to_numpy(dtype=np.float64),
            )
            overall_rows.append(
                {
                    "run": int(run_idx),
                    "random_state": int(seed),
                    "heuristic": heuristic_name,
                    "n_positive": int(scores_df["is_edge"].sum()),
                    "n_negative": int((1 - scores_df["is_edge"]).sum()),
                    "roc_auc": auc,
                    "average_precision": ap,
                }
            )

            for edge_type in EDGE_TYPES:
                subset = scores_df[scores_df["edge_type"] == edge_type]
                auc_t, ap_t = auc_and_ap(
                    subset["is_edge"].to_numpy(),
                    subset[score_column].to_numpy(dtype=np.float64),
                )
                type_rows.append(
                    {
                        "run": int(run_idx),
                        "random_state": int(seed),
                        "heuristic": heuristic_name,
                        "edge_type": edge_type,
                        "n_positive": int(subset["is_edge"].sum()),
                        "n_negative": int((1 - subset["is_edge"]).sum()),
                        "roc_auc": auc_t,
                        "average_precision": ap_t,
                    }
                )

            for edge_group in ("same-field", "cross-field"):
                subset = scores_df[scores_df["edge_group"] == edge_group]
                auc_g, ap_g = auc_and_ap(
                    subset["is_edge"].to_numpy(),
                    subset[score_column].to_numpy(dtype=np.float64),
                )
                group_rows.append(
                    {
                        "run": int(run_idx),
                        "random_state": int(seed),
                        "heuristic": heuristic_name,
                        "edge_group": edge_group,
                        "n_positive": int(subset["is_edge"].sum()),
                        "n_negative": int((1 - subset["is_edge"]).sum()),
                        "roc_auc": auc_g,
                        "average_precision": ap_g,
                    }
                )
        print(f"  completed in {time.time() - started:.1f}s")

    overall_df = pd.DataFrame(overall_rows)
    type_df = pd.DataFrame(type_rows)
    group_df = pd.DataFrame(group_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows).sort_values("run")

    overall_summary = _summarize(overall_df, ["heuristic"])
    type_summary = _summarize(type_df, ["heuristic", "edge_type"])
    group_summary = _summarize(group_df, ["heuristic", "edge_group"])

    group_pivot = group_summary.pivot(
        index="heuristic", columns="edge_group"
    )
    gaps_df = pd.DataFrame(
        {
            "heuristic": group_pivot.index,
            "roc_auc_same_minus_cross": (
                group_pivot[("roc_auc_mean", "same-field")]
                - group_pivot[("roc_auc_mean", "cross-field")]
            ),
            "average_precision_same_minus_cross": (
                group_pivot[("average_precision_mean", "same-field")]
                - group_pivot[("average_precision_mean", "cross-field")]
            ),
        }
    ).reset_index(drop=True)

    outputs = {
        "edge_reconstruction_results.csv": overall_df,
        "edge_reconstruction_summary.csv": overall_summary,
        "edge_reconstruction_by_type.csv": type_df,
        "edge_reconstruction_by_type_summary.csv": type_summary,
        "edge_reconstruction_same_vs_cross.csv": group_df,
        "edge_reconstruction_same_vs_cross_summary.csv": group_summary,
        "edge_reconstruction_same_vs_cross_gaps.csv": gaps_df,
        "edge_split_diagnostics.csv": diagnostics_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(tables_dir / filename, index=False)

    _plot_overall(
        overall_summary, figures_dir / "edge_reconstruction_auc.png"
    )
    _plot_same_cross(
        group_summary,
        figures_dir / "edge_reconstruction_same_vs_cross.png",
    )

    print("\nOverall reconstruction summary:")
    print(
        overall_summary[
            ["heuristic", "roc_auc_mean", "roc_auc_std",
             "average_precision_mean", "average_precision_std"]
        ].to_string(index=False)
    )
    print("\nSame-field minus cross-field gaps:")
    print(gaps_df.to_string(index=False))


if __name__ == "__main__":
    main()
