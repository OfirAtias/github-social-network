"""Run leakage-safe node-classification extension experiments.

Label Propagation, GCN, and GraphSAGE use the same five stratified 70/30
outer splits as the core models.  Each outer training set is split 80/20 for
validation.  Validation selects propagation settings or GNN hyperparameters
and stopping epochs; the selected method is then refit with every outer-train
label, while outer-test labels remain hidden until final scoring.

GCN and GraphSAGE are *transductive* here: they use the complete graph and
label-free covariates for every node.  This runner makes no inductive claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

EXTENSIONS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXTENSIONS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(EXTENSIONS_DIR))

from content_features import build_content_matrix, load_content_features
from data_loader import load_dataset
from gnn import build_mean_adjacency, build_normalized_adjacency, fit_gnn
from label_propagation import build_transition_matrix, propagation_checkpoints
from modeling import evaluate_predictions


SEEDS = (42, 7, 21, 84, 123)
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20
SVD_COMPONENTS = 64
PROPAGATION_ALPHAS = (0.60, 0.80, 0.90, 0.99)
PROPAGATION_ITERATIONS = (5, 10, 25, 50, 100)
GNN_CONFIGS = (
    {"hidden": 32, "dropout": 0.30, "learning_rate": 0.010, "weight_decay": 5e-4},
    {"hidden": 64, "dropout": 0.50, "learning_rate": 0.005, "weight_decay": 5e-4},
)
MAX_EPOCHS = 200
PATIENCE = 30
METRICS = ("roc_auc", "macro_f1", "f1", "balanced_accuracy", "precision", "recall")
EXTENSION_MODELS = ("Label Propagation", "GCN", "GraphSAGE")
CORE_MODELS = (
    "Logistic Regression - Hybrid (Network + Content)",
    "Logistic Regression - DeepWalk + Graph + Community",
)


def structural_input_features(graph: nx.Graph, node_order: Sequence[int]) -> np.ndarray:
    """Compute five label-free structural covariates."""
    clustering = nx.clustering(graph)
    core = nx.core_number(graph)
    triangles = nx.triangles(graph)
    pagerank = nx.pagerank(graph)
    return np.asarray(
        [
            (
                np.log1p(graph.degree(node)),
                clustering.get(node, 0.0),
                float(core.get(node, 0)),
                np.log1p(triangles.get(node, 0)),
                pagerank.get(node, 0.0),
            )
            for node in node_order
        ],
        dtype=np.float32,
    )


def build_gnn_inputs(
    graph: nx.Graph, labels_df: pd.DataFrame, data_dir: Path
) -> tuple[list[int], np.ndarray, np.ndarray, dict[str, float]]:
    """Build label-free structural + sparse-content-SVD inputs."""
    node_order = sorted(labels_df["id"].astype(int).tolist())
    labels = (
        labels_df.set_index("id").loc[node_order, "label"].to_numpy(dtype=np.int64)
    )
    structural = structural_input_features(graph, node_order)
    features_dict, n_features = load_content_features(
        data_dir / "musae_git_features.json"
    )
    sparse_content = build_content_matrix(features_dict, n_features, node_order)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEEDS[0])
    content_embedding = normalize(svd.fit_transform(sparse_content)).astype(np.float32)
    inputs = np.hstack((structural, content_embedding)).astype(np.float32)
    metadata = {
        "nodes": int(len(node_order)),
        "raw_content_dimensions": int(n_features),
        "svd_components": int(SVD_COMPONENTS),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "gnn_input_dimensions": int(inputs.shape[1]),
    }
    return node_order, labels, inputs, metadata


def _visible_labels(
    labels: np.ndarray, *visible_index_groups: np.ndarray
) -> np.ndarray:
    visible = np.full(labels.shape[0], -1, dtype=np.int64)
    for group in visible_index_groups:
        indices = np.asarray(group, dtype=np.int64)
        visible[indices] = labels[indices]
    return visible


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (np.asarray(scores) >= 0.5).astype(np.int64)
    values = evaluate_predictions(y_true, predictions, scores)
    return {metric: float(values[metric]) for metric in METRICS}


def _index_digest(values: np.ndarray) -> str:
    payload = np.sort(np.asarray(values, dtype=np.int64)).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("model", as_index=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "model" if column == ("model", "") else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary


def _select_row(rows: list[dict]) -> dict:
    """Rank candidates by validation Macro-F1, ROC-AUC, then lower loss."""
    def ranking(row: dict) -> tuple[float, float, float, int]:
        loss = float(row.get("validation_loss", float("inf")))
        if not np.isfinite(loss):
            loss = float("inf")
        return (
            float(row["validation_macro_f1"]),
            float(row["validation_roc_auc"]),
            -loss,
            -int(row["candidate_id"]),
        )

    return max(
        rows,
        key=ranking,
    )


def _load_core_results(tables_dir: Path) -> pd.DataFrame:
    """Load the two strongest complementary core baselines on the same splits."""
    ablation_path = PROJECT_DIR / "outputs" / "tables" / "ablation_model_results.csv"
    embedding_path = (
        PROJECT_DIR / "outputs" / "tables" / "embedding_classification_results.csv"
    )
    missing = [str(path) for path in (ablation_path, embedding_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Core comparison tables are missing; execute notebooks 05-06 first: "
            + ", ".join(missing)
        )
    core = pd.concat((pd.read_csv(ablation_path), pd.read_csv(embedding_path)))
    core = core[core["model"].isin(CORE_MODELS)].copy()
    expected = {(seed, model) for seed in SEEDS for model in CORE_MODELS}
    observed = set(zip(core["random_state"].astype(int), core["model"]))
    if observed != expected:
        raise ValueError("Core baseline rows do not match the five fixed extension splits.")
    core.loc[:, ["model", "run", "random_state", *METRICS]].to_csv(
        tables_dir / "core_baseline_results_same_splits.csv", index=False
    )
    return core.loc[:, ["model", "run", "random_state", *METRICS]]


def _paired_deltas(extension: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for extension_model in EXTENSION_MODELS:
        extension_rows = extension[extension["model"] == extension_model].set_index(
            "random_state"
        )
        for core_model in CORE_MODELS:
            core_rows = core[core["model"] == core_model].set_index("random_state")
            for metric in METRICS:
                delta = extension_rows[metric] - core_rows[metric]
                rows.append(
                    {
                        "extension_model": extension_model,
                        "core_model": core_model,
                        "metric": metric,
                        "mean_paired_delta": float(delta.mean()),
                        "std_paired_delta": float(delta.std()),
                        "wins_out_of_5": int((delta > 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    model_order = [*CORE_MODELS, *EXTENSION_MODELS]
    short_names = {
        CORE_MODELS[0]: "Core hybrid",
        CORE_MODELS[1]: "Core DeepWalk\n+ graph/community",
        "Label Propagation": "Label\nPropagation",
        "GCN": "GCN",
        "GraphSAGE": "GraphSAGE",
    }
    colors = ["#7A7A7A", "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    titles = {
        "roc_auc": "ROC-AUC",
        "macro_f1": "Macro-F1",
        "f1": "F1 (ML class)",
        "balanced_accuracy": "Balanced accuracy",
        "precision": "Precision",
        "recall": "Recall",
    }
    for ax, metric in zip(axes.ravel(), METRICS):
        means = [comparison.loc[model, metric].mean() for model in model_order]
        stds = [comparison.loc[model, metric].std() for model in model_order]
        bars = ax.bar(
            [short_names[model] for model in model_order],
            means,
            yerr=stds,
            capsize=3,
            color=colors,
        )
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
            )
        ax.set_title(titles[metric])
        ax.set_ylim(0.0 if metric in {"precision", "recall", "f1"} else 0.45, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Extension classification versus strongest core baselines\n"
        "Mean +/- sample standard deviation over the same five fixed splits",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(random_states: Sequence[int] = SEEDS) -> dict[str, Path]:
    """Execute all three classification extensions and write their artifacts."""
    started_total = time.perf_counter()
    random_states = tuple(int(seed) for seed in random_states)
    if random_states != SEEDS:
        raise ValueError(f"Use the fixed core splits exactly as declared: {SEEDS}.")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    data_dir = PROJECT_DIR / "data"
    output_root = PROJECT_DIR / "outputs" / "extensions"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    for directory in (tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    graph, _, labels_df, _ = load_dataset(data_dir)
    node_order, labels, raw_inputs, input_metadata = build_gnn_inputs(
        graph, labels_df, data_dir
    )
    index_of = {node: index for index, node in enumerate(node_order)}
    transition = build_transition_matrix(graph, node_order)
    gcn_adjacency = build_normalized_adjacency(graph, node_order)
    sage_adjacency = build_mean_adjacency(graph, node_order)

    results_rows: list[dict] = []
    validation_rows: list[dict] = []
    selected_rows: list[dict] = []
    leakage_rows: list[dict] = []
    split_rows: list[dict] = []
    runtime_rows: list[dict] = []
    gnn_history_rows: list[dict] = []

    all_nodes = np.asarray(node_order, dtype=np.int64)
    all_labels_for_stratification = labels.copy()

    for run, seed in enumerate(random_states, start=1):
        print(f"[extension split {run}/5] seed={seed}")
        outer_train_nodes, test_nodes = train_test_split(
            all_nodes,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=all_labels_for_stratification,
        )
        outer_train_labels = np.asarray(
            [labels[index_of[int(node)]] for node in outer_train_nodes], dtype=np.int64
        )
        inner_train_nodes, validation_nodes = train_test_split(
            outer_train_nodes,
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=outer_train_labels,
        )
        outer_idx = np.asarray([index_of[int(node)] for node in outer_train_nodes])
        inner_idx = np.asarray([index_of[int(node)] for node in inner_train_nodes])
        validation_idx = np.asarray([index_of[int(node)] for node in validation_nodes])
        test_idx = np.asarray([index_of[int(node)] for node in test_nodes])
        validation_truth = labels[validation_idx]
        pending_test_scores: dict[str, np.ndarray] = {}

        disjoint = (
            np.intersect1d(inner_idx, validation_idx).size == 0
            and np.intersect1d(outer_idx, test_idx).size == 0
            and np.array_equal(np.sort(np.union1d(inner_idx, validation_idx)), np.sort(outer_idx))
        )
        split_rows.append(
            {
                "run": run,
                "random_state": seed,
                "inner_train_nodes": len(inner_idx),
                "validation_nodes": len(validation_idx),
                "outer_train_nodes": len(outer_idx),
                "test_nodes": len(test_idx),
                "outer_train_digest": _index_digest(outer_idx),
                "test_digest": _index_digest(test_idx),
            }
        )

        # Label Propagation: validation nodes and test nodes stay unlabeled.
        model_started = time.perf_counter()
        selection_visible_lp = _visible_labels(labels, inner_idx)
        lp_candidates: list[dict] = []
        for candidate_id, alpha in enumerate(PROPAGATION_ALPHAS, start=1):
            checkpoint_scores = propagation_checkpoints(
                transition,
                selection_visible_lp,
                inner_idx,
                alpha=alpha,
                checkpoints=PROPAGATION_ITERATIONS,
            )
            for iterations in PROPAGATION_ITERATIONS:
                metrics = _metrics(validation_truth, checkpoint_scores[iterations][validation_idx])
                row = {
                    "run": run,
                    "random_state": seed,
                    "model": "Label Propagation",
                    "candidate_id": candidate_id * 1000 + iterations,
                    "alpha": alpha,
                    "iterations": iterations,
                    "hidden": np.nan,
                    "dropout": np.nan,
                    "learning_rate": np.nan,
                    "weight_decay": np.nan,
                    "best_epoch": np.nan,
                    "validation_loss": np.nan,
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_roc_auc": metrics["roc_auc"],
                }
                lp_candidates.append(row)
                validation_rows.append(row)
        best_lp = _select_row(lp_candidates)
        final_visible_lp = _visible_labels(labels, outer_idx)
        final_lp_scores = propagation_checkpoints(
            transition,
            final_visible_lp,
            outer_idx,
            alpha=float(best_lp["alpha"]),
            checkpoints=(int(best_lp["iterations"]),),
        )[int(best_lp["iterations"])]
        pending_test_scores["Label Propagation"] = final_lp_scores[test_idx]
        selected_rows.append({**best_lp, "selected": True})
        runtime_rows.append(
            {
                "run": run,
                "random_state": seed,
                "model": "Label Propagation",
                "seconds": time.perf_counter() - model_started,
            }
        )

        # GNN validation uses an inner-train scaler and labels only for inner/validation.
        selection_scaler = StandardScaler().fit(raw_inputs[inner_idx])
        selection_inputs = selection_scaler.transform(raw_inputs).astype(np.float32)
        final_scaler = StandardScaler().fit(raw_inputs[outer_idx])
        final_inputs = final_scaler.transform(raw_inputs).astype(np.float32)
        selection_visible_gnn = _visible_labels(labels, inner_idx, validation_idx)
        final_visible_gnn = _visible_labels(labels, outer_idx)

        for model_name, architecture, adjacency in (
            ("GCN", "gcn", gcn_adjacency),
            ("GraphSAGE", "graphsage", sage_adjacency),
        ):
            model_started = time.perf_counter()
            candidates: list[dict] = []
            candidate_histories: dict[int, list[dict[str, float]]] = {}
            for candidate_id, config in enumerate(GNN_CONFIGS, start=1):
                fitted = fit_gnn(
                    adjacency,
                    selection_inputs,
                    selection_visible_gnn,
                    inner_idx,
                    validation_idx,
                    architecture=architecture,
                    **config,
                    max_epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    seed=seed,
                )
                validation_metrics = _metrics(
                    validation_truth, fitted.probabilities[validation_idx, 1]
                )
                row = {
                    "run": run,
                    "random_state": seed,
                    "model": model_name,
                    "candidate_id": candidate_id,
                    "alpha": np.nan,
                    "iterations": np.nan,
                    **config,
                    "best_epoch": fitted.best_epoch,
                    "epochs_ran": fitted.epochs_ran,
                    "validation_loss": fitted.best_validation_loss,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_roc_auc": validation_metrics["roc_auc"],
                }
                candidates.append(row)
                validation_rows.append(row)
                candidate_histories[candidate_id] = fitted.history
            best = _select_row(candidates)
            for history_row in candidate_histories[int(best["candidate_id"])]:
                gnn_history_rows.append(
                    {
                        "run": run,
                        "random_state": seed,
                        "model": model_name,
                        "candidate_id": int(best["candidate_id"]),
                        "selected_epoch": int(best["best_epoch"]),
                        "hidden": int(best["hidden"]),
                        "dropout": float(best["dropout"]),
                        "learning_rate": float(best["learning_rate"]),
                        "weight_decay": float(best["weight_decay"]),
                        **history_row,
                    }
                )
            final_fitted = fit_gnn(
                adjacency,
                final_inputs,
                final_visible_gnn,
                outer_idx,
                validation_idx=None,
                architecture=architecture,
                hidden=int(best["hidden"]),
                dropout=float(best["dropout"]),
                learning_rate=float(best["learning_rate"]),
                weight_decay=float(best["weight_decay"]),
                fixed_epochs=int(best["best_epoch"]),
                seed=seed,
            )
            pending_test_scores[model_name] = final_fitted.probabilities[test_idx, 1]
            selected_rows.append({**best, "selected": True})
            runtime_rows.append(
                {
                    "run": run,
                    "random_state": seed,
                    "model": model_name,
                    "seconds": time.perf_counter() - model_started,
                }
            )

        leakage_rows.append(
            {
                "run": run,
                "random_state": seed,
                "partitions_disjoint_and_complete": bool(disjoint),
                "selection_test_labels_visible": int(np.sum(selection_visible_gnn[test_idx] != -1)),
                "final_fit_test_labels_visible": int(np.sum(final_visible_gnn[test_idx] != -1)),
                "lp_selection_test_labels_visible": int(np.sum(selection_visible_lp[test_idx] != -1)),
                "inner_scaler_fit_nodes": int(len(inner_idx)),
                "final_scaler_fit_nodes": int(len(outer_idx)),
                "test_labels_used_only_after_fit": True,
            }
        )

        # Only after every method has selected settings and completed its final
        # refit do we materialize the outer-test outcome vector.
        test_truth = labels[test_idx]
        for model_name in EXTENSION_MODELS:
            test_metrics = _metrics(test_truth, pending_test_scores[model_name])
            results_rows.append(
                {"run": run, "random_state": seed, "model": model_name, **test_metrics}
            )
            print(
                f"  {model_name}: Macro-F1={test_metrics['macro_f1']:.4f}, "
                f"ROC-AUC={test_metrics['roc_auc']:.4f}"
            )

    results = pd.DataFrame(results_rows)
    if set(results["model"]) != set(EXTENSION_MODELS) or len(results) != 15:
        raise AssertionError("Expected exactly three extension models on five splits.")
    summary = _summarize(results)
    validation = pd.DataFrame(validation_rows)
    selected = pd.DataFrame(selected_rows)
    leakage = pd.DataFrame(leakage_rows)
    splits = pd.DataFrame(split_rows)
    runtimes = pd.DataFrame(runtime_rows)
    gnn_history = pd.DataFrame(gnn_history_rows)
    if not leakage["partitions_disjoint_and_complete"].all():
        raise AssertionError("A split overlap or coverage failure was detected.")
    visible_columns = [
        "selection_test_labels_visible",
        "final_fit_test_labels_visible",
        "lp_selection_test_labels_visible",
    ]
    if leakage[visible_columns].to_numpy().sum() != 0:
        raise AssertionError("At least one test label was exposed before final evaluation.")

    core = _load_core_results(tables_dir)
    combined = pd.concat((core, results), ignore_index=True)
    comparison_summary = _summarize(combined)
    paired = _paired_deltas(results, core)

    paths = {
        "results": tables_dir / "node_classification_results.csv",
        "summary": tables_dir / "node_classification_summary.csv",
        "validation": tables_dir / "validation_search.csv",
        "selected": tables_dir / "selected_settings.csv",
        "leakage": tables_dir / "leakage_audit.csv",
        "splits": tables_dir / "fixed_split_manifest.csv",
        "runtime": tables_dir / "classification_runtime.csv",
        "comparison": tables_dir / "classification_comparison_summary.csv",
        "paired": tables_dir / "paired_core_deltas.csv",
        "gnn_history": tables_dir / "gnn_training_history.csv",
        "figure": figures_dir / "classification_comparison.png",
    }
    for frame, key in (
        (results, "results"),
        (summary, "summary"),
        (validation, "validation"),
        (selected, "selected"),
        (leakage, "leakage"),
        (splits, "splits"),
        (runtimes, "runtime"),
        (comparison_summary, "comparison"),
        (paired, "paired"),
        (gnn_history, "gnn_history"),
    ):
        frame.to_csv(paths[key], index=False)
    _plot_comparison(combined.set_index("model"), paths["figure"])

    total_seconds = time.perf_counter() - started_total
    metadata = {
        "experiment": "classification_extensions",
        "seeds": list(SEEDS),
        "outer_test_size": TEST_SIZE,
        "inner_validation_fraction_of_outer_train": VALIDATION_SIZE,
        "decision_threshold": 0.5,
        "transductive_gnn_evaluation": True,
        "inductive_evaluation": False,
        "input": input_metadata,
        "label_propagation_alphas": list(PROPAGATION_ALPHAS),
        "label_propagation_iterations": list(PROPAGATION_ITERATIONS),
        "gnn_configs": list(GNN_CONFIGS),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "total_seconds": total_seconds,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    metadata_path = tables_dir / "classification_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = metadata_path
    print(f"Classification extensions completed in {total_seconds:.1f}s")
    print(summary.to_string(index=False))
    return paths


if __name__ == "__main__":
    main()
