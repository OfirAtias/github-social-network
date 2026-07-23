# -*- coding: utf-8 -*-
"""Fail-fast final-project validation used by both CLI and notebook."""

from pathlib import Path
import json
import math
import re
from typing import Any
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
from pypdf import PdfReader

from data_loader import load_dataset
from link_prediction import sample_link_prediction_split


SEEDS = (42, 7, 21, 84, 123)
NOTEBOOK_RELATIVE_PATH = Path(
    "notebooks/GitHub_Developer_Field_Prediction_Network_Only.ipynb"
)
REQUIRED_TABLES = (
    "data_quality_checks.csv",
    "network_basic_stats.csv",
    "label_distribution.csv",
    "edge_type_counts.csv",
    "centrality_top_developers.csv",
    "centrality_top_overlap.csv",
    "community_summary.csv",
    "community_metrics.csv",
    "classification_results.csv",
    "classification_summary.csv",
    "threshold_tuning_results.csv",
    "threshold_tuning_summary.csv",
    "ablation_model_results.csv",
    "ablation_model_results_summary.csv",
    "embedding_classification_results.csv",
    "embedding_classification_summary.csv",
    "edge_reconstruction_results.csv",
    "edge_reconstruction_summary.csv",
    "edge_reconstruction_by_type.csv",
    "edge_reconstruction_by_type_summary.csv",
    "edge_reconstruction_same_vs_cross.csv",
    "edge_reconstruction_same_vs_cross_summary.csv",
    "edge_reconstruction_same_vs_cross_gaps.csv",
    "edge_split_diagnostics.csv",
    "null_model_homophily.csv",
    "null_model_modularity.csv",
    "network_vs_random_models.csv",
    "powerlaw_fit.csv",
)
REQUIRED_FIGURES = (
    "label_distribution.png",
    "degree_distribution.png",
    "edge_type_distribution.png",
    "sampled_network_by_label.png",
    "community_size_distribution.png",
    "community_size_and_composition.png",
    "classification_comparison.png",
    "threshold_tradeoff.png",
    "ablation_f1_auc_comparison.png",
    "representation_comparison.png",
    "embedding_tsne.png",
    "edge_reconstruction_auc.png",
    "edge_reconstruction_same_vs_cross.png",
    "homophily_null_distribution.png",
    "degree_distribution_loglog.png",
)
REQUIRED_EMBEDDINGS = ("deepwalk_64.npz", "node2vec_64.npz")

EXPECTED_CLASSIFICATION_MODELS = {
    "Majority Baseline",
    "Neighbor Majority",
    "Logistic Regression - Graph Features Only",
    "Logistic Regression - Graph + Community Features",
}
EXPECTED_ABLATION_MODELS = {
    "Logistic Regression - Content Features Only",
    "Logistic Regression - Network Only (Graph + Community)",
    "Logistic Regression - Hybrid (Network + Content)",
}
EXPECTED_EMBEDDING_MODELS = {
    "Logistic Regression - DeepWalk",
    "Logistic Regression - DeepWalk + Graph + Community",
    "Logistic Regression - Node2Vec",
    "Logistic Regression - Node2Vec + Graph + Community",
}
FORBIDDEN_PUBLIC_PATTERNS = (
    r"\bGCN\b",
    r"\bGraphSAGE\b",
    r"\bLabel\s+Propagation\b",
    r"\bTruncatedSVD\b",
    r"\bNMI\b",
    r"\bARI\b",
    r"\bSilhouette\b",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def collect_key_results(project_dir: Path) -> dict[str, Any]:
    """Collect the exact numerical claims embedded in public artifacts."""
    tables_dir = project_dir / "outputs" / "tables"
    basic = pd.read_csv(tables_dir / "network_basic_stats.csv").iloc[0]
    classification = pd.read_csv(
        tables_dir / "classification_summary.csv"
    )
    best_classification = classification.loc[
        classification["roc_auc_mean"].idxmax()
    ]
    embeddings = pd.read_csv(
        tables_dir / "embedding_classification_summary.csv"
    ).set_index("model")
    reconstruction = pd.read_csv(
        tables_dir / "edge_reconstruction_summary.csv"
    )
    best_reconstruction = reconstruction.loc[
        reconstruction["roc_auc_mean"].idxmax()
    ]
    groups = pd.read_csv(
        tables_dir / "edge_reconstruction_same_vs_cross_summary.csv"
    )
    best_groups = groups[
        groups["heuristic"] == best_reconstruction["heuristic"]
    ].set_index("edge_group")
    homophily_null = pd.read_csv(
        tables_dir / "null_model_homophily.csv"
    ).set_index("null_model")
    modularity_null = pd.read_csv(
        tables_dir / "null_model_modularity.csv"
    ).iloc[0]
    return {
        "network_nodes": int(basic["num_nodes"]),
        "network_edges": int(basic["num_edges"]),
        "homophily_rate": float(basic["homophily_rate"]),
        "best_classification_model": str(best_classification["model"]),
        "best_classification_roc_auc": float(
            best_classification["roc_auc_mean"]
        ),
        "deepwalk_roc_auc": float(
            embeddings.loc[
                "Logistic Regression - DeepWalk", "roc_auc_mean"
            ]
        ),
        "node2vec_roc_auc": float(
            embeddings.loc[
                "Logistic Regression - Node2Vec", "roc_auc_mean"
            ]
        ),
        "best_reconstruction_heuristic": str(
            best_reconstruction["heuristic"]
        ),
        "best_reconstruction_roc_auc": float(
            best_reconstruction["roc_auc_mean"]
        ),
        "best_reconstruction_same_field_roc_auc": float(
            best_groups.loc["same-field", "roc_auc_mean"]
        ),
        "best_reconstruction_cross_field_roc_auc": float(
            best_groups.loc["cross-field", "roc_auc_mean"]
        ),
        "label_shuffle_empirical_p": float(
            homophily_null.loc["label_shuffle", "empirical_p"]
        ),
        "degree_rewire_empirical_p": float(
            homophily_null.loc[
                "degree_preserving_rewire", "empirical_p"
            ]
        ),
        "modularity_empirical_p": float(modularity_null["empirical_p"]),
    }


def _check_required_files(project_dir: Path) -> None:
    for filename in REQUIRED_TABLES:
        path = project_dir / "outputs" / "tables" / filename
        _assert(path.is_file() and path.stat().st_size > 0, f"Missing table: {path}")
    for filename in REQUIRED_FIGURES:
        path = project_dir / "outputs" / "figures" / filename
        _assert(path.is_file() and path.stat().st_size > 0, f"Missing figure: {path}")
    for filename in REQUIRED_EMBEDDINGS:
        path = project_dir / "outputs" / "embeddings" / filename
        _assert(path.is_file() and path.stat().st_size > 0, f"Missing embedding: {path}")


def _check_csvs(project_dir: Path) -> None:
    tables_dir = project_dir / "outputs" / "tables"
    for filename in REQUIRED_TABLES:
        frame = pd.read_csv(tables_dir / filename)
        _assert(not frame.empty, f"CSV is empty: {filename}")
        numeric = frame.select_dtypes(include=[np.number])
        _assert(
            np.isfinite(numeric.to_numpy(dtype=float)).all(),
            f"NaN or Inf found in {filename}",
        )
        for column in numeric.columns:
            lower = column.lower()
            values = numeric[column].to_numpy(dtype=float)
            is_gap = "gap" in lower or "_minus_" in lower
            if is_gap:
                _assert(
                    ((-1 <= values) & (values <= 1)).all(),
                    f"Out-of-range [-1,1] gap {column} in {filename}",
                )
            elif lower.startswith("loglik_ratio"):
                # Likelihood-ratio statistics are signed and unbounded.
                continue
            elif any(
                token in lower
                for token in (
                    "roc_auc",
                    "average_precision",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "purity",
                    "ratio",
                    "fraction",
                )
            ):
                _assert(
                    ((0 <= values) & (values <= 1)).all(),
                    f"Out-of-range [0,1] metric {column} in {filename}",
                )
            if "modularity" in lower or "assortativity" in lower:
                _assert(
                    ((-1 <= values) & (values <= 1)).all(),
                    f"Out-of-range [-1,1] metric {column} in {filename}",
                )


def _check_graph_and_splits(project_dir: Path) -> None:
    G, edges_df, labels_df, labels_dict = load_dataset(project_dir / "data")
    _assert(G.number_of_nodes() == 37_700, "Unexpected node count.")
    _assert(G.number_of_edges() == 289_003, "Unexpected edge count.")
    _assert(nx.is_connected(G), "Observed graph is not connected.")
    _assert(nx.number_of_selfloops(G) == 0, "Self-loops found.")
    _assert(
        len(edges_df) == G.number_of_edges(),
        "Duplicate undirected edge rows found.",
    )
    _assert(not labels_df["id"].duplicated().any(), "Duplicate label IDs found.")
    _assert(not labels_df["label"].isna().any(), "Missing labels found.")
    _assert(set(G.nodes()) == set(labels_df["id"]), "Graph/label node mismatch.")

    diagnostics = pd.read_csv(
        project_dir / "outputs" / "tables" / "edge_split_diagnostics.csv"
    ).set_index("seed")
    _assert(set(diagnostics.index.astype(int)) == set(SEEDS), "Split seeds mismatch.")
    _assert(diagnostics["train_connected"].astype(bool).all(), "Disconnected train split.")
    _assert((diagnostics["isolated_nodes"] == 0).all(), "Isolates found in train split.")
    for edge_type in ("web_web", "ml_ml", "web_ml"):
        _assert(
            (
                diagnostics[f"positive_{edge_type}"]
                == diagnostics[f"negative_{edge_type}"]
            ).all(),
            f"Unmatched negative counts for {edge_type}.",
        )

    # Recompute the split invariant rather than trusting the diagnostic CSV.
    for seed in SEEDS:
        train_graph, _, _, split_diagnostics = sample_link_prediction_split(
            G,
            labels_dict,
            test_fraction=0.10,
            seed=seed,
        )
        _assert(nx.is_connected(train_graph), f"Seed {seed} disconnected.")
        _assert(
            split_diagnostics["isolated_nodes"] == 0,
            f"Seed {seed} created isolates.",
        )


def _check_empirical_p_values(project_dir: Path) -> None:
    tables_dir = project_dir / "outputs" / "tables"
    homophily = pd.read_csv(tables_dir / "null_model_homophily.csv")
    modularity = pd.read_csv(tables_dir / "null_model_modularity.csv")
    p_values = pd.concat(
        [homophily["empirical_p"], modularity["empirical_p"]],
        ignore_index=True,
    )
    _assert(((p_values > 0) & (p_values <= 1)).all(), "Invalid empirical p-value.")
    for _, row in homophily.iterrows():
        minimum = 1.0 / (int(row["n_samples"]) + 1)
        _assert(
            row["empirical_p"] + 1e-15 >= minimum,
            "Empirical p-value violates the finite-simulation correction.",
        )


def _check_model_sets(project_dir: Path) -> None:
    tables_dir = project_dir / "outputs" / "tables"
    classification = set(
        pd.read_csv(tables_dir / "classification_summary.csv")["model"]
    )
    ablation = set(
        pd.read_csv(tables_dir / "ablation_model_results_summary.csv")["model"]
    )
    embeddings = set(
        pd.read_csv(tables_dir / "embedding_classification_summary.csv")["model"]
    )
    _assert(classification == EXPECTED_CLASSIFICATION_MODELS, "Classification models mismatch.")
    _assert(ablation == EXPECTED_ABLATION_MODELS, "Ablation models mismatch.")
    _assert(embeddings == EXPECTED_EMBEDDING_MODELS, "Embedding models mismatch.")

    logical_models = set(classification) | set(embeddings) | set(ablation)
    logical_models.discard(
        "Logistic Regression - Network Only (Graph + Community)"
    )
    _assert(len(logical_models) == 10, f"Expected 10 logical models, got {len(logical_models)}.")

    forbidden_filenames = re.compile(
        r"(gnn|subfield|label[_ -]?propagation)", re.IGNORECASE
    )
    for path in (project_dir / "outputs").rglob("*"):
        if path.is_file():
            _assert(
                not forbidden_filenames.search(path.name),
                f"Removed-model output remains: {path.name}",
            )


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        chunks = []
        for name in archive.namelist():
            if name.startswith("word/") and name.endswith(".xml"):
                chunks.append(archive.read(name).decode("utf-8", errors="ignore"))
        return "\n".join(chunks)


def _check_public_artifacts(project_dir: Path) -> None:
    notebook_path = project_dir / NOTEBOOK_RELATIVE_PATH
    report_source = project_dir / "report" / "report_source.docx"
    report_pdf = project_dir / "report" / "final_report.pdf"
    _assert(notebook_path.is_file(), "Notebook is missing.")
    _assert(report_source.is_file(), "Editable report source is missing.")
    _assert(report_pdf.is_file(), "Final report PDF is missing.")
    pdf_reader = PdfReader(str(report_pdf))
    _assert(1 <= len(pdf_reader.pages) <= 8, "Final report exceeds eight pages.")

    public_texts = {
        "README.md": (project_dir / "README.md").read_text(encoding="utf-8"),
        "notebook": notebook_path.read_text(encoding="utf-8"),
        "report_source.docx": _docx_text(report_source),
        "final_report.pdf": "\n".join(
            page.extract_text() or "" for page in pdf_reader.pages
        ),
    }
    for label, text in public_texts.items():
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            _assert(
                re.search(pattern, text, flags=re.IGNORECASE) is None,
                f"Forbidden method remains in {label}: {pattern}",
            )
        _assert(
            re.search(r"חיזוי\s+עתידי|future\s+link", text, flags=re.IGNORECASE)
            is None,
            f"Future-prediction wording remains in {label}.",
        )


def _check_notebook(
    project_dir: Path, verify_execution: bool
) -> None:
    notebook_path = project_dir / NOTEBOOK_RELATIVE_PATH
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    embedded = notebook.get("metadata", {}).get(
        "final_project_expected_results"
    )
    _assert(isinstance(embedded, dict) and embedded, "Notebook expected-results metadata missing.")
    actual = collect_key_results(project_dir)
    _assert(set(embedded) == set(actual), "Notebook result keys do not match CSV result keys.")
    for key, expected in embedded.items():
        observed = actual[key]
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            _assert(
                math.isclose(float(expected), float(observed), rel_tol=1e-7, abs_tol=1e-9),
                f"Notebook/CSV mismatch for {key}: {expected} != {observed}",
            )
        else:
            _assert(expected == observed, f"Notebook/CSV mismatch for {key}.")

    if verify_execution:
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        _assert(code_cells, "Notebook has no code cells.")
        _assert(
            all(cell.get("execution_count") is not None for cell in code_cells),
            "Notebook contains unexecuted code cells.",
        )
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        _assert(not errors, "Notebook contains error outputs.")
        final_text = json.dumps(
            code_cells[-1].get("outputs", []), ensure_ascii=False
        )
        _assert(
            "ALL FINAL PROJECT VALIDATIONS PASSED" in final_text,
            "Notebook final validation success message is missing.",
        )


def run_all_validations(
    project_dir: str | Path | None = None,
    *,
    verify_notebook_execution: bool = True,
) -> None:
    """Run every final-project validation and print success only at the end."""
    root = (
        Path(project_dir).resolve()
        if project_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    _check_required_files(root)
    _check_csvs(root)
    _check_graph_and_splits(root)
    _check_empirical_p_values(root)
    _check_model_sets(root)
    _check_public_artifacts(root)
    _check_notebook(root, verify_notebook_execution)
    print("ALL FINAL PROJECT VALIDATIONS PASSED")


if __name__ == "__main__":
    run_all_validations()
