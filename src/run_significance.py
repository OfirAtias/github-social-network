# -*- coding: utf-8 -*-
"""
Paired significance tests over the repeated-split results.

All experiments were run on the exact same stratified splits
(random_states 42, 7, 21, 84, 123), so per-split scores are paired samples.
For each model comparison we report:

- mean paired difference and its 95% CI (t-distribution)
- paired t-test p-value (primary, valid for n=5)
- Wilcoxon signed-rank p-value (reported for completeness; note that with
  n=5 pairs its two-sided p-value cannot go below 0.0625)

Outputs:
    outputs/tables/significance_tests.csv
    outputs/figures/significance_forest_plot.png
"""
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent


METRICS = ["f1", "macro_f1", "roc_auc"]

# (baseline model, improved model, short label for plots)
COMPARISONS = [
    ("Majority Baseline", "Neighbor Majority", "Neighbor vs Majority"),
    ("Neighbor Majority", "Label Propagation", "LabelProp vs Neighbor"),
    ("Label Propagation", "Logistic Regression - Graph Features Only", "GraphFeat vs LabelProp"),
    (
        "Logistic Regression - Graph Features Only",
        "Logistic Regression - Graph + Community Features",
        "+Community vs GraphFeat",
    ),
    (
        "Logistic Regression - Network Only (Graph + Community)",
        "Logistic Regression - Content Features Only",
        "Content vs Network",
    ),
    (
        "Logistic Regression - Content Features Only",
        "Logistic Regression - Hybrid (Network + Content)",
        "Hybrid vs Content",
    ),
    (
        "Logistic Regression - Network Only (Graph + Community)",
        "Logistic Regression - Hybrid (Network + Content)",
        "Hybrid vs Network",
    ),
    (
        "Logistic Regression - Graph + Community Features",
        "Logistic Regression - DeepWalk Embeddings Only",
        "DeepWalk vs Handcrafted",
    ),
    (
        "Logistic Regression - DeepWalk Embeddings Only",
        "Logistic Regression - Content Features Only",
        "Content vs DeepWalk (gap test)",
    ),
    (
        "Logistic Regression - DeepWalk Embeddings Only",
        "Logistic Regression - DeepWalk + Graph + Community Features",
        "DW+Handcrafted vs DeepWalk",
    ),
    (
        "Logistic Regression - DeepWalk Embeddings Only",
        "GCN - DeepWalk Input",
        "GCN(DW) vs DeepWalk",
    ),
    (
        "Logistic Regression - DeepWalk + Graph + Community Features",
        "GCN - DeepWalk Input",
        "GCN(DW) vs DW+Handcrafted (tie test)",
    ),
]


def load_all_results(tables_dir: Path) -> pd.DataFrame:
    """Load per-run results of the original models and the ablation models."""
    frames = []
    for filename in (
        "repeated_model_results_with_communities.csv",
        "ablation_model_results.csv",
        "deepwalk_model_results.csv",
        "gnn_model_results.csv",
    ):
        path = tables_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        frames.append(pd.read_csv(path))

    results_df = pd.concat(frames, ignore_index=True)
    return results_df


def paired_comparison(
    results_df: pd.DataFrame,
    model_baseline: str,
    model_improved: str,
    metric: str,
) -> dict:
    """Paired stats for improved - baseline on a shared set of random states."""
    base = (
        results_df[results_df["model"] == model_baseline]
        .set_index("random_state")[metric]
        .sort_index()
    )
    improved = (
        results_df[results_df["model"] == model_improved]
        .set_index("random_state")[metric]
        .sort_index()
    )

    shared_states = base.index.intersection(improved.index)
    if len(shared_states) < 2:
        raise ValueError(
            f"Not enough shared splits between '{model_baseline}' and '{model_improved}'."
        )

    diffs = (improved.loc[shared_states] - base.loc[shared_states]).to_numpy(dtype=np.float64)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    sem = std_diff / np.sqrt(n)
    ci_half = float(stats.t.ppf(0.975, df=n - 1) * sem)

    t_stat, t_pvalue = stats.ttest_rel(improved.loc[shared_states], base.loc[shared_states])

    try:
        _, w_pvalue = stats.wilcoxon(diffs)
    except ValueError:
        w_pvalue = float("nan")

    return {
        "n_splits": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_low": mean_diff - ci_half,
        "ci95_high": mean_diff + ci_half,
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "wilcoxon_pvalue": float(w_pvalue),
    }


def plot_forest(tests_df: pd.DataFrame, metric: str, ax: plt.Axes) -> None:
    """Horizontal mean-difference plot with 95% CI per comparison."""
    subset = tests_df[tests_df["metric"] == metric].reset_index(drop=True)
    y = np.arange(len(subset))[::-1]

    for yi, (_, row) in zip(y, subset.iterrows()):
        color = "#2c7a2c" if row["t_pvalue"] < 0.05 else "#a0a0a0"
        ax.errorbar(
            row["mean_diff"],
            yi,
            xerr=[[row["mean_diff"] - row["ci95_low"]], [row["ci95_high"] - row["mean_diff"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            linewidth=2,
            markersize=6,
        )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(subset["comparison"])
    ax.set_xlabel(f"Mean paired difference in {metric} (95% CI)")
    ax.set_title(metric)
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"

    results_df = load_all_results(tables_dir)

    rows = []
    for model_baseline, model_improved, label in COMPARISONS:
        for metric in METRICS:
            comparison_stats = paired_comparison(results_df, model_baseline, model_improved, metric)
            rows.append(
                {
                    "comparison": label,
                    "baseline_model": model_baseline,
                    "improved_model": model_improved,
                    "metric": metric,
                    **comparison_stats,
                }
            )

    tests_df = pd.DataFrame(rows)
    tests_path = tables_dir / "significance_tests.csv"
    tests_df.to_csv(tests_path, index=False)
    print(f"Saved: {tests_path}\n")

    display_cols = ["comparison", "metric", "mean_diff", "ci95_low", "ci95_high", "t_pvalue", "wilcoxon_pvalue"]
    print(tests_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    plot_forest(tests_df, "macro_f1", axes[0])
    plot_forest(tests_df, "roc_auc", axes[1])
    fig.suptitle(
        "Paired Model Comparisons over Shared Splits\n"
        "(green = paired t-test p < 0.05; CI crossing 0 means no reliable difference)",
        fontsize=12,
    )
    fig.tight_layout()
    figure_path = figures_dir / "significance_forest_plot.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {figure_path}")


if __name__ == "__main__":
    main()
