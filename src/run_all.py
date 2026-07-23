# -*- coding: utf-8 -*-
"""Rebuild every experiment, public output, notebook source and report source."""

import argparse
import gc
from pathlib import Path
import random
import shutil

import numpy as np

from run_ablation import main as run_ablation
from run_classification import main as run_classification
from run_core_analysis import main as run_core_analysis
from run_embeddings import main as run_embeddings
from run_link_prediction import main as run_edge_reconstruction
from run_null_models import main as run_null_models


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def _clean_outputs() -> None:
    """Delete only the resolved project outputs directory."""
    root = PROJECT_DIR.resolve()
    target = OUTPUTS_DIR.resolve()
    if target.parent != root or target.name != "outputs":
        raise RuntimeError(f"Refusing to clean unexpected path: {target}")
    if target.exists():
        shutil.rmtree(target)
    (target / "tables").mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    (target / "embeddings").mkdir(parents=True, exist_ok=True)


def main(clean: bool = True) -> None:
    random.seed(42)
    np.random.seed(42)
    if clean:
        _clean_outputs()

    stages = (
        ("core analysis", run_core_analysis),
        ("node classification", run_classification),
        ("content/network ablation", run_ablation),
        ("DeepWalk and Node2Vec", run_embeddings),
        ("connected edge reconstruction", run_edge_reconstruction),
        ("null and network models", run_null_models),
    )
    for name, function in stages:
        print(f"\n=== {name} ===")
        function()
        gc.collect()

    from build_notebook import main as build_notebook
    from build_readme import main as build_readme
    from build_report import main as build_report
    from build_report_pdf import main as build_report_pdf

    print("\n=== notebook source ===")
    build_notebook()
    print("\n=== README ===")
    build_readme()
    print("\n=== editable report source ===")
    build_report()
    print("\n=== final report PDF ===")
    build_report_pdf()
    print(
        "\nExperiments and public artifacts rebuilt. Execute the notebook, then "
        "run python src/final_validation.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove old output files before rebuilding.",
    )
    args = parser.parse_args()
    main(clean=not args.keep_existing)
