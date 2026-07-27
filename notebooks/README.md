# Research Notebook Series

This is the executed English research workflow for Noa Yarin Levy, Jimi Halawi, and Ofir Atias. The numbered notebooks keep the project code, results, figures, research questions, and main findings in a clean submission order.

## Execution order

| Order | Notebook | Purpose |
|---:|---|---|
| 01 | [`01_environment_and_reproducibility.ipynb`](01_environment_and_reproducibility.ipynb) | Workspace, Dependencies, and Reproducibility |
| 02 | [`02_data_loading_and_preparation.ipynb`](02_data_loading_and_preparation.ipynb) | Data Loading and Base Objects |
| 03 | [`03_cleaning_eda_and_graph_statistics.ipynb`](03_cleaning_eda_and_graph_statistics.ipynb) | Data Quality, Exploratory Analysis, and Graph Statistics |
| 04 | [`04_communities_and_problem_definition.ipynb`](04_communities_and_problem_definition.ipynb) | Louvain Communities and Operational Problem Definition |
| 05 | [`05_node_classification_algorithms.ipynb`](05_node_classification_algorithms.ipynb) | Node-Classification Algorithms and Repeated Evaluation |
| 06 | [`06_embeddings_content_and_ablation.ipynb`](06_embeddings_content_and_ablation.ipynb) | DeepWalk, Node2Vec, Content Features, and Ablation |
| 07 | [`07_link_reconstruction_experiments.ipynb`](07_link_reconstruction_experiments.ipynb) | Static Link Reconstruction: Protocol, Algorithms, and Experiments |
| 08 | [`08_null_models_results_and_research_answers.ipynb`](08_null_models_results_and_research_answers.ipynb) | Null Models, Statistical Significance, and Research Answers |
| 09 | [`09_EXTENTIONS_and_validation.ipynb`](09_EXTENTIONS_and_validation.ipynb) | EXTENTIONS: Optional Work and Project Checks |

`GitHub_Developer_Field_Prediction_Network_Only.ipynb` is retained as a consolidated compatibility notebook. Its unique explanations have been merged into the numbered series without duplicating identical workflow calls.

## Full execution on Windows PowerShell

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:PYTHONHASHSEED="0"
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.11 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
New-Item -ItemType Directory -Force .\tmp\jupyter-runtime | Out-Null
.\.venv\Scripts\python.exe -m ipykernel install --prefix .\tmp\jupyter-runtime --name github-social-network --display-name "GitHub Social Network (.venv)"
$env:JUPYTER_PATH = (Resolve-Path .\tmp\jupyter-runtime\share\jupyter).Path
$analysis = Get-ChildItem .\notebooks -Filter "*.ipynb" | Where-Object Name -Match "^0[1-8]_" | Sort-Object Name
$executionOrder = @($analysis.FullName)
$executionOrder += (Resolve-Path ".\notebooks\GitHub_Developer_Field_Prediction_Network_Only.ipynb").Path
$executionOrder += (Resolve-Path ".\notebooks\09_EXTENTIONS_and_validation.ipynb").Path
foreach ($notebookPath in $executionOrder) {
    .\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=github-social-network --ExecutePreprocessor.timeout=-1 $notebookPath
}
```

The compatibility notebook is kept for comparison with the original single-notebook version. Every notebook locates the repository root automatically. Results are written to `outputs/tables`, `outputs/figures`, and `outputs/embeddings`. Each code cell has short local context before it and a brief result note after it.
