# GitHub Developer Field from a Mutual-Follow Network

Final project for the Graph Algorithms course.

This project studies an undirected network of 37,700 GitHub developers and 289,003 mutual-follow edges. It asks how much information about developer field, Web or Machine Learning, is present in neighborhoods, centrality, communities, graph embeddings, and sparse content features.

The main analysis uses course methods. Optional Label Propagation, graph-neural-network, and content-subfield experiments remain isolated under `extensions/` and are not mixed into the main result tables.

## Research Questions

1. **RQ1 - Node classification:** Can network neighborhoods, centrality, communities, and learned graph representations classify developer field without using test labels?
2. **RQ2 - Network versus content:** How much predictive value comes from network structure compared with content features, and what does their combination add?
3. **RQ3 - Static edge reconstruction:** Can randomly hidden mutual-follow edges be reconstructed, and are same-field edges easier to reconstruct than cross-field edges?
4. **RQ4 - Null models:** Are observed homophily and modularity unusual compared with label shuffling and degree-preserving graphs?

RQ3 reconstructs hidden edges from one static snapshot. It is not a time-based forecast and does not identify a causal link-formation mechanism.

## Data

The project uses the June 2019 MUSAE GitHub Social Network:

- `data/musae_git_edges.csv`: 289,003 undirected mutual-follow edges.
- `data/musae_git_target.csv`: one binary target for each of 37,700 nodes.
- `data/musae_git_features.json`: sparse content features for the network/content comparison.

The graph used in the notebooks has one connected component, no self-loops, no duplicate undirected edges, and complete target coverage.

## Methods and Main Findings

- Graph and community features classify developer field well; the strongest handcrafted model is logistic regression with graph and community features, with mean ROC-AUC near 0.869.
- Content features alone reach ROC-AUC near 0.889, while the hybrid network-content model reaches about 0.898.
- DeepWalk and Node2Vec are strong learned representations; combining them with handcrafted structure reaches ROC-AUC near 0.900.
- Preferential Attachment is the strongest static edge-reconstruction method, with mean ROC-AUC near 0.924.
- Observed homophily is about 84.5%. Both homophily and Louvain modularity are unusually high under the tested null models.
- Classification errors are additionally analyzed by node degree and community purity, showing where structural evidence is weakest.

These findings are descriptive and predictive, not causal.

## Notebook Series

Run exactly these nine notebooks in order:

| Order | Notebook | Purpose |
|---:|---|---|
| 01 | `01_environment_and_reproducibility.ipynb` | Environment and input inventory |
| 02 | `02_data_loading_and_preparation.ipynb` | Validated graph construction |
| 03 | `03_cleaning_eda_and_graph_statistics.ipynb` | Data quality, EDA, and graph statistics |
| 04 | `04_communities_and_problem_definition.ipynb` | Louvain communities and leakage-safe features |
| 05 | `05_node_classification_algorithms.ipynb` | Classification, threshold tuning, and subgroup errors |
| 06 | `06_embeddings_content_and_ablation.ipynb` | DeepWalk, Node2Vec, content, and ablation |
| 07 | `07_link_reconstruction_experiments.ipynb` | Connected static edge reconstruction |
| 08 | `08_null_models_results_and_research_answers.ipynb` | Null models and consolidated answers |
| 09 | `09_extensions_and_conclusions.ipynb` | Optional extensions and final synthesis |

## Execution

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv

# Activate the new environment in your shell, then run the remaining commands
# with that environment's python executable.
python -m pip install -r .\requirements.txt
python -m ipykernel install --user --name github-social-network --display-name "GitHub Social Network (.venv)"
$env:PYTHONHASHSEED = "0"

$notebooks = Get-ChildItem .\notebooks -Filter "[0-9][0-9]_*.ipynb" | Sort-Object Name
foreach ($notebook in $notebooks) {
    python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=github-social-network --ExecutePreprocessor.timeout=-1 $notebook.FullName
}
```

The notebooks create temporary experiment artifacts under `outputs/` while they run. Those files are reproducible from the notebooks and are not part of the submitted repository.

## Submission Structure

```text
data/              Original MUSAE inputs
notebooks/         Exactly the executed notebooks 01-09
src/               Research implementation imported by the notebooks
extensions/        Optional exploratory work outside the core claims
requirements.txt   Python dependencies
README.md          Project overview and execution guide
.gitignore         Local and generated-file exclusions
```

## Limitations

- The data are a June 2019 snapshot and may not represent GitHub today.
- The binary target may be noisy and omits finer professional identities.
- Node classification is transductive: graph structure is known during evaluation.
- Betweenness is approximated because the graph is large.
- Embedding hyperparameters use a fixed budget rather than exhaustive search.
- Hidden-edge reconstruction is static rather than temporal.
- Null models isolate selected properties and do not model every network-formation process.
