# GitHub Developer Field from a Mutual-Follow Network

Final project for the Graph Algorithms course.

This project studies an undirected network of 37,700 GitHub developers and 289,003 mutual-follow edges. It asks how much information about developer field—Web or Machine Learning—is present in neighborhoods, centrality, communities, graph embeddings, and sparse content features.

The main analysis uses course methods. Optional Label Propagation, GCN, GraphSAGE, and content-subfield experiments remain isolated under `extensions/`; notebook 09 executes and reports them separately, without changing the core results in notebooks 01-08 or the final report.

## Research Questions

1. **RQ1 - Node classification:** Can network neighborhoods, centrality, communities, and learned graph representations classify developer field without using test labels?
2. **RQ2 - Network versus content:** How much predictive value comes from network structure compared with content features, and what does their combination add?
3. **RQ3 - Static edge reconstruction:** Can randomly hidden mutual-follow edges be reconstructed, and are same-field edges easier to reconstruct than cross-field edges?
4. **RQ4 - Null models:** Are observed homophily and modularity unusual compared with label shuffling and degree-preserving graphs?

RQ3 reconstructs hidden edges from one static snapshot. It is not a time-based forecast and does not identify a causal link-formation mechanism.

## Data Access

The repository includes the June 2019 MUSAE GitHub Social Network inputs under `data/`:

- `musae_git_edges.csv`: 289,003 undirected mutual-follow edges.
- `musae_git_target.csv`: one binary target for each of 37,700 nodes.
- `musae_git_features.json`: sparse content features for all 37,700 nodes.

No download step is required. The loaders reject missing or non-numeric IDs, self-loops, duplicate undirected edges, duplicate target IDs, invalid labels, and graph nodes without labels. The validated graph has one connected component and complete target coverage.

The content file defines 4,005 sparse binary dimensions derived from developer profile and activity information such as starred repositories, location, employer, and email-related fields. These are attributes of developers, unlike graph features such as degree, PageRank, clustering, labeled-neighbor ratios, and community membership. The released data use encoded indices without a complete feature-name mapping, so individual indices may not have direct human-readable meanings.

## Methods and Main Findings

- Graph and community features classify developer field well; the strongest handcrafted model has mean ROC-AUC near 0.869.
- Content features alone reach ROC-AUC near 0.889, while the network–content hybrid reaches about 0.898.
- DeepWalk and Node2Vec are strong learned representations; combining them with handcrafted structure reaches ROC-AUC near 0.900.
- Preferential Attachment is the strongest static edge-reconstruction method, with mean ROC-AUC near 0.924.
- Observed homophily is about 84.5%, and observed Louvain modularity is about 0.453. Both lie above the tested null distributions, subject to the finite-simulation limitation below.
- Classification errors are analyzed by node degree and community purity. Community purity is used only for error analysis, never as a predictive feature.

These findings support a strong relationship between developer field and network structure in this dataset. They are descriptive and predictive, not causal.

## Notebook Order

Run exactly these nine notebooks in order:

| Order | Notebook | Purpose |
|---:|---|---|
| 01 | `01_environment_and_reproducibility.ipynb` | Environment, research questions, and input inventory |
| 02 | `02_data_loading_and_preparation.ipynb` | Strict validation and graph construction |
| 03 | `03_cleaning_eda_and_graph_statistics.ipynb` | Data quality, EDA, and graph statistics |
| 04 | `04_communities_and_problem_definition.ipynb` | Louvain communities and leakage-safe community features |
| 05 | `05_node_classification_algorithms.ipynb` | Classification, validation-only threshold tuning, and error analysis |
| 06 | `06_embeddings_content_and_ablation.ipynb` | DeepWalk, Node2Vec, content features, and ablation |
| 07 | `07_link_reconstruction_experiments.ipynb` | Connected static edge reconstruction |
| 08 | `08_null_models_results_and_research_answers.ipynb` | Null models and consolidated research answers |
| 09 | `09_extensions_and_conclusions.ipynb` | Executed optional extensions and final synthesis |

Every notebook includes its own imports, project-root discovery, and data loading. It can start in a fresh kernel without variables or imports from another notebook. Running in order remains recommended because it presents the research argument in sequence and regenerates all artifacts predictably.

## Installation and Reproducible Execution

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
$env:PYTHONHASHSEED = "0"

$notebooks = Get-ChildItem .\notebooks -Filter "*.ipynb" |
    Where-Object { $_.Name -match '^0[1-9]_' } |
    Sort-Object Name

foreach ($notebook in $notebooks) {
    python -m nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.kernel_name=python3 `
        --ExecutePreprocessor.timeout=-1 `
        $notebook.FullName
}
```

Each `nbconvert` call starts a new kernel. Running the loop therefore checks both clean-kernel execution and the complete 01–09 order. The notebooks save final cell outputs in place and regenerate temporary tables, figures, and embeddings under `outputs/`. Those artifacts are reproducible and are excluded from the submitted repository.

## Leakage Prevention

- Train/test splits are stratified and fixed by seed.
- Scaling, one-hot column layout, logistic regression, and other fitted preprocessing are learned from training data only.
- Probability thresholds are selected on an inner validation split only; test labels are used only for final metrics.
- Neighbor-label and community-label features receive dictionaries containing training labels only.
- Louvain membership, community size, and unsupervised node embeddings use graph structure without labels.
- Full-label community purity is used only after prediction for error analysis; it is never a model input.
- In every link-reconstruction split, test edges are removed first. Louvain, DeepWalk, and Node2Vec are then retrained on that split's training graph only.

## Project Structure

```text
data/              Original MUSAE inputs
notebooks/         Executed notebooks 01–09
src/               Reusable research implementation
extensions/        Executed optional methods, kept separate from core results
requirements.txt   Pinned Python dependencies
README.md          Project overview and execution guide
.gitignore         Local and generated-file exclusions
```

## Limitations

- The data are a June 2019 snapshot and may not represent GitHub today.
- The binary target may be noisy and omits finer professional identities.
- Node classification is transductive: graph structure is known during evaluation.
- Betweenness is approximated because the graph is large.
- Embedding hyperparameters use a fixed budget rather than exhaustive tuning.
- Hidden-edge reconstruction is static rather than temporal.
- Random non-edge sampling may make reconstruction easier and may favor degree-based methods such as Preferential Attachment. Degree-matched or distance-two negatives would provide a harder evaluation.
- The degree-preserving null uses 20 rewired graphs because rerunning both swaps and Louvain 99 times is not practical for the complete reproducibility run. With 20 samples, the smallest empirical p-value is `1/21 = 0.0476`, so the rewiring experiment has coarse p-value resolution.
- Null models isolate selected properties and do not reproduce every network-formation process.
