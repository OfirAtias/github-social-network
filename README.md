# GitHub Developer Field from a Mutual-Follow Network

Final project for the Graph Algorithms course.

This project studies an undirected network of 37,700 GitHub developers and 289,003 mutual-follow edges. The goal is to understand how much information about developer field, Web versus Machine Learning, is present in neighborhoods, centrality, communities, graph embeddings, and content features.

The main pipeline uses the course methods. The `extensions/` folder contains optional experiments that are documented separately and are not mixed into the main claims.

## Research Questions

1. **RQ1: Node classification.** Can network neighborhoods, centrality, communities, and learned graph representations classify developer field without using test labels?
2. **RQ2: Network versus content.** How much predictive value comes from network structure compared with content features, and what does their combination add?
3. **RQ3: Static edge reconstruction.** Can randomly hidden mutual-follow edges be reconstructed, and are same-field edges easier to reconstruct than cross-field edges?
4. **RQ4: Null models.** Are the observed homophily and modularity unusual compared with label shuffling and degree-preserving graphs?

RQ3 reconstructs hidden edges from one static snapshot. It is not a time-based forecast and does not prove how links are formed.

## Data

The project uses the June 2019 MUSAE GitHub Social Network:

- `data/musae_git_edges.csv`: 289,003 undirected mutual-follow edges.
- `data/musae_git_target.csv`: one binary label for each of the 37,700 nodes.
- `data/musae_git_features.json`: sparse content features used in the network/content comparison.

The graph has one connected component, no self-loops, no duplicate undirected edges, and complete label coverage.

## Main Methods

- Structural analysis: Degree, clustering coefficient, PageRank, sampled betweenness, HITS hub score, Louvain communities, and modularity.
- Node classification: majority baseline, neighbor-majority baseline, and logistic regression with graph, community, content, and hybrid features.
- Learned representations: DeepWalk and Node2Vec, tested alone and with handcrafted graph features.
- Link reconstruction: Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, Same Community, DeepWalk, and Node2Vec.
- Null models: label shuffling, degree-preserving rewiring, ER, BA, Watts-Strogatz, configuration model, and power-law diagnostics.

## Main Findings

- The best handcrafted node-classification model is `Logistic Regression - Graph + Community Features`, with mean ROC-AUC 0.869.
- Content features alone reach ROC-AUC 0.889, while the hybrid network-content model reaches ROC-AUC 0.898.
- DeepWalk and Node2Vec alone are close to ROC-AUC 0.891 and 0.890; adding graph and community features raises performance to about 0.900.
- Preferential Attachment is the strongest static edge-reconstruction score, with mean ROC-AUC 0.924.
- Observed homophily is high: 84.5% of labeled edges connect developers from the same field.
- The observed modularity is 0.453, compared with 0.189 on average under degree-preserving rewiring.
- The empirical p-values are 0.000999 for label shuffling, 0.047619 for degree-preserving homophily, and 0.047619 for modularity.

These findings show a strong descriptive link between developer field and network structure. They do not establish causality.

## Repository Structure

```text
data/                 Original MUSAE input files
src/                  Pipeline code, experiment runners, and artifact builders
extensions/           Optional work beyond the core pipeline
notebooks/            Final executed notebook series and notebook index
outputs/tables/       Current CSV result tables
outputs/figures/      Current saved figures
outputs/embeddings/   DeepWalk and Node2Vec vector files
report/               Final report artifacts
tests/                Focused tests for critical protocols
```

## Notebook Series

The final notebook order is documented in `notebooks/README.md`. The numbered notebooks 01-09 contain the cleaned English research workflow. The consolidated compatibility notebook, `GitHub_Developer_Field_Prediction_Network_Only.ipynb`, is also kept.

## Reproducible Run

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
$env:PYTHONHASHSEED="0"
.\.venv\Scripts\python.exe .\src\run_all.py
```

Run the final project check:

```powershell
.\.venv\Scripts\python.exe .\src\final_validation.py
```

Expected success message:

```text
ALL FINAL PROJECT VALIDATIONS PASSED
```

## Limitations

- The data are a June 2019 snapshot and may not represent GitHub today.
- The target label is binary and may contain noise.
- Node classification is transductive: the graph structure is known during evaluation.
- Betweenness is approximated because the graph is large.
- Hidden-edge reconstruction is a static task, not a time-based forecast.
- Null models isolate specific properties; they are not full models of all network formation processes.
