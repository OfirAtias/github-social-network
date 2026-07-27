"""Convert project notebooks to standalone professional English documentation.

The numbered notebooks contain exact, byte-for-byte copies of ``src/*.py``.
Those tagged source cells are never modified.  This module rewrites only
notebook narration and notebook-specific orchestration strings, then merges the
unique explanatory material from the legacy consolidated notebook into the
numbered research sequence.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from textwrap import dedent

import nbformat


HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
SERIES_GLOB = "[0-9][0-9]_*.ipynb"
LEGACY_NAME = "GitHub_Developer_Field_Prediction_Network_Only.ipynb"


INTRO = {
    "01_environment_and_reproducibility.ipynb": (
        "Workspace, Dependencies, and Reproducibility",
        "Verify that the repository, data files, Python environment, and required packages are available before any analysis begins.",
        "The project directory, `requirements.txt`, and the files under `data/`.",
        "Package versions, data-file inventory, cryptographic hashes, and pre-run checks.",
        "This is the reproducibility foundation shared by RQ1-RQ4.",
    ),
    "02_data_loading_and_preparation.ipynb": (
        "Data Loading and Base Objects",
        "Load the MUSAE edge list and developer labels, validate their schemas, and build the undirected graph used throughout the project.",
        "`musae_git_edges.csv`, `musae_git_target.csv`, and the dataset metadata.",
        "`G`, `edges_df`, `labels_df`, and `labels_dict`, together with quality checks and sample records.",
        "Defines the common analysis population for RQ1-RQ4 and prevents silent data-quality failures.",
    ),
    "03_cleaning_eda_and_graph_statistics.ipynb": (
        "Data Quality, Exploratory Analysis, and Graph Statistics",
        "Measure data quality, label balance, homophily, centrality, and the main descriptive properties of the network.",
        "The graph and labels created from the validated MUSAE files.",
        "Auditable CSV tables and figures for structure, centrality, label composition, and initial communities.",
        "Provides the structural evidence used by RQ1 and motivates the homophily test in RQ4.",
    ),
    "04_communities_and_problem_definition.ipynb": (
        "Louvain Communities and Operational Problem Definition",
        "Detect communities, measure modularity and purity, and construct leakage-safe community features.",
        "The complete graph and labels; supervised community ratios use training labels only.",
        "A node-to-community partition, modularity, community summaries, and node-level community features.",
        "Defines the community component of RQ1 and the observed modularity tested in RQ4.",
    ),
    "05_node_classification_algorithms.ipynb": (
        "Node-Classification Algorithms and Repeated Evaluation",
        "Implement baselines, graph and community features, logistic regression, and validation-only threshold tuning without test leakage.",
        "The graph, labels, communities, and fixed random seeds for stratified splits.",
        "Five-run results, means, standard deviations, classification metrics, and comparison figures.",
        "Answers RQ1 and establishes the controlled network baseline used in RQ2.",
    ),
    "06_embeddings_content_and_ablation.ipynb": (
        "DeepWalk, Node2Vec, Content Features, and Ablation",
        "Learn node representations from graph structure and compare network, content, and hybrid feature families under one protocol.",
        "The graph, labels, `musae_git_features.json`, and identical train/test splits.",
        "Embedding files, repeated performance tables, ablation figures, representation comparisons, and t-SNE views.",
        "Answers RQ2 and extends RQ1 with unsupervised learned graph representations.",
    ),
    "07_link_reconstruction_experiments.ipynb": (
        "Static Link Reconstruction: Protocol, Algorithms, and Experiments",
        "Hide edges while preserving connectivity, then compare structural heuristics with learned embeddings.",
        "The full graph, endpoint labels, and fixed seeds; negatives are true non-edges matched by endpoint-label type.",
        "Overall and stratified ROC-AUC and Average Precision, including same-field versus cross-field comparisons.",
        "Answers RQ3. The task reconstructs hidden edges in a static snapshot; it is not temporal forecasting.",
    ),
    "08_null_models_results_and_research_answers.ipynb": (
        "Null Models, Statistical Significance, and Research Answers",
        "Compare the observed network with label shuffles, degree-preserving rewires, and standard random graph models, then synthesize RQ1-RQ4.",
        "The original graph and the result tables produced by the classification, ablation, embedding, and link-reconstruction experiments.",
        "Empirical p-values, random-model comparisons, power-law diagnostics, and a consolidated research-answer table.",
        "Directly answers RQ4 and combines the quantitative evidence for RQ1-RQ3.",
    ),
    "09_EXTENTIONS_and_validation.ipynb": (
        "EXTENTIONS: Optional Work and Project Checks",
        "Separate optional experiments from the core claims and keep the project artifacts reproducible.",
        "All project outputs, builder and validation code, and the `extensions/` directory.",
        "A short extension inventory, reproducibility checks, and exact `src` coverage.",
        "Distinguishes the validated contribution from future extensions, as required by the project guidelines.",
    ),
}


MODULE_ROLE = {
    "data_loader.py": "This module validates the raw CSV schemas and constructs the graph objects used by every experiment.",
    "graph_analysis.py": "This module computes descriptive network statistics, label balance, edge composition, homophily, and assortativity.",
    "visualization.py": "This module creates consistent publication-ready figures with explicit titles, axes, legends, and deterministic output paths.",
    "run_core_analysis.py": "This orchestration module runs data-quality checks, descriptive analysis, centrality, Louvain analysis, and figure generation.",
    "community_analysis.py": "This module implements Louvain detection, community summaries, and leakage-safe community features.",
    "modeling.py": "This module contains the complete node-classification protocol: splits, baselines, feature extraction, logistic regression, metrics, and threshold selection.",
    "run_classification.py": "This orchestration module repeats node classification over five fixed splits and saves auditable summaries and figures.",
    "content_features.py": "This module loads sparse MUSAE content features and implements content-only and hybrid ablation models.",
    "embeddings.py": "This module implements DeepWalk and Node2Vec walks, skip-gram training, and conversion of learned vectors to feature frames.",
    "run_ablation.py": "This module compares content-only, network-only, and hybrid representations under identical splits and metrics.",
    "run_embeddings.py": "This module trains graph embeddings, evaluates them over repeated splits, and creates representation and t-SNE figures.",
    "link_prediction.py": "This module implements connectivity-preserving edge splits, matched non-edge sampling, structural scores, embedding scores, and ranking metrics.",
    "run_link_prediction.py": "This module executes the repeated static edge-reconstruction experiment and reports overall and label-stratified results.",
    "run_null_models.py": "This module implements label shuffles, degree-preserving rewires, generative graph comparisons, and power-law diagnostics.",
    "final_validation.py": "This module performs fail-fast checks on files, tables, metrics, graph splits, notebook execution, and public report artifacts.",
    "run_all.py": "This module is the authoritative end-to-end orchestration entry point for rebuilding experiments and public artifacts.",
    "build_notebook.py": "This legacy builder creates the original consolidated notebook from current result tables.",
    "build_readme.py": "This builder regenerates the project README from current results so public numerical claims do not become stale.",
    "build_report.py": "This builder creates the editable Word report with controlled layout, tables, figures, headers, and page structure.",
    "build_report_pdf.py": "This builder renders the final PDF directly from current tables and figures without requiring Microsoft Office.",
}


FUNCTION_DETAIL = {
    "load_edges": "Read the edge CSV, standardize endpoint columns, reject missing endpoints and self-loops, and preserve one row per undirected edge. The scan is linear in the number of input rows, O(E).",
    "load_targets": "Read node metadata, normalize the target to a binary `label` column, reject duplicate IDs or missing labels, and return one validated row per developer. The scan is O(V).",
    "build_graph": "Insert every validated endpoint pair into an undirected NetworkX graph. Construction requires O(V+E) time and storage.",
    "load_dataset": "Compose edge loading, target loading, graph construction, graph-node completion, and label-dictionary creation into one checked dataset API.",
    "compute_basic_stats": "Calculate node and edge counts, density, average degree, connected components, label coverage, and same-label edge rates directly from the graph and labels.",
    "compute_label_distribution": "Count each binary class and divide by the total number of labeled nodes to expose class imbalance before modeling.",
    "compute_edge_type_counts": "Classify each edge as Web-Web, ML-ML, Web-ML, or missing-label by inspecting both endpoint labels. The pass is O(E).",
    "compute_homophily": "Measure the fraction of labeled edges whose endpoints share a label. This is descriptive association, not causal evidence.",
    "compute_label_assortativity": "Create a labeled subgraph and apply categorical assortativity; an optional node subset supports train-only diagnostics without test-label leakage.",
    "detect_louvain_communities": "Run seeded Louvain modularity optimization on graph structure only. Louvain alternates local node moves with community aggregation; it is fast in practice but heuristic, seed-sensitive, and subject to the resolution limit.",
    "summarize_communities": "Group nodes by community and compute size, class counts, class ratios, dominant label, and purity for every group.",
    "add_community_features": "Attach community ID, size, global composition, and train-label-only composition to requested nodes. Restricting label-derived ratios to training nodes prevents leakage.",
    "prepare_train_test_split": "Create a seeded stratified 70/30 split of labeled node IDs so every run preserves the global class ratio.",
    "majority_class_baseline": "Predict the training-set majority class for every test node and use the training positive-class prior as a constant score. This establishes the imbalance-only baseline.",
    "neighbor_majority_classifier": "Predict from labels of training neighbors only; use the training majority and prior when a node has no labeled training neighbors.",
    "extract_graph_neighbor_features": "Compute structural features and training-neighbor label ratios for each requested node. Test labels are never used to construct a feature.",
    "_prepare_feature_matrix": "Remove identifiers, one-hot encode categorical community IDs, align columns between train and test, and return a numeric matrix suitable for scikit-learn.",
    "train_logistic_regression_graph_features": "Fit a preprocessing and logistic-regression pipeline. Logistic regression models the log odds as a linear function of standardized features and optimizes regularized log loss.",
    "evaluate_predictions": "Compute Accuracy, Balanced Accuracy, Precision, Recall, F1, Macro F1, Weighted F1, ROC-AUC, and confusion-matrix counts from one prediction set.",
    "evaluate_thresholds": "Convert probability scores to labels across candidate thresholds and evaluate each operating point with the same metric suite.",
    "run_repeated_threshold_tuning": "For every seed, split training data again into inner-train and validation sets, choose the threshold on validation F1 only, and evaluate the untouched test set once.",
    "run_repeated_experiments": "Repeat all four classification models over fixed stratified splits and store one auditable metric row per model and seed.",
    "load_content_features": "Parse the JSON mapping from node IDs to active sparse content-feature IDs and infer the full feature dimension.",
    "build_content_matrix": "Construct a CSR binary matrix whose rows follow an explicit node order; sparse storage avoids allocating a dense V-by-D matrix.",
    "train_logistic_regression_content_only": "Fit logistic regression directly on sparse content indicators without any network feature.",
    "train_logistic_regression_hybrid": "Scale dense network features, horizontally stack them with sparse content indicators, and fit one logistic-regression model on the combined representation.",
    "generate_random_walks": "Start uniform random walks from every node for DeepWalk. Runtime is O(V * num_walks * walk_length).",
    "generate_node2vec_walks": "Generate second-order walks whose transition weights depend on the previous node; p controls return probability and q controls local versus outward exploration.",
    "train_deepwalk": "Train deterministic single-worker skip-gram Word2Vec on uniform graph walks so nodes with similar walk contexts receive nearby vectors.",
    "train_node2vec": "Train the same skip-gram objective on Node2Vec-biased walks, isolating the effect of the transition policy.",
    "embeddings_to_frame": "Read each requested node vector from Word2Vec and return an ID-aligned DataFrame for downstream model joins.",
    "canonical_pair": "Sort the endpoints of an undirected pair so set membership and duplicate checks use a stable representation.",
    "edge_label_type": "Map a pair to Web-Web, ML-ML, or Web-ML from its endpoint labels.",
    "edge_field_group": "Collapse the three endpoint-label strata to same-field or cross-field for the research comparison.",
    "_protected_spanning_tree_edges": "Build a seeded spanning tree and mark its edges as protected so removing test positives cannot disconnect the training graph.",
    "_sample_negative_pairs_by_type": "Sample true non-edges against the complete graph while matching the positive count within every endpoint-label stratum.",
    "sample_link_prediction_split": "Protect a spanning tree, hide only eligible non-tree edges, sample matched true non-edges, and assert connectivity and class balance for the resulting training graph.",
    "score_heuristics": "Score each candidate pair with Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, and Same Community using the training graph only.",
    "score_embedding_dot": "Use the dot product of two train-graph embedding vectors as a similarity score for one candidate edge.",
    "auc_and_ap": "Compute ROC-AUC for global ranking quality and Average Precision for precision across recall levels; both require a non-degenerate positive/negative stratum.",
    "same_label_fraction": "Vectorize the fraction of edges whose endpoint labels match, allowing fast evaluation across many null samples.",
    "sampled_average_shortest_path": "Run BFS from a reproducible sample of source nodes and average reachable shortest-path distances to estimate the graph-wide mean.",
    "empirical_upper_tail_p": "Count null statistics at least as large as the observation and apply (b+1)/(B+1), preventing an impossible zero p-value in a finite simulation.",
    "run_all_validations": "Run every artifact, table, graph-protocol, model-set, notebook, and report check; print success only after every assertion passes.",
}


COMPLEXITY = {
    "plot_sampled_network": "The dominant work is graph sampling, layout, and drawing on at most the configured sample size, not on all nodes.",
    "_centrality_tables": "Degree is O(V+E); PageRank and HITS are iterative O(kE); sampled betweenness is approximately O(s(V+E)) for s source nodes.",
    "run_repeated_experiments": "Total cost is five times the feature extraction and model-fitting cost because the protocol uses five fixed splits.",
    "run_ablation_experiments": "Sparse content operations scale with the number of active entries, while the dense network branch scales with rows times network-feature columns.",
    "_learn_embeddings": "Walk generation dominates at O(V * walks * length); skip-gram cost then scales with the generated token-context pairs.",
    "sample_link_prediction_split": "Spanning-tree construction is O(V+E); negative sampling also depends on rejection frequency in the sparse graph.",
    "main": "This entry point executes the module's complete workflow; its runtime is the sum of the algorithms, model fits, file writes, and figures described in the surrounding module.",
}


COMMON_SETUP = """## Portable project initialization

This cell finds the project root, moves execution there, and adds `src` to the import path. It also sets deterministic defaults. Expected result: the absolute project-root path."""

COMMON_SETUP_OUTPUT = """**Result.** The printed path must end in `github-social-network`. This confirms that all relative paths point to the submitted project folder."""


SPECIAL_MD = {
    ("01_environment_and_reproducibility.ipynb", 4): """## Conditional dependency check

The cell checks every import required by `requirements.txt`. It runs `pip install -r requirements.txt` only when a dependency is missing, so a clean environment is repaired while repeated runs remain fast. Expected output: `All required imports are available`, or the package-installation log.""",
    ("01_environment_and_reproducibility.ipynb", 6): """**Output and interpretation.** The availability message confirms that no manual package correction is required. Conditional installation makes the notebook portable without reinstalling the environment on every run.""",
    ("01_environment_and_reproducibility.ipynb", 7): """## Package versions, data files, and hashes

This cell records installed package versions, file sizes, and SHA-256 hashes for the MUSAE inputs. Hashes make data provenance auditable: a third party can distinguish a changed dataset from a changed algorithm. Expected output: one version row per major dependency and one inventory row per MUSAE data file.""",
    ("01_environment_and_reproducibility.ipynb", 9): """**Output and interpretation.** The version and hash tables capture the exact execution context. They are especially useful when investigating small numerical differences in Word2Vec, linear algebra, or other stochastic and platform-dependent routines.""",
    ("02_data_loading_and_preparation.ipynb", 4): """## Data model and conservative cleaning policy

Each edge row represents a mutual follow relation, so the project models an undirected graph. `ml_target=0` denotes Web and `ml_target=1` denotes Machine Learning. Cleaning is deliberately conservative: the loader does not impute labels, silently drop nodes, or rewrite edges. Invalid schemas, duplicates, self-loops, and missing values raise explicit errors, preserving fidelity to the original MUSAE snapshot.""",
    ("02_data_loading_and_preparation.ipynb", 24): """## Full dataset load and validation display

This cell calls the complete loader, builds `networkx.Graph`, and displays sizes, connectivity, label coverage, and sample records. Expected result: 37,700 nodes, 289,003 edges, one connected component, and no graph node without a label.""",
    ("02_data_loading_and_preparation.ipynb", 26): """**Output, findings, and meaning.** The graph contains 37,700 nodes and 289,003 edges, is connected, and has complete label coverage. No imputation or giant-component filtering is needed, so RQ1-RQ4 use exactly the same population.""",
    ("03_cleaning_eda_and_graph_statistics.ipynb", 4): """## Structural measures and computational cost

- **Degree / Degree Centrality:** number of neighbors, optionally divided by |V|-1; O(V+E). It measures local exposure but strongly favors hubs.
- **PageRank:** power iteration on a random-walk transition matrix; approximately O(k(V+E)) for k iterations. Links from important neighbors receive greater weight.
- **Betweenness:** fraction of shortest paths that pass through a node. Exact computation is expensive, so the project samples 100 sources, approximately O(100(V+E)).
- **Clustering coefficient:** density of triangles in a node's neighborhood. It captures local closure but can be high for low-degree nodes.
- **HITS hub score:** spectral iteration, approximately O(kE). In an undirected graph hub and authority scores are redundant, so only one is retained.
- **Homophily:** fraction of labeled edges joining equal labels. It is descriptive association; the null-model test quantifies how unusual it is without claiming causality.""",
    ("04_communities_and_problem_definition.ipynb", 4): """## Louvain, modularity, and community purity

Louvain seeks a partition that increases

\[Q=\\frac{1}{2m}\\sum_{ij}\\left(A_{ij}-\\frac{k_i k_j}{2m}\\right)\\mathbf{1}[c_i=c_j].\]

It repeatedly moves nodes to neighboring communities when the move improves Q, then contracts communities into supernodes and repeats. The method is fast on large sparse graphs, but it is heuristic, seed-sensitive, can stop at a local optimum, and has a resolution limit. Community purity is the majority-label share inside a community; because purity is affected by class imbalance, it is descriptive and not a replacement for held-out classification metrics.""",
    ("04_communities_and_problem_definition.ipynb", 21): """## Community detection and a leakage-safe feature example

The cell loads the graph, runs Louvain with seed 42, and displays the largest communities. It then creates a stratified split and constructs community features for sample test nodes. Any ratio derived from labels uses training labels only. Expected result: 31 communities and modularity near 0.453.""",
    ("04_communities_and_problem_definition.ipynb", 23): """**Output, findings, and meaning.** Louvain finds 31 communities with Q≈0.453 and size-weighted purity≈0.830. The largest community is Web-dominant and the second-largest is ML-dominant, so community membership provides useful RQ1 signal; mixed communities show that it is not a perfect proxy for field.""",
    ("05_node_classification_algorithms.ipynb", 4): """## Classification design

Four levels are evaluated: Majority Baseline, Neighbor Majority, logistic regression on graph features, and logistic regression on graph plus community features. Logistic regression models `P(y=1|x)=σ(wᵀx+b)` and minimizes regularized log loss. It is interpretable and efficient but assumes a linear decision boundary after preprocessing. Results are means and sample standard deviations over seeds 42, 7, 21, 84, and 123. Test labels never enter features or training, and probability thresholds are selected on an inner validation split only.""",
    ("06_embeddings_content_and_ablation.ipynb", 4): """## Learned representations and controlled ablation

DeepWalk treats uniform random walks as sentences; Node2Vec changes second-order transition probabilities with p and q. Both use skip-gram to place nodes with similar walk contexts near one another. They can capture nonlinear structure without labels, but require walk generation and Word2Vec training, depend on hyperparameters, and do not naturally embed unseen nodes. The ablation holds seeds, splits, classifier, and metrics fixed while changing only the input family—content, network, or hybrid—so performance differences estimate the marginal predictive value of each source.""",
    ("07_link_reconstruction_experiments.ipynb", 4): """## Connectivity-preserving reconstruction protocol and scores

A spanning tree is protected in O(V+E), and 10% of eligible non-tree edges are hidden; the training graph therefore remains connected with no isolated nodes. Every positive is paired with a true non-edge of the same Web-Web, ML-ML, or Web-ML type. Scores are Common Neighbors |Γ(u)∩Γ(v)|, Jaccard intersection over union, Adamic-Adar with 1/log-degree weighting, Preferential Attachment d(u)d(v), Same Community, and embedding dot products. Local heuristic cost depends on neighborhood intersection; embedding scores additionally require train-graph walk training.""",
    ("08_null_models_results_and_research_answers.ipynb", 4): """## Null-model methodology

Label shuffling keeps the graph and class counts fixed and asks whether label placement alone explains homophily. Degree-preserving rewiring keeps labels and the degree sequence fixed and asks whether structure beyond degree-label correlation matters. Erdős-Rényi matches graph size and expected density; Barabási-Albert models preferential attachment; Watts-Strogatz models local clustering and short paths; the configuration null preserves the observed degree sequence. Empirical upper-tail p-values use `(b+1)/(B+1)`, so p is never zero. With 20 rewires the smallest attainable p is 1/21≈0.0476, which limits inferential resolution.""",
    ("08_null_models_results_and_research_answers.ipynb", 45): """## Consolidated answers to RQ1-RQ4

The code loads the latest summary tables and constructs one evidence table dynamically. Numerical statements therefore stay synchronized with the files produced in this execution rather than being copied manually.""",
    ("08_null_models_results_and_research_answers.ipynb", 47): """**Output, findings, and meaning.** RQ1 receives a qualified positive answer: graph structure predicts field well but imperfectly. RQ2 shows complementary information in network and content. RQ3 shows strong static reconstruction with method-dependent same/cross-field differences. RQ4 shows unusually high homophily and modularity at the available simulation resolution. These are associational results, not causal mechanisms.""",
    ("09_EXTENTIONS_and_validation.ipynb", 4): """## What counts as an EXTENTION in this project

The validated core is everything executed by `src/run_all.py` and reported in the README and final report. `extensions/` preserves work beyond that core: Label Propagation, GCN/GraphSAGE, and SVD+KMeans content subfield discovery. These directions test semi-supervision, message passing, and richer label resolution, but they are not mixed into the main result tables because some require PyTorch or additional experiment budgets and some fall outside the selected course scope. This separation prevents an exploratory result from being presented as a validated project claim.

Useful future improvements include nested cross-validation, more degree-preserving rewires for finer p-value resolution, a genuinely temporal link-forecasting dataset, probability calibration, and fairness checks beyond the binary target.

**Submission note.** The course guidelines require a factual individual-contribution statement for Noa Yarin Levy, Jimi Halawi, and Ofir Atias. The repository does not contain enough evidence to infer that division, so the notebooks do not invent it; the submitted report must include the team's verified contribution statement.""",
    ("09_EXTENTIONS_and_validation.ipynb", 5): """## Existing extension inventory

The cell reads `extensions/README.md` and inventories the Python files without executing experiments that are outside the validated pipeline. Expected output: four source files and the README explanation of their separation from the core.""",
    ("09_EXTENTIONS_and_validation.ipynb", 7): """**Output and interpretation.** The optional implementations are present and inspectable, but they do not change the answers to RQ1-RQ4. The inventory makes the boundary between the validated contribution and future exploration explicit.""",
    ("09_EXTENTIONS_and_validation.ipynb", 216): """## Validate all public project artifacts

This cell runs the original validation suite: required files, readable finite CSVs, graph and edge-split invariants, empirical p-values, exact model sets, the executed legacy notebook, and the report's eight-page limit. Expected output: `ALL FINAL PROJECT VALIDATIONS PASSED`.""",
    ("09_EXTENTIONS_and_validation.ipynb", 218): """**Output and interpretation.** The success message is printed only after every original quality condition passes. Any missing artifact or numerical contradiction raises an explicit error instead of being hidden.""",
    ("09_EXTENTIONS_and_validation.ipynb", 219): """## Prove exact byte-for-byte coverage of every `src/*.py` file

Every copied source cell stores its original path and line interval. This cell concatenates those cells in source order, compares them with the current files, and verifies that all 20 Python files are represented exactly once in the series. It is an automated test of the no-omissions requirement.""",
    ("09_EXTENTIONS_and_validation.ipynb", 221): """**Output and interpretation.** `exact_match=True` for every row proves that the notebooks preserve every source character without modifying `src`. This documentation audit complements the experimental validation suite.""",
}


RUN_INFO = {
    "03_cleaning_eda_and_graph_statistics.ipynb": [("Run the complete core-analysis entry point: quality checks, descriptive statistics, centrality, Louvain summaries, and figures.", "CSV tables under `outputs/tables` and at least six PNG figures.")],
    "05_node_classification_algorithms.ipynb": [("Run four classifiers over five stratified splits and perform validation-only threshold tuning.", "Twenty base-model rows, ten threshold rows, two summaries, and two figures.")],
    "06_embeddings_content_and_ablation.ipynb": [
        ("Run the content-only, network-only, and hybrid ablation models over the same five splits.", "Fifteen result rows, a three-model summary, and the ablation figure."),
        ("Train DeepWalk and Node2Vec, save their vectors, and evaluate four embedding-based classifiers.", "Two NPZ files, twenty result rows, a summary table, a t-SNE figure, and a representation comparison."),
    ],
    "07_link_reconstruction_experiments.ipynb": [("Run five connectivity-preserving splits, score seven methods, and save overall and stratified reconstruction results.", "Thirty-five overall rows, one diagnostic row per split, and results by edge type and same/cross-field group.")],
    "08_null_models_results_and_research_answers.ipynb": [("Run 1,000 label shuffles, 20 degree-preserving rewires, generative graph comparisons, and power-law diagnostics.", "Null-model tables, network-model comparisons, power-law results, and the null-distribution figure.")],
}


TABLE_INFO = {
    "outputs/tables/data_quality_checks.csv": ("Physical and logical data-quality checks.", "All nine checks pass: no loops, duplicates, missing labels, or disconnected components."),
    "outputs/tables/network_basic_stats.csv": ("Core descriptive network statistics.", "The graph is sparse (density≈0.000407) with mean degree≈15.33, yet 84.5% of labeled edges are same-field. That strong association motivates the null-model test."),
    "outputs/tables/centrality_top_overlap.csv": ("Top-20 overlap between centrality rankings.", "Degree and PageRank have complete top-20 overlap, while Clustering has no overlap with them; the measures capture different structural roles."),
    "outputs/tables/classification_summary.csv": ("Means and standard deviations for every node-classification model.", "Graph + Community is the strongest handcrafted model with ROC-AUC≈0.869 and F1≈0.672. The Majority Baseline has superficially high Accuracy but ML-class F1=0, so it is not a useful classifier."),
    "outputs/tables/threshold_tuning_summary.csv": ("Default threshold 0.5 versus a threshold selected on validation F1.", "The selected threshold is 0.55±0.05. Precision increases and Recall falls while F1 changes little; threshold choice changes the operating point, not ROC-AUC ranking quality."),
    "outputs/tables/ablation_model_results_summary.csv": ("Controlled comparison of content, network, and hybrid information.", "Content-only reaches ROC-AUC≈0.889, network-only≈0.869, and hybrid≈0.898. Content is stronger alone, while network structure adds complementary signal above content."),
    "outputs/tables/embedding_classification_summary.csv": ("DeepWalk and Node2Vec alone and combined with handcrafted graph features.", "DeepWalk and Node2Vec alone are near ROC-AUC 0.891; adding graph and community features raises performance to about 0.900. Similar scores do not prove equivalent representations."),
    "outputs/tables/edge_split_diagnostics.csv": ("Connectivity, isolation, and stratum-balance checks for each hidden-edge split.", "Every seed hides 28,900 edges, keeps the training graph connected, creates no isolated nodes, and exactly matches positive and negative counts within every endpoint-label stratum."),
    "outputs/tables/edge_reconstruction_summary.csv": ("Mean ROC-AUC and Average Precision over five reconstruction runs.", "Preferential Attachment leads with ROC-AUC≈0.924 and AP≈0.931, followed by Node2Vec and DeepWalk. Endpoint degree is therefore a powerful signal in this static reconstruction task."),
    "outputs/tables/edge_reconstruction_same_vs_cross_gaps.csv": ("Same-field minus cross-field performance for every method.", "The gap is not universal: Preferential Attachment has a small same-field advantage (≈0.008 AUC), while the embeddings are slightly stronger on cross-field pairs. Ease of reconstruction depends on the scoring method."),
    "outputs/tables/null_model_homophily.csv": ("Observed homophily against label-shuffle and degree-preserving nulls.", "Observed homophily is 0.845 versus 0.617 after label shuffling (p≈0.001) and 0.751 after rewiring (p≈0.0476). Excess homophily remains after controlling the degree sequence."),
    "outputs/tables/null_model_modularity.csv": ("Observed Louvain modularity against degree-preserving rewires.", "Observed Q≈0.453 versus null mean≈0.189 with p≈0.0476, so the degree sequence alone does not reproduce the community strength."),
    "outputs/tables/network_vs_random_models.csv": ("Observed network statistics versus standard generative models.", "No single model reproduces all properties: ER misses hubs and clustering; BA creates a tail but too little clustering; Watts-Strogatz overproduces clustering and path length; the configuration null matches degrees but not all local closure."),
    "outputs/tables/powerlaw_fit.csv": ("Tail-fit diagnostics for the observed degree distribution.", "The fitted exponent is α≈2.54 above xmin=55. Power law beats an exponential tail, but is not clearly distinguishable from lognormal (p≈0.783), so the evidence supports a heavy tail rather than a unique scale-free claim."),
}


FIGURE_INFO = {
    "03_cleaning_eda_and_graph_statistics.ipynb": ("The figures show class imbalance, the degree tail, edge-label composition, and a labeled network sample. Each saved figure includes a title, named axes, and a legend when multiple visual groups are present.", "About 74.2% of nodes are Web, degrees are heavy-tailed, and most edges are within-field. Accuracy alone is therefore insufficient; later tables report Balanced Accuracy, F1, and ROC-AUC."),
    "04_communities_and_problem_definition.ipynb": ("One figure shows community-size rank and the other shows label composition in the largest communities. Titles, axes, and the Web/ML legend are included in the saved images.", "A small number of large communities contain most nodes, followed by a long tail of small groups. Mixed label composition explains why community membership is informative but not perfectly predictive."),
    "05_node_classification_algorithms.ipynb": ("The first figure compares model metrics; the second shows the Precision-Recall-F1 trade-off after validation-only threshold selection. Metric names, model labels, and the threshold-policy legend identify every axis and series.", "Community features consistently improve the handcrafted model. The threshold chart separates ranking quality, measured by ROC-AUC, from a binary decision rule chosen for a particular operating trade-off."),
    "06_embeddings_content_and_ablation.ipynb": ("The figures compare information sources, representation families, and a 4,000-node t-SNE projection. Bar-chart categories and metric panels label their axes; the t-SNE axes are intentionally suppressed because their coordinates have no intrinsic units, while titles and the class legend preserve interpretation.", "Hybrid information leads the ablation and handcrafted additions slightly improve graph embeddings. t-SNE shows overlap and some field-enriched regions, but it is a two-dimensional visualization rather than a classification metric."),
    "07_link_reconstruction_experiments.ipynb": ("The overall figure and the same/cross-field figure include method labels, metric axes, error bars, titles, and a group legend.", "The overall ranking is stable across seeds. The grouped view exposes an interaction between method and edge type, preventing an overgeneralized same-field conclusion."),
    "08_null_models_results_and_research_answers.ipynb": ("The observed homophily line is plotted against 1,000 shuffled-label values and 20 degree-preserving rewires. Both panels identify the statistic, count axis, null model, and observed-value legend.", "The observation lies far in the upper tail of both null distributions. This supports statistical association between field labels and network structure, not causality."),
}


SUMMARY = {
    "01_environment_and_reproducibility.ipynb": ("Located the repository, checked or installed dependencies, and recorded package versions and data hashes.", "The environment is complete and all three MUSAE inputs are directly accessible.", "The validated inputs are ready for graph construction and every research question."),
    "02_data_loading_and_preparation.ipynb": ("Loaded edges and labels, validated schemas, and built the undirected graph.", "The data are complete, connected, and require no silent correction or imputation.", "The resulting graph and label objects are the inputs to descriptive and model-based analysis."),
    "03_cleaning_eda_and_graph_statistics.ipynb": ("Completed quality checks, EDA, centrality analysis, and structural visualizations.", "The network is sparse, heavy-tailed, class-imbalanced, and strongly homophilous in descriptive terms.", "These properties motivate community analysis, leakage-safe prediction, and explicit null-model controls."),
    "04_communities_and_problem_definition.ipynb": ("Defined Louvain, modularity, purity, and leakage-safe community features.", "Communities express strong but incomplete professional-field structure.", "These features can now be evaluated on held-out node labels rather than interpreted from training composition alone."),
    "05_node_classification_algorithms.ipynb": ("Implemented and evaluated four classification models plus validation-only threshold tuning.", "Graph + Community is the strongest handcrafted model with ROC-AUC≈0.869.", "The controlled baseline enables a fair test of content and learned graph representations."),
    "06_embeddings_content_and_ablation.ipynb": ("Trained graph embeddings and ran a controlled network/content/hybrid ablation.", "Content is strong, network structure is complementary, hybrid reaches ROC-AUC≈0.898, and learned-plus-handcrafted graph representations are near 0.900.", "The learned structure is next evaluated on a distinct edge-reconstruction task."),
    "07_link_reconstruction_experiments.ipynb": ("Ran leakage-safe static reconstruction with seven methods over five connected splits.", "Preferential Attachment leads overall; its same-field advantage is small and not shared by every method.", "Null models are required to determine whether the observed label and community structure is unusual beyond simple graph constraints."),
    "08_null_models_results_and_research_answers.ipynb": ("Compared focused null models and generative networks and built a dynamic RQ1-RQ4 answer table.", "Professional-field structure remains unusual after controlling class counts and degrees, while no simple random model reproduces every observed property.", "The validated scientific results are now ready for artifact and source-coverage checks."),
    "09_EXTENTIONS_and_validation.ipynb": ("Separated optional work, documented every builder and validation module, ran artifact validation, and verified byte-for-byte source coverage.", "The core pipeline is reproducible and internally consistent; optional experiments are clearly excluded from the core claims.", "This completes the research sequence. The commands in `notebooks/README.md` reproduce all notebooks and outputs."),
}


LEGACY_MARKDOWN = {
    0: """# Predicting GitHub Developer Fields from a Mutual-Follow Network

**Final project — Graph Algorithms**

Noa Yarin Levy · Jimi Halawi · Ofir Atias  
Instructor: Dr. Avner Priel

This project studies an undirected network of 37,700 GitHub developers and 289,003 mutual-follow links. The validated core uses course methods; sparse content features provide a controlled reference point rather than replacing network analysis.""",
    1: """## 1. Motivation and research questions

GitHub follow relationships may carry descriptive information about professional interests. The analysis does not claim causality; it measures which structural properties are associated with Web and Machine Learning labels and how much predictive information they contain.

1. **RQ1 — Node classification:** To what extent can neighborhoods, centrality, communities, and learned representations classify developer field without using test labels?
2. **RQ2 — Network versus content:** What is the marginal predictive value of network structure relative to content features, and what does their combination add?
3. **RQ3 — Static edge reconstruction:** Can randomly hidden mutual-follow edges be reconstructed with structural heuristics and learned representations, and are within-field edges easier than cross-field edges?
4. **RQ4 — Null models:** Are observed homophily and modularity unusual relative to label shuffling and degree-preserving graphs, and which properties are reproduced by ER, BA, Small-World, and Configuration models?

RQ3 reconstructs hidden edges from one static snapshot. It is neither temporal forecasting nor a model of link formation.""",
    2: """## 2. Execution environment and reproducibility

All repeated experiments use fixed seeds 42, 7, 21, 84, and 123. Word2Vec uses one worker for deterministic training. For the closest bit-level reproduction, set `PYTHONHASHSEED=0` before starting the kernel. The setup cell locates the project, registers source and output paths, and records the hash-seed state.""",
    4: """## 3. Data preparation and quality checks

The input is the June 2019 MUSAE GitHub Social Network: an edge list, one binary target per node, and sparse content features. Before analysis, the pipeline checks duplicate undirected edges, self-loops, missing labels, node-set alignment, and graph connectivity. The following workflow call regenerates the core tables and displays both the validation results and basic network statistics.""",
    6: """### Network description

The network contains 37,700 nodes and 289,003 edges, with no self-loops or duplicate undirected edges, and forms one connected component. Same-field endpoints account for 84.5% of edges. This is a strong descriptive relationship, but by itself it does not separate class imbalance, degree distribution, and community structure.""",
    8: """## 4. Centrality: popularity, recursive importance, and brokerage

- **Degree Centrality** emphasizes direct connectivity; dalinhuang99 (node 31890, degree 9,458) ranks first.
- **PageRank** also weights neighbor importance; dalinhuang99 remains first.
- **Betweenness Centrality** emphasizes shortest-path brokerage; with 100 sampled sources, nfultz (node 27803, degree 7,085) ranks first.
- Top-20 Jaccard overlap is 1.000 between Degree and PageRank but 0.600 between Degree and sampled Betweenness. The measures are therefore not interchangeable: Degree and PageRank identify hubs, while Betweenness can identify bridges.

Only one HITS hub score is kept because hub and authority scores are redundant in an undirected graph.""",
    10: """## 5. Louvain communities

Louvain finds 31 communities with modularity 0.453. The largest community contains 8,732 developers and size-weighted purity is 0.830.

- Community 7: 8,732 nodes, 6.5% ML
- Community 2: 7,664 nodes, 71.5% ML
- Community 13: 6,454 nodes, 19.4% ML

The composition differences show that community structure is associated with field, but mixed communities prevent perfect identification and should not be interpreted as rigid professional identities.""",
    12: """## 6. RQ1 — Node classification without information leakage

Each of five runs uses a stratified 70/30 split. Test labels are excluded from training, neighbor and community feature construction, and threshold calibration. Label-derived community features use training counts only. The workflow below regenerates repeated model results and validation-only threshold analysis.""",
    14: """### RQ1 results

| Model | Mean Macro F1 | SD Macro F1 | Mean ROC-AUC | SD ROC-AUC |
|---|---:|---:|---:|---:|
| Majority Baseline | 0.426 | 0.000 | 0.500 | 0.000 |
| Neighbor Majority | 0.702 | 0.006 | 0.785 | 0.005 |
| Logistic Regression — Graph Features Only | 0.751 | 0.005 | 0.841 | 0.004 |
| Logistic Regression — Graph + Community Features | 0.772 | 0.004 | 0.869 | 0.005 |

Graph + Community is the strongest handcrafted model at mean ROC-AUC 0.869. Its advantage over neighbor voting shows the cumulative value of centrality and community features. Validation-only threshold selection changes the Precision-Recall balance but not ROC-AUC.""",
    15: """## 7. RQ2 — Ablation: network, content, and hybrid information

All three information sources use the same classifier, seeds, splits, and metrics. Performance differences can therefore be attributed to representation choice rather than a different evaluation protocol.""",
    17: """### RQ2 results

| Information source | Macro F1 | ROC-AUC | AUC SD |
|---|---:|---:|---:|
| Content Features Only | 0.790 | 0.889 | 0.004 |
| Hybrid Network + Content | 0.797 | 0.898 | 0.004 |
| Network Only, Graph + Community | 0.772 | 0.869 | 0.005 |

Content is a strong reference and exceeds the handcrafted network representation alone. The hybrid improves the mean, indicating complementary information. This is predictive evidence under the tested splits, not proof that either source causes the label.""",
    18: """## 8. Learned representations: DeepWalk and Node2Vec

Both representations are learned without labels or content using the same budget: 3 walks per node, walk length 40, 64 dimensions, window 5, and 5 epochs. Node2Vec uses p=1 and q=2 for a modest local-neighborhood bias. Each representation is tested alone and together with handcrafted graph features.""",
    20: """### Representation comparison

| Model | Macro F1 | ROC-AUC | AUC SD |
|---|---:|---:|---:|
| Node2Vec | 0.783 | 0.890 | 0.005 |
| DeepWalk | 0.783 | 0.891 | 0.004 |
| Node2Vec + Graph + Community | 0.791 | 0.899 | 0.005 |
| DeepWalk + Graph + Community | 0.791 | 0.900 | 0.005 |

DeepWalk alone reaches ROC-AUC 0.891 and Node2Vec 0.890. The comparison shows how effectively random-walk contexts compress useful network structure. Similar performance does not prove representation equivalence; it applies only to the tested data, hyperparameters, and classifier.""",
    21: """## 9. RQ3 — Edge reconstruction while preserving connectivity

For each seed, a spanning tree is protected and test positives are sampled only from non-tree edges. After removal, the workflow explicitly checks `nx.is_connected(G_train)` and verifies that no node is isolated. Connectivity is necessary for path-based measures, Louvain, and graph walks.

Negative examples are true non-edges of the complete graph and match the positive count within Web-Web, ML-ML, and Web-ML. Same-field and cross-field results therefore use the same sampling rule.""",
    23: """### RQ3 results

Preferential Attachment is strongest overall with ROC-AUC 0.924. For this method, same-field ROC-AUC is 0.926 versus 0.918 cross-field, a gap of 0.008; Average Precision is 0.933 versus 0.925.

Within this balanced protocol, same-field reconstruction is slightly stronger for the leading heuristic. The pattern is not universal across methods, does not identify a causal link-formation mechanism, and cannot be generalized to temporal forecasting.""",
    24: """## 10. RQ4 — Null models and random graph models

The analysis uses 1,000 label shuffles and 20 degree-preserving rewired graphs. Empirical p-values use the finite-simulation correction `(b+1)/(B+1)`, so they cannot be zero. Observed statistics are also compared with Erdős-Rényi, Barabási-Albert, Watts-Strogatz, and a simple degree-preserving configuration null.""",
    26: """### RQ4 results

- Observed homophily is 0.845, versus 0.617 under label shuffling and 0.751 under degree-preserving rewiring. Corrected p-values are 0.000999 and 0.047619.
- Observed modularity is 0.453, versus 0.189 in rewired graphs; empirical p=0.047619.
- Observed average clustering is 0.168, versus 0.0004 in ER, 0.0034 in BA, 0.512 in Small-World, and 0.091 in the degree-preserving configuration null.
- The fitted tail exponent is α=2.54, but the comparison with lognormal does not justify a blanket scale-free claim. The defensible conclusion is a heavy-tailed degree distribution.""",
    27: """## 11. Discussion, limitations, and conclusions

**RQ1:** Network structure provides consistent predictive signal; communities and learned representations improve simple baselines.  
**RQ2:** Content is stronger than handcrafted network features, while their combination adds information; this is partial overlap, not equivalence.  
**RQ3:** Hidden-edge reconstruction performs well above chance while connectivity is preserved. Same/cross-field differences depend on method and protocol.  
**RQ4:** Homophily and modularity are unusual under the tested nulls; degree differences explain part, but not all, of the structure.

### Limitations

- The data are a June 2019 snapshot and the binary target may be noisy.
- Node classification is transductive: the complete graph structure is visible.
- Betweenness is approximated because of graph size.
- Embedding hyperparameters were fixed in advance rather than extensively searched.
- Hidden-edge reconstruction is not temporal forecasting; direction, tie strength, and causality are not modeled.
- Null and generative models isolate selected mechanisms and are not complete models of the network process.

The project evaluates 10 logical classification models: four baselines/handcrafted models, content and hybrid models, and four DeepWalk/Node2Vec models alone or combined with handcrafted features.""",
    28: """## 12. References

1. Rozemberczki, B., Allen, C., & Sarkar, R. (2019). *Multi-scale Attributed Node Embedding (MUSAE)*.
2. Blondel, V. D., et al. (2008). *Fast unfolding of communities in large networks*.
3. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). *DeepWalk*.
4. Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*.
5. Liben-Nowell, D., & Kleinberg, J. (2003). *The Link Prediction Problem for Social Networks*.
6. Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). *Power-law distributions in empirical data*.
7. Page, L., et al. (1999). *The PageRank Citation Ranking*.
8. Course lectures and tutorials on centrality, communities, network models, link reconstruction, and graph embeddings.""",
    29: """## 13. Final validation

The following cell fails if an output is missing, a CSV is empty or non-finite, a metric is outside its valid range, a model set is inconsistent, an empirical p-value is invalid, a training graph is disconnected, a node is isolated, notebook claims disagree with CSV values, or the final report exceeds eight pages. Expected output: `ALL FINAL PROJECT VALIDATIONS PASSED`.""",
}


LEGACY_INSERTIONS = {
    "01_environment_and_reproducibility.ipynb": (
        "project_scope_and_rqs",
        1,
        """## Project scope and research questions

The project studies 37,700 GitHub developers and 289,003 mutual-follow links. It asks whether network structure helps explain and predict the binary Web/ML field label. The analysis is descriptive and does not claim causality.

1. **RQ1:** Can graph features classify developer field without test-label leakage?
2. **RQ2:** What does content add to network structure, and what does their hybrid add?
3. **RQ3:** Can hidden static edges be reconstructed, and does same-field reconstruction differ from cross-field reconstruction?
4. **RQ4:** Are homophily and modularity unusual under simple null models?""",
    ),
    "02_data_loading_and_preparation.ipynb": (
        "dataset_context",
        5,
        """## Dataset context

The MUSAE GitHub Social Network is a June 2019 snapshot. The project uses a mutual-follow edge list, one binary Web/ML label per developer, and sparse content indicators derived from profile and repository information. The label is an analysis target rather than a complete professional identity; its binary form and age are explicit limitations.""",
    ),
    "03_cleaning_eda_and_graph_statistics.ipynb": (
        "named_centrality_findings",
        -1,
        """## Named centrality findings and interpretation

Degree Centrality and PageRank both rank dalinhuang99 (node 31890, degree 9,458) first. Approximate Betweenness with 100 sampled sources ranks nfultz (node 27803, degree 7,085) first. Degree and PageRank have top-20 Jaccard overlap 1.000, whereas Degree and Betweenness overlap 0.600. The first two measures emphasize hubs; Betweenness can reveal brokers between regions. Only one HITS hub feature is retained because hub and authority scores are redundant in an undirected graph.""",
    ),
    "04_communities_and_problem_definition.ipynb": (
        "largest_community_findings",
        -1,
        """## Composition of the largest communities

- Community 7: 8,732 developers, 6.5% Machine Learning.
- Community 2: 7,664 developers, 71.5% Machine Learning.
- Community 13: 6,454 developers, 19.4% Machine Learning.

The contrast supports a relationship between community structure and field, while the mixed proportions rule out treating communities as fixed professional identities.""",
    ),
    "05_node_classification_algorithms.ipynb": (
        "rq1_protocol_and_result",
        -1,
        """## RQ1 conclusion in protocol context

Every run uses a stratified 70/30 split, and test labels are excluded from training, neighborhood features, community ratios, and threshold calibration. Graph + Community reaches mean Macro F1 0.772 and ROC-AUC 0.869, compared with 0.702 and 0.785 for Neighbor Majority. The gain shows that centrality and community information add signal beyond immediate labeled neighbors; it does not imply causal influence.""",
    ),
    "06_embeddings_content_and_ablation.ipynb": (
        "embedding_budget_and_rq2_context",
        5,
        """## Exact embedding budget and RQ2 interpretation

DeepWalk and Node2Vec use the same budget: 3 walks per node, walk length 40, 64 dimensions, window 5, and 5 epochs. Node2Vec uses p=1 and q=2 for a modest local-neighborhood bias. Content-only ROC-AUC is about 0.889, network-only about 0.869, and hybrid about 0.898. The hybrid gain is predictive evidence of complementary information under this fixed protocol, not evidence that either source causes the label.""",
    ),
    "07_link_reconstruction_experiments.ipynb": (
        "rq3_leading_method_context",
        -1,
        """## RQ3 conclusion for the leading method

Preferential Attachment is strongest overall at ROC-AUC 0.924. Its same-field ROC-AUC is 0.926 versus 0.918 cross-field, and Average Precision is 0.933 versus 0.925. This small within-field advantage applies to the leading heuristic under matched sampling; other methods show different gaps, so the result is not universal and does not identify a temporal or causal link-formation process.""",
    ),
    "08_null_models_results_and_research_answers.ipynb": (
        "discussion_limitations_references",
        -1,
        """## Discussion, limitations, and references

**RQ1:** Network structure provides consistent but imperfect predictive signal.  
**RQ2:** Content is stronger than handcrafted network features, while hybrid models show complementary information.  
**RQ3:** Static hidden-edge reconstruction is strong, but same/cross-field differences depend on the method.  
**RQ4:** Homophily and modularity remain unusually high after focused null controls.

### Limitations

- The data are a June 2019 snapshot and the binary target may be noisy.
- Classification is transductive because the complete graph structure is visible.
- Betweenness is approximated with 100 sources.
- Embedding hyperparameters use a fixed budget rather than an extensive search.
- Hidden-edge reconstruction is not temporal forecasting and does not model direction, tie strength, or causality.
- Null and generative models isolate selected mechanisms rather than fully reproduce the social process.

### Core references

MUSAE (Rozemberczki, Allen, and Sarkar, 2019); Louvain (Blondel et al., 2008); DeepWalk (Perozzi, Al-Rfou, and Skiena, 2014); Node2Vec (Grover and Leskovec, 2016); link prediction (Liben-Nowell and Kleinberg, 2003); power-law analysis (Clauset, Shalizi, and Newman, 2009); PageRank (Page et al., 1999); and course material on centrality, communities, random graphs, reconstruction, and embeddings.""",
    ),
    "09_EXTENTIONS_and_validation.ipynb": (
        "legacy_integration_map",
        8,
        """## Legacy notebook integration map

All relevant material from `GitHub_Developer_Field_Prediction_Network_Only.ipynb` is now distributed by topic:

- motivation, research questions, and reproducibility → notebook 01;
- dataset context → notebook 02;
- named centrality results → notebook 03;
- largest-community composition → notebook 04;
- RQ1 protocol and conclusion → notebook 05;
- exact embedding budget and RQ2 context → notebook 06;
- leading-method RQ3 interpretation → notebook 07;
- consolidated discussion, limitations, and references → notebook 08.

The legacy workflow calls were not duplicated because the numbered notebooks already execute the same authoritative entry points and preserve every `src` block. This keeps all code, results, and explanations while avoiding a second copy of identical computation inside the series.""",
    ),
}


def _intro(filename: str, order: int) -> str:
    title, purpose, inputs, outputs, research = INTRO[filename]
    return dedent(
        f"""# {order:02d}. {title}

        Noa Yarin Levy · Jimi Halawi · Ofir Atias

        **Purpose.** {purpose}

        **Inputs.** {inputs}

        **Outputs.** {outputs}

        **Research role.** {research}

        The notebook is self-contained: it initializes the project paths, defines the needed code, runs the relevant step, and explains the saved results."""
    ).strip()


def _block_identity(cell) -> tuple[str, str]:
    source = cell.source
    match = re.search(r"^(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.M)
    if match:
        return "function", match.group(1)
    match = re.search(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.M)
    if match:
        return "class", match.group(1)
    if re.search(r"^if\s+__name__\s*==", source, re.M):
        return "main_guard", "__main__ guard"
    return "prologue", "imports, constants, and module setup"


def _docstring(cell) -> str:
    try:
        tree = ast.parse(cell.source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return " ".join((ast.get_docstring(node) or "").split())
    return ""


def _source_pre(cell, source_cell) -> str:
    path = source_cell.metadata["source_file"]
    filename = Path(path).name
    start = source_cell.metadata["source_start_line"]
    end = source_cell.metadata["source_end_line"]
    kind, name = _block_identity(source_cell)
    doc = _docstring(source_cell)
    if kind == "prologue":
        purpose = MODULE_ROLE.get(filename, "This block imports libraries and sets constants used by the module.")
        expected = "Expected result: imports and constants are available in memory."
    elif kind == "main_guard":
        purpose = "This block preserves the command-line entry guard from the original source file."
        expected = "Expected result: no output in the notebook."
    else:
        detail = FUNCTION_DETAIL.get(name) or doc or f"Define `{name}` for reuse later in the notebook."
        purpose = re.split(r"(?<=[.!?])\s+", detail.strip())[0]
        expected = "Expected result: the function is defined; visible output appears only when it is called."
    label = f"`{name}`" if kind != "prologue" else name
    return dedent(
        f"""### {filename}, lines {start}-{end}: {label}

        {purpose}

        {expected}"""
    ).strip()


def _source_post(source_cell) -> str:
    _kind, name = _block_identity(source_cell)
    if name == "__main__ guard":
        return "**Result.** No visible result is expected; the copied command-line guard stays inactive in the notebook."
    return f"**Result.** `{name}` is now defined. This cell usually has no visible output."


def _source_heading(nb, index: int) -> str | None:
    if index + 2 >= len(nb.cells):
        return None
    context = nb.cells[index + 2]
    if context.cell_type != "code" or "__file__ = str(ROOT /" not in context.source:
        return None
    match = re.search(r"ROOT\s*/\s*['\"]([^'\"]+\.py)['\"]", context.source)
    if not match:
        return None
    path = match.group(1).replace("\\", "/")
    filename = Path(path).name
    source_cells = [c for c in nb.cells if c.cell_type == "code" and c.metadata.get("source_file") == path]
    digest = source_cells[0].metadata.get("source_sha256", "not recorded") if source_cells else "not recorded"
    return dedent(
        f"""## Source code: `{path}`

        {MODULE_ROLE.get(filename, 'This module is reproduced here so the notebook exposes the implementation it uses.')}

        The following cells copy the source exactly. A short setup cell gives the copied code the same file path context it has in `src`. Source SHA-256: `{digest}`."""
    ).strip()


def _summary(filename: str) -> str:
    completed, findings, _connection = SUMMARY[filename]
    return dedent(
        f"""## Summary

        {completed}

        Main finding: {findings}"""
    ).strip()


def _run_markdown(filename: str, occurrence: int) -> str:
    description, expected = RUN_INFO[filename][occurrence]
    return dedent(
        f"""## Run this stage

        {description}

        Expected result: {expected}"""
    ).strip()


def _table_markdown(path: str) -> str:
    intro, _ = TABLE_INFO[path]
    return f"### Table: `{path}`\n\n{intro} The cell reloads the saved CSV so the displayed numbers match the project artifacts."


def _table_interpretation(path: str) -> str:
    return f"**Result.** {TABLE_INFO[path][1]}"


def _translate_numbered(nb, filename: str) -> None:
    run_occurrence = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        prev = nb.cells[i - 1] if i else None
        nxt = nb.cells[i + 1] if i + 1 < len(nb.cells) else None
        if i == 0:
            cell.source = _intro(filename, int(filename[:2]))
        elif nxt and nxt.cell_type == "code" and nxt.metadata.get("source_file"):
            cell.source = _source_pre(cell, nxt)
        elif prev and prev.cell_type == "code" and prev.metadata.get("source_file"):
            cell.source = _source_post(prev)
        elif nxt and nxt.cell_type == "code" and "__file__ = str(ROOT /" in nxt.source:
            cell.source = "This cell sets the file path context for the copied source block. It does not modify any file. Expected result: the source path."
        elif prev and prev.cell_type == "code" and "__file__ = str(ROOT /" in prev.source:
            cell.source = "**Result.** The printed path confirms that this source block reads and writes inside the project folder."
        elif (heading := _source_heading(nb, i)) is not None:
            cell.source = heading
        elif i == 1:
            cell.source = COMMON_SETUP
        elif i == 3:
            cell.source = COMMON_SETUP_OUTPUT
        elif cell.source.lstrip().startswith("## \u05d4\u05e8\u05e6\u05ea \u05d4\u05e9\u05dc\u05d1"):
            cell.source = _run_markdown(filename, run_occurrence)
            run_occurrence += 1
        elif "\u05d4\u05d5\u05d3\u05e2\u05d5\u05ea \u05d4\u05d4\u05ea\u05e7\u05d3\u05de\u05d5\u05ea" in cell.source:
            cell.source = "**Result.** The stage completed and wrote its tables and figures. The next cells read those saved files."
        elif (match := re.search(r"outputs/tables/[^`]+\.csv", cell.source)):
            path = match.group(0)
            if cell.source.lstrip().startswith("###"):
                cell.source = _table_markdown(path)
            elif path in TABLE_INFO:
                cell.source = _table_interpretation(path)
        elif (
            prev
            and prev.cell_type == "code"
            and (match := re.search(r"outputs/tables/[^\"']+\.csv", prev.source))
            and match.group(0) in TABLE_INFO
        ):
            # Result-comment cells name the table in the preceding loader cell,
            # not in their own Markdown.  Use that local relationship so every
            # persisted table receives its specific English interpretation.
            cell.source = _table_interpretation(match.group(0))
        elif cell.source.lstrip().startswith("### \u05ea\u05e8\u05e9\u05d9\u05de\u05d9\u05dd \u05e9\u05de\u05d5\u05e8\u05d9\u05dd"):
            intro, _ = FIGURE_INFO[filename]
            cell.source = f"### Saved figures\n\n{intro}"
        elif cell.source.lstrip().startswith("**\u05e4\u05dc\u05d8, \u05de\u05de\u05e6\u05d0\u05d9\u05dd \u05d5\u05e4\u05e8\u05e9\u05e0\u05d5\u05ea.**"):
            cell.source = f"**Result.** {FIGURE_INFO[filename][1]}"
        elif cell.source.lstrip().startswith("## \u05e1\u05d9\u05db\u05d5\u05dd \u05d4\u05de\u05d7\u05d1\u05e8\u05ea"):
            cell.source = _summary(filename)
        elif (filename, i) in SPECIAL_MD:
            cell.source = dedent(SPECIAL_MD[(filename, i)]).strip()


def _englishize_notebook_code(nb, filename: str) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code" or cell.metadata.get("source_file"):
            continue
        if "def _find_project_root" in cell.source:
            cell.source = re.sub(
                r'raise FileNotFoundError\([^\n]+\)',
                'raise FileNotFoundError("Could not locate a project root containing requirements.txt and data/")',
                cell.source,
            )
        if filename.startswith("08_") and "rq_answers = pd.DataFrame" in cell.source:
            cell.source = dedent(
                """
                classification = pd.read_csv(ROOT / "outputs/tables/classification_summary.csv")
                ablation = pd.read_csv(ROOT / "outputs/tables/ablation_model_results_summary.csv")
                embeddings = pd.read_csv(ROOT / "outputs/tables/embedding_classification_summary.csv")
                links = pd.read_csv(ROOT / "outputs/tables/edge_reconstruction_summary.csv")
                gaps = pd.read_csv(ROOT / "outputs/tables/edge_reconstruction_same_vs_cross_gaps.csv")
                null_h = pd.read_csv(ROOT / "outputs/tables/null_model_homophily.csv")
                null_q = pd.read_csv(ROOT / "outputs/tables/null_model_modularity.csv")

                rq_answers = pd.DataFrame([
                    {"RQ": "RQ1", "evidence": f"Best handcrafted AUC={classification.roc_auc_mean.max():.3f}; best embedding+handcrafted AUC={embeddings.roc_auc_mean.max():.3f}", "answer": "Network structure predicts developer field well, but not perfectly."},
                    {"RQ": "RQ2", "evidence": f"Network={ablation.loc[ablation.model.str.contains('Network Only'), 'roc_auc_mean'].iloc[0]:.3f}; Content={ablation.loc[ablation.model.str.contains('Content Features Only'), 'roc_auc_mean'].iloc[0]:.3f}; Hybrid={ablation.loc[ablation.model.str.contains('Hybrid'), 'roc_auc_mean'].iloc[0]:.3f}", "answer": "Content is stronger alone, while network structure adds complementary information."},
                    {"RQ": "RQ3", "evidence": f"Best={links.loc[links.roc_auc_mean.idxmax(), 'heuristic']} AUC={links.roc_auc_mean.max():.3f}; same-cross gap={gaps.loc[gaps.heuristic.eq('Preferential Attachment'), 'roc_auc_same_minus_cross'].iloc[0]:.3f}", "answer": "Static reconstruction is strong; same-field advantage depends on the scoring method."},
                    {"RQ": "RQ4", "evidence": f"Homophily p={null_h.loc[null_h.null_model.eq('label_shuffle'), 'empirical_p'].iloc[0]:.6f}; rewired homophily p={null_h.loc[null_h.null_model.eq('degree_preserving_rewire'), 'empirical_p'].iloc[0]:.6f}; modularity p={null_q.empirical_p.iloc[0]:.6f}", "answer": "Observed homophily and modularity are unusually high under the tested null models."},
                ])
                display(rq_answers)
                """
            ).strip()


def _insert_legacy_content(nb, filename: str) -> None:
    key, index, content = LEGACY_INSERTIONS[filename]
    cell = nbformat.v4.new_markdown_cell(dedent(content).strip())
    cell.metadata["legacy_integration_key"] = key
    if index >= 0:
        nb.cells.insert(index, cell)
        return
    if filename.startswith("03_"):
        marker = "Degree and PageRank have complete top-20 overlap"
    elif filename.startswith("04_"):
        marker = "Louvain finds 31 communities"
    elif filename.startswith("05_"):
        marker = "Graph + Community is the strongest handcrafted model"
    elif filename.startswith("07_"):
        marker = "The gap is not universal"
    else:
        marker = "## Summary"
    for i, existing in enumerate(nb.cells):
        if existing.cell_type == "markdown" and marker in existing.source:
            nb.cells.insert(i + (0 if marker == "## Summary" else 1), cell)
            return
    raise ValueError(f"Could not place legacy integration {key} in {filename}")


def _add_alias_context(nb) -> None:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and cell.source.strip() == "_run_all_validations = run_all_validations":
            before = nbformat.v4.new_markdown_cell(
                "### Preserve the validation entry point\n\nThe next source modules also define functions named `main`. This one-line alias keeps the already loaded `run_all_validations` function available under an unambiguous name. Expected output: none."
            )
            before.metadata["validation_alias_context"] = "before"
            after = nbformat.v4.new_markdown_cell(
                "**Output and interpretation.** The validation callable is now stored as `_run_all_validations`; later builder definitions cannot overwrite that reference."
            )
            after.metadata["validation_alias_context"] = "after"
            nb.cells[i:i] = [before]
            nb.cells.insert(i + 2, after)
            return


def _translate_legacy(nb) -> None:
    for index, text in LEGACY_MARKDOWN.items():
        if index >= len(nb.cells) or nb.cells[index].cell_type != "markdown":
            raise ValueError(f"Legacy markdown cell {index} is missing")
        nb.cells[index].source = dedent(text).strip()
    final = nbformat.v4.new_markdown_cell(
        "**Output and interpretation.** The success message confirms that the consolidated notebook agrees with the current result tables and satisfies the original artifact checks."
    )
    final.metadata["legacy_final_output_explanation"] = True
    nb.cells.append(final)
    summary = nbformat.v4.new_markdown_cell(
        "## Summary\n\nThis consolidated notebook presents the same project in one compact workflow. Main finding: network structure, content features, learned representations, and null models all show a strong descriptive link between developer field and the mutual-follow network."
    )
    summary.metadata["legacy_final_summary"] = True
    nb.cells.append(summary)


def _clean_submission_text(text: str) -> str:
    replacements = {
        "**Output and interpretation.**": "**Result.**",
        "**Output, findings, and meaning.**": "**Result.**",
        "**Output, findings, and interpretation.**": "**Result.**",
        "Expected output:": "Expected result:",
        "Expected output.": "Expected result.",
        "Final Validation": "Project Check",
        "final validation": "project check",
        "## Notebook summary": "## Summary",
        "Validate all public project artifacts": "Check project artifacts",
        "Prove exact byte-for-byte coverage of every `src/*.py` file": "Check source-code coverage",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\n\n\*\*Submission note\.\*\*.*", "", text, flags=re.S)
    text = text.replace(
        "Definition cells normally produce no display; the block is now available to the explicitly documented workflow or validation step in this notebook.",
        "This definition cell usually has no visible output.",
    )
    text = text.replace(
        "The progress log confirms that the entry point completed. The following cells reload the written artifacts, so numerical findings are verified from disk rather than inferred from console text.",
        "The stage completed and wrote its tables and figures. The next cells read those saved files.",
    )
    return text.strip()


def _submission_markdown_override(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("## Structural measures and computational cost"):
        return """## Structural measures used here

This notebook uses degree, PageRank, sampled betweenness, clustering, HITS, and homophily. Degree and PageRank identify hubs, betweenness identifies bridge nodes, clustering measures triangles, and homophily measures same-label edges. These measures describe the network; they do not prove causality."""
    if stripped.startswith("## Louvain, modularity, and community purity"):
        return """## Louvain, modularity, and community purity

Louvain groups nodes into communities by trying to increase modularity, a score that is high when many edges stay inside communities. Community purity is the majority-label share inside a community. Both measures are useful descriptions, but mixed communities mean they cannot replace held-out prediction."""
    if stripped.startswith("## Classification design"):
        return """## Classification design

The notebook compares four models: a majority baseline, a neighbor-majority baseline, logistic regression on graph features, and logistic regression on graph plus community features. Logistic regression is a linear classifier that estimates the probability of the ML label from numeric features. All models use the same five stratified splits, and test labels are not used to build features."""
    if stripped.startswith("## Learned representations and controlled ablation"):
        return """## Learned representations and controlled ablation

DeepWalk and Node2Vec learn node vectors from random walks on the graph. DeepWalk uses uniform walks; Node2Vec biases the walks to control local versus wider exploration. The ablation compares network, content, and hybrid inputs under the same splits and metrics, so the difference comes from the information source."""
    if stripped.startswith("## Connectivity-preserving reconstruction protocol and scores"):
        return """## Static link-reconstruction protocol

The experiment hides 10% of eligible edges while keeping a spanning tree, so the training graph stays connected. Each hidden edge is compared with a true non-edge of the same endpoint-label type. The scores include Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, Same Community, and embedding dot products."""
    if stripped.startswith("## Null-model methodology"):
        return """## Null-model methodology

A null model is a comparison graph or label assignment used to ask whether an observed pattern is unusual. Label shuffling keeps the graph fixed. Degree-preserving rewiring keeps node degrees fixed. ER, BA, Watts-Strogatz, and configuration models test which basic network properties can be reproduced by simple random mechanisms."""
    if stripped.startswith("## What counts as an EXTENTION in this project"):
        return """## EXTENTIONS in this project

The core project is the validated pipeline run by `src/run_all.py`. The `extensions/` folder keeps extra work outside that core: Label Propagation, GCN/GraphSAGE, and SVD+KMeans content subfield discovery. These extensions are useful because they test semi-supervised learning, graph neural networks, and finer content groups, but they are not mixed into the main result tables."""
    if stripped.startswith("## Existing extension inventory"):
        return """## Extension inventory

This cell lists the files in `extensions/` and reads the extension README. It documents what exists without running optional experiments that are outside the main pipeline."""
    if stripped.startswith("## Legacy notebook integration map"):
        return ""
    if stripped.startswith("### Preserve the validation entry point"):
        return """### Store the project-check function

This short alias keeps the loaded `run_all_validations` function available after later source modules define their own `main` functions. Expected result: no visible output."""
    if stripped.startswith("## Check project artifacts"):
        return """## Check project artifacts

This cell checks that the required tables, figures, reports, notebook execution state, and graph split constraints are present and consistent. Expected result: the success message from the project checker."""
    if stripped.startswith("## Check source-code coverage"):
        return """## Check source-code coverage

Each copied source cell stores its original file and line numbers. This cell confirms that every `src/*.py` file is represented exactly once in the notebook series."""
    if stripped.startswith("## Discussion, limitations, and references"):
        return """## Discussion, limitations, and references

The results are descriptive and predictive, not causal. Main limitations are the 2019 snapshot, the binary target label, approximate betweenness, fixed embedding budgets, and static link reconstruction instead of temporal forecasting. The references list the dataset and the main graph algorithms used in the project."""
    if stripped.startswith("## 1. Motivation and research questions"):
        return """## 1. Motivation and research questions

The project asks whether GitHub mutual-follow structure is associated with developer field. It studies node classification, network versus content features, static edge reconstruction, and null-model comparisons. All claims are descriptive; the analysis does not claim causality."""
    if stripped.startswith("## 11. Discussion, limitations, and conclusions"):
        return """## 11. Discussion, limitations, and conclusions

The main conclusion is that network structure contains useful information about developer field, especially through communities and learned graph representations. Content features are also strong, and the hybrid model performs best. The limitations are the static 2019 snapshot, the binary target label, approximate centrality, and static reconstruction of hidden edges."""
    if stripped.startswith("## 13. Project Check") or stripped.startswith("## 13. Final validation"):
        return """## 13. Project check

This final cell checks that the consolidated notebook agrees with the saved result tables and required project artifacts."""
    return None


def _finalize_submission_notebook(nb, filename: str, *, numbered: bool) -> None:
    if numbered and filename.startswith("09_"):
        nb.cells = [
            cell for cell in nb.cells
            if not (
                cell.cell_type == "markdown"
                and (
                    cell.metadata.get("legacy_integration_key") == "legacy_integration_map"
                    or cell.source.strip().startswith("## Legacy notebook integration map")
                )
            )
        ]

    if numbered:
        nb.metadata.setdefault("project", {})["title"] = INTRO[filename][0]
    elif filename == LEGACY_NAME:
        nb.metadata.setdefault("project", {})["title"] = "Predicting GitHub Developer Fields from a Mutual-Follow Network"

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        if numbered and i == 0:
            cell.source = _intro(filename, int(filename[:2]))
        override = _submission_markdown_override(cell.source)
        if override == "":
            cell.source = ""
        elif override is not None:
            cell.source = override
        else:
            cell.source = _clean_submission_text(cell.source)

    nb.cells = [cell for cell in nb.cells if not (cell.cell_type == "markdown" and not cell.source.strip())]


def _write_readme(notebooks_dir: Path) -> None:
    rows = []
    for order, filename in enumerate(INTRO, 1):
        title = INTRO[filename][0]
        rows.append(f"| {order:02d} | [`{filename}`]({filename}) | {title} |")
    text = f"""# Research Notebook Series

This is the executed English research workflow for Noa Yarin Levy, Jimi Halawi, and Ofir Atias. The numbered notebooks keep the project code, results, figures, research questions, and main findings in a clean submission order.

## Execution order

| Order | Notebook | Purpose |
|---:|---|---|
{chr(10).join(rows)}

`{LEGACY_NAME}` is retained as a consolidated compatibility notebook. Its unique explanations have been merged into the numbered series without duplicating identical workflow calls.

## Full execution on Windows PowerShell

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:PYTHONHASHSEED="0"
if (-not (Test-Path .\\.venv\\Scripts\\python.exe)) {{
    py -3.11 -m venv .venv
}}
.\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements.txt
New-Item -ItemType Directory -Force .\\tmp\\jupyter-runtime | Out-Null
.\\.venv\\Scripts\\python.exe -m ipykernel install --prefix .\\tmp\\jupyter-runtime --name github-social-network --display-name "GitHub Social Network (.venv)"
$env:JUPYTER_PATH = (Resolve-Path .\\tmp\\jupyter-runtime\\share\\jupyter).Path
$analysis = Get-ChildItem .\\notebooks -Filter "*.ipynb" | Where-Object Name -Match "^0[1-8]_" | Sort-Object Name
$executionOrder = @($analysis.FullName)
$executionOrder += (Resolve-Path ".\\notebooks\\{LEGACY_NAME}").Path
$executionOrder += (Resolve-Path ".\\notebooks\\09_EXTENTIONS_and_validation.ipynb").Path
foreach ($notebookPath in $executionOrder) {{
    .\\.venv\\Scripts\\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=github-social-network --ExecutePreprocessor.timeout=-1 $notebookPath
}}
```

The compatibility notebook is kept for comparison with the original single-notebook version. Every notebook locates the repository root automatically. Results are written to `outputs/tables`, `outputs/figures`, and `outputs/embeddings`. Each code cell has short local context before it and a brief result note after it.
"""
    (notebooks_dir / "README.md").write_text(text, encoding="utf-8")


def englishize_all(root: str | Path | None = None) -> None:
    project = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    notebooks_dir = project / "notebooks"
    numbered = sorted(notebooks_dir.glob(SERIES_GLOB))
    if len(numbered) != 9:
        raise ValueError(f"Expected 9 numbered notebooks, found {len(numbered)}")

    for path in numbered:
        nb = nbformat.read(path, as_version=4)
        nb.cells = [
            c for c in nb.cells
            if not c.metadata.get("legacy_integration_key")
            and not c.metadata.get("validation_alias_context")
        ]
        _translate_numbered(nb, path.name)
        _englishize_notebook_code(nb, path.name)
        _insert_legacy_content(nb, path.name)
        if path.name.startswith("09_"):
            _add_alias_context(nb)
        _finalize_submission_notebook(nb, path.name, numbered=True)
        for cell in nb.cells:
            if cell.cell_type == "markdown" and HEBREW_RE.search(cell.source):
                raise ValueError(f"Untranslated Hebrew markdown in {path.name}: {cell.source[:120]!r}")
            if cell.cell_type == "markdown" and re.search(r"^#+\s*\?+", cell.source, re.M):
                raise ValueError(f"Question-mark heading remains in {path.name}")
            if cell.cell_type == "code" and not cell.metadata.get("source_file") and HEBREW_RE.search(cell.source):
                raise ValueError(f"Untranslated notebook code string in {path.name}: {cell.source[:120]!r}")
        nbformat.write(nb, path)
        print(f"Englishized {path.name}: {len(nb.cells)} cells")

    legacy_path = notebooks_dir / LEGACY_NAME
    legacy = nbformat.read(legacy_path, as_version=4)
    legacy.cells = [
        c for c in legacy.cells
        if not c.metadata.get("legacy_final_output_explanation")
        and not c.metadata.get("legacy_final_summary")
    ]
    _translate_legacy(legacy)
    _finalize_submission_notebook(legacy, LEGACY_NAME, numbered=False)
    for cell in legacy.cells:
        if cell.cell_type == "markdown" and HEBREW_RE.search(cell.source):
            raise ValueError(f"Untranslated Hebrew markdown in {LEGACY_NAME}")
    nbformat.write(legacy, legacy_path)
    print(f"Englishized {LEGACY_NAME}: {len(legacy.cells)} cells")
    _write_readme(notebooks_dir)
    print("Wrote English notebooks/README.md")


if __name__ == "__main__":
    englishize_all()
