# Executed optional extensions

These methods remain separate from the core findings in notebooks 01-08, but
they are real, reproducible experiments executed and reported by notebook 09.

- `label_propagation.py`: clamped propagation with explicit hidden labels.
- `gnn.py`: full-batch GCN and mean-aggregating GraphSAGE in PyTorch.
- `run_gnn.py`: the three classification extensions on the same five fixed
  splits, with inner validation, final refitting, leakage audits, and direct
  comparison to the strongest complementary core baselines.
- `run_subfields.py`: 64-component TruncatedSVD and K-means clustering across
  `k=3..10`, with silhouette, five-seed stability, cluster sizes, and cautious
  post-hoc alignment checks.

All extension artifacts are written below `outputs/extensions/`. GCN and
GraphSAGE are evaluated **transductively**: the full graph and label-free node
covariates are visible, while only outer-training labels supervise the final
fit. No inductive performance is claimed.
