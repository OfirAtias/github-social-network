# Optional extensions

This directory preserves exploratory implementations that are **not** part of
the final course-only project, its reproducibility pipeline, its reported
results, or its grading claims.

- `gnn.py` and `run_gnn.py`: GCN and GraphSAGE experiments.
- `run_subfields.py`: TruncatedSVD/K-means subfield exploration with
  Silhouette, NMI, and ARI evaluation.
- `label_propagation.py`: the former label-propagation classifier.

The authoritative project code lives in `src/`. Running `python src/run_all.py`
does not import or execute anything in this directory.
