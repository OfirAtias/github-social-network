from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import pandas as pd


EDGE_COLUMNS = ("id_1", "id_2")
TARGET_ID_COLUMN = "id"
TARGET_LABEL_COLUMN = "ml_target"


def load_edges(path: str | Path) -> pd.DataFrame:
    """Load and validate the edge list."""
    path = Path(path)
    edges_df = pd.read_csv(path)

    missing_columns = [col for col in EDGE_COLUMNS if col not in edges_df.columns]
    if missing_columns:
        raise ValueError(f"Edge file is missing required columns: {missing_columns}")

    edges_df = edges_df.loc[:, list(EDGE_COLUMNS)].copy()
    edges_df["id_1"] = pd.to_numeric(edges_df["id_1"], errors="coerce")
    edges_df["id_2"] = pd.to_numeric(edges_df["id_2"], errors="coerce")
    if edges_df[list(EDGE_COLUMNS)].isna().any().any():
        raise ValueError("Edge file contains missing or non-numeric endpoints.")
    edges_df = edges_df.astype({"id_1": "int64", "id_2": "int64"})

    if (edges_df["id_1"] == edges_df["id_2"]).any():
        raise ValueError("Edge file contains self-loops.")

    canonical_edges = pd.DataFrame(
        {
            "u": edges_df[["id_1", "id_2"]].min(axis=1),
            "v": edges_df[["id_1", "id_2"]].max(axis=1),
        }
    )
    if canonical_edges.duplicated().any():
        raise ValueError("Edge file contains duplicate undirected edges.")

    return edges_df


def load_targets(path: str | Path) -> pd.DataFrame:
    """Load and validate node labels, returning a standardized labels DataFrame."""
    path = Path(path)
    targets_df = pd.read_csv(path)

    required_columns = [TARGET_ID_COLUMN, TARGET_LABEL_COLUMN]
    missing_columns = [col for col in required_columns if col not in targets_df.columns]
    if missing_columns:
        raise ValueError(f"Target file is missing required columns: {missing_columns}")

    keep_columns = [TARGET_ID_COLUMN, TARGET_LABEL_COLUMN]
    if "name" in targets_df.columns:
        keep_columns.insert(1, "name")

    labels_df = targets_df.loc[:, keep_columns].copy()
    labels_df[TARGET_ID_COLUMN] = pd.to_numeric(labels_df[TARGET_ID_COLUMN], errors="coerce")
    labels_df[TARGET_LABEL_COLUMN] = pd.to_numeric(labels_df[TARGET_LABEL_COLUMN], errors="coerce")
    if labels_df[[TARGET_ID_COLUMN, TARGET_LABEL_COLUMN]].isna().any().any():
        raise ValueError("Target file contains missing or non-numeric IDs or labels.")
    labels_df[TARGET_ID_COLUMN] = labels_df[TARGET_ID_COLUMN].astype("int64")
    labels_df[TARGET_LABEL_COLUMN] = labels_df[TARGET_LABEL_COLUMN].astype("int64")
    if labels_df[TARGET_ID_COLUMN].duplicated().any():
        raise ValueError("Target file contains duplicate node IDs.")

    unique_labels = set(labels_df[TARGET_LABEL_COLUMN].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Unexpected labels detected in target file: {sorted(unique_labels)}")

    labels_df = labels_df.rename(columns={TARGET_LABEL_COLUMN: "label"})

    return labels_df


def build_graph(edges_df: pd.DataFrame) -> nx.Graph:
    """Build an undirected NetworkX graph from an edges DataFrame."""
    missing_columns = [col for col in EDGE_COLUMNS if col not in edges_df.columns]
    if missing_columns:
        raise ValueError(f"Edges DataFrame is missing required columns: {missing_columns}")

    graph = nx.from_pandas_edgelist(edges_df, source="id_1", target="id_2", create_using=nx.Graph())
    return graph


def load_dataset(data_dir: str | Path) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Load the dataset from a directory and build all base objects.

    Returns:
        G: Undirected graph
        edges_df: Edge list DataFrame
        labels_df: Labels DataFrame with standardized 'label' column
        labels_dict: Mapping node_id -> label
    """
    data_dir = Path(data_dir)
    edges_path = data_dir / "musae_git_edges.csv"
    targets_path = data_dir / "musae_git_target.csv"

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edge file: {edges_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing target file: {targets_path}")

    edges_df = load_edges(edges_path)
    labels_df = load_targets(targets_path)
    graph = build_graph(edges_df)

    labels_dict = dict(zip(labels_df["id"], labels_df["label"]))
    missing_labels = set(graph.nodes()) - set(labels_dict)
    if missing_labels:
        raise ValueError(f"Graph contains {len(missing_labels)} nodes without labels.")
    nx.set_node_attributes(graph, labels_dict, name="label")

    return graph, edges_df, labels_df, labels_dict
