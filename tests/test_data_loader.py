from pathlib import Path
import sys

import networkx as nx

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import load_dataset


def test_repository_dataset_quality() -> None:
    graph, edges, labels, label_map = load_dataset(PROJECT_DIR / "data")
    assert graph.number_of_nodes() == 37_700
    assert graph.number_of_edges() == 289_003
    assert len(edges) == graph.number_of_edges()
    assert nx.is_connected(graph)
    assert nx.number_of_selfloops(graph) == 0
    assert not labels["id"].duplicated().any()
    assert not labels["label"].isna().any()
    assert set(graph.nodes()) == set(label_map)
