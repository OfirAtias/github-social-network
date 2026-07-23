from collections import Counter
from pathlib import Path
import sys

import networkx as nx

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from link_prediction import edge_label_type, sample_link_prediction_split


def _fixture_graph() -> tuple[nx.Graph, dict[int, int]]:
    graph = nx.cycle_graph(10)
    graph.add_edges_from(
        [(0, 2), (1, 4), (2, 5), (3, 7), (4, 8), (5, 9), (0, 6)]
    )
    labels = {node: int(node >= 5) for node in graph}
    return graph, labels


def test_split_preserves_connectivity_and_nodes() -> None:
    graph, labels = _fixture_graph()
    train, positives, negatives, diagnostics = sample_link_prediction_split(
        graph, labels, test_fraction=0.25, seed=42
    )
    assert nx.is_connected(train)
    assert train.number_of_nodes() == graph.number_of_nodes()
    assert list(nx.isolates(train)) == []
    assert diagnostics["train_connected"] is True
    assert diagnostics["isolated_nodes"] == 0
    assert len(positives) == len(negatives)


def test_negative_pairs_match_positive_endpoint_types() -> None:
    graph, labels = _fixture_graph()
    _, positives, negatives, _ = sample_link_prediction_split(
        graph, labels, test_fraction=0.25, seed=7
    )
    positive_counts = Counter(
        edge_label_type(u, v, labels) for u, v in positives
    )
    negative_counts = Counter(
        edge_label_type(u, v, labels) for u, v in negatives
    )
    assert positive_counts == negative_counts
    assert all(not graph.has_edge(u, v) for u, v in negatives)
    assert all(u != v for u, v in negatives)


def test_split_is_reproducible_for_a_fixed_seed() -> None:
    graph, labels = _fixture_graph()
    first = sample_link_prediction_split(
        graph, labels, test_fraction=0.25, seed=123
    )
    second = sample_link_prediction_split(
        graph, labels, test_fraction=0.25, seed=123
    )
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
