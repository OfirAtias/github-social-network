# -*- coding: utf-8 -*-
"""
DeepWalk and Node2Vec node embeddings.

DeepWalk treats truncated random walks over the graph as "sentences" and
learns node embeddings with Word2Vec (skip-gram), following the idea from
the course DeepWalk tutorial, scaled up to the full GitHub network.

The embeddings are unsupervised and depend only on graph structure, so they
respect the project's network-only principle: no labels and no content
features are used at any point of the embedding computation.
"""
import random
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def generate_random_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    seed: int = 42,
) -> List[List[str]]:
    """
    Generate uniform truncated random walks from every node.

    Returns walks as lists of string node ids (gensim expects string tokens).
    """
    rng = random.Random(seed)

    # Precompute neighbor lists once; per-step attribute access would dominate runtime.
    neighbors: Dict[int, List[int]] = {node: list(G.neighbors(node)) for node in G.nodes()}

    nodes = list(G.nodes())
    walks: List[List[str]] = []

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                current_neighbors = neighbors[current]
                if not current_neighbors:
                    break
                current = current_neighbors[rng.randrange(len(current_neighbors))]
                walk.append(current)
            walks.append([str(node) for node in walk])

    return walks


def generate_node2vec_walks(
    G: nx.Graph,
    num_walks: int = 5,
    walk_length: int = 40,
    p: float = 1.0,
    q: float = 2.0,
    seed: int = 42,
) -> List[List[str]]:
    """Generate second-order biased walks using the Node2Vec transition rule.

    ``p`` controls immediate returns and ``q`` controls outward exploration.
    The default ``q=2`` gives a modest breadth-first bias that is appropriate
    for testing whether structurally similar neighborhoods help field
    classification.  The implementation is deterministic for a fixed graph
    order and seed.
    """
    if p <= 0 or q <= 0:
        raise ValueError("Node2Vec parameters p and q must be positive.")

    rng = random.Random(seed)
    neighbors: Dict[int, List[int]] = {
        int(node): [int(nbr) for nbr in G.neighbors(node)] for node in G.nodes()
    }
    neighbor_sets = {node: set(nbrs) for node, nbrs in neighbors.items()}
    nodes = [int(node) for node in G.nodes()]
    walks: List[List[str]] = []

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                current = walk[-1]
                candidates = neighbors[current]
                if not candidates:
                    break
                if len(walk) == 1:
                    next_node = candidates[rng.randrange(len(candidates))]
                else:
                    previous = walk[-2]
                    weights = []
                    previous_neighbors = neighbor_sets[previous]
                    for candidate in candidates:
                        if candidate == previous:
                            weights.append(1.0 / p)
                        elif candidate in previous_neighbors:
                            weights.append(1.0)
                        else:
                            weights.append(1.0 / q)
                    next_node = rng.choices(candidates, weights=weights, k=1)[0]
                walk.append(next_node)
            walks.append([str(node) for node in walk])
    return walks


def train_deepwalk(
    walks: List[List[str]],
    dimensions: int = 128,
    window: int = 10,
    epochs: int = 5,
    seed: int = 42,
    workers: int = 1,
) -> Word2Vec:
    """
    Train skip-gram Word2Vec over the random walks.

    workers=1 keeps training deterministic for a fixed seed (multi-worker
    training is faster but yields slightly different vectors on every run,
    which would make all downstream numbers non-reproducible). For full
    determinism run with the environment variable PYTHONHASHSEED=0.
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=0,
        sg=1,  # Skip-gram, as in DeepWalk
        workers=workers,
        seed=seed,
        epochs=epochs,
    )
    return model


def train_node2vec(
    walks: List[List[str]],
    dimensions: int = 64,
    window: int = 5,
    epochs: int = 5,
    seed: int = 42,
    workers: int = 1,
) -> Word2Vec:
    """Train skip-gram embeddings from Node2Vec biased walks."""
    return Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=0,
        sg=1,
        workers=workers,
        seed=seed,
        epochs=epochs,
    )


def embeddings_to_frame(model: Word2Vec, nodes: List[int]) -> pd.DataFrame:
    """
    Extract embeddings for the given nodes into a DataFrame with a node_id
    column, compatible with the existing feature-frame merging utilities.
    """
    dimensions = model.wv.vector_size
    rows = np.zeros((len(nodes), dimensions), dtype=np.float64)
    for i, node in enumerate(nodes):
        key = str(int(node))
        if key in model.wv:
            rows[i] = model.wv[key]

    frame = pd.DataFrame(rows, columns=[f"deepwalk_{d}" for d in range(dimensions)])
    frame.insert(0, "node_id", [int(node) for node in nodes])
    return frame
