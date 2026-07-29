from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple

import networkx as nx
import pandas as pd
import community as community_louvain


def detect_louvain_communities(
    G: nx.Graph,
    random_state: int = 42,
) -> Tuple[Dict[int, int], float]:
    """
    Run Louvain community detection on the full graph structure.

    Returns:
        partition: mapping node_id -> community_id
        modularity: Louvain modularity score
    """
    partition = community_louvain.best_partition(G, random_state=random_state)
    modularity = community_louvain.modularity(partition, G)
    return partition, float(modularity)


def summarize_communities(
    G: nx.Graph,
    labels_dict: Dict[int, int],
    partition: Dict[int, int],
) -> pd.DataFrame:
    """
    Build one summary row per community.

    Output columns:
      community_id, size, web_count, ml_count, web_ratio, ml_ratio,
      dominant_label, purity
    """
    community_nodes: Dict[int, list[int]] = defaultdict(list)
    for node in G.nodes():
        community_id = int(partition.get(node, -1))
        community_nodes[community_id].append(int(node))

    rows = []
    for community_id, nodes in community_nodes.items():
        size = len(nodes)
        web_count = 0
        ml_count = 0
        for node in nodes:
            label = labels_dict.get(node)
            if label == 0:
                web_count += 1
            elif label == 1:
                ml_count += 1

        labeled_total = web_count + ml_count
        if labeled_total == 0:
            web_ratio = 0.0
            ml_ratio = 0.0
            dominant_label = "Web"
            purity = 0.0
        else:
            web_ratio = web_count / labeled_total
            ml_ratio = ml_count / labeled_total
            dominant_label = "Web" if web_count >= ml_count else "ML"
            purity = max(web_ratio, ml_ratio)

        rows.append(
            {
                "community_id": community_id,
                "size": size,
                "web_count": web_count,
                "ml_count": ml_count,
                "web_ratio": web_ratio,
                "ml_ratio": ml_ratio,
                "dominant_label": dominant_label,
                "purity": purity,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(by="size", ascending=False).reset_index(drop=True)
    return summary_df


def add_community_features(
    G: nx.Graph,
    labels_dict: Dict[int, int],
    train_nodes: Iterable[int],
    nodes: Iterable[int],
    partition: Dict[int, int],
) -> pd.DataFrame:
    """
    Build community-based node features.

    Label-based features are calculated from train labels only.
    """
    train_set = {int(node) for node in train_nodes}
    # Community sizes are structural.  Derive them directly from the
    # label-free Louvain partition rather than from the descriptive summary,
    # which also contains full-dataset label counts and purity values.
    community_size_map = Counter(int(community_id) for community_id in partition.values())

    train_web_count_by_community: Counter[int] = Counter()
    train_ml_count_by_community: Counter[int] = Counter()

    for node in train_set:
        label = labels_dict.get(node)
        if label not in (0, 1):
            continue
        community_id = int(partition.get(node, -1))
        if label == 0:
            train_web_count_by_community[community_id] += 1
        else:
            train_ml_count_by_community[community_id] += 1

    rows = []
    for node in nodes:
        node_int = int(node)
        community_id = int(partition.get(node_int, -1))
        community_size = int(community_size_map.get(community_id, 0))

        train_web = int(train_web_count_by_community.get(community_id, 0))
        train_ml = int(train_ml_count_by_community.get(community_id, 0))
        train_labeled_total = train_web + train_ml

        if train_labeled_total > 0:
            train_ml_ratio = train_ml / train_labeled_total
            train_web_ratio = train_web / train_labeled_total
        else:
            train_ml_ratio = 0.0
            train_web_ratio = 0.0

        rows.append(
            {
                "node_id": node_int,
                "community_id": community_id,
                "community_size": community_size,
                "train_web_in_community": train_web,
                "train_ml_in_community": train_ml,
                "train_ml_ratio_in_community": train_ml_ratio,
                "train_web_ratio_in_community": train_web_ratio,
                "community_train_label_gap": train_ml - train_web,
            }
        )

    return pd.DataFrame(rows)
