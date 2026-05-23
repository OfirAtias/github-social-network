import pandas as pd
import os
import networkx as nx

def get_data_paths():
    """Returns the absolute paths to the local dataset files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edges_path = os.path.join(base_dir, "data", "musae_git_edges.csv")
    targets_path = os.path.join(base_dir, "data", "musae_git_target.csv")
    return edges_path, targets_path

def load_github_graph():
    """Loads edges and node features, and returns a compiled NetworkX Graph object."""
    edges_path, targets_path = get_data_paths()
    
    if not os.path.exists(edges_path) or not os.path.exists(targets_path):
        raise FileNotFoundError("[-] Error: Dataset files missing inside the 'data' folder.")
        
    print("[+] Loading CSV files...")
    edges_df = pd.read_csv(edges_path)
    nodes_df = pd.read_csv(targets_path)
    
    print("[+] Building network graph using NetworkX...")
    # Create an undirected graph from the edges dataframe
    G = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')
    
    print("[+] Attaching target classifications to nodes...")
    # Add target classifications (Web=0, ML=1) as node attributes
    # mapping node IDs to their respective target class
    class_mapping = dict(zip(nodes_df['id'], nodes_df['ml_target']))
    nx.set_node_attributes(G, class_mapping, name='target')
    
    return G, nodes_df