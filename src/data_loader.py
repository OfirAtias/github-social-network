import pandas as pd
import networkx as nx
from pathlib import Path
import os

def get_data_paths():
    """Returns the absolute paths to the local dataset files."""
    # מוצא את המיקום של הקובץ הנוכחי (בתוך src)
    current_dir = Path(__file__).resolve().parent
    # הולך צעד אחד אחורה לתיקיית הפרויקט הראשית
    project_root = current_dir.parent
    
    edges_path = project_root / "data" / "musae_git_edges.csv"
    targets_path = project_root / "data" / "musae_git_target.csv"
    
    return str(edges_path), str(targets_path)

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
    class_mapping = dict(zip(nodes_df['id'], nodes_df['ml_target']))
    nx.set_node_attributes(G, class_mapping, name='target')
    
    return G, nodes_df