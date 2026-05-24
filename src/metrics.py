import networkx as nx
from prettytable import PrettyTable

def calculate_top_centralities(G, top_n=5):
    """Calculates Degree and Betweenness centrality, returns top nodes."""
    print("[+] Calculating Degree Centrality...")
    degree_cent = nx.degree_centrality(G)
    
    print("[+] Calculating Betweenness Centrality (this might take a moment)...")
    # Using k=100 for approximation to save time on large networks
    betweenness_cent = nx.betweenness_centrality(G, k=100, seed=42)
    
    # Sort and get top N
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_betweenness = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Print beautiful tables
    deg_table = PrettyTable(["Rank", "Developer ID", "Degree Centrality Score"])
    for i, (node, score) in enumerate(top_degree, 1):
        deg_table.add_row([i, node, f"{score:.4f}"])
        
    bet_table = PrettyTable(["Rank", "Developer ID", "Betweenness Score"])
    for i, (node, score) in enumerate(top_betweenness, 1):
        bet_table.add_row([i, node, f"{score:.4f}"])
        
    print("\n--- Top 5 Developers by Degree (The 'Rockstars') ---")
    print(deg_table)
    print("\n--- Top 5 Developers by Betweenness (The 'Bridges') ---")
    print(bet_table)
    
    return top_degree, top_betweenness

def detect_communities_louvain(G):
    """Runs Louvain community detection on the network."""
    print("\n[+] Running Louvain Community Detection...")
    communities = nx.community.louvain_communities(G, seed=42)
    
    # Sort communities by size
    communities = sorted(communities, key=len, reverse=True)
    
    print(f"Total Communities Detected (Louvain): {len(communities)}")
    
    comm_table = PrettyTable(["Community Rank", "Number of Developers"])
    for i, comm in enumerate(communities[:5], 1): # Show top 5 largest
        comm_table.add_row([i, len(comm)])
        
    print("\n--- Top 5 Largest Developer Communities (Louvain) ---")
    print(comm_table)
    
    return communities

def detect_communities_girvan_newman(G, top_nodes=500):
    """
    Runs Girvan-Newman community detection.
    To avoid infinite runtime on large graphs, we extract a subgraph of the top connected nodes.
    """
    print(f"\n[+] Extracting Subgraph of top {top_nodes} nodes for Girvan-Newman...")
    
    # Extract the subgraph of the most connected nodes to make the algorithm feasible
    top_degree_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_nodes]
    top_nodes_list = [node[0] for node in top_degree_nodes]
    subgraph = G.subgraph(top_nodes_list).copy()
    
    print("[+] Running Girvan-Newman Community Detection (First level split)...")
    # The algorithm returns an iterator over community divisions at each level
    comp = nx.community.girvan_newman(subgraph)
    
    # We only take the first level of division (the primary split)
    first_level_communities = tuple(sorted(c) for c in next(comp))
    
    # Sort communities by size
    first_level_communities = sorted(first_level_communities, key=len, reverse=True)
    
    print(f"Total Communities Detected in Subgraph (Girvan-Newman): {len(first_level_communities)}")
    
    gn_table = PrettyTable(["Community Rank", "Number of Developers"])
    for i, comm in enumerate(first_level_communities, 1):
        gn_table.add_row([i, len(comm)])
        
    print("\n--- Girvan-Newman Communities (Top Nodes Subgraph) ---")
    print(gn_table)
    
    return first_level_communities