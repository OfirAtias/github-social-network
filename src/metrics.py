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
    
    print(f"Total Communities Detected: {len(communities)}")
    
    comm_table = PrettyTable(["Community Rank", "Number of Developers"])
    for i, comm in enumerate(communities[:5], 1): # Show top 5 largest
        comm_table.add_row([i, len(comm)])
        
    print("\n--- Top 5 Largest Developer Communities ---")
    print(comm_table)
    
    return communities