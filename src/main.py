import networkx as nx
from data_loader import load_github_graph
from visualizer import plot_class_distribution, plot_degree_distribution
from metrics import calculate_top_centralities, detect_communities_louvain, detect_communities_girvan_newman

def main():
    print("==================================================")
    print("   GITHUB SOCIAL NETWORK - ADVANCED PIPELINE      ")
    print("==================================================")
    
    try:
        # 1. Load Data
        G, nodes_df = load_github_graph()
        
        # 2. Basic Network Overview
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        
        print("\n--- Network Overview Summary ---")
        print(f"Total Nodes (Developers): {total_nodes:,}")
        print(f"Total Edges (Connections): {total_edges:,}")
        print(f"Graph Density: {nx.density(G):.5f}")
        print("---------------------------------")
        
        # 3. Advanced Metrics & Centralities
        top_degree, top_between = calculate_top_centralities(G, top_n=5)
        
        # 4. Community Detection (Exercise 2 Focus)
        communities_louvain = detect_communities_louvain(G)
        communities_gn = detect_communities_girvan_newman(G, top_nodes=500)
        
        # 5. Visualizations
        print("\n--- Generating Visualizations ---")
        plot_class_distribution(nodes_df)
        plot_degree_distribution(G)
        print("[+] Visualizations saved successfully.")
        
        print("\n==================================================")
        print(" [+] Pipeline execution completed successfully! ")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[-] Pipeline failed: {e}")

if __name__ == "__main__":
    main()