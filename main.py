from data_loader import load_github_graph
from visualizer import plot_class_distribution, plot_degree_distribution

def main():
    print("==================================================")
    print("   GITHUB SOCIAL NETWORK - MODULAR PIPELINE      ")
    print("==================================================")
    
    try:
        # 1. Load data and construct graph
        G, nodes_df = load_github_graph()
        
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        avg_degree = (2 * total_edges) / total_nodes if total_nodes > 0 else 0
        
        print("\n--- Network Overview Summary ---")
        print(f"Total Nodes (Developers): {total_nodes:,}")
        print(f"Total Edges (Connections): {total_edges:,}")
        print(f"Average Degree: {avg_degree:.2f}")
        print("---------------------------------")
        
        # 2. Trigger Visualizations for presentation
        print("\n--- Generating Visualizations for Canva ---")
        plot_class_distribution(nodes_df)
        plot_degree_distribution(G)
        
        print("\n[+] Success! All milestone outputs generated.")
        print("You can now find 'class_distribution.png' and 'degree_distribution.png' in your project folder.")
        
    except Exception as e:
        print(f"\n[-] Pipeline failed: {e}")

if __name__ == "__main__":
    main()