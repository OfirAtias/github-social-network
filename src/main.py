import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_github_graph
from visualizer import plot_class_distribution, plot_degree_distribution
from metrics import calculate_top_centralities, detect_communities_louvain, analyze_community_labels, detect_communities_girvan_newman

def run_ml_classification(G, nodes_df):
    """
    The Crown Jewel: Node2Vec embeddings + Machine Learning Classification.
    Predicts if a developer is Web (0) or ML (1) based ONLY on network structure.
    """
    print("\n==================================================")
    print("   ADVANCED ML PIPELINE (Node Classification)     ")
    print("==================================================")
    
    # 1. Generate Embeddings using Node2Vec
    print("[+] Generating Node Embeddings (Node2Vec) - This may take a few minutes...")
    # Parameters are kept small to run reasonably fast. 
    # For a real run on 37k nodes, you might need more time or a smaller subgraph.
    node2vec = Node2Vec(G, dimensions=64, walk_length=15, num_walks=10, workers=4, quiet=True)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    
    # Extract features (embeddings) and labels for the nodes
    print("[+] Preparing Data for Machine Learning...")
    X = [] # Features (Embeddings)
    y = [] # Labels (0 for Web, 1 for ML)
    
    target_col = 'ml_target' if 'ml_target' in nodes_df.columns else nodes_df.columns[-1]
    
    for node in G.nodes():
        node_str = str(node)
        # Check if we have an embedding and a label for this node
        if node_str in model.wv and int(node) in nodes_df.index:
            try:
                embedding = model.wv[node_str]
                label = int(nodes_df.loc[int(node), target_col])
                X.append(embedding)
                y.append(label)
            except KeyError:
                continue
                
    # 2. Train-Test Split
    print(f"[+] Training data size: {len(X)} nodes.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Random Forest Classifier
    print("[+] Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Predict and Evaluate
    print("[+] Evaluating Model Performance...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n---> Classification Accuracy: {accuracy * 100:.2f}% <---")
    
    print("\nClassification Report:")
    # target_names represent [0: Web Dev, 1: ML Dev]
    print(classification_report(y_test, y_pred, target_names=["Web Devs", "ML Devs"]))
    
    return clf


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
        
        # 4. Community Detection & Analysis
        communities_louvain = detect_communities_louvain(G)
        
        # Correlating structure with labels
        analyze_community_labels(communities_louvain, nodes_df, top_k=5)
        
        # Run Girvan-Newman on subgraph
        communities_gn = detect_communities_girvan_newman(G, top_nodes=500)
        
        # 5. Visualizations
        print("\n--- Generating Visualizations ---")
        plot_class_distribution(nodes_df)
        plot_degree_distribution(G)
        print("[+] Visualizations saved successfully.")
        
        # 6. Machine Learning Classification
        # We run this on a subgraph if the full graph is too slow for testing
        print("\n[!] To save time during testing, we'll run ML on the top 1000 nodes.")
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:1000]
        subgraph_nodes = [n[0] for n in top_nodes]
        G_sub = G.subgraph(subgraph_nodes)
        
        run_ml_classification(G_sub, nodes_df)
        
        print("\n==================================================")
        print(" [+] Pipeline execution completed successfully! ")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[-] Pipeline failed: {e}")

if __name__ == "__main__":
    main()