import pandas as pd
import os

print("========================")
print("   GITHUB SOCIAL NETWORK ")
print("========================")

base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Define exact paths based on the debug output
edges_path = os.path.join(base_dir, "data", "musae_git_edges.csv")
targets_path = os.path.join(base_dir, "data", "musae_git_target.csv")

if os.path.exists(edges_path) and os.path.exists(targets_path):
    # 2. Load the Edges (Connections)
    edges_df = pd.read_csv(edges_path)
    total_connections = len(edges_df)
    
    # 3. Load the Targets (Nodes and Classifications)
    nodes_df = pd.read_csv(targets_path)
    total_users = len(nodes_df)
    
    # 4. Calculate network metrics
    avg_connections = (2 * total_connections) / total_users if total_users > 0 else 0
    
    # Print clean results for your presentation slides
    print(f"Total Nodes (Number of Developers): {total_users:,}")
    print(f"Total Edges (Number of Connections): {total_connections:,}")
    print(f"Average Number of Followers per User: {avg_connections:.2f}")
    
    # 5. Calculate Web vs ML distribution
    # Let's check the column name for the classification (usually 'ml_target')
    target_col = [col for col in nodes_df.columns if 'target' in col or 'ml' in col]
    
    if target_col:
        col_name = target_col[0]
        counts = nodes_df[col_name].value_counts()
        web_devs = counts.get(0, 0)
        ml_devs = counts.get(1, 0)
        
        print("\n--- Developer Specialization Split (EDA) ---")
        print(f" -> Web Developers count (Class 0): {web_devs:,} ({web_devs/total_users*100:.1f}%)")
        print(f" -> Machine Learning Developers count (Class 1): {ml_devs:,} ({ml_devs/total_users*100:.1f}%)")
    else:
        print("\n--- Target File Structure ---")
        print(nodes_df.head())
        
else:
    print("[-] Error: Files not found. Please check paths.")