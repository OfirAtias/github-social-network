import pandas as pd
import requests
import json

# 1. Define direct URLs to the Stanford SNAP datasets
edges_url = "https://snap.stanford.edu/data/musae_git_edges.csv"
targets_url = "https://snap.stanford.edu/data/musae_git_target_idents.json"

# 2. Download and load the network edges (connections) directly into a Pandas DataFrame
print("Starting to download network edges...")
edges_df = pd.read_csv(edges_url)
print(f"Edges loaded successfully! Total connections found: {len(edges_df)}")

# 3. Download and load the node features/targets (user classification)
print("\nStarting to download node targets...")
response = requests.get(targets_url)
target_data = response.json()
print(f"Targets loaded successfully! Total users found: {len(target_data)}")

# 4. Preview the first 5 rows of the dataset to verify everything works
print("\nPreviewing the first 5 edges in the graph:")
print(edges_df.head())