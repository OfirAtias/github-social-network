import matplotlib.pyplot as plt
import collections
import networkx as nx
from pathlib import Path

def get_output_dir():
    """Creates and returns the path to the outputs/figures directory."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    output_dir = project_root / "outputs" / "figures"
    
    # מוודא שהתיקייה קיימת, ואם לא - יוצר אותה אוטומטית
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_class_distribution(nodes_df):
    print("[+] Generating Class Distribution Pie Chart...")
    counts = nodes_df['ml_target'].value_counts()
    
    labels = ['Web Developers (Class 0)', 'ML Developers (Class 1)']
    sizes = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#1f77b4', '#ff7f0e'] 
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
            textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title("GitHub Dataset: Developer Specialization Split", fontsize=14, weight='bold', pad=20)
    
    # שמירה לתיקיית היעדים החדשה
    output_path = get_output_dir() / 'class_distribution.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved: {output_path}")

def plot_degree_distribution(G):
    print("[+] Generating Degree Distribution Plot...")
    
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_counts = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_counts.items())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(deg, cnt, width=0.80, color='#2ca02c')
    
    plt.title("Network Degree Distribution", fontsize=14, weight='bold')
    plt.ylabel("Number of Nodes (Developers)", fontsize=11)
    plt.xlabel("Degree (Number of Connections)", fontsize=11)
    
    plt.xlim(0, 100) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # שמירה לתיקיית היעדים החדשה
    output_path = get_output_dir() / 'degree_distribution.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved: {output_path}")