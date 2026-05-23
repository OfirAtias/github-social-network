import matplotlib.pyplot as plt
import collections
import networkx as nx
import os

def plot_class_distribution(nodes_df):
    """Generates a pie chart showing the Web vs ML developer split."""
    print("[+] Generating Class Distribution Pie Chart...")
    counts = nodes_df['ml_target'].value_counts()
    
    labels = ['Web Developers (Class 0)', 'ML Developers (Class 1)']
    sizes = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#1f77b4', '#ff7f0e']  # Tech blue and ML orange
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
            textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title("GitHub Dataset: Developer Specialization Split", fontsize=14, weight='bold', pad=20)
    
    # Save the plot as an image for Canva
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)
    plt.close()
    print(" -> Saved: class_distribution.png")

def plot_degree_distribution(G):
    """Generates a Degree Distribution plot showing the structural 'Power Law' of the network."""
    print("[+] Generating Degree Distribution Plot...")
    
    # Calculate degrees for all nodes
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_counts = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_counts.items())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(deg, cnt, width=0.80, color='#2ca02c')
    
    plt.title("Network Degree Distribution", fontsize=14, weight='bold')
    plt.ylabel("Number of Nodes (Developers)", fontsize=11)
    plt.xlabel("Degree (Number of Connections)", fontsize=11)
    
    # Zooming in slightly on the main dense area of the distribution for clarity
    plt.xlim(0, 100) 
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('degree_distribution.png', dpi=300)
    plt.close()
    print(" -> Saved: degree_distribution.png")