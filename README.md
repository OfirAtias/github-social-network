# GitHub Social Network Analysis

An academic project for the **Network Algorithms** course. This project focuses on analyzing the social network structure of GitHub developers and performing node classification using advanced network algorithms and machine learning techniques.

## Project Overview

The goal of this project is to predict whether a GitHub user is a **Web Developer** or a **Machine Learning Developer** based on their mutual follower connections and user features.

We utilize the **MUSAE GitHub Social Network dataset** provided by Stanford University (SNAP).

### Dataset Statistics

- **Nodes (Users):** ~37,700
- **Edges (Mutual Followers):** ~289,003
- **Task:** Binary Node Classification (Web vs. ML)

## Methodology & Planned Approach

1. **Exploratory Data Analysis (EDA):** Analyze network degree distribution, density, centrality metrics (PageRank), and giant components.
2. **Graph Embeddings:** Transform structural graph features into numerical vectors using techniques like `Node2Vec`.
3. **Graph Neural Networks (GNNs):** Implement a `Graph Convolutional Network (GCN)` using PyTorch Geometric to perform node classification.

## Project Structure

```text
github-social-network/
│
├── main.py            # Main script to load and fetch data
├── .gitignore         # Specifies intentionally untracked files to ignore
└── README.md          # Project documentation (this file)
```
