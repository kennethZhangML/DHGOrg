# DHGOrg
# DiHypergraph: Analyzing and Visualizing GitHub Repositories

Dihypergraph offers a suite of tools designed to construct, visualize, and analyze data from GitHub repositories. Our primary data structure is a directed hypergraph, a generalization of a standard directed graph, which enables the representation of complex relationships in GitHub repositories. By using state-of-the-art techniques, we provide insightful visualizations and identify potential anomalies in commit messages using autoencoders.

## Features

1. **Construct Directed Hypergraphs from GitHub Data**: Convert GitHub issues, commits, and pull requests into nodes of a dihypergraph, capturing rich relationships among them.
2. **Visualize Dihypergraph Data**: Plot the directed hypergraph using NetworkX and Matplotlib.
3. **Anomaly Detection in Commit Messages**: Utilizing the power of Transformer-based autoencoders to identify potentially anomalous commit messages.

## Quick Start

```bash
# Clone the repository
git clone <repository_link>

# Navigate to the repository folder
cd dihypergraph
```

## Usage
### 1. Constructing Dihypergraph
To generate a dihypergraph JSON representation from a GitHub repository:

```python
from DHG import DHGConstructor

BASE_URL = "https://api.github.com/repos/YOUR_REPO_NAME/"
TOKEN = "YOUR_GITHUB_TOKEN"

my_dhg = DHGConstructor(BASE_URL, HEADERS, "output_name")
my_dhg.run_loop()
```

### 2. Visualize Dihypergraph
Visualize the dihypergraph and compute associations between issues:

```python
from DHGVisualizer import DHGVisualizer

dhg_vis = DHGVisualizer("dihypergraph_output_name.json")
dhg_vis.create_graph()
dhg_vis.visualize()
```

