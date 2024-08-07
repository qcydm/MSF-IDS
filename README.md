# README.md
# Unsupervised Intrusion Detection based on Multi-Scale Spatial-Frequency Lenses

## Overview
This project is a PyTorch-based implementation of an advanced Unsupervised Intrusion Detection System (UNIDS) that utilizes a novel approach involving multi-scale spatial-frequency lenses for the detection of network anomalies. The system is designed to operate without the need for labeled data, making it highly adaptable to various network environments.

## Features
- **Multi-Scale Analysis**: Captures traffic features at different scales using wavelet transformations.
- **Self-Supervised Learning**: Employs self-expressiveness to enhance feature representation.
- **Graph Neural Networks (GNNs)**: Utilizes graph convolutional networks for learning from graph-structured data.
- **Spectral Clustering**: Integrates spectral clustering for unsupervised classification.
- **TensorBoard Integration**: Provides real-time monitoring of training and evaluation metrics.

## Requirements
- Python
- PyTorch
- time
- NumPy
- SciPy
- scikit-learn
- tqdm
- YAML
- os

## Installation
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset
Benchmark Datasets The system is trained and tested on several benchmark datasets, which are commonly used in the field of network traffic analysis and intrusion detection. These datasets provide a rich source of labeled data for training and evaluating models.

- CIC-ToN-IoT: A dataset designed for evaluating network-based intrusion detection systems in the context of IoT networks.
- CIC-BoT-IoT: Another IoT-focused dataset that simulates botnet attacks in a network environment.
- NF-UNSW-NB15-v2: A network flow dataset containing modern normal and attack traffic traces from a university network.
- NF-UQ-NIDS：A dataset from the University of Queensland for testing NIDS, featuring various network traffic scenarios.
- NF-BoT-IoT-v2：An updated dataset focusing on IoT botnet attacks, providing refined data for network security research.

You can download above datasets used in this paper from the following URLs: [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/). Then, select one of the three datasets.py files under the data directory to process the corresponding dataset.


## Usage
### Configuration
Modify the `args_set` in the script `options` to configure the system parameters according to your dataset and requirements.

### Training
Run the training script to start the UNIDS training process:
```bash
python main.py
```


## Model Components
1. **Memory Module**: Manages the state of nodes in the graph.
2. **Merge Layer**: Combines node features for graph convolution.
3. **Feature Extractor**: Processes raw traffic data into a suitable format for GNNs.
4. **Self-Expressive Model**: A module that promotes self-expressiveness in node embeddings.
5. **Cluster Model**: Performs clustering on the learned embeddings.

## Loss Functions
- **Graph Convolutional Loss**: Measures the difference between input and output of the GCN layers.
- **Self-Expressiveness Loss**: Encourages the model to learn expressive node embeddings.
- **Cluster Loss**: Evaluates the model's ability to cluster similar traffic patterns.


## Spatial-Frequency Encoders

### Spatial Encoder NAPH (Network-Aware Persistent Homology)

#### Table of Contents

- [README.md](#readmemd)
  - [NAPH (Network-Aware Persistent Homology)](#naph-network-aware-persistent-homology)
    - [Table of Contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Requirements](#requirements)
    - [Functions](#functions)
      - [ph\_enc](#ph_enc)
    - [Example](#example)

#### Introduction

NAPH, or Network-Aware Persistent Homology, is a comprehensive Python toolkit purpose-built for the analysis of temporal graphs through the application of persistent homology—a cutting-edge technique within the realm of Topological Data Analysis (TDA). This versatile suite equips researchers and developers with the means to distill significant topological features from dynamic graph data, features that can be harnessed for a range of machine learning applications such as graph classification, node classification, and link prediction. The library's compatibility with various graph representations, including those from PyG and networkx, ensures broad applicability and seamless integration into diverse analytical workflows.

#### Requirements

Please refer to the `requirements.txt` file for a list of required packages and their versions. You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

#### Functions

##### ph_enc

The `ph_enc` function computes persistent homology embeddings for a set of nodes. Here is a detailed explanation of each parameter:

- `source_nodes (torch.Tensor)`: A tensor containing the source nodes of the edges in the graph.
- `destination_nodes (torch.Tensor)`: A tensor containing the destination nodes of the edges in the graph.
- `edge_timestamps (torch.Tensor)`: A tensor containing the timestamps of the edges in the graph.
- `node_ids (torch.Tensor)`: A tensor containing the unique node IDs in the graph.
- `logger`: A logger object for logging purposes (optional).
- `pixel_size (float, optional)`: The size of the pixels in the persistence image. Default is 0.1.
- `num_hops (int, optional)`: The number of hops for constructing the subgraph. Default is 2.
- `max_subgraph_nodes (int, optional)`: The maximum size of the subgraph. Default is 1024.
- `birth_range (tuple, optional)`: The range of birth values for persistence diagram normalization. Default is (0, 1).
- `persistence_range (tuple, optional)`: The range of persistence values for persistence diagram normalization. Default is (0, 1).
- `retain_infinite (bool, optional)`: Whether to retain infinite persistence points. Default is True.
- `normalize (bool, optional)`: Whether to normalize the embeddings. Default is False.

#### Example

Here is an example demonstrating how to use the `ph_enc` function to compute persistent homology embeddings for a set of nodes:

```python
import torch
from script_name import ph_enc

# Example data
source_nodes = torch.tensor([0, 1, 1, 2], dtype=torch.long)
destination_nodes = torch.tensor([1, 0, 2, 1], dtype=torch.long)
edge_timestamps = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32)
node_ids = torch.tensor([0, 1, 2], dtype=torch.long)

# Compute persistent homology embeddings
embeddings = ph_enc(source_nodes, destination_nodes, edge_timestamps, node_ids)

# Print the embeddings
print(embeddings)
```


### Frequency Encoder

This Python script is designed to augment temporal network data with frequency and wavelet features extracted from the network's graph structure. It utilizes libraries such as PyTorch, NetworkX, NumPy, Pandas, and PyWavelets to process graph data and generate additional features that can be useful for anomaly detection, particularly in the context of cyber security applications like intrusion detection systems.

#### Prerequisites

Before running this script, ensure you have the following packages installed:

- `pandas`
- `numpy`
- `networkx`
- `torch`
- `torch_geometric`
- `scipy`
- `pywt`

You can install these packages using pip:

```bash
pip install pandas numpy networkx torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric scipy pywt
```

#### Usage

1. **Data Loading**:
   The script loads temporal data from a `.pt` file using the `load_data` function. Ensure your data is in the correct format and the path to your data file is specified correctly.

2. **Feature Extraction**:
   The `add_frequency_feature_to_temporal_data` function adds frequency features based on the graph's Laplacian matrix and wavelet features using Beta and Daubechies wavelets. These features are calculated for each node in the graph and then concatenated into the data object.

3. **Saving Extended Data**:
   The `save_data` function saves the extended data back to a new `.pt` file.

#### Running the Script

To run the script, simply execute it in your Python environment. You may need to adjust the `input_path` and `output_path` variables to point to the correct input and output files.

```python
if __name__ == "__main__":
    C = 4  # Wavelet parameter C
    input_path = ''
    output_path = ''

    temporal_data = load_data(input_path)
    extended_data = add_frequency_feature_to_temporal_data(temporal_data, C)
    save_data(extended_data, output_path)
```

#### Functions

- `load_data(file_path)`: Loads the data from a given file path.
- `add_frequency_feature_to_temporal_data(data, C)`: Adds frequency and wavelet features to the data.
- `eigenvalues_and_vectors_from_graph(G)`: Computes the eigenvalues and eigenvectors of a graph's Laplacian matrix.
- `calculate_frequency_feature(eigenvalues)`: Calculates frequency features based on the distribution of eigenvalues.
- `calculate_beta_wavelet_features(eigenvectors, C)`: Applies the Beta wavelet transform to eigenvectors.
- `calculate_daubechies_wavelet_features(eigenvectors)`: Applies the Daubechies wavelet transform to eigenvectors.
- `save_data(data, file_path)`: Saves the data to a file at the given path.

#### Notes

- This script assumes that the input data is structured in a way that it can be loaded using `torch.load` and that it contains attributes `src`, `dst`, and `attack` which represent source nodes, destination nodes, and attack labels respectively.
- The wavelet feature calculation is simplified for demonstration purposes and might require adjustments depending on the specific requirements of your application.
- Make sure to handle exceptions and edge cases properly when integrating this script into a larger project.

#### Acknowledgments

This script was developed as part of a research effort to improve the robustness of machine learning models in detecting anomalies within network traffic data. Contributions from the cybersecurity and signal processing communities are gratefully acknowledged.
