import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import TemporalData, Data
from scipy.special import beta
import pywt

def load_data(file_path):
    data = torch.load(file_path)
    return data

def add_frequency_feature_to_temporal_data(data, C):
    G = nx.Graph()
    src_nodes = data.src.numpy()
    dst_nodes = data.dst.numpy()
    attack_labels = data.attack.numpy()

    num_edges = len(src_nodes)
    print("Adding edges to graph...")
    for i in range(num_edges):
        if i % 1000 == 0:
            print(f"Progress: {i}/{num_edges} edges added")
        G.add_edge(src_nodes[i], dst_nodes[i], attack=attack_labels[i])

    unique_nodes = set(src_nodes).union(set(dst_nodes))
    num_nodes = len(unique_nodes)
    frequency_features = {}
    feature_vectors = {}
    print("Calculating frequency features for each node...")
    for idx, node in enumerate(unique_nodes):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{num_nodes} nodes processed")
        neighbors = list(G.neighbors(node))[:10]
        subgraph = G.subgraph(neighbors + [node])
        eigenvalues, eigenvectors = eigenvalues_and_vectors_from_graph(subgraph)
        if len(eigenvalues) > 0:
            frequency_features[node] = calculate_frequency_feature(eigenvalues)
            beta_features = calculate_beta_wavelet_features(eigenvectors, C)
            daubechies_features = calculate_daubechies_wavelet_features(eigenvectors)
            feature_vectors[node] = np.concatenate((beta_features, daubechies_features))
        else:
            frequency_features[node] = np.zeros(10)  
            feature_vectors[node] = np.zeros(40)  # 20 (Beta) + 20 (Daubechies)

    features_list = [frequency_features.get(node, np.zeros(10)) for node in src_nodes]
    vectors_list = [feature_vectors.get(node, np.zeros(40)) for node in src_nodes]
    
    features_array = np.array(features_list)
    vectors_array = np.array(vectors_list)
    
    features_tensor = torch.tensor(features_array).view(-1, 10)
    vectors_tensor = torch.tensor(vectors_array).view(-1, 40)
    
    combined_tensor = torch.cat((features_tensor, vectors_tensor), dim=1)

    if hasattr(data, 'msg'):
        original_features = data.msg[:, :79]
        original_features = original_features.unsqueeze(2)
        
        combined_tensor = combined_tensor.unsqueeze(1).expand(-1, original_features.size(1), combined_tensor.size(1))
        data.msg = torch.cat([original_features, combined_tensor], dim=2)
    else:
        combined_tensor = combined_tensor.unsqueeze(1).expand(-1, 79, 50)
        data.msg = combined_tensor

    return data

def eigenvalues_and_vectors_from_graph(G):
    if len(G) > 1:
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        return eigenvalues, eigenvectors
    else:
        return np.array([]), np.array([[]])

def calculate_frequency_feature(eigenvalues):
    counts, bins = np.histogram(eigenvalues, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return counts.astype(float) / len(eigenvalues)

def calculate_beta_wavelet_features(eigenvectors, C):
    def beta_wavelet_transform(x, p, q):
        if 0 <= x <= 2:
            return (x/2)**p * (1 - x/2)**q / beta(p+1, q+1)
        else:
            return 0

    transformed_vectors = []
    for p in range(C+1):
        q = C - p
        transformed_vector = np.array([beta_wavelet_transform(x, p, q) for x in eigenvectors[:, 0]])
        transformed_vectors.append(transformed_vector)

    transformed_vectors = np.concatenate(transformed_vectors)
    if transformed_vectors.shape[0] < 20:
        transformed_vectors = np.pad(transformed_vectors, (0, 20 - transformed_vectors.shape[0]), 'constant')
    elif transformed_vectors.shape[0] > 20:
        transformed_vectors = transformed_vectors[:20]
    return transformed_vectors

def calculate_daubechies_wavelet_features(eigenvectors):
    coeffs = pywt.wavedec(eigenvectors[:, 0], 'db1', level=2)
    low_freq_vector = np.mean(coeffs[0]) if len(coeffs[0]) > 0 else np.zeros(1)
    high_freq_vector = np.mean(coeffs[1]) if len(coeffs[1]) > 0 else np.zeros(1)
    combined_vector = np.concatenate((low_freq_vector.reshape(-1), high_freq_vector.reshape(-1)))
    if combined_vector.shape[0] < 20:
        combined_vector = np.pad(combined_vector, (0, 20 - combined_vector.shape[0]), 'constant')
    elif combined_vector.shape[0] > 20:
        combined_vector = combined_vector[:20]
    return combined_vector

input_path = '../data/CIC-BoT-IoT.pt'
output_path = 'data/CIC-BoT-IoT_w.pt'

C = 4

temporal_data = load_data(input_path)
extended_data = add_frequency_feature_to_temporal_data(temporal_data, C)

def save_data(data, file_path):
    torch.save(data, file_path)
    print("Data saved successfully.")

save_data(extended_data, output_path)
