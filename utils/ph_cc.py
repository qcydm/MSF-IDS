import networkx as nx
import gudhi as gd
from persim import PersistenceImager
import numpy as np
from tqdm import tqdm
from line_profiler import LineProfiler
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch
import networkx as nx
from networkx import MultiDiGraph
from tqdm import tqdm
import gudhi as gd
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import sys
import torch.nn as nn


def ph_emb(G,timestamp_name: str,pixel_size=0.1,hops=4,graph_type="nx",birth_range=(0, 1),pers_range=(0, 1),):
    embedding = []
    embedding_dim = int((1 / pixel_size) ** 2)
    counter = 0
    if graph_type == "nx":
        node_itr=G.nodes
    elif graph_type == "pyg":
        node_itr=range(G.num_nodes)
    for node in tqdm(node_itr):
        if graph_type == "nx":
            subgraph = nx.ego_graph(G, node, radius=hops)
            subgraph_nodes = sorted(list(subgraph.nodes()))
            node_map = {node: i for i, node in enumerate(subgraph_nodes)}
            simplices = gd.SimplexTree()
            for u, v, d in subgraph.edges(data=True):
                simplices.insert([node_map[u], node_map[v]], filtration=d[timestamp_name])
        elif graph_type == "pyg":
            subset, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(node, hops, G.edge_index)
            subgraph_edge_attr = G.edge_attr[edge_mask]
            node_map={}
            for i,node in enumerate(subset):
                node_map[node.item()]=i
            simplices = gd.SimplexTree()
            for (u,v),d in zip(subgraph_edge_index.T,subgraph_edge_attr):
                simplices.insert([node_map[u.item()], node_map[v.item()]], filtration=d.item())
           
        diagrams = simplices.persistence(persistence_dim_max=True)
        diagrams = [(birth, death) if not np.isinf(death) else (birth, 1.0)for dim, (birth, death) in diagrams]
        if not diagrams:
            embedding.append(np.zeros(embedding_dim))
            continue
        pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range)
        pimgr.fit(diagrams)
        img = pimgr.transform(diagrams)
        img = img.flatten()
        embedding.append(img)
    return embedding

import time
import torch
import numpy as np
from tqdm import tqdm

def ph_append(data, pixel_size=0.1, hops=4, birth_range=(0, 1), pers_range=(0, 1), max_subgraph_size=1024):
    max_t = data.t.max()
    min_t = data.t.min()
    timestamp = (data.t - min_t) / (max_t - min_t)
    edge_index = data.edge_index
    embedding_dim = int((1 / pixel_size) ** 2)
    embedding = torch.ones((data.num_nodes, embedding_dim), dtype=torch.float32)
    
    for node in tqdm(range(data.num_nodes)):
        # start_time = time.time()  # 记录开始时间
        subset, _, _, edge_mask = k_hop_subgraph(node, hops, edge_index)
        # subgraph_sampling_end_time=time.time()
        # check if the subgraph is too large
        if len(subset) > max_subgraph_size:
            # randomly sample a subset of nodes
            subset_indices = torch.randperm(len(subset))[:max_subgraph_size]
            subset = subset[subset_indices]
            # make sure the node itself is included
            if node not in subset:
                subset[0] = node
            # for edge_id in range(edge_mask.shape[0]):
            #     if edge_mask[edge_id] and (data.src[edge_id] not in subset or data.dst[edge_id] not in subset):
            #         edge_mask[edge_id] = False
            edge_mask = edge_mask & torch.isin(data.src, subset) & torch.isin(data.dst, subset)
            
        subgraph_t = timestamp[edge_mask]
        srcs = data.src[edge_mask]
        dsts = data.dst[edge_mask]
        node_map = {}
        for i, node in enumerate(subset):
            node_map[node.item()] = i
        simplices = gd.SimplexTree()
        for u, v, d in zip(srcs, dsts, subgraph_t):
            simplices.insert([node_map[u.item()], node_map[v.item()]], d.item())
        diagrams = simplices.persistence(persistence_dim_max=True)
        diagrams = [(birth, death) if not np.isinf(death) else (birth, 1.0) for dim, (birth, death) in diagrams]
        if not diagrams:
            continue
        pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range)
        img = pimgr.transform(diagrams)
        img = img.flatten()
        embedding[node] = torch.tensor(img, dtype=torch.float32)

        # elapsed_time = time.time() - start_time  # 计算该循环的耗时
        # sampling_time=subgraph_sampling_end_time-start_time
        # if elapsed_time > 1:  # 如果执行时间超过1秒
        #     print(f"Node {node}: Processing time exceeded 1 second ({elapsed_time:.2f} seconds)")
        #     print("Sampling time:",sampling_time)
        #     # print more info
        #     print(f"subset: {subset}")
        #     print("num nodes:",len(subset))

    if hasattr(data, 'x'):
        data.x = torch.cat([data.x, embedding], dim=1)
    else:
        data.x = embedding

    return data

# def ph_batch(src:torch.Tensor,dst:torch.Tensor,t:torch.Tensor,pixel_size=0.1,hops=2,birth_range=(0, 1),pers_range=(0, 1),max_subgraph_size=1024):
#     n_id=torch.cat([src,dst]).unique()
#     max_t=t.max()
#     min_t=t.min()
#     timestamp=(t-min_t)/(max_t-min_t)
#     edge_index=torch.stack([src,dst],dim=0)
#     print(edge_index)
#     print(n_id)
#     embedding_dim=int((1/pixel_size)**2)
#     embedding=torch.zeros((n_id.size(0),embedding_dim),dtype=torch.float32)
#     node_id_set=set(n_id.tolist())
#     node_id_map={node:i for i,node in enumerate(n_id)}
#     # use node id map to map the edge index
#     src=node_id_map[src]
#     dst=node_id_map[dst]
#     for node in n_id:
#         subset,_,_,edge_mask=k_hop_subgraph(node,hops,edge_index)
#         if len(subset)>max_subgraph_size:
#             subset_indices=torch.randperm(len(subset))[:max_subgraph_size]
#             subset=subset[subset_indices]
#             if node not in subset:
#                 subset[0]=node
#             edge_mask=edge_mask&torch.isin(src,subset)&torch.isin(dst,subset)
#         subgraph_t=timestamp[edge_mask]
#         srcs=src[edge_mask]
#         dsts=dst[edge_mask]
#         node_map={}
#         for i,node in enumerate(subset):
#             node_map[node.item()]=i
#         simplices=gd.SimplexTree()
#         for u,v,d in zip(srcs,dsts,subgraph_t):
#             simplices.insert([node_map[u.item()],node_map[v.item()]],d.item())
#         diagrams=simplices.persistence(persistence_dim_max=True)
#         diagrams=[(birth,death) if not np.isinf(death) else (birth,1.0) for dim,(birth,death) in diagrams]
#         if not diagrams:
#             continue
#         pimgr=PersistenceImager(pixel_size=pixel_size,birth_range=birth_range,pers_range=pers_range)
#         img=pimgr.transform(diagrams)
#         img=img.flatten()
#         embedding[node]=torch.tensor(img,dtype=torch.float32)
#     return embedding


def get_subgraph(edge_index:torch.Tensor, center_node:int, hops:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
    nodes = torch.tensor([center_node], device=device)
    src=edge_index[0]
    dst=edge_index[1]
    src_us=edge_index[0].unsqueeze(1)
    dst_us=edge_index[1].unsqueeze(1)
    for _ in range(hops):
        mask = (src_us == nodes).any(dim=1) | (dst_us == nodes).any(dim=1)
        edge_mask = edge_mask | mask
        nodes = torch.cat([nodes, src[mask], dst[mask]]).unique()
    return nodes.cpu().numpy(), edge_mask.cpu()

# import torch
# 
# def get_subgraph(edge_index: torch.Tensor, center_node: int, hops: int):
#     device = torch.device("cpu")
    
#     # 获得最大节点索引以构建邻接矩阵
#     num_nodes = edge_index.max() + 1
#     adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    
#     # 填充邻接矩阵
#     adjacency_matrix[edge_index[0], edge_index[1]] = True
#     adjacency_matrix[edge_index[1], edge_index[0]] = True  # 如果是无向图

#     # 初始化节点集
#     current_nodes = torch.tensor([center_node], device=device)
#     visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
#     visited[current_nodes] = True
    
#     # 使用BFS进行跳数限制的节点扩展
#     for _ in range(hops):
#         neighbors = adjacency_matrix[current_nodes].any(dim=0)
#         neighbors &= ~visited  # 只关注未访问过的节点
#         visited |= neighbors
#         current_nodes = neighbors.nonzero(as_tuple=True)[0]

#     edge_mask = adjacency_matrix[current_nodes].any(dim=0)
#     return current_nodes.cpu().numpy(), edge_mask.cpu()




def ph_batch(
    src: torch.Tensor,
    dst: torch.Tensor,
    t: torch.Tensor,
    n_id: torch.Tensor,
    pixel_size=0.1,
    hops=2,
    birth_range=(0, 1),
    pers_range=(0, 1),
    retain_inf=True
):
    """
    Compute persistent homology features for a batch of graphs.

    Args:
        src (torch.Tensor): Source nodes of the edges in the graph batch.
        dst (torch.Tensor): Destination nodes of the edges in the graph batch.
        t (torch.Tensor): Timestamps of the edges in the graph batch.
        pixel_size (float, optional): Size of the pixels in the persistence image. Defaults to 0.1.
        hops (int, optional): Number of hops for constructing the subgraph. Defaults to 32.
        birth_range (tuple, optional): Range of birth values for persistence diagram normalization. Defaults to (0, 1).
        pers_range (tuple, optional): Range of persistence values for persistence diagram normalization. Defaults to (0, 1).
        max_subgraph_size (int, optional): Maximum size of the subgraph. Defaults to 1024.
        retain_inf (bool, optional): Whether to retain infinite persistence points. Defaults to True.

    Returns:
        torch.Tensor: Embedding matrix containing the persistent homology features for each graph in the batch.
    """
    max_t = t.max()
    min_t = t.min()
    if max_t==min_t:
        max_t+=1
    timestamp = (t - min_t) / (max_t - min_t)
    timestamp = (t - min_t) / (max_t - min_t)
    n_id_map={}
    for i,node in enumerate(n_id):
        n_id_map[node.item()]=i
    num_nodes=n_id.size(0)
    src=torch.tensor([n_id_map[node.item()] for node in src],dtype=torch.long)
    dst=torch.tensor([n_id_map[node.item()] for node in dst],dtype=torch.long)
    embedding_dim = int((1 / pixel_size) ** 2)
    embedding=torch.zeros((num_nodes, embedding_dim), dtype=torch.float32)
    edge_index=torch.stack([src,dst],dim=0)
    for i in range(num_nodes):
        node=i
        subset, _, _, edge_mask = k_hop_subgraph(node, hops, edge_index,num_nodes=num_nodes)
        subgraph_t = timestamp[edge_mask]
        subgraph_edge_index=edge_index[:,edge_mask]
        node_map = {}
        for i, node in enumerate(subset):
            node_map[node.item()] = i
        simplices = gd.SimplexTree()
        for edge,d in zip(subgraph_edge_index.T,subgraph_t):
            u=edge[0].item()
            v=edge[1].item()
            simplices.insert([node_map[u], node_map[v]], d.item())
        diagrams = simplices.persistence(persistence_dim_max=True)
        if retain_inf:
            diagrams = [
                (birth, death) if not np.isinf(death) else (birth, 1.0)
                for dim, (birth, death) in diagrams
            ]
        else:
            diagrams = [
                (birth, death) for dim, (birth, death) in diagrams if not np.isinf(death)
            ]
        if not diagrams:
            continue
        pimgr = PersistenceImager(
            pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range
        )
        img = pimgr.transform(diagrams)
        img = img.flatten()
        embedding[i] = torch.tensor(img, dtype=torch.float32)
    return embedding

import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import gudhi as gd
from persim import PersistenceImager
from torch_scatter import scatter

def ph_enc(
    src: torch.Tensor,
    dst: torch.Tensor,
    t: torch.Tensor,
    n_id: torch.Tensor,
    pixel_size=0.1,
    hops=2,
    birth_range=(0, 1),
    pers_range=(0, 1),
    retain_inf=True
):
    max_t, min_t = t.max(), t.min()
    timestamp = (t - min_t) / (max_t - min_t + 1e-8)  # Add small epsilon to avoid division by zero
    
    n_id_map = {node.item(): i for i, node in enumerate(n_id)}
    num_nodes = n_id.size(0)
    
    src = torch.tensor([n_id_map[node.item()] for node in src], dtype=torch.long)
    dst = torch.tensor([n_id_map[node.item()] for node in dst], dtype=torch.long)
    
    embedding_dim = int((1 / pixel_size) ** 2)
    embedding = torch.zeros((num_nodes, embedding_dim), dtype=torch.float32)
    edge_index = torch.stack([src, dst], dim=0)
    
    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range)
    
    for i in range(num_nodes):
        subset, _, _, edge_mask = k_hop_subgraph(i, hops, edge_index, num_nodes=num_nodes)
        subgraph_t = timestamp[edge_mask]
        subgraph_edge_index = edge_index[:, edge_mask]
        
        node_map = {n.item(): j for j, n in enumerate(subset)}
        
        simplices = gd.SimplexTree()
        for edge, d in zip(subgraph_edge_index.T, subgraph_t):
            u, v = node_map[edge[0].item()], node_map[edge[1].item()]
            simplices.insert([u, v], d.item())
        
        diagrams = simplices.persistence(persistence_dim_max=True)
        
        if retain_inf:
            diagrams = [(birth, 1.0 if np.isinf(death) else death) for _, (birth, death) in diagrams]
        else:
            diagrams = [(birth, death) for _, (birth, death) in diagrams if not np.isinf(death)]
        
        if diagrams:
            img = pimgr.transform(diagrams).flatten()
            embedding[i] = torch.tensor(img, dtype=torch.float32)
    
    return embedding



class PH_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1):
        super(PH_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out



def main():
    # Check if a filename was provided as a command line argument
    if len(sys.argv) < 2:
        print("Please provide a filename as a command-line argument.")
        return

    # Get the filename from the command line arguments
    filename = sys.argv[1]

    data=torch.load(filename)
    data=ph_append(data)
    torch.save(data,filename+"_ph")



if __name__ == "__main__":
    main()