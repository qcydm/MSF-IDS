import gudhi as gd
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from matplotlib.patches import Rectangle
import networkx as nx
import matplotlib as mpl
from persim import PersistenceImager
import random
import pandas as pd
import os
import re

import torch

def pre(extracted_data):
    unique_values = pd.unique(extracted_data[['Dst IP', 'Src IP']].values.ravel('K'))
    mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
    extracted_data['Dst IP'] = extracted_data['Dst IP'].map(mapping)
    extracted_data['Src IP'] = extracted_data['Src IP'].map(mapping)
    extracted_data['Timestamp'] = pd.to_datetime(extracted_data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    extracted_data.sort_values('Timestamp', inplace=True)
    
    extracted_data.reset_index(drop=True, inplace=True)
    extracted_data['Timestamp'] = range(1, len(extracted_data) + 1)
    tuple_data = list(extracted_data.itertuples(index=False, name=None))
    
    return tuple_data

def gen(data):
    # 从字符串数据中提取元组

    temp_dir = os.path.join(os.environ['temp'],str(random.randint(100000000000,999999999999)))
    # temp_dir = os.path.join('temp',str(random.randint(100000000000,999999999999)))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    traffic_data = data

    # 持续图像器配置
    pixel_size = 0.1
    birth_range = (0, 1)
    pers_range = (0, 1)
    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range)

    # 数据归一化
    max_t = max(t for _, _, t in traffic_data)
    min_t = min(t for _, _, t in traffic_data)
    normalized_traffic = [[u, v, (t - min_t) / (max_t - min_t)] for u, v, t in traffic_data]

    # 构建复形
    simplices = gd.SimplexTree()
    for u, v, t in normalized_traffic:
        simplices.insert([u, v], filtration=t)

    # 计算持续同调
    diagram = simplices.persistence(persistence_dim_max=True)

    # 绘制并显示持续图
    original_figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = [6,6]
    gd.plot_persistence_diagram(diagram)
    plt.savefig(os.path.join(temp_dir,'c.png'))

    plt.rcParams['figure.figsize'] = original_figsize

    # 绘制并显示条形码
    gd.plot_persistence_barcode(diagram)
    plt.savefig(os.path.join(temp_dir,'b2.png'))

    # 转换为持续图像
    diagrams = [
        (birth, death) if not np.isinf(death) else (birth, 1.0)
        for dim, (birth, death) in diagram if dim == 1  # Assuming we're interested in dimension 1
    ]
    img = pimgr.transform(diagrams)

    # 显示持续图像
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='hot', interpolation='nearest')

    plt.savefig(os.path.join(temp_dir,'d.png'))


    # 处理图像数据，打平并转换为 torch 张量
    flat_img = img.flatten()
    torch.set_printoptions(linewidth=150)
    tensor_img = torch.tensor(flat_img, dtype=torch.float32)

    # 绘制几何空间连接
    plt.figure(figsize=(16, 12))

    random.seed(10)

    # 构建点集
    points = {i: (np.random.random(), np.random.random()) for i in set(sum(([(u, v) for u, v, _ in normalized_traffic]), ()))}

    # 选择几个阈值
    thresholds = [0.2, 0.4, 0.8, 0.9, 1.0]

    # 绘图
    fig, axs = plt.subplots(1, len(thresholds), figsize=(15, 3))

    for ax, threshold in zip(axs, thresholds):
        # 创建连接矩阵
        connectivity = {k: [] for k in points}
        for u, v, t in normalized_traffic:
            if t <= threshold:
                connectivity[u].append(v)
                connectivity[v].append(u)
                if 6 in (u, v):  # 特别标记包含节点 6 的线为红色
                    # line_color = '#ff8884'
                    line_color = '#2878b5'
                else:
                    line_color = '#2878b5'
                ax.plot([points[u][0], points[v][0]], [points[u][1], points[v][1]], color=line_color, linestyle='-')

        # 检测并填充三角形
        for u, v, w in combinations(points, 3):
            if v in connectivity[u] and w in connectivity[v] and u in connectivity[w]:
                triangle = np.array([points[u], points[v], points[w]])
                ax.fill(triangle[:, 0], triangle[:, 1], '#2878b5', alpha=0.4, edgecolor='#2878b5')

        for k, (x, y) in points.items():
            if k == 6:
                # ax.scatter(x, y, c='#E88482')
                ax.scatter(x, y, c='#6F6F6F')
            else:
                ax.scatter(x, y, c='#6F6F6F')

        ax.set_title(f't = {threshold:.1f}')

        # 添加黑色方框
        # 获取轴的高度和宽度
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]

        # 添加方框
        rect = Rectangle((xlim[0], ylim[0]), width, height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        ax.axis('off')

    plt.savefig(os.path.join(temp_dir,'b1.png'))

    # 创建一个空的有向图
    G = nx.DiGraph()

    # 设置字体为 Times New Roman
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'

    plt.figure(figsize=(6,12))

    # 添加边和节点到图中
    for src, dst, time in traffic_data:
        G.add_edge(src, dst, time=time)

    # 绘制网络图
    pos = nx.spring_layout(G)

    # 设置节点和边的颜色
    # edge_colors = ['#87B1D4' if src != 6 and dst != 6 else '#EE7577' for src, dst in G.edges()]
    # node_colors = ['#87B1D4' if node != 6 else '#EE7577' for node in G.nodes()]
    edge_colors = ['#87B1D4']
    node_colors = ['#87B1D4']

    # 绘制图形
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, font_size=20, font_color='black', font_weight='light', node_size=800)

    # 保存图形为SVG格式
    plt.title('Network Graph', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(temp_dir,"a.png"))

    return temp_dir,str(tensor_img)


# if __name__ == '__main__':

#     data = [
#     (1, 2, 1),
#     (1, 2, 2),
#     (1, 2, 3),
#     (2, 5, 4),
#     (5, 3, 5),
#     (3, 1, 6),
#     (5, 1, 7),
#     (5, 2, 8),
#     (4, 5, 9),
#     (4, 3, 10),
#     (1, 7, 1),
#     (1, 8, 2),
#     (7, 1, 3),
#     (8, 1, 4),
#     (2, 9, 5),
#     (2, 10, 6),
#     (9, 2, 7),
#     (10, 2, 8),
#     (3, 11, 9),
#     (11, 3, 10),
#     (4, 11, 11),
#     (7, 8, 12),
#     (9, 12, 13),
#     (12, 9, 14),
#     (3, 8, 15),
#     (8, 3, 16),
#     (1, 11, 17),
#     (11, 1, 18),
#     (8, 11, 19)
#     ]

#     print(gen(data))