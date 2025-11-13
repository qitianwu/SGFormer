import numpy as np
import torch
from torch import long
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def build_graph_by_pos(height: long, width: long, node_features=None):
    """
    简单的八向建图 8*height*width
    """
    edge_index = []
    for i in range(height):
        for j in range(width):
            node_idx = i * width + j
            if i > 0:  # up
                edge_index.append([node_idx, node_idx - width])
            if i < height - 1:  # down
                edge_index.append([node_idx, node_idx + width])
            if j > 0:  # left
                edge_index.append([node_idx, node_idx - 1])
            if j < width - 1:  # right
                edge_index.append([node_idx, node_idx + 1])
            if i > 0 and j > 0:  # left and up
                edge_index.append([node_idx, node_idx - width - 1])
            if i > 0 and j < width - 1:  # right and up
                edge_index.append([node_idx, node_idx - width + 1])
            if i < height - 1 and j > 0:  # left and down
                edge_index.append([node_idx, node_idx + width - 1])
            if i < height - 1 and j < width - 1:  # right and down
                edge_index.append([node_idx, node_idx + width + 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index

def build_graph_by_Knn(height: long, width: long, node_features=None, k=10):
    num_nodes = height * width
    edge_index = []
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(node_features)
    distances, indices = nbrs.kneighbors(node_features)

    # 遍历每个节点的近邻
    for i in range(num_nodes):
        for j in range(1, k):  # 从1开始，跳过自己
            neighbor_index = indices[i][j]
            edge_index.append((i, neighbor_index))

    # 将边列表转换为 NumPy 数组
    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 转置为 [2, num_edges]

    return edge_index


def get_weight(node_features, edge_index, sigma=10):
    """根据节点特征和边索引计算边权重，并返回张量形式的权重"""
    num_edges = edge_index.shape[1]  # 边的数量
    edge_weights = torch.zeros(num_edges, dtype=torch.float)

    def gaussian_kernel(x, y, sig=1.0):
        return torch.exp(-torch.norm(x - y) ** 2 / (sig ** 2))

    # 遍历每条边
    for i in range(num_edges):
        start_node = edge_index[0, i]  # 从节点
        end_node = edge_index[1, i]  # 到节点

        # 提取节点特征
        x_i = node_features[start_node]  # 从节点特征
        x_j = node_features[end_node]  # 到节点特征

        # 计算高斯核权重
        edge_weights[i] = gaussian_kernel(x_i, x_j, sigma)

    return edge_weights


def build_graph_by_fix(node_fea, ground_truth, train_idx, row, col):
    """
    先对数据进行LDA降维之后 对降维光谱特征+位置特征进行knn建图
    主要是加入pos 但是单纯的加入pos对于欧式距离的衡量改变不大，所以先降维成channel再增加pos信息
    这样pos信息所占比重就能上升，以此查看与单纯的220个波段的knn之间的区别
    """
    edge_index = []
    k = 15  # 邻居数量
    weight_factor = 15  # 权重系数

    x = node_fea[train_idx]
    y = ground_truth[train_idx]
    lda = LinearDiscriminantAnalysis()
    # 调用fit方法　对训练集像素点的特征和标签进行拟合
    lda.fit(x, y - 1)
    x_new = lda.transform(node_fea)

    all_indices = np.arange(x_new.shape[0])  # 计算 x_new 中所有节点的索引
    x_coords = all_indices // col  # 计算 x 坐标
    y_coords = all_indices % col  # 计算 y 坐标

    # 将坐标信息添加到 x_new 中
    coords = np.column_stack((x_coords, y_coords))  # 合并 x 和 y 坐标
    x_new_with_coords = np.hstack((x_new, coords))  # 将坐标与 x_new 合并

    # 进行最小-最大标准化
    x_min = np.min(x_new_with_coords, axis=0)  # 每列的最小值
    x_max = np.max(x_new_with_coords, axis=0)  # 每列的最大值

    # 标准化
    x_normalized = (x_new_with_coords - x_min) / (x_max - x_min)

    x_weighted = x_normalized.copy()
    x_weighted[:, -2:] *= weight_factor  # 增加最后两个维度的权重

    knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_weighted)
    distances, indices = knn.kneighbors(x_weighted)

    # 遍历每个节点的近邻
    for i in range(row * col):
        for j in range(1, k):  # 从1开始，跳过自己
            neighbor_index = indices[i][j]
            edge_index.append((i, neighbor_index))

    # 将边列表转换为 NumPy 数组
    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 转置为 [2, num_edges]

    return edge_index