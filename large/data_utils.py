import os
from collections import defaultdict
import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
import spectral as spy
from matplotlib import cm
from google_drive_downloader import GoogleDriveDownloader as gdd


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def load_fixed_splits(data_dir, dataset, name, protocol):
    splits_lst = []
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_idx)
        splits['valid'] = torch.as_tensor(dataset.valid_idx)
        splits['test'] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']:
        for i in range(10):
            splits_file_path = '{}/geom-gcn/splits/{}'.format(data_dir, name) + '_split_0.6_0.2_' + str(i) + '.npz'
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits['train'] = torch.BoolTensor(splits_file['train_mask'])
                splits['valid'] = torch.BoolTensor(splits_file['val_mask'])
                splits['test'] = torch.BoolTensor(splits_file['test_mask'])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    train_idx, non_train_idx = [], []  # 训练集、非训练集
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num + test_num]

    return train_idx, valid_idx, test_idx


def hsi_splits(label, train_prop=0.1, valid_prop=0.01):
    """
    hsi数据集划分 每个类都选取0.1的样本
    return idx都是tensor形式
    """
    classCount = label.max().item()
    print(f"class count = {classCount}")
    train_rand_idx = []
    for i in range(classCount):
        idx = torch.where(label == i + 1)[0]
        samplesCount = len(idx)
        if samplesCount > 0:  # 只有当类别样本存在时才进行抽样
            # 0~len随机抽取idx
            rand_list = [j for j in range(samplesCount)]
            rand_idx = random.sample(rand_list,
                                     int(np.ceil(samplesCount * train_prop)))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
    # train_rand_idx是二维数组形式[[1, 2], [4, 7][8, 11]] 需要转换成一维
    """
        train_idx = []
        # 使用 len() 函数获取 train_rand_idx 的长度
        for i in range(len(train_rand_idx)):
            # 获取第 i 个类别的随机样本索引数组
            temp = train_rand_idx[i]
            train_idx.extend(temp)
    """
    train_idx = torch.cat(train_rand_idx).tolist() if train_rand_idx else []

    # train_labels = label[train_idx]
    # if (train_labels == 0).any():
    #     print("Warning: Train indices contain background samples (label=0).")
    # else:
    #     print("train_correct")

    # 这里还要将这些idx变成set去重 得到测试集
    train_set = set(train_idx)

    # 所有数据的set
    """
    all_set = [i for i in range(len(label))]
    all_set = set(all_set)
    """
    all_set = set(range(len(label)))  # fixed

    """
    # 得到背景元素下标
    background_set = torch.where(label == 0)[0]
    background_set = set(background_set)
    # 测试集
    test_set = all_set - background_set - train_set
    # 转化成列表方便选取验证集元素
    test_set = list(test_set)
    """

    # 得到背景元素下标
    background_set = set(torch.where(label == 0)[0].tolist())

    # 计算测试集
    test_set = all_set - background_set - train_set

    # 转换为列表
    test_set = list(test_set)

    # 检查测试集是否包含背景类
    test_labels = label[torch.tensor(test_set)]

    # 这个检查是为了确保测试集不包含背景类
    if (test_labels == 0).any():
        print("Warning: test indices contain background samples (label=0).")
    else:
        print("test correct")

    # 生成验证集
    val_count = int(valid_prop * (len(train_set) + len(test_set)))
    valid_idx = random.sample(test_set, min(val_count, len(test_set)))  # 防止val_count超出范围

    # 更新测试集，排除验证集
    valid_set = set(valid_idx)
    test_set = list(set(test_set) - valid_set)  # 转换为列表并去重
    """
    # 验证集大小
    valCount = int(valid_prop * (len(train_set) + len(test_set)))
    valid_idx = random.sample(test_set, valCount)

    # 求出剩余的测试集
    valid_set = set(valid_idx)
    test_set = set(test_set) - valid_set
    test_set = list(test_set)
    """

    # train_labels = label[train_idx]
    # if (train_labels == 0).any():
    #     print("Warning: Train indices contain background samples (label=0).")
    # test_labels = label[test_set]
    # if (test_labels == 0).any():
    #     print("Warning: test indices contain background samples (label=0).")
    # valid_labels = label[valid_idx]
    # if (valid_labels == 0).any():
    #     print("Warning: valid indices contain background samples (label=0).")

    # 最后返回tensor形式
    test_idx = torch.tensor(test_set)
    valid_idx = torch.tensor(valid_idx)
    train_idx = torch.as_tensor(train_idx)
    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:
        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_oa(y_true, y_pred):  # 直接修改了标签导致下标越界
    with torch.no_grad():
        y_true_detached = y_true.detach().cpu().numpy()
        y_true_detached = np.squeeze(y_true_detached)
        y_true_detached -= 1

        y_pred_probs = torch.softmax(y_pred, dim=-1)  # 计算每个类的概率
        y_pred_labels = y_pred_probs.argmax(dim=-1, keepdim=True).detach().cpu().numpy()  # 获取预测的类别

        total_samples = y_true_detached.shape[0]  # 总样本数
        correct_samples = np.sum(y_pred_labels.flatten() == y_true_detached)  # 计算正确样本数

        oa = correct_samples / total_samples if total_samples > 0 else 0.0  # 避免除以零
        return oa

def eval_aa(y_true, y_pred, class_count: int):
    with torch.no_grad():
        # 先获取正确下标 从0开始
        y_true_detached = y_true.detach().cpu().numpy()
        y_true_detached = np.squeeze(y_true_detached)
        y_true_detached -= 1

        y_pred_probs = torch.softmax(y_pred, dim=-1)  # 计算每个类的概率
        y_pred_labels = y_pred_probs.argmax(dim=-1, keepdim=True).detach().cpu().numpy()  # 获取预测的类别

        # 初始化每个种类的统计情况
        correct_counts = np.zeros(class_count, dtype=int)
        total_counts = np.zeros(class_count, dtype=int)
        class_acc = np.zeros(class_count)

        for i in range(len(y_true_detached)):
            true_class = int(y_true_detached[i])
            pred_class = int(y_pred_labels[i])
            # 类别是0~classCount-1
            if 0 <= true_class < class_count:
                total_counts[true_class] += 1
                if true_class == pred_class:
                    correct_counts[true_class] += 1

        for i in range(class_count):
            if total_counts[i] > 0:
                class_acc[i] = correct_counts[i] / total_counts[i]
            else:
                class_acc[i] = 0.0

        aa = np.mean(class_acc)

        return aa

def eval_kappa(y_true, y_pred):  # 检查标签一致性
    with torch.no_grad():
        y_true_detached = y_true.detach().cpu().numpy()
        y_true_detached = np.squeeze(y_true_detached)
        y_true_detached -= 1

        # 计算预测下标
        y_pred_probs = torch.softmax(y_pred, dim=-1)  # 计算每个类的概率
        # 一维的分类结果 argmax
        y_pred_labels = y_pred_probs.argmax(dim=-1, keepdim=False).detach().cpu().numpy()  # 获取预测的类别

        kappa = cohen_kappa_score(y_true_detached.astype(np.int16), y_pred_labels.astype(np.int16))

        return kappa

def draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D 二维数组
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK <===> 400点每英寸
    :return: null
    '''
    cmap = cm.get_cmap('jet', label.max().item())
    plt.set_cmap(cmap)

    fig, ax = plt.subplots()
    num_label = np.array(label)
    v = spy.imshow(classes=num_label.astype(np.int16), fignum=fig.number)
    # 关闭坐标轴的显示。 X,Y的可见性设置为false
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # 设置图像大小
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    # 'get current figure' 获取当前图像
    foo_fig = plt.gcf()
    # 移除X轴和Y轴的主要刻度标记。nullLocataor
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 整子图参数，使得图像边缘没有空白
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def convert_to_adj(edge_index, n_node):
    '''convert from pyg format edge_index to n by n adj matrix'''
    adj = torch.zeros((n_node, n_node))
    row, col = edge_index
    adj[row, col] = 1
    return adj


def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j


import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory


import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}
