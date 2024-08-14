import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
from torch_sparse import SparseTensor


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    train_idx, non_train_idx = [], []
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
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num : valid_num + test_num],
    )
    print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx


def normalize_feat(mx):
    """Row-normalize np or sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


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
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        if len(np.unique(y_true_i)) == 1:
            continue
        rocauc_i = roc_auc_score(y_true_i, y_pred_i)
        rocauc_list.append(rocauc_i)
    return np.mean(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if args.method == "fast_transgnn" or args.method == "glcn" or args.method == "nodeformer":
            out, _ = model(dataset)
        else:
            out = model(dataset)
    
    if out.size()[0] == 1:
        out = out[0]

    train_acc = eval_func(dataset.label[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(dataset.label[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(dataset.label[split_idx["test"]], out[split_idx["test"]])
    if args.dataset in (
        "yelp-chi",
        "deezer-europe",
        "twitch-e",
        "fb100",
        "ogbn-proteins",
    ):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(
            out[split_idx["valid"]],
            true_label.squeeze(1)[split_idx["valid"]].to(torch.float),
        )
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx["valid"]], dataset.label.squeeze(1)[split_idx["valid"]]
        )

    return train_acc, valid_acc, test_acc, valid_loss, out


def load_fixed_splits(dataset, name, protocol):

    splits_lst = []
    if name in ["cora", "citeseer", "pubmed"] and protocol == "semi":
        splits = {}
        splits["train"] = torch.as_tensor(dataset.train_idx)
        splits["valid"] = torch.as_tensor(dataset.valid_idx)
        splits["test"] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ["chameleon", "squirrel"]:
        file_path = f"../data/wiki_new/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)

    elif name in ["film"]:
        for i in range(10):
            splits_file_path = (
                "../data/geom-gcn/{}/{}".format(name, name)
                + "_split_0.6_0.2_"
                + str(i)
                + ".npz"
            )
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits["train"] = torch.BoolTensor(splits_file["train_mask"])
                splits["valid"] = torch.BoolTensor(splits_file["val_mask"])
                splits["test"] = torch.BoolTensor(splits_file["test_mask"])
            splits_lst.append(splits)
    
    elif name in ['deezer-europe']:
        splits_lst = np.load(f'../data/deezer/{name}-splits.npy', allow_pickle=True)
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])

    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        for i in range(10):
            i = (i+1) % 10
            splits = {}
            splits["train"] = dataset.train_idx[i]
            splits["valid"] = dataset.valid_idx[i]
            splits["test"] = dataset.test_idx[i]
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

@torch.jit.script
def convert_to_single_emb(x, offset: int = 2):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_graph(graph):
    edge_feat, edge_index, x = None, graph['edge_index'], graph['node_feat']
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # # edge feature here
    # if len(edge_feat.size()) == 1:
    #     edge_feat = edge_feat[:, None]
    # attn_edge_type = torch.zeros([N, N, edge_feat.size(-1)], dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
    #     convert_to_single_emb(edge_feat) + 1
    # )

    # shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    # max_dist = np.amax(shortest_path_result)
    # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    max_node_num = x.size(0)
    x = x.unsqueeze(0)
    print('x',x.size())
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    spatial_pos = torch.randint(0,1000,size=(max_node_num,max_node_num))
    attn_bias = torch.zeros([N, N], dtype=torch.float)  # with graph token
    spatial_pos = pad_spatial_pos_unsqueeze(spatial_pos, max_node_num)
    attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node_num)
    in_degree = adj.long().sum(dim=1).view(-1)
    in_degree = pad_1d_unsqueeze(in_degree, max_node_num)
    # combine
    graph['x'] = x
    graph['attn_bias'] = attn_bias
    # graph['attn_edge_type'] = attn_edge_type
    graph['spatial_pos'] = spatial_pos
    graph['in_degree'] = in_degree
    graph['out_degree'] = in_degree  # for undirected graph
    # graph['edge_input'] = torch.from_numpy(edge_input).long()

    return graph

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