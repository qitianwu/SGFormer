import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters, eval_oa, draw_Classification_Map, \
    eval_aa, eval_kappa
from eval import evaluate
from parse import parse_method, parser_add_main_args
from build_graph import get_weight, build_graph_by_fix

import time
import pickle

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ----Parse args----
# 创建一个新的参数解析器对象，并提供描述信息。
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
# 该函数向解析器添加了一系列命令行参数
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

# 设置随机数
fix_seed(args.seed)

# 检测设备
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)


# 转化维度 [num] ---> [num, 1] 将一维tensor转换为二维tensor
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
# print(len(dataset.label.shape))

# get the splits for all runs 划分数据集
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
# hsi划分数据集
elif args.dataset in 'Indian_pines':
    split_idx_lst = [dataset.get_idx_split(split_type='hsi', train_prop=0.1, valid_prop=0.01)
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

# 创建边集
dataset.graph['edge_index'] = build_graph_by_fix(dataset.graph['node_feat'], dataset.label,
                                                 split_idx_lst[0]['train'], dataset.row, dataset.col)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item(), dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

#  symmetrize 对称化
if not args.directed and args.dataset not in ['ogbn-proteins']:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

# 添加自环
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)


# 计算边权
dataset.graph['edge_weight'] = get_weight(dataset.graph['node_feat'], dataset.graph['edge_index'])
dataset.graph['edge_weight'] = dataset.graph['edge_weight'].to(device)


# 将edge_index和node_fea移动到GPU上
# ver2 添加了label到GPU上
dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.label = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device), \
    dataset.label.to(device)



### Load method ###
model = parse_method(args, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()  # 使用于二分类 二元交叉熵损失函数带
elif args.dataset in 'Indian_pines':
    criterion = nn.CrossEntropyLoss()  # 使用于多分类
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
elif args.metric == 'oa':
    eval_func = eval_oa
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    # 重置参数
    model.reset_parameters()
    if args.method == 'sgformer':
        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    # 记录最佳损失
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()  # 清楚梯度

        train_start = time.time()
        # out返回提取后的node_fea
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'], dataset.graph['edge_weight'])
        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max()).squeeze(1).to(device)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        elif args.dataset in 'Indian_pines':
            train_labels = dataset.label[train_idx].squeeze(1).to(device)
            train_labels -= 1
            loss = criterion(out[train_idx], train_labels)
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % args.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            cur_loss = result[1]  # 当前模型验证集损失
            logger.add_result(run, result[:-1])  # check result length
            # default args.display_step=1
            if epoch % args.display_step == 0:
                print_str = f'Epoch: {epoch:02d}, ' + \
                            f'Train Loss: {loss:.4f}, ' + \
                            f'Valid Loss: {result[1]:.4f}, ' + \
                            f'Test OA: {100 * result[0]:.2f}% '
                print(print_str)
            if cur_loss < best_loss:
                best_loss = cur_loss  # 更新最佳验证集损失
                # 保存模型到指定路径
                torch.save(model.state_dict(), "best_model.pt")  # 保存到当前文件夹
                print(f'Save best model in current folder, epoch = {epoch}\n')
    # logger.print_statistics(run, mode="minLoss", step=args.eval_step)

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.load_state_dict(torch.load("best_model.pt"))
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'], dataset.graph['edge_weight'])
        # 提取的特征

        # oa
        test_oa = eval_func(
            dataset.label[split_idx['test']], out[split_idx['test']])
        train_oa = eval_func(
            dataset.label[split_idx['train']], out[split_idx['train']])
        valid_oa = eval_func(
            dataset.label[split_idx['valid']], out[split_idx['valid']])

        # aa
        train_aa = eval_aa(
            dataset.label[split_idx['train']], out[split_idx['train']], dataset.label.max())
        valid_aa = eval_aa(
            dataset.label[split_idx['valid']], out[split_idx['valid']], dataset.label.max())
        test_aa = eval_aa(
            dataset.label[split_idx['test']], out[split_idx['test']], dataset.label.max())

        # kpp
        train_kpp = eval_kappa(
            dataset.label[split_idx['train']], out[split_idx['train']])
        valid_kpp = eval_kappa(
            dataset.label[split_idx['valid']], out[split_idx['valid']])
        test_kpp = eval_kappa(
            dataset.label[split_idx['test']], out[split_idx['test']])

        logger.print_best_epoch()
        # 打印结果，AA、OA、Kappa以百分比形式输出，损失保留四位小数
        print(f"Final Train OA: {train_oa * 100:.2f}%")
        print(f"Final Valid OA: {valid_oa * 100:.2f}%")
        print(f"Final Test OA: {test_oa * 100:.2f}%")
        print(f"Final Train AA: {train_aa * 100:.2f}%")
        print(f"Final Valid AA: {valid_aa * 100:.2f}%")
        print(f"Final Test AA: {test_aa * 100:.2f}%")
        print(f"Final Train Kappa: {train_kpp * 100:.2f}%")
        print(f"Final Valid Kappa: {valid_kpp * 100:.2f}%")
        print(f"Final Test Kappa: {test_kpp * 100:.2f}%")

        # draw graph
        out = F.softmax(out, dim=-1)
        predicted_labels = torch.argmax(out, 1).reshape([dataset.row, dataset.col]).cpu() + 1
        draw_Classification_Map(predicted_labels, dataset.name)
    torch.cuda.empty_cache()


#logger.print_statistics()
