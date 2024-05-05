'''tick time'''

import argparse
import sys
import os
import random
import numpy as np
import torch
import copy
import time, subprocess

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter
from sklearn.neighbors import kneighbors_graph

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import eval_f1, evaluate, eval_acc, to_sparse_tensor, \
    load_fixed_splits, class_rand_splits
from parse import parse_method, parser_add_main_args, parser_add_default_args

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx

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

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# seed=random.randint(1,10000)
# print(f'============Generating random seed {seed}==========')
# fix_seed(seed)


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

if args.dataset == 'ogbn-proteins':
    if args.method == 'mlp' or args.method == 'cs':
        dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
                                             dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
    else:
        dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
                                                       dataset.graph['edge_feat'], dataset.graph['num_nodes'])
        dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
        dataset.graph['edge_index'].set_value_(None)
    dataset.graph['edge_feat'] = None

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if not args.directed and args.dataset not in {'ogbn-proteins', 'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")
# exit()

### Load method ###
model = parse_method(args.method, args, c, d, device)

# using rocauc as the eval function
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()
if args.dataset in {'disease', 'airport'}:
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()

### Training loop ###
best_emb, best_model = None, None
patience = 0
if args.method == 'ours' and args.use_graph:
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ],
        lr=args.lr)
else:
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

train_start=time.time()
epoch_cnt=0

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        # split_idx = split_idx_lst[0]
        split_idx = split_idx_lst[0]
    elif args.dataset in ['chain']:
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    # train_idx = torch.arange(0, n)
    # split_idx['train'] = train_idx
    model.reset_parameters()

    # best_val = float('-inf')
    # patience = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        # if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        #     if dataset.label.shape[1] == 1:
        #         true_label = F.one_hot(
        #             dataset.label, dataset.label.max() + 1).squeeze(1)
        #     else:
        #         true_label = dataset.label
        #     loss = criterion(out[train_idx], true_label.squeeze(1)[
        #         train_idx].to(torch.float))
        # else:
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        # result = evaluate(model, dataset, split_idx,
        #                   eval_func, criterion, args)

        # quick evaluation
        train_acc=eval_func(dataset.label[split_idx['train']], out[split_idx['train']])
        valid_acc=eval_func(dataset.label[split_idx['valid']], out[split_idx['valid']])
        test_acc=eval_func(dataset.label[split_idx['test']], out[split_idx['test']])
        result = [train_acc, valid_acc, test_acc, 0, 0] 

        logger.add_result(run, result[:-1])

        # if result[1] > best_val:
            # patientce = 0
            # best_val=result[1]
        # else:
        #     patience += 1
        #     if patience >= args.patience:
        #         epoch_cnt+=epoch+1
        #         break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    # logger.print_statistics(run)

train_end=time.time()
train_time=train_end-train_start
# train_time_per_epoch=train_time/epoch_cnt if epoch_cnt>0 else train_time/args.epochs
train_time_per_epoch=train_time/args.epochs

# testing time
test_start=time.time()
with torch.no_grad():
    model(dataset)
test_time=time.time()-test_start
test_mem = get_gpu_memory_map()[int(args.device)]

results = logger.print_statistics()
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)


def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    if method == 'gcn':
        print_str += f'method: {args.method} layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} \n'
    elif method == 'ours':
        print_str += f'method: {args.method} hidden: {args.hidden_channels} ours_layers:{args.ours_layers} lr:{args.lr} k:{args.k} use_graph:{args.use_graph} graph_weight:{args.graph_weight} alpha:{args.alpha} ours_decay:{args.ours_weight_decay} ours_dropout:{args.ours_dropout} epochs:{args.epochs} use_feat_norm:{not args.no_feat_norm} use_bn:{args.use_bn}\n'
        if not args.use_graph:
            return print_str
        if args.backbone == 'gcn':
            print_str += f'backbone:{args.backbone}, layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay} dropout:{args.dropout}\n'
    elif method == 'gat':
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr} k:{args.k} heads:{args.gat_heads}\n'
    elif method == 'mlp':
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr} k:{args.k} n_layers:{args.num_layers}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr} k:{args.k}\n'
    return print_str


file_name = f'{args.dataset}_{args.method}'
if args.method == 'ours' and args.use_graph:
    file_name += '_' + args.backbone
file_name += '.txt'
out_path = os.path.join(out_folder, file_name)
with open(out_path, 'a+') as f:
    print_str = make_print(args.method)
    f.write(print_str)
    f.write(results)

    print_time=f'\ntotal train time:{train_time:.4f}, train per epoch:{train_time_per_epoch:.4f}, test time:{test_time:.4f} gpu:{test_mem}'
    f.write(print_time)
    print(print_time)
    f.write('\n\n')