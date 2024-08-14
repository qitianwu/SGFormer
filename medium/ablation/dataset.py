import os
import pickle as pkl
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from data_utils import normalize_feat, rand_train_test_idx
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, to_dense_adj

DATAPATH = '../../data/'

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction: 

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(args):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    global DATAPATH
    
    DATAPATH=args.data_dir
    dataname = args.dataset
    print(dataname)
    if dataname == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, args.no_feat_norm)
    elif dataname in ('film'):
        dataset = load_geom_gcn_dataset(dataname)
    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(dataname,args.no_feat_norm)
        # dataset = load_wikipedia(dataname,args.no_feat_norm)
    elif dataname in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_heterophily_dataset(dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_deezer_dataset():

    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}/deezer/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_planetoid_dataset(name, no_feat_norm=False):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_geom_gcn_dataset(name):
    # graph_adjacency_list_file_path = '../../data/geom-gcn/{}/out1_graph_edges.txt'.format(
    #     name)
    # graph_node_features_and_labels_file_path = '../../data/geom-gcn/{}/out1_node_feature_label.txt'.format(
    #     name)
    graph_adjacency_list_file_path = os.path.join(DATAPATH, 'geom-gcn/{}/out1_graph_edges.txt'.format(name) )
    graph_node_features_and_labels_file_path = os.path.join(DATAPATH, 'geom-gcn/{}/out1_node_feature_label.txt'.format(name) )

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(
                    line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print(features.shape)

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat
    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels

    return dataset

def load_wiki_new(name, no_feat_norm=False):
    path=os.path.join(DATAPATH, f'wiki_new/{name}/{name}_filtered.npz')
    data=np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat=data['node_features'] # unnormalized
    labels=data['node_labels']
    edges=data['edges'] #(E, 2)
    edge_index=edges.T

    if not no_feat_norm:
        node_feat=normalize_feat(node_feat)

    dataset = NCDataset(name)

    edge_index=torch.as_tensor(edge_index)
    node_feat=torch.as_tensor(node_feat)
    labels=torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels

    return dataset

def load_heterophily_dataset(name):
    # load data elements
    data = np.load(os.path.join('../../heterophilous-graphs/data', f'{name.replace("-", "_")}.npz'))   # 这句路径可以抽象一下-1
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'])
    num_nodes = labels.shape[0]

    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
    val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
    test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

    # to_undirected = ToUndirected()
    # graph = torch.stack((edges[:, 0], edges[:, 1]), dim=0)

    # if 'directed' not in name:
    #     graph = to_undirected(graph)

    edge_index = torch.stack((edges[:, 0], edges[:, 1]), dim=0)
    node_features = augment_node_features(edge_index=edge_index,
                                                   node_features=node_features,
                                                   use_sgc_features=False,
                                                   use_identity_features=False,
                                                   use_adjacency_features=False,
                                                   do_not_use_original_features=False)

    # create dataset class
    dataset = NCDataset(name)
    dataset.graph = {'edge_index': edges,
                     'edge_feat': None,
                     'node_feat': node_features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = train_idx_list
    dataset.valid_idx = val_idx_list
    dataset.test_idx = test_idx_list

    return dataset

def compute_sgc_features(edge_index, node_features, num_props=5):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    row, col = edge_index
    degrees = degree(row).float()
    degree_edge_products = degrees[row] * degrees[col]
    norm_coefs = 1 / degree_edge_products.clamp(min=1) ** 0.5

    for _ in range(num_props):
        node_features = norm_coefs.view(-1, 1) * node_features[col]
        node_features = node_features.new_zeros(node_features.size()).scatter_add_(0, row.unsqueeze(1).expand_as(node_features), node_features)

    return node_features


def augment_node_features(edge_index, node_features, use_sgc_features, use_identity_features, use_adjacency_features,
                              do_not_use_original_features):

        n = node_features.size(0)
        original_node_features = node_features

        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = compute_sgc_features(edge_index, original_node_features)
            if node_features.size(0) != sgc_features.size(0):
                sgc_features = sgc_features.resize_(node_features.size(0), sgc_features.size(1))
            node_features = torch.cat([node_features, sgc_features], dim=1)

        if use_identity_features:
            node_features = torch.cat([node_features, torch.eye(n)], dim=1)

        if use_adjacency_features:
            edge_index, _ = remove_self_loops(edge_index)
            adj_matrix = to_dense_adj(edge_index).squeeze()
            node_features = torch.cat([node_features, adj_matrix], dim=1)

        return node_features

if __name__ == '__main__':
    # load_airport()
    # load_wikipedia('squirrel')
    # load_wiki_new('chameleon')
    load_heterophily_dataset('roman-empire')
    pass
