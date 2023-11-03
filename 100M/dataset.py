from os import path

import torch
from data_utils import class_rand_splits, rand_train_test_idx
from ogb.nodeproppred import PygNodePropPredDataset


class NCDataset(object):
    def __init__(self, name):
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

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
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
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
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


def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    print(dataname)
    if dataname == 'ogbn-papers100M':
        dataset = load_papers100M(data_dir)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_papers100M(data_dir):
    ogb_dataset = PygNodePropPredDataset('ogbn-papers100M',root=data_dir)
    ogb_data=ogb_dataset[0]
    dataset = NCDataset('ogbn-papers100M')
    dataset.graph = dict()
    dataset.graph['edge_index'] = torch.as_tensor(ogb_data.edge_index)
    dataset.graph['node_feat'] = torch.as_tensor(ogb_data.x)
    dataset.graph['num_nodes'] = ogb_data.num_nodes
    
    # Use mapped train, valid and test index, same as OGB.
    split_idx = ogb_dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    # dataset.label = torch.as_tensor(ogb_data.y.data[all_idx], dtype=int).reshape(-1, 1)
    dataset.label = torch.as_tensor(ogb_data.y.data, dtype=int).reshape(-1, 1) # 99% labels are nan, not available
    # print(f'f1:{dataset.label.shape}')
    # print(f'{ogb_data.num_nodes}')
    def get_idx_split():
        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx,
        }
        return split_idx

    dataset.load_fixed_splits = get_idx_split

    return dataset

if __name__=='__main__':
    # load_papers100M('../data')
    pass