# https://github.com/divelab/DIG/blob/dig/dig/xgraph/dataset/syn_dataset.py

import os
import yaml
import torch
import pickle
import numpy as np
import os.path as osp
from pathlib import Path
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, download_url


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        x = torch.from_numpy(node_features[graph_idx]).float()
        edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
        y = torch.from_numpy(np.where(graph_labels[graph_idx])[0]).reshape(-1, 1).float()

        node_label = torch.zeros(x.shape[0]).float()
        node_label[20:] = 1
        edge_label = ((edge_index[0] >= 20) & (edge_index[0] < 25) & (edge_index[1] >= 20) & (edge_index[1] < 25)).float()

        data_list.append(Data(x=x, edge_index=edge_index, y=y, node_label=node_label, edge_label=edge_label))
    return data_list


class SynGraphDataset(InMemoryDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.
    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'
    # Format: name: [display_name, url_name, filename]
    names = {
        'ba_shapes': ['BA_shapes', 'BA_shapes.pkl', 'BA_shapes'],
        'ba_community': ['BA_Community', 'BA_Community.pkl', 'BA_Community'],
        'tree_grid': ['Tree_Grid', 'Tree_Grid.pkl', 'Tree_Grid'],
        'tree_cycle': ['Tree_Cycle', 'Tree_Cycles.pkl', 'Tree_Cycles'],
        'ba_2motifs': ['BA_2Motifs', 'BA_2Motifs.pkl', 'BA_2Motifs']
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.pkl'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = self.url.format(self.names[self.name][1])
        download_url(url, self.raw_dir)

    def process(self):
        if self.name.lower() == 'BA_2Motifs'.lower():
            data_list = read_ba2motif_data(self.raw_dir, self.names[self.name][2])

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [data for data in data_list if self.pre_filter(data)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)
        else:
            # Read data into huge `Data` list.
            data = self.read_syn_data()
            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list = [data]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

    def read_syn_data(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

        x = torch.from_numpy(features).float()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        y = torch.from_numpy(np.where(y)[1])
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        return data
