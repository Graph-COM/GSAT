# From Discovering Invariant Rationales for Graph Neural Networks

import os.path as osp
import pickle as pkl

import yaml
import torch
import torch.nn.functional as F
import random
import numpy as np
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

try:
    from .spmotif_utils import gen_dataset
except ImportError:
    from spmotif_utils import gen_dataset


class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, b, mode, transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.b = b
        self.mode = mode
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    def download(self):
        print('[INFO] Generating SPMotif dataset...')
        gen_dataset(self.b, Path(self.raw_dir))

    def process(self):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index).long()
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            # x = torch.zeros(node_idx.size(0), 4)
            # index = [i for i in range(node_idx.size(0))]
            # x[index, z] = 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).reshape(-1)

            node_label = torch.tensor(z, dtype=torch.float)
            node_label[node_label != 0] = 1
            edge_label = torch.tensor(ground_truth, dtype=torch.float)

            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])
