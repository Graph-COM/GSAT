# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .conv_layers import GINConv, GINEConv


class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        if model_config.get('atom_encoder', False):
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc_out(self.pool(x, batch))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
