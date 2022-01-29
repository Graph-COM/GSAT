# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/example.py

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import BatchNorm, global_mean_pool
from .conv_layers import PNAConvSimple


class PNA(torch.nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()
        hidden_size = model_config['hidden_size']
        self.n_layers = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.edge_attr_dim = edge_attr_dim

        if model_config.get('atom_encoder', False):
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and model_config.get('use_edge_attr', True):
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and model_config.get('use_edge_attr', True):
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        aggregators = model_config['aggregators']
        scalers = ['identity', 'amplification', 'attenuation'] if model_config['scalers'] else ['identity']
        deg = model_config['deg']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        if model_config.get('use_edge_attr', True):
            in_channels = hidden_size * 2 if edge_attr_dim == 0 else hidden_size * 3
        else:
            in_channels = hidden_size * 2

        for _ in range(self.n_layers):
            conv = PNAConvSimple(in_channels=in_channels, out_channels=hidden_size, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.pool = global_mean_pool
        self.fc_out = Sequential(Linear(hidden_size, hidden_size//2), ReLU(),
                                 Linear(hidden_size//2, hidden_size//4), ReLU(),
                                 Linear(hidden_size//4, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            x = h + x  # residual#
            x = F.dropout(x, self.dropout_p, training=self.training)

        x = self.pool(x, batch)
        return self.fc_out(x)

    def get_emb(self, x, edge_index, batch, edge_attr, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            x = h + x  # residual#
            x = F.dropout(x, self.dropout_p, training=self.training)

        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
