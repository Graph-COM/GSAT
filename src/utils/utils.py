import torch
import random
import numpy as np
import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, sort_edge_index
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


init_metric_dict = {'metric/best_clf_epoch': 0, 'metric/best_clf_valid_loss': 0,
                    'metric/best_clf_train': 0, 'metric/best_clf_valid': 0, 'metric/best_clf_test': 0,
                    'metric/best_x_roc_train': 0, 'metric/best_x_roc_valid': 0, 'metric/best_x_roc_test': 0,
                    'metric/best_x_precision_train': 0, 'metric/best_x_precision_valid': 0, 'metric/best_x_precision_test': 0}


def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]


def process_data(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data


def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir / (model_name + '.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, model_dir / (model_name + '.pt'))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_local_config_name(model_name, dataset_name):
    if 'ogbg_mol' in dataset_name:
        local_config_name = f'{model_name}-ogbg_mol.yml'
    elif 'spmotif' in dataset_name:
        local_config_name = f'{model_name}-spmotif.yml'
    else:
        local_config_name = f'{model_name}-{dataset_name}.yml'
    return local_config_name


def write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer):
    res = {metric: {'value': [], 'mean': 0, 'std': 0} for metric in metric_dicts[0].keys()}

    for metric_dict in metric_dicts:
        for metric, value in metric_dict.items():
            res[metric]['value'].append(value)

    stat = {}
    for metric, value in res.items():
        stat[metric] = np.mean(value['value'])
        stat[metric+'/std'] = np.std(value['value'])

    writer.add_hparams(hparam_dict, stat)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Writer(SummaryWriter):
    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, coor=None, norm=False, mol_type=None, nodesize=300):
    plt.clf()
    if norm:
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')

    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image
