import sys
sys.path.append('../src')

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, to_networkx

from tqdm import tqdm
import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from utils import process_data, get_preds, save_checkpoint


@torch.no_grad()
def eval_one_batch(gsat, data, epoch):
    gsat.extractor.eval()
    gsat.clf.eval()

    att, loss, loss_dict, clf_logits = gsat.forward_pass(data, epoch, training=False)
    return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()


def train_one_batch(gsat, data, epoch):
    gsat.extractor.train()
    gsat.clf.train()

    att, loss, loss_dict, clf_logits = gsat.forward_pass(data, epoch, training=True)
    gsat.optimizer.zero_grad()
    loss.backward()
    gsat.optimizer.step()
    return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()


def run_one_epoch(gsat, data_loader, epoch, phase, dataset_name, seed, use_edge_attr, multi_label):
    loader_len = len(data_loader)
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    all_loss_dict = {}
    all_exp_labels, all_att, all_clf_labels, all_clf_logits = ([] for i in range(4))
    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        data = process_data(data, use_edge_attr)
        att, loss_dict, clf_logits = run_one_batch(gsat, data.to(gsat.device), epoch)

        exp_labels = data.edge_label.data.cpu()
        desc, _, _, _, _ = log_epoch(epoch, phase, loss_dict, exp_labels, att, data.y.data.cpu(), clf_logits,
                                     dataset_name, seed, multi_label, batch=True)
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v

        all_exp_labels.append(exp_labels), all_att.append(att)
        all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

        if idx == loader_len - 1:
            all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
            all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, att_auroc, clf_acc, clf_roc, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att, all_clf_labels, all_clf_logits,
                                                                    dataset_name, seed, multi_label, batch=False)
        pbar.set_description(desc)
    return att_auroc, None, clf_acc, clf_roc, avg_loss


def log_epoch(epoch, phase, loss_dict, exp_labels, att, clf_labels, clf_logits, dataset_name, seed, multi_label, batch):
    desc = f'[Seed {seed}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {seed}, Epoch: {epoch}]: gsat_{phase} finished, '
    for k, v in loss_dict.items():
        desc += f'{k}: {v:.3f}, '

    eval_desc, att_auroc, clf_acc, clf_roc = get_eval_score(exp_labels, att, clf_labels, clf_logits, dataset_name, multi_label, batch)
    desc += eval_desc
    return desc, att_auroc, clf_acc, clf_roc, loss_dict['pred']


def get_eval_score(exp_labels, att, clf_labels, clf_logits, dataset_name, multi_label, batch):
    clf_preds = get_preds(clf_logits, multi_label)
    clf_acc = 0 if multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

    if batch:
        return f'clf_acc: {clf_acc:.3f}', None, None, None

    clf_roc = 0
    if 'ogb' in dataset_name:
        evaluator = Evaluator(name='-'.join(dataset_name.split('_')))
        clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']
    att_auroc = roc_auc_score(exp_labels, att) if np.unique(exp_labels).shape[0] > 1 else 0

    desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, att_roc: {att_auroc:.3f}'
    return desc, att_auroc, clf_acc, clf_roc


def update_best_epoch_res(gsat, train_res, valid_res, test_res, metric_dict, dataset_name, epoch, model_dir):
    assert len(train_res) == 5
    main_metric_idx = 3 if 'ogb' in dataset_name else 2  # clf_roc or clf_acc
    current_r = gsat.get_r(gsat.decay_interval, gsat.decay_r, epoch, final_r=gsat.final_r)
    model_dir.mkdir(parents=True, exist_ok=True)

    if (current_r == gsat.final_r) and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                        or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid'] and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):

        metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                       'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                       'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                       'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
        save_checkpoint(gsat, model_dir, model_name='gsat_epoch_' + str(epoch))
    return metric_dict


def get_viz_idx(test_set, dataset_name, num_viz_samples):
    y_dist = test_set.data.y.numpy().reshape(-1)
    num_nodes = np.array([each.x.shape[0] for each in test_set])
    classes = np.unique(y_dist)
    res = []
    for each_class in classes:
        tag = 'class_' + str(each_class)
        if dataset_name == 'Graph-SST2':
            condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
            candidate_set = np.nonzero(condi)[0]
        else:
            candidate_set = np.nonzero(y_dist == each_class)[0]
        idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
        res.append((idx, tag))

    if dataset_name == 'mutag':
        for each_class in classes:
            tag = 'class_' + str(each_class)
            candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
            res.append((idx, tag))
    return res


def visualize_results(gsat, all_viz_set, test_set, num_viz_samples, dataset_name, use_edge_attr):
    figsize = 10
    fig, axes = plt.subplots(len(all_viz_set), num_viz_samples, figsize=(figsize*num_viz_samples, figsize*len(all_viz_set)*0.8))

    for class_idx, (idx, tag) in enumerate(all_viz_set):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, _ = eval_one_batch(gsat, data.to(gsat.device), epoch=500)

        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_mask = subgraph(node_subset.cpu(), data.edge_index.cpu(), edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            visualize_a_graph(viz_set[i].edge_index, edge_mask, node_label, dataset_name, axes[class_idx, i], norm=True, mol_type=mol_type, coor=coor)
            # axes[class_idx, i].axis('off')
        fig.tight_layout()

    each_plot_len = 1/len(viz_set)
    for num in range(1, len(viz_set)):
        line = plt.Line2D((each_plot_len*num, each_plot_len*num), (0, 1), color="gray", linewidth=1, linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)

    each_plot_width = 1/len(all_viz_set)
    for num in range(1, len(all_viz_set)):
        line = plt.Line2D((0, 1), (each_plot_width*num, each_plot_width*num), color="gray", linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)


def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, ax, coor=None, norm=False, mol_type=None, nodesize=300):
    if norm:  # for better visualization
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
