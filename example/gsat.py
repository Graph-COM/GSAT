import sys
sys.path.append('../src')

import scipy
import torch
import torch.nn as nn
from torch_sparse import transpose
from torch_geometric.utils import is_undirected
from utils import MLP, reorder_like


class GSAT(nn.Module):

    def __init__(self, clf, extractor, criterion, optimizer, learn_edge_att=True, final_r=0.7, decay_interval=10, decay_r=0.1):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = next(self.parameters()).device

        self.learn_edge_att = learn_edge_att
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
        return edge_att, loss, loss_dict, clf_logits

    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.5)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits
