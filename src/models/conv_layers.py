from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor

import torch
from torch import Tensor
from torch_geometric.nn import GINEConv as BaseGINEConv, GINConv as BaseGINConv, LEConv as BaseLEConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree
from torch_scatter import scatter


class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j


class GINEConv(BaseGINEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        m = (x_j + edge_attr).relu()

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


class LEConv(BaseLEConv):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_atten: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        a = self.lin1(x[0])
        b = self.lin2(x[1])

        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight, edge_atten=edge_atten, size=None)

        return out + self.lin3(x[1])

    def message(self, a_j: Tensor, b_i: Tensor, edge_weight: OptTensor, edge_atten: OptTensor = None) -> Tensor:
        out = a_j - b_i
        m = out if edge_weight is None else out * edge_weight.view(-1, 1)

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/pna.py
class PNAConvSimple(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper
        .. math::
            \bigoplus = \underbrace{\begin{bmatrix}I \\ S(D, \alpha=1) \\
            S(D, \alpha=-1) \end{bmatrix} }_{\text{scalers}}
            \otimes \underbrace{\begin{bmatrix} \mu \\ \sigma \\ \max \\ \min
            \end{bmatrix}}_{\text{aggregators}},
        in:
        .. math::
            X_i^{(t+1)} = U \left( \underset{(j,i) \in E}{\bigoplus}
            M \left(X_j^{(t)} \right) \right)
        where :math:`U` denote the MLP referred to with posttrans.
        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            aggregators (list of str): Set of aggregation function identifiers,
                namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
                :obj:`"var"` and :obj:`"std"`.
            scalers: (list of str): Set of scaling function identifiers, namely
                :obj:`"identity"`, :obj:`"amplification"`,
                :obj:`"attenuation"`, :obj:`"linear"` and
                :obj:`"inverse_linear"`.
            deg (Tensor): Histogram of in-degrees of nodes in the training set,
                used by scalers to normalize.
            post_layers (int, optional): Number of transformation layers after
                aggregation (default: :obj:`1`).
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 post_layers: int = 1, **kwargs):

        super(PNAConvSimple, self).__init__(aggr=None, node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        self.F_in = in_channels
        self.F_out = self.out_channels

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        in_channels = (len(aggregators) * len(scalers)) * self.F_in
        modules = [Linear(in_channels, self.F_out)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_out, self.F_out)]
        self.post_nn = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.post_nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_atten=None) -> Tensor:

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None, edge_atten=edge_atten)
        return self.post_nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr=None, edge_atten=None) -> Tensor:
        if edge_attr is not None:
            m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            m = torch.cat([x_i, x_j], dim=-1)

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')
        raise NotImplementedError


def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='min')


def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='max')


def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'max': aggregate_max,
    'var': aggregate_var,
    'std': aggregate_std,
}


def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['lin'] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}
