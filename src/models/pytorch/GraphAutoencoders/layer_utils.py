import torch
import torch.nn as nn
from torch_geometric import EdgeIndex
from torch_geometric.nn import GATConv, GINConv, SAGEConv
from torch_geometric.nn.conv.cugraph import CuGraphSAGEConv, CuGraphGATConv

@torch.compiler.disable
def prepare_edge_index(edge_index, num_nodes):
    if not isinstance(edge_index, EdgeIndex):
        return EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes))
    return edge_index

@torch.compiler.disable
def apply_conv(conv, x, edge_index):
    return conv(x, edge_index)

def create_layer(layer_type, in_dim, out_dim, **kwargs):
    if layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv)):
        train_eps = kwargs.get('train_eps', False)
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        return layer_type(mlp, train_eps=train_eps)
    elif layer_type.__name__ in ['GATConv', 'CuGraphGATConv'] or (isinstance(layer_type, type) and issubclass(layer_type, (GATConv, CuGraphGATConv))):
        heads = kwargs.get('heads', 1)
        dropout = kwargs.get('dropout', 0.0)
        return layer_type(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
    elif layer_type.__name__ in ['SAGEConv', 'CuGraphSAGEConv'] or (isinstance(layer_type, type) and issubclass(layer_type, (SAGEConv, CuGraphSAGEConv))):
        aggr = kwargs.get('aggr', 'mean')
        return layer_type(in_dim, out_dim, aggr=aggr)
    else:
        return layer_type(in_dim, out_dim)
