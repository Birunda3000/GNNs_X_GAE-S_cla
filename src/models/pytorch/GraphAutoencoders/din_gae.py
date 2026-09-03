import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import GATConv, GINConv, SAGEConv
from torch_geometric.nn.conv.cugraph import CuGraphSAGEConv, CuGraphGATConv
from torch_geometric import EdgeIndex

# 🔥 FASE 3: Importação da Fronteira Tecnológica de Atenção (PyTorch 2.6)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

from src.models.pytorch.GraphAutoencoders.base_graph_autoenconders_model import BaseGAE, BaseVGAE
from src.models.base_model import get_activation_fn

# ==============================================================================
# 🔥 RESOLUÇÃO ESTRUTURAL JIT
# Isola os objetos obscuros da NVIDIA do radar do compilador TorchDynamo
# ==============================================================================

@torch.compiler.disable
def prepare_edge_index(edge_index, num_nodes):
    """Protege a classe opaca EdgeIndex do rastreamento do compilador."""
    if not isinstance(edge_index, EdgeIndex):
        return EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes))
    return edge_index

@torch.compiler.disable
def apply_conv(conv, x, edge_index):
    """Protege os kernels bare-metal do NVIDIA RAPIDS contra dissecção do JIT."""
    return conv(x, edge_index)

# ==============================================================================

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


class DynamicGAE(BaseGAE):
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim, 
                 layer_type, num_layers, activation=nn.ReLU, dropout=0.5, normalize_embeddings=True, 
                 **kwargs): 
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings

        layers = []
        layers.append(create_layer(layer_type, embedding_dim, hidden_dim, **kwargs))

        for _ in range(num_layers - 2):
            layers.append(create_layer(layer_type, hidden_dim, hidden_dim, **kwargs))

        layers.append(create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs))

        self.convs = nn.ModuleList(layers)

    def encode(self, data):
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # O compilador quebra o grafo intencionalmente e em silêncio aqui
        edge_index = prepare_edge_index(data.edge_index, x.size(0))

        for i, conv in enumerate(self.convs):
            # O kernel da NVIDIA roda blindado do JIT
            x = apply_conv(conv, x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.normalize_embeddings:
            return F.normalize(x, p=2, dim=-1)
        return x

class DynamicVGAE(BaseVGAE):
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim, 
                 layer_type, num_layers, activation=nn.ReLU, dropout=0.5, normalize_embeddings=True, 
                 **kwargs):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings

        hidden_layers = []
        hidden_layers.append(create_layer(layer_type, embedding_dim, hidden_dim, **kwargs))
        
        for _ in range(num_layers - 2):
            hidden_layers.append(create_layer(layer_type, hidden_dim, hidden_dim, **kwargs))
            
        self.convs_hidden = nn.ModuleList(hidden_layers)

        self.conv_mu = create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs)
        self.conv_logstd = create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs)

    def encode(self, data):
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # O compilador quebra o grafo intencionalmente e em silêncio aqui
        edge_index = prepare_edge_index(data.edge_index, x.size(0))

        for conv in self.convs_hidden:
            x = apply_conv(conv, x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        self.__mu__ = apply_conv(self.conv_mu, x, edge_index)
        self.__logstd__ = apply_conv(self.conv_logstd, x, edge_index)
        
        eps = torch.randn_like(self.__mu__)
        z = self.__mu__ + eps * torch.exp(self.__logstd__)

        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=-1)
        return z


