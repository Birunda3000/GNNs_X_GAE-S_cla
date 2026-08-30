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

from src.models.embedding_models.base_graph_autoenconders_model import BaseGAE, BaseVGAE
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


class FacebookGAE(DynamicGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,       
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.GELU,         
            dropout=0.5,                
            normalize_embeddings=True,  
            heads=1,                    
        )
        self.model_name = "FacebookGAE"

class FacebookVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,       
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.GELU,         
            dropout=0.5,                
            normalize_embeddings=True,  
            heads=1,                    
        )
        self.model_name = "FacebookVGAE"

class GithubVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=256,          
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,       
            layer_type=CuGraphSAGEConv,
            num_layers=4,               
            activation=nn.LeakyReLU,    
            dropout=0.1,                
            normalize_embeddings=True,  
            aggr='mean',                
        )
        self.model_name = "GithubVGAE"

class TwitchVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim: int):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,  
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.ReLU,         
            dropout=0.2,                
            normalize_embeddings=False, 
            heads=1,                    
        )
        self.model_name = "TwitchVGAE"

class RedditVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim: int):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=128,          
            hidden_dim=64,              
            out_embedding_dim=out_embedding_dim,  
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.LeakyReLU,    
            dropout=0.2,                
            normalize_embeddings=True,  
            heads=1,                    
        )
        self.model_name = "RedditVGAE"

# ==============================================================================
# A FRONTEIRA: GRAPH TRANSFORMERS COM FLEX ATTENTION (FASE 3)
# ==============================================================================

class FlexGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        
        self.q_proj = nn.Linear(in_dim, out_dim * heads)
        self.k_proj = nn.Linear(in_dim, out_dim * heads)
        self.v_proj = nn.Linear(in_dim, out_dim * heads)
        self.out_proj = nn.Linear(out_dim * heads, out_dim)

    def forward(self, x, edge_index):
        if not HAS_FLEX:
            raise ImportError("FlexAttention requer PyTorch 2.5/2.6. Atualize o ambiente para utilizar Graph Transformers.")
            
        N = x.size(0)
        device = x.device
        
        adj = torch.zeros((N, N), dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        adj.fill_diagonal_(True)

        def graph_mask_mod(b, h, q_idx, kv_idx):
            return adj[q_idx, kv_idx]

        block_mask = create_block_mask(graph_mask_mod, B=1, H=self.heads, Q_LEN=N, KV_LEN=N)

        q = self.q_proj(x).view(1, N, self.heads, self.out_dim).transpose(1, 2)
        k = self.k_proj(x).view(1, N, self.heads, self.out_dim).transpose(1, 2)
        v = self.v_proj(x).view(1, N, self.heads, self.out_dim).transpose(1, 2)

        out = flex_attention(q, k, v, block_mask=block_mask)

        out = out.transpose(1, 2).reshape(N, self.heads * self.out_dim)
        return self.out_proj(out)


class FlexTransformerGAE(BaseGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim: int):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=128,          
            hidden_dim=128,             
            out_embedding_dim=out_embedding_dim,
        )
        self.model_name = "FlexTransformerGAE"
        self.activation_fn = nn.GELU()
        self.dropout = 0.2
        
        self.layer1 = FlexGraphAttentionLayer(128, 128, heads=4)
        self.layer2 = FlexGraphAttentionLayer(128, out_embedding_dim, heads=4)

    def encode(self, data):
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layer1(x, data.edge_index)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layer2(x, data.edge_index)
        
        return F.normalize(x, p=2, dim=-1)