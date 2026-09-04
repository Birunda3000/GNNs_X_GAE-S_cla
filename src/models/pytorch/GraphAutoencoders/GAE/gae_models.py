import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv.cugraph import CuGraphGATConv
from src.models.base_model import get_activation_fn
from src.models.pytorch.GraphAutoencoders.GAE.base_gae import BaseGAE

# Importação atualizada com a nova função forward_hidden_layers
from src.models.pytorch.layer_utils import (
    prepare_edge_index, 
    apply_conv, 
    create_layer, 
    forward_hidden_layers
)

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

class GCNGAE(DynamicGAE):
    """Implementação Clássica do GCN-GAE usando o motor dinâmico."""
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(
            config=config, 
            num_total_features=num_total_features, 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            out_embedding_dim=out_embedding_dim,
            layer_type=GCNConv,       # Injeta o GCNConv
            num_layers=2,             # 1 Oculta + 1 Saída
            activation=nn.ReLU,
            dropout=0.5,
            normalize_embeddings=True
        )
        self.model_name = "GCNGAE"

class GraphSageGAE(DynamicGAE):
    """Implementação Clássica do GraphSAGE-GAE usando o motor dinâmico."""
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(
            config=config, 
            num_total_features=num_total_features, 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            out_embedding_dim=out_embedding_dim,
            layer_type=SAGEConv,      # Injeta o SAGEConv
            num_layers=2,             # 1 Oculta + 1 Saída
            activation=nn.ReLU,
            dropout=0.5,
            normalize_embeddings=True
        )
        self.model_name = "GraphSageGAE"

class DynamicGAE(BaseGAE):
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim, layer_type, num_layers, activation=nn.ReLU, dropout=0.5, normalize_embeddings=True, **kwargs):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings
        layers = [create_layer(layer_type, embedding_dim, hidden_dim, **kwargs)]
        for _ in range(num_layers - 2):
            layers.append(create_layer(layer_type, hidden_dim, hidden_dim, **kwargs))
        layers.append(create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs))
        self.convs = nn.ModuleList(layers)

    def encode(self, data):
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        edge_index = prepare_edge_index(data.edge_index, x.size(0))
        
        # Propaga as camadas ocultas utilizando a função unificada
        x = forward_hidden_layers(
            x, edge_index, self.convs[:-1], self.activation_fn, self.dropout, self.training
        )
        
        # Aplica a camada final
        x = apply_conv(self.convs[-1], x, edge_index)
        
        if self.normalize_embeddings:
            return F.normalize(x, p=2, dim=-1)
        return x

class FacebookGAE(DynamicGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(config=config, num_total_features=num_total_features, embedding_dim=64, hidden_dim=256, out_embedding_dim=out_embedding_dim, layer_type=CuGraphGATConv, num_layers=4, activation=nn.GELU, dropout=0.5, normalize_embeddings=True, heads=1)
        self.model_name = "FacebookGAE"

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
        super().__init__(config=config, num_total_features=num_total_features, embedding_dim=128, hidden_dim=128, out_embedding_dim=out_embedding_dim)
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
