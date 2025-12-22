import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GINConv, SAGEConv
from src.models.embedding_models.base_graph_autoenconders_model import BaseGAE, BaseVGAE
from src.models.base_model import get_activation_fn

def create_layer(layer_type, in_dim, out_dim, **kwargs):
    """
    Fábrica de camadas que distribui os parâmetros corretos para cada tipo.
    """
    # 1. GIN: Precisa de MLP interno e train_eps
    if layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv)):
        train_eps = kwargs.get('train_eps', False) # Default False
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        return layer_type(mlp, train_eps=train_eps)
    
    # 2. GAT: Precisa de heads, concat e dropout
    elif layer_type.__name__ == 'GATConv' or (isinstance(layer_type, type) and issubclass(layer_type, GATConv)):
        heads = kwargs.get('heads', 1)
        dropout = kwargs.get('dropout', 0.0)
        # Forçamos concat=False no Autoencoder para manter dimensões controladas
        return layer_type(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
    
    # 3. SAGE: Precisa de aggregator (aggr)
    elif layer_type.__name__ == 'SAGEConv' or (isinstance(layer_type, type) and issubclass(layer_type, SAGEConv)):
        aggr = kwargs.get('aggr', 'mean')
        return layer_type(in_dim, out_dim, aggr=aggr)
    
    # 4. GCN (Baseline) ou Outros
    else:
        return layer_type(in_dim, out_dim)

# --- Classes DynamicGAE e DynamicVGAE (Atualizadas) ---

class DynamicGAE(BaseGAE):
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim, 
                 layer_type, num_layers, activation=nn.ReLU, dropout=0.5, normalize_embeddings=True, 
                 **kwargs): # <--- Captura kwargs extras (heads, aggr, train_eps)
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings

        layers = []
        # Camada 1
        layers.append(create_layer(layer_type, embedding_dim, hidden_dim, **kwargs))

        # Camadas Ocultas
        for _ in range(num_layers - 2):
            layers.append(create_layer(layer_type, hidden_dim, hidden_dim, **kwargs))

        # Camada Final
        layers.append(create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs))

        self.convs = nn.ModuleList(layers)

    def encode(self, data):
        # ... (implementação do encode igual ao anterior) ...
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.normalize_embeddings:
            return F.normalize(x, p=2, dim=-1)
        return x

class DynamicVGAE(BaseVGAE):
    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim, 
                 layer_type, num_layers, activation=nn.ReLU, dropout=0.5, normalize_embeddings=True, 
                 **kwargs): # <--- Captura kwargs extras
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings

        hidden_layers = []
        # Camada 1
        hidden_layers.append(create_layer(layer_type, embedding_dim, hidden_dim, **kwargs))
        
        # Camadas Intermediárias
        for _ in range(num_layers - 2):
            hidden_layers.append(create_layer(layer_type, hidden_dim, hidden_dim, **kwargs))
            
        self.convs_hidden = nn.ModuleList(hidden_layers)

        # Camadas Mu e LogStd (Finais)
        self.conv_mu = create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs)
        self.conv_logstd = create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs)

    def encode(self, data):
        # ... (implementação do encode igual ao anterior) ...
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index = data.edge_index

        for conv in self.convs_hidden:
            x = conv(x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        self.__mu__ = self.conv_mu(x, edge_index)
        self.__logstd__ = self.conv_logstd(x, edge_index)
        
        eps = torch.randn_like(self.__mu__)
        z = self.__mu__ + eps * torch.exp(self.__logstd__)

        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=-1)
        return z

# Classe otimizada para Facebook
class FacebookGAE(DynamicGAE):
    """
    Modelo GAE otimizado para o dataset Facebook (MUSAE).
    
    Hiperparâmetros selecionados:
    - Layer: GATConv (Atenção é fundamental para redes sociais)
    - Layers: 4 (Profundo, mas o GAT mitiga oversmoothing)
    - Heads: 1 (Simples e eficiente)
    - Hidden Dim: 256
    - Dropout: 0.5
    - Activation: GELU (Mais suave que ReLU)
    - Embedding Dim (Input): 64
    - Out Embedding Dim (Latente/Z): 32
    - Normalize Embeddings: True (Crítico para qualidade dos embeddings)
    """

    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            # Dimensões
            embedding_dim=64,           # Input embedding
            hidden_dim=256,             # Camadas ocultas
            out_embedding_dim=out_embedding_dim,       # Latente Z
            # Arquitetura
            layer_type=GATConv,         # Mecanismo de atenção
            num_layers=4,               # Profundidade
            # Regularização
            activation=nn.GELU,         # Ativação suave
            dropout=0.5,                # Dropout padrão
            normalize_embeddings=True,  # Normalização L2
            # Parâmetros específicos do GATConv
            heads=1,                    # Número de cabeças de atenção
        )
        self.model_name = "FacebookGAE"


# Classe otimizada para Github
class GithubVGAE(DynamicVGAE):
    """
    Modelo VGAE otimizado para o dataset Github (DEV).
    
    Hiperparâmetros selecionados:
    - Layer: SAGEConv (Boa generalização em grafos de colaboração)
    - Agregador: mean (Simples e eficaz)
    - Layers: 4 (Profundo, mas controlado)
    - Hidden Dim: 256
    - Dropout: 0.1 (Baixo para preservar informação)
    - Activation: LeakyReLU (Evita neurônios mortos)
    - Embedding Dim (Input): 256
    - Out Embedding Dim (Latente/Z): 32
    - Normalize Embeddings: True (Importante para qualidade dos embeddings)
    """

    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            # Dimensões
            embedding_dim=256,          # Input embedding
            hidden_dim=256,             # Camadas ocultas
            out_embedding_dim=out_embedding_dim,       # Latente Z
            # Arquitetura
            layer_type=SAGEConv,        # GraphSAGE
            num_layers=4,               # Profundidade
            # Regularização
            activation=nn.LeakyReLU,    # Evita neurônios mortos
            dropout=0.1,                # Baixo dropout
            normalize_embeddings=True,  # Normalização L2
            # Parâmetros específicos do SAGEConv
            aggr='mean',                # Agregador mean
        )
        self.model_name = "GithubVGAE"