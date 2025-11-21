import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from src.models.embedding_models.base_graph_autoenconders_model import BaseGAE, BaseVGAE


def get_activation_fn(activation):
    """
    Converte ativação (classe ou instância) para função.

    Args:
        activation: nn.Module class (nn.ReLU) ou instância (nn.ReLU())

    Returns:
        Função de ativação (F.relu, F.elu, etc.)
    """
    # ✅ Suporta tanto classe quanto instância
    if isinstance(activation, type):
        activation = activation()

    mapping = {
        nn.ReLU: F.relu,
        nn.ELU: F.elu,
        nn.LeakyReLU: F.leaky_relu,
        nn.Tanh: torch.tanh,
    }

    # Busca por tipo da instância
    for cls, fn in mapping.items():
        if isinstance(activation, cls):
            return fn

    # Fallback para ReLU
    print(f"[WARNING] Ativação {type(activation)} não reconhecida, usando ReLU")
    return F.relu


class DynamicGAE(BaseGAE):
    """
    Dynamic Graph Autoencoder for grid search experiments.

    Architecture:
        EmbeddingBag → Dropout → (Conv → Act → Dropout)* → Conv → Normalize(optional)
    """

    def __init__(
        self,
        config,
        num_total_features,
        embedding_dim,
        hidden_dim,
        out_embedding_dim,
        layer_type: MessagePassing,
        num_layers: int,
        activation=nn.ReLU,
        dropout=0.5,
        normalize_embeddings=True,  # ✅ ADICIONADO
    ):
        super().__init__(
            config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim
        )

        assert num_layers >= 2, "num_layers must be >= 2"
        assert 0.0 <= dropout <= 1.0
        assert embedding_dim > 0 and hidden_dim > 0 and out_embedding_dim > 0

        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings  # ✅ ADICIONADO

        layers = []
        layers.append(layer_type(embedding_dim, hidden_dim))

        for _ in range(num_layers - 2):
            layers.append(layer_type(hidden_dim, hidden_dim))

        layers.append(layer_type(hidden_dim, out_embedding_dim))

        self.convs = nn.ModuleList(layers)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.activation_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ✅ Normalização condicional
        if self.normalize_embeddings:
            return F.normalize(x, p=2, dim=-1)
        return x


class DynamicVGAE(BaseVGAE):
    """
    Dynamic Variational Graph Autoencoder for grid search.

    Architecture:
        EmbeddingBag → Dropout → (Conv → Act → Dropout)* → mu/logstd → Reparam → Normalize(optional)
    """

    def __init__(
        self,
        config,
        num_total_features,
        embedding_dim,
        hidden_dim,
        out_embedding_dim,
        layer_type: MessagePassing,
        num_layers: int,
        activation=nn.ReLU,
        dropout=0.5,
        normalize_embeddings=True,
    ):
        super().__init__(
            config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim
        )

        assert num_layers >= 2
        assert 0.0 <= dropout <= 1.0

        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings

        hidden_layers = []
        hidden_layers.append(layer_type(embedding_dim, hidden_dim))

        for _ in range(num_layers - 2):
            hidden_layers.append(layer_type(hidden_dim, hidden_dim))

        self.convs_hidden = nn.ModuleList(hidden_layers)

        self.conv_mu = layer_type(hidden_dim, out_embedding_dim)
        self.conv_logstd = layer_type(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )

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
