import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv.cugraph import CuGraphSAGEConv, CuGraphGATConv
from src.models.base_model import get_activation_fn
from src.models.pytorch.GraphAutoencoders.VGAE.base_vgae import BaseVGAE
from src.models.pytorch.layer_utils import prepare_edge_index, apply_conv, create_layer, forward_hidden_layers


class DynamicVGAE(BaseVGAE):
    def __init__(
        self,
        config,
        num_total_features,
        embedding_dim,
        hidden_dim,
        out_embedding_dim,
        layer_type,
        num_layers,
        activation=nn.ReLU,
        dropout=0.5,
        normalize_embeddings=True,
        **kwargs,
    ):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.activation_fn = get_activation_fn(activation)
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings
        hidden_layers = [create_layer(layer_type, embedding_dim, hidden_dim, **kwargs)]
        for _ in range(num_layers - 2):
            hidden_layers.append(create_layer(layer_type, hidden_dim, hidden_dim, **kwargs))
        self.convs_hidden = nn.ModuleList(hidden_layers)
        self.conv_mu = create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs)
        self.conv_logstd = create_layer(layer_type, hidden_dim, out_embedding_dim, **kwargs)


def encode(self, data):
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_index = prepare_edge_index(data.edge_index, x.size(0))

        # Propaga as camadas ocultas 
        x = forward_hidden_layers(
            x, edge_index, self.convs_hidden, self.activation_fn, self.dropout, self.training
        )

        # Gera os parâmetros da distribuição
        self.__mu__ = apply_conv(self.conv_mu, x, edge_index)
        self.__logstd__ = apply_conv(self.conv_logstd, x, edge_index)

        # HOISTING: Usa a função matemática teórica da classe Base!
        z = self.reparameterize(self.__mu__, self.__logstd__)

        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=-1)
        return z


class GCNVGAE(DynamicVGAE):
    """Implementação Clássica do GCN-VGAE usando o motor dinâmico."""

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_embedding_dim=out_embedding_dim,
            layer_type=GCNConv,  # Injeta o GCNConv aqui
            num_layers=2,  # 1 Camada Oculta + Camada de Saída (mu/logstd)
            activation=nn.ReLU,
            dropout=0.5,
            normalize_embeddings=True,
        )
        self.model_name = "GCNVGAE"


class GraphSageVGAE(DynamicVGAE):
    """Implementação Clássica do GraphSAGE-VGAE usando o motor dinâmico."""

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_embedding_dim=out_embedding_dim,
            layer_type=SAGEConv,  # Injeta o SAGEConv aqui
            num_layers=2,  # 1 Camada Oculta + Camada de Saída (mu/logstd)
            activation=nn.ReLU,
            dropout=0.5,
            normalize_embeddings=True,
        )
        self.model_name = "GraphSageVGAE"


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
            aggr="mean",
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
