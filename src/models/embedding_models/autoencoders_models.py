import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from src.models.embedding_models.base_graph_autoenconders_model import BaseGAE, BaseVGAE


class GCNGAE(BaseGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )
        z = self.conv2(x, data.edge_index)

        return F.normalize(z, p=2, dim=-1)


class GCNVGAE(BaseVGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, out_embedding_dim)
        self.conv_logstd = GCNConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )

        self.__mu__ = self.conv_mu(x, data.edge_index)
        self.__logstd__ = self.conv_logstd(x, data.edge_index)

        z = self.__mu__ + torch.randn_like(self.__mu__) * torch.exp(self.__logstd__)

        return F.normalize(z, p=2, dim=-1)



class GraphSageGAE(BaseGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )
        z = self.conv2(x, data.edge_index)

        return F.normalize(z, p=2, dim=-1)



class GraphSageVGAE(BaseVGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv_mu = SAGEConv(hidden_dim, out_embedding_dim)
        self.conv_logstd = SAGEConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )

        self.__mu__ = self.conv_mu(x, data.edge_index)
        self.__logstd__ = self.conv_logstd(x, data.edge_index)

        z = self.__mu__ + torch.randn_like(self.__mu__) * torch.exp(self.__logstd__)

        return F.normalize(z, p=2, dim=-1)

