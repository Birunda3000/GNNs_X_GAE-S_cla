import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.cugraph import CuGraphSAGEConv
from src.models.pytorch.Classifiers.base_classifier import PyTorchClassifier
from src.models.pytorch.Classifiers.GNN.dynamic_gnn import DynamicGNNClassifier, DynamicEmbeddingGNNClassifier

class GCNClassifier(PyTorchClassifier):
    use_gnn = True
    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
        
    def verify_train_input_data(self, data: Data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index."

class GATClassifier(PyTorchClassifier):
    use_gnn = True
    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)
        
    def verify_train_input_data(self, data: Data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index."

class FacebookGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(config=config, input_dim=input_dim, output_dim=output_dim, layer_type=CuGraphSAGEConv, num_layers=4, hidden_dim=256, dropout=0.5, activation=nn.ReLU, aggr='mean')

class GitHubGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(config=config, input_dim=input_dim, output_dim=output_dim, layer_type=CuGraphSAGEConv, num_layers=3, hidden_dim=256, dropout=0.5, activation=nn.LeakyReLU, aggr='mean')

class FacebookEmbeddingGNN(DynamicEmbeddingGNNClassifier):
    def __init__(self, config, num_total_features, output_dim):
        super().__init__(config=config, num_total_features=num_total_features, embedding_dim=64, output_dim=output_dim, layer_type=CuGraphSAGEConv, num_layers=4, hidden_dim=256, dropout=0.5, activation=nn.ReLU, aggr='mean')
        self.model_name = "FacebookEmbeddingGNN"

class GithubEmbeddingGNN(DynamicEmbeddingGNNClassifier):
    def __init__(self, config, num_total_features, output_dim):
        super().__init__(config=config, num_total_features=num_total_features, embedding_dim=256, output_dim=output_dim, layer_type=CuGraphSAGEConv, num_layers=3, hidden_dim=256, dropout=0.5, activation=nn.LeakyReLU, aggr='mean')
        self.model_name = "GithubEmbeddingGNN"
