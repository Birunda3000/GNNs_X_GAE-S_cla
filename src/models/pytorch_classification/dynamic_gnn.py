import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from src.models.pytorch_classification.base_classifiers import PyTorchClassifier

def get_activation_fn(activation):
    """Auxiliar para converter classe/instância de ativação em função funcional."""
    if isinstance(activation, type):
        activation = activation()
    
    if isinstance(activation, nn.ReLU): return F.relu
    if isinstance(activation, nn.ELU): return F.elu
    if isinstance(activation, nn.LeakyReLU): return F.leaky_relu
    if isinstance(activation, nn.Tanh): return torch.tanh
    return F.relu

class DynamicGNNClassifier(PyTorchClassifier):
    """
    Classificador GNN Dinâmico para Grid Search (Motor 3).
    Suporta: GCNConv, SAGEConv, GATConv.
    """
    
    use_gnn = True

    def __init__(
        self,
        config,
        input_dim: int,
        output_dim: int,
        # Parâmetros do Grid
        layer_type: MessagePassing,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.5,
        activation = nn.ReLU,
        heads: int = 1, # Específico para GAT
        **kwargs # Captura outros argumentos para evitar erros
    ):
        super().__init__(config, input_dim, hidden_dim, output_dim)

        self.dropout_rate = dropout
        self.activation_fn = get_activation_fn(activation)
        self.num_layers = num_layers
        
        # Verifica se é GAT para ajustar dimensões de entrada/saída (multi-head)
        self.is_gat = layer_type.__name__ == 'GATConv' or (isinstance(layer_type, type) and issubclass(layer_type, GATConv))
        
        self.convs = nn.ModuleList()
        
        # --- 1. Camada de Entrada ---
        # Se for GAT, a saída é hidden_dim * heads
        if self.is_gat:
            self.convs.append(layer_type(input_dim, hidden_dim, heads=heads, dropout=dropout))
            current_dim = hidden_dim * heads
        else:
            self.convs.append(layer_type(input_dim, hidden_dim))
            current_dim = hidden_dim

        # --- 2. Camadas Ocultas (se num_layers > 2) ---
        for _ in range(num_layers - 2):
            if self.is_gat:
                self.convs.append(layer_type(current_dim, hidden_dim, heads=heads, dropout=dropout))
                current_dim = hidden_dim * heads
            else:
                self.convs.append(layer_type(current_dim, hidden_dim))
                current_dim = hidden_dim

        # --- 3. Camada de Saída ---
        # A última camada projeta para output_dim (classes)
        # Para GAT, geralmente concat=False na última camada para tirar a média ou somar
        if self.is_gat:
            self.convs.append(layer_type(current_dim, output_dim, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(layer_type(current_dim, output_dim))

    def forward(self, x, edge_index):
        # Aplica camadas exceto a última
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Última camada (Sem ativação/dropout, retorna logits brutos)
        x = self.convs[-1](x, edge_index)
        
        return x

    def verify_train_input_data(self, data):
        # Chama verificação da base e adiciona verificação de arestas
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "DynamicGNN requer edge_index."