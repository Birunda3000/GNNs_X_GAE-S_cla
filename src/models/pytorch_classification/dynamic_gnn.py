import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GINConv, SAGEConv # Importar SAGEConv explicitamente para checagem
from src.models.pytorch_classification.base_classifiers import PyTorchClassifier
from src.models.base_model import get_activation_fn

class DynamicGNNClassifier(PyTorchClassifier):
    
    use_gnn = True

    def __init__(
        self,
        config,
        input_dim: int,
        output_dim: int,
        layer_type: MessagePassing,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.5,
        activation = nn.ReLU,
        heads: int = 1,
        **kwargs  # <--- Captura aggr, train_eps, etc.
    ):
        super().__init__(config, input_dim, hidden_dim, output_dim)

        self.dropout_rate = dropout
        self.activation_fn = get_activation_fn(activation)
        self.num_layers = num_layers
        
        # Identificadores de tipo
        self.is_gat = layer_type.__name__ == 'GATConv' or (isinstance(layer_type, type) and issubclass(layer_type, GATConv))
        self.is_gin = layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv))
        self.is_sage = layer_type.__name__ == 'SAGEConv' or (isinstance(layer_type, type) and issubclass(layer_type, SAGEConv))

        # Captura parâmetros específicos do kwargs
        self.sage_aggr = kwargs.get('aggr', 'mean')
        self.gin_train_eps = kwargs.get('train_eps', False)

        self.convs = nn.ModuleList()
        
        # --- Helper para instanciar camadas ---
        def build_layer(in_d, out_d, is_last=False):
            if self.is_gin:
                mlp = nn.Sequential(
                    nn.Linear(in_d, out_d),
                    nn.ReLU(),
                    nn.Linear(out_d, out_d)
                )
                return layer_type(mlp, train_eps=self.gin_train_eps) # Passa train_eps
            
            elif self.is_gat:
                # Última camada: concat=False, heads=1 (padrão p/ classificação)
                if is_last:
                    return layer_type(in_d, out_d, heads=1, concat=False, dropout=dropout)
                else:
                    return layer_type(in_d, out_d, heads=heads, dropout=dropout)
            
            elif self.is_sage:
                return layer_type(in_d, out_d, aggr=self.sage_aggr) # Passa aggr
            
            else:
                # GCN e outros
                return layer_type(in_d, out_d)

        # --- 1. Camada de Entrada ---
        self.convs.append(build_layer(input_dim, hidden_dim))
        
        # Ajuste de dimensão atual
        if self.is_gat:
            current_dim = hidden_dim * heads
        else:
            current_dim = hidden_dim

        # --- 2. Camadas Ocultas ---
        for _ in range(num_layers - 2):
            self.convs.append(build_layer(current_dim, hidden_dim))
            
            if self.is_gat:
                current_dim = hidden_dim * heads
            else:
                current_dim = hidden_dim

        # --- 3. Camada de Saída ---
        self.convs.append(build_layer(current_dim, output_dim, is_last=True))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

    def verify_train_input_data(self, data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "DynamicGNN requer edge_index."






'''Tabela 16 – D1 (MUSAE-Facebook): Top-1 por família em GNN end-to-end (baseline
supervisionado).
Arquitetura
(E2E)
F1-weighted
(Val.)
Configuração Top-1 (principais hiperparâ-
metros)
SAGEConv 0.9559 
camadas: 4; hidden: 256; ativação: ReLU;
dropout: 0.5; agregador: mean

Tabela 20 – D2 (MUSAE-GitHub): Top-1 por família em GNN end-to-end (baseline
supervisionado).
Arquitetura
(E2E)
F1-weighted
(Val.)
Configuração Top-1 (principais hiperparâ-
metros)
SAGEConv 0.8705 
camadas: 3; hidden: 256; ativação: Leaky-
ReLU; dropout: 0.5; agregador: mean'''


class FacebookGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=SAGEConv,
            num_layers=4,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.ReLU,
            aggr='mean'
        )

class GitHubGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=SAGEConv,
            num_layers=3,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.LeakyReLU,
            aggr='mean'
        )