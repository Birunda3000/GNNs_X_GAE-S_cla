import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from src.models.pytorch_classification.base_classifiers import PyTorchClassifier
from torch_geometric.data import Data
from typing import Dict, Any
from src.early_stopper import EarlyStopper
import torch.optim as optim
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLPClassifier(PyTorchClassifier):
    """Classificador MLP que opera em um tensor de features denso."""

    use_gnn = False

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_model(self, data: Data, epochs: Optional[int]=None, early_stopper: Optional[EarlyStopper]=None, scheduler: Optional[ReduceLROnPlateau] = None, optimizer: Optional[optim.Optimizer] = None, criterion = nn.CrossEntropyLoss()) -> Dict[str, Any]:

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=self.config.SCHEDULER_PATIENCE, factor=self.config.SCHEDULER_FACTOR, min_lr=self.config.MIN_LR)
        
        if epochs is None:
            epochs = self.config.EPOCHS
        
        if early_stopper is None:
            early_stopper = EarlyStopper(
                patience=self.config.EARLY_STOPPING_PATIENCE,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                mode="max",
                metric_name="val_f1",
            )

        return self.internal_train_model(
            data,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=early_stopper,
            scheduler=scheduler,
            use_gnn=self.use_gnn,
            criterion=criterion
        )


class GCNClassifier(PyTorchClassifier):
    """Classificador GCN que opera em features e na estrutura do grafo."""

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
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index (data.edge_index)."

    def train_model(self, data: Data, epochs: Optional[int]=None, early_stopper: Optional[EarlyStopper]=None, scheduler: Optional[ReduceLROnPlateau] = None, optimizer: Optional[optim.Optimizer] = None, criterion = nn.CrossEntropyLoss()) -> Dict[str, Any]:

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=self.config.SCHEDULER_PATIENCE, factor=self.config.SCHEDULER_FACTOR, min_lr=self.config.MIN_LR)
        
        if epochs is None:
            epochs = self.config.EPOCHS
        
        if early_stopper is None:
            early_stopper = EarlyStopper(
                patience=self.config.EARLY_STOPPING_PATIENCE,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                mode="max",
                metric_name="val_f1",
            )

        return self.internal_train_model(
            data,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=early_stopper,
            scheduler=scheduler,
            use_gnn=self.use_gnn,
            criterion=criterion
        )


class GATClassifier(PyTorchClassifier):
    """Classificador GAT que utiliza mecanismos de atenção."""

    use_gnn = True

    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)
    
    def verify_train_input_data(self, data: Data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index (data.edge_index)."

    def train_model(self, data: Data, epochs: Optional[int]=None, early_stopper: Optional[EarlyStopper]=None, scheduler: Optional[ReduceLROnPlateau] = None, optimizer: Optional[optim.Optimizer] = None, criterion = nn.CrossEntropyLoss()) -> Dict[str, Any]:

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=self.config.SCHEDULER_PATIENCE, factor=self.config.SCHEDULER_FACTOR, min_lr=self.config.MIN_LR)
        
        if epochs is None:
            epochs = self.config.EPOCHS
        
        if early_stopper is None:
            early_stopper = EarlyStopper(
                patience=self.config.EARLY_STOPPING_PATIENCE,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                mode="max",
                metric_name="val_f1",
            )

        return self.internal_train_model(
            data,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=early_stopper,
            scheduler=scheduler,
            use_gnn=self.use_gnn,
            criterion=criterion
        )