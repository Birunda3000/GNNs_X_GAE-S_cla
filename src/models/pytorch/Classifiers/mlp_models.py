import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pytorch.Classifiers.base_classifier import PyTorchClassifier

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