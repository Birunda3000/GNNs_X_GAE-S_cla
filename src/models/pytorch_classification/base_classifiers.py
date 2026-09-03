import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.config import Config


class PyTorchClassifier(nn.Module):
    """Base pura para classificadores PyTorch (``nn.Module``).

    Após a migração para o padrão Trainer, contém apenas a inicialização
    básica (device, nome) e um helper de inferência. Todo o loop de treino
    vive em ``src.trainer.Trainer``; a validação de dados vive em
    ``src.datamodule.GraphDataModule``.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.model_name = self.__class__.__name__
        self.device = torch.device(config.DEVICE)
        self.to(self.device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Método forward() deve ser implementado na subclasse."
        )

    def inference(self, data: Data) -> torch.Tensor:
        """Inferência em eval/no_grad, cobrindo GNN, MLP e EmbeddingBag."""
        device = self.device
        data = data.to(device)

        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                if hasattr(data, "feature_indices"):
                    # EmbeddingBag GNN: (indices, offsets, weights, edge_index)
                    out = self(
                        data.feature_indices,
                        data.feature_offsets,
                        data.feature_weights,
                        data.edge_index,
                    )
                elif getattr(self, "use_gnn", False):
                    out = self(data.x, data.edge_index)
                else:
                    out = self(data.x)
        finally:
            if training_was:
                self.train()

        return out
