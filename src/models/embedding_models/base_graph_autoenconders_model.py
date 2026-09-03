import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.utils import negative_sampling


class BaseGAECommon(nn.Module):
    """
    Classe intermediária base para todos os Autoencoders de Grafo (GAE/VGAE).
    Contém apenas:
        - feature_embedder (EmbeddingBag)
        - decodificador e função de reconstrução

    Após a migração para o padrão Trainer, o loop de treino foi removido e
    vive em ``src.trainer.Trainer.fit_gae``.
    """

    def __init__(
        self,
        config,
        num_total_features: int,
        embedding_dim: int,
        hidden_dim: int,
        out_embedding_dim: int,
    ):
        super().__init__()
        self.config = config
        self.model_name = self.__class__.__name__

        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum",
        )

    # ========== MÉTODOS GENÉRICOS ==========

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Produto escalar entre embeddings de nós conectados."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def reconstruction_loss(
        self, z: torch.Tensor, pos_edge_index: torch.Tensor
    ) -> torch.Tensor:
        pos_logits = self.decode(z, pos_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, z.new_ones(pos_edge_index.size(1))
        )

        neg_edge_index = negative_sampling(
            pos_edge_index, num_nodes=z.size(0), num_neg_samples=pos_edge_index.size(1)
        )
        neg_logits = self.decode(z, neg_edge_index)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, z.new_zeros(neg_edge_index.size(1))
        )
        return pos_loss + neg_loss

    # ========== MÉTODOS A SEREM IMPLEMENTADOS ==========

    def encode(self, data: Data) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the encode method.")

    def compute_total_loss(self, z: torch.Tensor, data: Data, edge_index: torch.Tensor):
        raise NotImplementedError("Subclasses must implement the compute_total_loss method.")

    def inference(self, input_data: Data) -> torch.Tensor:
        device = next(self.parameters()).device
        input_data.to(device)

        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                z = self.encode(input_data)
        finally:
            if training_was:
                self.train()

        return z


class BaseGAE(BaseGAECommon):
    def compute_total_loss(self, z, data, edge_index):
        return self.reconstruction_loss(z, edge_index)


class BaseVGAE(BaseGAECommon):
    conv1: MessagePassing
    conv_mu: MessagePassing
    conv_logstd: MessagePassing

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mu__ = self.__logstd__ = None

    def kl_loss(self) -> torch.Tensor:
        if self.__mu__ is None or self.__logstd__ is None:
            return torch.tensor(0.0)
        return -0.5 * torch.mean(
            torch.sum(
                1
                + 2 * self.__logstd__
                - self.__mu__.pow(2)
                - self.__logstd__.exp().pow(2),
                dim=1,
            )
        )

    def compute_total_loss(self, z, data, edge_index):
        assert data.num_nodes is not None, "data.num_nodes must be valid."
        return self.reconstruction_loss(z, edge_index) + (
            1.0 / float(data.num_nodes)
        ) * self.kl_loss()

    def inference(self, input_data: Data) -> Tensor:
        device = next(self.parameters()).device
        input_data.to(device)

        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                self.encode(input_data)
        finally:
            if training_was:
                self.train()

        if self.__mu__ is None:
            raise RuntimeError("O atributo `__mu__` não foi definido pelo método `encode`.")

        return self.__mu__
