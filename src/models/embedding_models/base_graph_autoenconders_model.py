import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from typing import Dict, Any, List, Optional, cast
from tqdm import tqdm
import time
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.utils import negative_sampling

from src.models.base_model import BaseModel
from src.early_stopper import EarlyStopper


class BaseGAECommon(BaseModel, nn.Module):
    """
    Classe intermediária base para todos os Autoencoders de Grafo (GAE/VGAE).
    Contém:
        - feature_embedder (EmbeddingBag)
        - verificação de dados
        - decodificador e função de reconstrução
        - loop de treino genérico
    """

    def __init__(
        self,
        config,
        num_total_features: int,
        embedding_dim: int,
        hidden_dim: int,
        out_embedding_dim: int,
    ):
        BaseModel.__init__(self, config)
        nn.Module.__init__(self)

        # Camada de embedding para features esparsas
        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum",  # ***Uso "sum" para agregar embeddings de features, media não implementada***
        )

    # ========== MÉTODOS GENÉRICOS ==========

    def verify_train_input_data(self, data: Data):
        assert data.edge_index is not None, "Input data must contain edge_index."
        assert (
            data.feature_indices is not None
        ), "Input data must contain feature_indices."
        assert (
            data.feature_offsets is not None
        ), "Input data must contain feature_offsets."
        assert (
            data.feature_weights is not None
        ), "Input data must contain feature_weights."
        assert (
            data.num_nodes is not None and data.num_nodes > 0
        ), "data.num_nodes must be valid."

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

    def train_model(
        self,
        data: Data,
        optimizer: optim.Optimizer,
        epochs: int,
        early_stopper: EarlyStopper,
        scheduler,
    ) -> Dict[str, Any]:
        """Loop de treino genérico, compartilhado entre GAE e VGAE."""
        self.verify_train_input_data(data)

        # Move Data para o mesmo device do modelo
        device = next(self.parameters()).device
        data = data.to(device)

        edge_index = cast(torch.Tensor, data.edge_index)
        training_history: List[Dict[str, Any]] = []

        score: Optional[float] = None
        report: Optional[Dict[str, Any]] = None
        stop_now: bool = False
        best_epoch: Optional[int] = None

        pbar = tqdm(
            range(1, epochs + 1), desc=f"Treinando {self.model_name}", leave=False
        )

        start_time = time.process_time()

        for epoch in pbar:
            self.train()
            optimizer.zero_grad()
            z = self.encode(data)
            total_loss = self.compute_total_loss(z, data, edge_index)
            total_loss.backward()
            optimizer.step()

            stop_now, score, best_epoch, report = early_stopper.check(self, epoch=epoch)
            scheduler.step(score)

            training_history.append(
                {
                    "epoch": epoch,
                    "Time_per_epoch": time.process_time() - start_time,
                    "train_total_loss": total_loss.item(),
                    "test_total_loss": None,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "early_stopping_score": score,
                    "early_stopping_report": report,
                }
            )

            pbar.set_postfix(
                {"loss": f"{total_loss.item():.4f}", "score": f"{score:.4f}"}
            )

            if early_stopper is not None and stop_now:
                print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                early_stopper.restore_best_state(self)
                break

        return {
            "total_training_time": time.process_time() - start_time,
            "best_epoch": best_epoch,
            "best_score": early_stopper.best_value,
            "training_history": training_history,
        }

    # ========== MÉTODOS A SEREM IMPLEMENTADOS ==========

    def encode(self, data: Data) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the encode method.")

    def compute_total_loss(self, z: torch.Tensor, data: Data, edge_index: torch.Tensor):
        raise NotImplementedError(
            "Subclasses must implement the compute_total_loss method."
        )

    def inference(self, input_data: Data) -> torch.Tensor:
        """
        Inferência padrão para modelos determinísticos (GAE).
        Chama encode() dentro de no_grad() e em modo eval() para garantir comportamento determinístico.
        """
        device = next(self.parameters()).device
        input_data.to(device)

        # ✅ Garante comportamento determinístico (desliga dropout)
        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                z = self.encode(input_data)
        finally:
            # Restaura o modo anterior (train ou eval)
            if training_was:
                self.train()

        return z

    def evaluate(self, input_data: Data) -> Any:
        """
        Implementação mínima de evaluate para satisfazer a interface da BaseModel.
        Por padrão, retorna os embeddings (resultado de inference).
        Subclasses podem sobrescrever para retornar métricas específicas.
        """
        return self.inference(input_data)


class BaseGAE(BaseGAECommon):
    """Versão determinística (GAE)."""

    def compute_total_loss(self, z, data, edge_index):
        return self.reconstruction_loss(z, edge_index)


class BaseVGAE(BaseGAECommon):
    """Versão variacional (VGAE), com perda KL."""

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
        return (
            self.reconstruction_loss(z, edge_index)
            + (1.0 / float(data.num_nodes)) * self.kl_loss()
        )

    def inference(self, input_data: Data) -> Tensor:
        """
        Inferência para o VGAE: usa a média (mu) em vez de amostragem.
        Chama o método `encode` da subclasse em modo de avaliação e retorna a média `__mu__`.
        """
        device = next(self.parameters()).device
        input_data.to(device)

        # Garante comportamento determinístico (desliga dropout) e restaura estado
        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                # Chama o método encode da subclasse (GCNVGAE, etc.)
                # que irá popular self.__mu__
                self.encode(input_data)
        finally:
            if training_was:
                self.train()

        if self.__mu__ is None:
            raise RuntimeError("O atributo `__mu__` não foi definido pelo método `encode`.")

        return self.__mu__
