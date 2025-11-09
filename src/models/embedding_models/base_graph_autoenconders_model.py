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
from torch.optim import lr_scheduler

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
            mode="sum",# ***Uso "sum" para agregar embeddings de features, media não implementada***
        )

    # ========== MÉTODOS GENÉRICOS ==========

    def verify_train_input_data(self, data: Data):
        assert data.edge_index is not None, "Input data must contain edge_index."
        assert data.feature_indices is not None, "Input data must contain feature_indices."
        assert data.feature_offsets is not None, "Input data must contain feature_offsets."
        assert data.feature_weights is not None, "Input data must contain feature_weights."
        assert data.num_nodes is not None and data.num_nodes > 0, "data.num_nodes must be valid."

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Produto escalar entre embeddings de nós conectados."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def reconstruction_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor) -> torch.Tensor:
        """Loss de reconstrução com amostragem negativa."""
        pos_logits = self.decode(z, pos_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, z.new_ones(pos_edge_index.size(1)))

        num_neg_samples = pos_edge_index.size(1)
        neg_edge_index = torch.randint(0, z.size(0), (2, num_neg_samples), dtype=torch.long, device=z.device)
        neg_logits = self.decode(z, neg_edge_index)
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, z.new_zeros(num_neg_samples))

        return pos_loss + neg_loss

    def train_model(self, data: Data, optimizer: optim.Optimizer, epochs: int, early_stopper: EarlyStopper, scheduler) -> Dict[str, Any]:
        """Loop de treino genérico, compartilhado entre GAE e VGAE."""
        self.verify_train_input_data(data)
        edge_index = cast(torch.Tensor, data.edge_index)
        training_history: List[Dict[str, Any]] = []

        score: Optional[float] = None
        report: Optional[Dict[str, Any]] = None
        stop_now: bool = False
        best_epoch: Optional[int] = None

        pbar = tqdm(range(1, epochs + 1), desc=f"Treinando {self.model_name}", leave=False)

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

            training_history.append({
                "epoch": epoch,
                "Time_per_epoch": time.process_time() - start_time,
                "train_total_loss": total_loss.item(),
                "test_total_loss": None,
                "learning_rate": scheduler.get_last_lr()[0],
                "early_stopping_score": score,
                "early_stopping_report": report
            })

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            if early_stopper is not None and stop_now:
                print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                early_stopper.restore_best_state(self)
                break

        return {
            "total_training_time": time.process_time() - start_time,
            "best_epoch": best_epoch,
            "best_score": score,
            "final_train_loss": total_loss.item(),
            "training_history": training_history,
        }

    # ========== MÉTODOS A SEREM IMPLEMENTADOS ==========

    def encode(self, data: Data) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the encode method.")

    def compute_total_loss(self, z: torch.Tensor, data: Data, edge_index: torch.Tensor):
        raise NotImplementedError("Subclasses must implement the compute_total_loss method.")
    
    def inference(self, input_data: Data) -> torch.Tensor:
        """
        Inferência padrão: para modelos determinísticos (GAE),
        chamamos encode() dentro de no_grad() para obter embeddings.
        Para modelos variacionais, BaseVGAE sobrescreverá esse método
        para retornar mu (média) ao invés de uma amostra.
        """
        with torch.no_grad():
            z = self.encode(input_data)
            # garante que retornamos um Tensor CPU/GPU consistente e normalizado
            if isinstance(z, torch.Tensor):
                return z
            else:
                return torch.as_tensor(z)
    
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
                1 + 2 * self.__logstd__ - self.__mu__.pow(2) - self.__logstd__.exp().pow(2),
                dim=1,
            )
        )

    def compute_total_loss(self, z, data, edge_index):
        assert data.num_nodes is not None, "data.num_nodes must be valid."
        return self.reconstruction_loss(z, edge_index) + (1.0 / float(data.num_nodes)) * self.kl_loss()
    
    def inference(self, input_data: Data) -> Tensor:
        """
        Inferência para o VGAE: usa a média (mu) em vez de amostragem.
        Supõe que a subclasse define `conv1` e `conv_mu` (como GCNVGAE ou GraphSageVGAE).
        """
        with torch.no_grad():
            # Passo 1: gerar embeddings iniciais
            x = self.feature_embedder(
                input_data.feature_indices,
                input_data.feature_offsets,
                per_sample_weights=input_data.feature_weights,
            )
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

            # Passo 2: obtém referências seguras e com casting explícito de tipo
            conv1 = cast(MessagePassing, getattr(self, "conv1", None))
            conv_mu = cast(MessagePassing, getattr(self, "conv_mu", None))

            if conv1 is None or conv_mu is None:
                raise AttributeError(
                    "A subclasse VGAE deve definir camadas `conv1` e `conv_mu` antes de usar inference()."
                )

            # Passo 3: aplica o encoder determinístico (garantido que o retorno é Tensor)
            x = torch.nn.functional.relu(conv1(x, input_data.edge_index))
            mu: Tensor = conv_mu(x, input_data.edge_index)

            # Passo 4: retorna a média latente (Tensor)
            return mu
