import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from typing import Dict, Any, List, Optional, cast
from tqdm import tqdm
import gc
from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.utils import negative_sampling

from src.models.base_model import BaseModel
# Importa o novo orquestrador (ajuste o nome se você o nomeou diferente no early_stopper.py)
from src.early_stopper import UniversalEarlyStopper 
from src.utils import DeviceTimer

# 🔥 FASE 3: Suporte ao NVIDIA Transformer Engine para FP8
try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False

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

        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum",
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
        early_stopper: UniversalEarlyStopper, # ✅ Atualizado para a tipagem correta
        scheduler,
        scheduler_metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Loop de treino genérico, compartilhado entre GAE e VGAE."""
        self.verify_train_input_data(data)

        device = next(self.parameters()).device
        data = data.to(device)

        edge_index = cast(torch.Tensor, data.edge_index)
        training_history: List[Dict[str, Any]] = []

        stop_now: bool = False

        pbar = tqdm(range(1, epochs + 1), desc=f"Treinando {self.model_name}", leave=False)
        epoch_timer = DeviceTimer(self.config.DEVICE, disable_gc=False)

        self.compile_methods(["encode", "decode"], dynamic=True)

        with DeviceTimer(self.config.DEVICE, disable_gc=True) as total_timer:

            for epoch in pbar:
                with epoch_timer:
                    self.train()
                    optimizer.zero_grad()

                    if HAS_TE:
                        with te.fp8_autocast(enabled=True):
                            z = self.encode(data)
                            total_loss = self.compute_total_loss(z, data, edge_index)
                    else:
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            z = self.encode(data)
                            total_loss = self.compute_total_loss(z, data, edge_index)

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()


                stop_now, report = early_stopper.check(
                    epoch=epoch,
                    model=self,
                    data=data,
                    train_mask=data.train_mask,
                    eval_mask=data.val_mask
                )
                scheduler.step(report[scheduler_metric_name] if scheduler_metric_name else total_loss.item())

                training_history.append(
                    {
                        "epoch": epoch,
                        "Time_per_epoch": epoch_timer.duration,
                        "train_total_loss": total_loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "early_stopping_report": report,
                    }
                )
                gc.collect()
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

                if stop_now:
                    print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                    early_stopper.restore_best_state(self)
                    break

        self.decompile_methods()

        return {
            "total_training_time": total_timer.duration,
            "best_epoch": early_stopper.best_epoch,
            "best_scores": early_stopper.best_values,
            "training_history": training_history,
        }

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

    def evaluate(self, input_data: Data) -> Any:
        return self.inference(input_data)


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
                1 + 2 * self.__logstd__ - self.__mu__.pow(2) - self.__logstd__.exp().pow(2),
                dim=1,
            )
        )

    def compute_total_loss(self, z, data, edge_index):
        assert data.num_nodes is not None, "data.num_nodes must be valid."
        return self.reconstruction_loss(z, edge_index) + (1.0 / float(data.num_nodes)) * self.kl_loss()

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