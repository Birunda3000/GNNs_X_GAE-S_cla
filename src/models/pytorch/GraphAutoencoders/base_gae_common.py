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

# 🔥 NOVA IMPORTAÇÃO: Usando os Loaders do PyG e removendo o negative_sampling manual
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from src.models.pytorch.pytorch_base_model import PyTorchBaseModel
from src.early_stopper import UniversalEarlyStopper 
from src.utils import DeviceTimer

# Suporte ao NVIDIA Transformer Engine para FP8
try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False

# HERANÇA SIMPLIFICADA: Apenas PyTorchBaseModel (já inclui nn.Module e BaseModel)
class BaseGAECommon(PyTorchBaseModel):
    """
    Classe intermediária base para todos os Autoencoders de Grafo (GAE/VGAE).
    Contém:
        - feature_embedder (EmbeddingBag) -> [AVISO: Será refatorado em breve]
        - verificação de dados
        - decodificador e função de reconstrução adaptada para Mini-Batch
        - loop de treino genérico em lotes
    """

    def __init__(
        self,
        config,
        num_total_features: int,
        embedding_dim: int,
        hidden_dim: int,
        out_embedding_dim: int,
    ):
        # INICIALIZAÇÃO CENTRALIZADA
        super().__init__(config)

        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum",
        )

    # ========== MÉTODOS GENÉRICOS ==========

    def verify_train_input_data(self, data: Data):
        assert data.edge_index is not None, "Input data must contain edge_index."
        # assert data.feature_indices is not None, "Input data must contain feature_indices."
        # assert data.feature_offsets is not None, "Input data must contain feature_offsets."
        # assert data.feature_weights is not None, "Input data must contain feature_weights."
        assert data.num_nodes is not None and data.num_nodes > 0, "data.num_nodes must be valid."
        assert hasattr(data, 'train_mask') and data.train_mask is not None, "Input data must contain train_mask."
        assert hasattr(data, 'val_mask') and data.val_mask is not None, "Input data must contain val_mask."

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Produto escalar entre embeddings de nós conectados."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def reconstruction_loss(self, z: torch.Tensor, edge_label_index: torch.Tensor, edge_label: torch.Tensor) -> torch.Tensor:
        """
        Calcula a perda (BCE) comparando a predição das arestas contra o gabarito.
        O 'edge_label_index' e 'edge_label' já vêm do LinkNeighborLoader contendo 
        arestas reais e falsas (amostradas dinamicamente em CPU).
        """
        logits = self.decode(z, edge_label_index)
        return F.binary_cross_entropy_with_logits(logits, edge_label.float())

    def train_model(
        self,
        data: Data,
        optimizer: optim.Optimizer,
        epochs: int,
        early_stopper: UniversalEarlyStopper,
        scheduler,
        scheduler_metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Loop de treino genérico adaptado para Mini-Batch."""
        self.verify_train_input_data(data)
        device = self.device

        # 🔥 O Grafo inteiro NÃO vai mais para a GPU.
        # O Loader fatia o grafo e já gera as amostras negativas em CPU
        train_loader = LinkNeighborLoader(
            data,
            num_neighbors=self.config.GAE_NUM_NEIGHBORS,
            batch_size=self.config.GAE_BATCH_SIZE,
            edge_label_index=data.edge_index,
            neg_sampling_ratio=1.0, 
            shuffle=True,
        )

        training_history: List[Dict[str, Any]] = []
        stop_now: bool = False

        pbar = tqdm(range(1, epochs + 1), desc=f"Treinando {self.model_name} (Mini-Batch)", leave=False)
        epoch_timer = DeviceTimer(self.config.DEVICE, disable_gc=False)

        self.compile_methods(["encode", "decode"], dynamic=True)

        with DeviceTimer(self.config.DEVICE, disable_gc=True) as total_timer:
            for epoch in pbar:
                self.train()
                total_loss_epoch = 0.0
                num_batches = 0

                with epoch_timer:
                    # Iteramos pelos subgrafos (batches) gerados em tempo real
                    for batch in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()

                        if HAS_TE:
                            with te.fp8_autocast(enabled=True):
                                z = self.encode(batch)
                                loss = self.compute_total_loss(z, batch)
                        else:
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                z = self.encode(batch)
                                loss = self.compute_total_loss(z, batch)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_loss_epoch += loss.item()
                        num_batches += 1

                avg_train_loss = total_loss_epoch / num_batches if num_batches > 0 else 0.0

                # O Early Stopper avaliará o modelo usando a função segura _predict_step
                stop_now, report = early_stopper.check(
                    epoch=epoch,
                    model=self,
                    data=data, # Mandamos o grafo inteiro, o _predict_step fatia para avaliar
                    train_mask=data.train_mask,
                    eval_mask=data.val_mask
                )
                
                scheduler.step(report[scheduler_metric_name] if scheduler_metric_name else avg_train_loss)

                training_history.append(
                    {
                        "epoch": epoch,
                        "Time_per_epoch": epoch_timer.duration,
                        "train_total_loss": avg_train_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "early_stopping_report": report,
                    }
                )
                gc.collect()
                pbar.set_postfix({"loss": f"{avg_train_loss:.4f}"})

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

    def encode(self, batch: Data) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the encode method.")

    def compute_total_loss(self, z: torch.Tensor, batch: Data):
        """Agora recebe o batch inteiro em vez de apenas data e edge_index"""
        raise NotImplementedError("Subclasses must implement the compute_total_loss method.")

    def _predict_step(self, data: Data) -> torch.Tensor:
        """
        🔥 Inferência Segura: Usa o NeighborLoader para processar o grafo
        em lotes focados nos nós, evitando estouro de VRAM na extração final de Z.
        """
        device = self.device
        loader = NeighborLoader(
            data,
            num_neighbors=self.config.GAE_NUM_NEIGHBORS,
            batch_size=self.config.INFERENCE_BATCH_SIZE,
            shuffle=False,
        )

        all_z = []
        for batch in loader:
            batch = batch.to(device)
            # O PyG insere o tamanho original do lote (target nodes) em batch.batch_size
            batch_size = batch.batch_size 
            z = self.encode(batch)

            # Guardamos os embeddings latentes apenas dos nós originais do batch
            # e os movemos para a RAM (CPU) imediatamente
            all_z.append(z[:batch_size].cpu())

        return torch.cat(all_z, dim=0)

    def evaluate(self, input_data: Data) -> Any:
        return self.inference(input_data)