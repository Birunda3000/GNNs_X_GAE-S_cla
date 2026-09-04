import torch
import torch.nn as nn
import time
from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import Config
from typing import List, Dict, Any, Optional, Tuple, cast
from src.early_stopper import EarlyStopper
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Importe a nova classe base
from src.models.pytorch.pytorch_base_model import PyTorchBaseModel

# 🔥 FASE 2: DataLoader otimizado da NVIDIA
try:
    from torch_geometric.loader.cugraph import CuGraphNeighborLoader
except ImportError:
    # Fallback seguro caso a imagem NGC precise de ajustes finos
    from torch_geometric.loader import NeighborLoader as CuGraphNeighborLoader


class PyTorchClassifier(PyTorchBaseModel):
    """
    Classe base para classificadores PyTorch. Contém o loop de treino completo.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        # Agora chamamos apenas o super() da nova base, que já resolve o nn.Module e o device
        super().__init__(config)
        
        self.input_dim = input_dim
        self.model_name = self.__class__.__name__

    def verify_train_input_data(self, data: Data):
        assert (
            data.x is not None
        ), "Os dados de entrada devem conter atributos de nó (data.x)."
        assert (
            data.y is not None
        ), "Os dados de entrada devem conter rótulos de nó (data.y)."
        assert (
            data.train_mask is not None
        ), "Os dados de entrada devem conter uma máscara de treino (data.train_mask)."
        assert (
            data.val_mask is not None
        ), "Os dados de entrada devem conter uma máscara de validação (data.val_mask)."
        assert (
            data.test_mask is not None
        ), "Os dados de entrada devem conter uma máscara de teste (data.test_mask)."

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Método forward() deve ser implementado na subclasse."
        )

    def _train_step(
        self, optimizer, criterion, use_gnn, x, y, edge_index=None, train_mask=None
    ):
        self.train()
        optimizer.zero_grad()

        if use_gnn and edge_index is not None:
            args = [x, edge_index]
        else:
            print("[INFO]: Usando classificador sem informações de arestas.")
            args = [x]

        # AQUI ONDE "forward" É CHAMADO "out = self.forward(*args)"
        out = self(*args)

        train_loss = criterion(out[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()
        return train_loss.item()

    @torch.no_grad()
    def evaluate(
        self, x, y, use_gnn, train_or_test_mask, edge_index=None
    ) -> Tuple[float, float, Dict[str, Any], Any]:  # <- ajustar tipagem para Any
        self.eval()

        if use_gnn and edge_index is not None:
            args = [x, edge_index]
        elif not use_gnn and edge_index is not None:
            print(
                "[WARNING]: edge_index fornecido, mas use_gnn está definido como False. Ignorando edge_index."
            )
            args = [x]
        else:
            args = [x]

        out = self(*args)
        pred = out.argmax(dim=1)

        y_true = y[train_or_test_mask]
        y_pred = pred[train_or_test_mask]

        acc = float(accuracy_score(y_true.cpu(), y_pred.cpu()))
        f1 = float(f1_score(y_true.cpu(), y_pred.cpu(), average="weighted"))
        f1_macro = float(f1_score(y_true.cpu(), y_pred.cpu(), average="macro"))
        report = cast(
            Dict[str, Any],
            classification_report(
                y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
            ),
        )

        conf_mat = confusion_matrix(y_true.cpu(), y_pred.cpu())

        return acc, f1, report, conf_mat

    def train_model(
        self,
        data: Data,
        epochs: Optional[int] = None,
        early_stopper: Optional[EarlyStopper] = None,
        scheduler: Optional[ReduceLROnPlateau] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion=nn.CrossEntropyLoss(),
    ) -> Dict[str, Any]:
        """
        Método de treino padrão unificado para todos os classificadores PyTorch.
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        if scheduler is None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=self.config.SCHEDULER_PATIENCE,
                factor=self.config.SCHEDULER_FACTOR,
                min_lr=self.config.MIN_LR,
            )

        if epochs is None:
            epochs = self.config.EPOCHS

        if early_stopper is None:
            early_stopper = EarlyStopper(
                patience=self.config.EARLY_STOPPING_PATIENCE,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                mode="max",
                metric_name="val_f1",
            )

        # Assume que self.use_gnn é definido na classe filha ou default False
        use_gnn = getattr(self, "use_gnn", False)

        return self.internal_train_model(
            data,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=early_stopper,
            scheduler=scheduler,
            use_gnn=use_gnn,
            criterion=criterion,
        )

    def internal_train_model(
        self,
        data: Data,
        use_gnn: bool,
        epochs: int,
        optimizer: optim.Optimizer,
        early_stopper: EarlyStopper,
        scheduler,
        criterion=nn.CrossEntropyLoss(),
    ):
        self.verify_train_input_data(data)
        device = self.device

        # 🔥 FASE 2: Configuração do Loader de Alta Performance
        # Se for GNN, mapeamos os vizinhos usando a profundidade da rede (num_layers).
        # Se for MLP (use_gnn=False), não precisamos de vizinhos.
        num_layers = getattr(self, 'num_layers', 2)
        neighbors_sample = [15] * num_layers if use_gnn else []

        print("\n🚀 Inicializando CuGraphNeighborLoader (Amostragem em UVM)...")
        train_loader = CuGraphNeighborLoader(
            data,
            num_neighbors=neighbors_sample,
            batch_size=1024,
            input_nodes=data.train_mask,
            shuffle=True,
        )

        training_history: List[Dict[str, Any]] = []
        stop_now: bool = False
        best_epoch: Optional[int] = None

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name} (Mini-Batch)",
            leave=False,
        )

        start_time = time.perf_counter()

        # 🔥 FASE 1: Aciona a compilação JIT apenas no passe frontal
        self.compile_methods(["forward"], dynamic=True)

        for epoch in pbar:
            self.train()
            total_loss = 0.0

            # Iterando sobre os subgrafos ultra-rápidos injetados pelo RAPIDS
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                if use_gnn and hasattr(batch, 'edge_index'):
                    out = self(batch.x, batch.edge_index)
                else:
                    out = self(batch.x)

                # No mini-batch, calculamos a loss apenas para os nós alvo do batch
                # que estão nas primeiras posições (tamanho do batch_size)
                batch_size = batch.batch_size
                loss = criterion(out[:batch_size], batch.y[:batch_size])

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Para manter a avaliação ágil durante o treino, usamos o full-batch tradicional
            x_full = data.x.to(device)
            y_full = data.y.to(device)
            edge_index_full = getattr(data, "edge_index", None)
            if edge_index_full is not None:
                edge_index_full = edge_index_full.to(device)

            train_acc, train_f1, _, _ = self.evaluate(
                x=x_full, y=y_full, use_gnn=use_gnn, train_or_test_mask=data.train_mask.to(device), edge_index=edge_index_full
            )
            val_acc, val_f1, _, _ = self.evaluate(
                x=x_full, y=y_full, use_gnn=use_gnn, train_or_test_mask=data.val_mask.to(device), edge_index=edge_index_full
            )

            stop_now, f1, best_epoch, _ = early_stopper.check(self, epoch=epoch, current_value=val_f1)
            scheduler.step(f1)

            training_history.append({
                "epoch": epoch,
                "train_f1": train_f1,
                "train_accuracy": train_acc,
                "train_loss": avg_train_loss,
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "Time_per_epoch": time.perf_counter() - start_time,
                "learning_rate": scheduler.get_last_lr()[0],
            })

            pbar.set_postfix({"loss": f"{avg_train_loss:.4f}", "val_f1": f"{val_f1:.4f}"})

            if early_stopper is not None and stop_now:
                print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                early_stopper.restore_best_state(self)
                break

        # Relatórios Finais após a restauração do melhor modelo
        train_acc, train_f1, train_report, train_confusion_matrix = self.evaluate(
            x=x_full, y=y_full, use_gnn=use_gnn, train_or_test_mask=data.train_mask.to(device), edge_index=edge_index_full
        )
        val_acc, val_f1, val_report, val_confusion_matrix = self.evaluate(
            x=x_full, y=y_full, use_gnn=use_gnn, train_or_test_mask=data.val_mask.to(device), edge_index=edge_index_full
        )
        test_acc, test_f1, test_report, test_confusion_matrix = self.evaluate(
            x=x_full, y=y_full, use_gnn=use_gnn, train_or_test_mask=data.test_mask.to(device), edge_index=edge_index_full
        )

        self.decompile_methods()
        return {
            "total_training_time": time.perf_counter() - start_time,

            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_report": test_report,
            "test_confusion_matrix": test_confusion_matrix,

            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_report": val_report,
            "val_confusion_matrix": val_confusion_matrix,

            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_report": train_report,
            "train_confusion_matrix": train_confusion_matrix,

            "best_epoch": best_epoch,

            "training_history": training_history,
        }

    def inference(self, data: Data) -> torch.Tensor:
            """
            Executa a inferência no modelo (GNN ou MLP).
            Segue estritamente o padrão do modelo de Embedding:
            1. Recebe 'data'.
            2. Move para device.
            3. Configura eval/no_grad.
            4. Retorna Logits.
            """
            # Garante device
            device = self.device
            data = data.to(device)

            # Garante comportamento determinístico (desliga dropout) e restaura estado anterior
            training_was = self.training
            self.eval()

            try:
                with torch.no_grad():
                    use_gnn = getattr(self, "use_gnn", False)

                    # Seleciona argumentos baseado na arquitetura
                    if use_gnn:
                        # GNN: precisa de X e Arestas
                        out = self(data.x, data.edge_index)
                    else:
                        # MLP: precisa apenas de X
                        out = self(data.x)

            finally:
                # Restaura o modo anterior (se estava treinando, volta a treinar)
                if training_was:
                    self.train()

            return out