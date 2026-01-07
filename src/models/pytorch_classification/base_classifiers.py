import torch
import torch.nn as nn
import src.models.base_model as basemodel
import time
from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import Config
from typing import List, Dict, Any, Optional, Tuple, cast
from src.early_stopper import EarlyStopper
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PyTorchClassifier(basemodel.BaseModel, nn.Module):
    """
    Classe base para classificadores PyTorch. Contém o loop de treino completo.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        basemodel.BaseModel.__init__(self, config)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.model_name = self.__class__.__name__
        self.device = torch.device(self.config.DEVICE)
        self.to(self.device)

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

        # Garante que tensores e máscaras estão no mesmo device do modelo
        device = self.device
        x = data.x.to(device)
        y = data.y.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)
        test_mask = data.test_mask.to(device)

        training_history: List[Dict[str, Any]] = []
        stop_now: bool = False
        best_epoch: Optional[int] = None

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )

        start_time = time.perf_counter()

        for epoch in pbar:
            train_loss = self._train_step(
                optimizer,
                criterion,
                use_gnn,
                x=x,
                y=y,
                edge_index=edge_index,
                train_mask=train_mask,
            )

            train_acc, train_f1, _, _ = self.evaluate(
                x=x,
                y=y,
                use_gnn=use_gnn,
                train_or_test_mask=train_mask,
                edge_index=edge_index,
            )

            val_acc, val_f1, _, _ = self.evaluate(
                x=x,
                y=y,
                use_gnn=use_gnn,
                train_or_test_mask=val_mask,
                edge_index=edge_index,
            )

            stop_now, f1, best_epoch, _ = early_stopper.check(
                self,
                epoch=epoch,
                current_value=val_f1,
            )
            scheduler.step(f1)

            training_history.append(
                {
                    "epoch": epoch,
                    "train_f1": train_f1,
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                    "val_f1": val_f1,
                    "val_accuracy": val_acc,
                    "Time_per_epoch": time.perf_counter() - start_time,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

            pbar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "val_f1": f"{val_f1:.4f}"}
            )

            if early_stopper is not None and stop_now:
                print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                early_stopper.restore_best_state(self)
                break


        train_acc, train_f1, train_report, train_confusion_matrix = self.evaluate(
            x=x,
            y=y,
            use_gnn=use_gnn,
            train_or_test_mask=train_mask,
            edge_index=edge_index,
        )
        val_acc, val_f1, val_report, val_confusion_matrix = self.evaluate(
            x=x,
            y=y,
            use_gnn=use_gnn,
            train_or_test_mask=val_mask,
            edge_index=edge_index,
        )
        test_acc, test_f1, test_report, test_confusion_matrix = self.evaluate(
            x=x,
            y=y,
            use_gnn=use_gnn,
            train_or_test_mask=test_mask,
            edge_index=edge_index,
        )

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
