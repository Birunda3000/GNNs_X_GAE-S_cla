"""Trainer pattern: loop de treinamento composto, desacoplado do modelo.

O modelo deve ser um ``nn.Module`` puro (apenas propagação de tensores via
``forward``/``encode``). Tudo o mais — loop, loaders, early stopping, JIT —
vive aqui ou no ``GraphDataModule``.
"""

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils import DeviceTimer

try:
    import transformer_engine.pytorch as te

    HAS_TE = True
except ImportError:
    HAS_TE = False


class Trainer:
    """Treina um ``nn.Module`` via injeção total de dependências."""

    def __init__(
        self,
        model: nn.Module,
        datamodule,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module],
        scheduler,
        device,
    ):
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model_name = getattr(model, "model_name", model.__class__.__name__)
        self._is_compiled = False
        self._compiled_method_names: List[str] = []

    # ------------------------------------------------------------------
    # JIT helpers (movidos do BaseModel para cá — são preocupação de treino)
    # ------------------------------------------------------------------
    def _compile(self, method_names: List[str]) -> None:
        if self._is_compiled:
            return
        print(
            f"\n⚡ [JIT] Compilando {self.model_name} "
            f"(Métodos: {method_names} | dynamic=True)..."
        )
        for name in method_names:
            if hasattr(self.model, name):
                setattr(
                    self.model,
                    name,
                    torch.compile(getattr(self.model, name), dynamic=True),
                )
                self._compiled_method_names.append(name)
        self._is_compiled = True

    def _decompile(self) -> None:
        if not self._is_compiled:
            return
        for name in self._compiled_method_names:
            if name in self.model.__dict__:
                del self.model.__dict__[name]
        self._is_compiled = False
        self._compiled_method_names = []

    # ------------------------------------------------------------------
    # Forward / avaliação compartilhados pelos caminhos supervisionados
    # ------------------------------------------------------------------
    def _predict(self, data: Data) -> torch.Tensor:
        """Forward full-batch -> logits, conforme a assinatura do modelo."""
        if hasattr(data, "feature_indices"):
            # EmbeddingBag GNN: (indices, offsets, weights, edge_index)
            return self.model(
                data.feature_indices,
                data.feature_offsets,
                data.feature_weights,
                data.edge_index,
            )
        if getattr(self.model, "use_gnn", False):
            return self.model(data.x, data.edge_index)
        return self.model(data.x)

    @torch.no_grad()
    def evaluate(self, mask) -> Tuple[float, float, Dict[str, Any], Any]:
        """acc, f1 weighted, classification_report, confusion_matrix sobre ``mask``."""
        self.model.eval()
        data = self.datamodule.data
        out = self._predict(data)
        pred = out.argmax(dim=1)

        y_true = data.y[mask].cpu()
        y_pred = pred[mask].cpu()

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="weighted"))
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        return acc, f1, report, cm

    def _final_supervised_report(
        self, data, best_epoch, training_history, start_time
    ) -> Dict[str, Any]:
        """Formato de retorno esperado pelo ``ExperimentRunner`` (inalterado)."""
        train_acc, train_f1, train_rep, train_cm = self.evaluate(data.train_mask)
        val_acc, val_f1, val_rep, val_cm = self.evaluate(data.val_mask)
        test_acc, test_f1, test_rep, test_cm = self.evaluate(data.test_mask)

        return {
            "total_training_time": time.perf_counter() - start_time,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_report": test_rep,
            "test_confusion_matrix": test_cm,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_report": val_rep,
            "val_confusion_matrix": val_cm,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_report": train_rep,
            "train_confusion_matrix": train_cm,
            "best_epoch": best_epoch,
            "training_history": training_history,
        }

    # ------------------------------------------------------------------
    # Caminho supervisionado
    # ------------------------------------------------------------------
    def fit(self, epochs: int, early_stopper) -> Dict[str, Any]:
        """Loop supervisionado. Despacha para mini-batch ou full-batch.

        - ``early_stopper``: ``EarlyStopper`` (interface de 4 tuplas).
        """
        self.early_stopper = early_stopper
        self.datamodule.prepare(self.device)

        data = self.datamodule.data
        use_embedding_bag = (
            hasattr(data, "feature_indices")
            and getattr(self.model, "use_gnn", False)
        )
        if use_embedding_bag:
            return self._fit_full_batch(epochs)
        return self._fit_mini_batch(epochs)

    def _fit_mini_batch(self, epochs: int) -> Dict[str, Any]:
        model = self.model
        data = self.datamodule.data
        device = self.device
        use_gnn = getattr(model, "use_gnn", False)
        optimizer = self.optimizer
        criterion = self.criterion
        scheduler = self.scheduler
        early_stopper = self.early_stopper

        train_loader = self.datamodule.train_dataloader()

        training_history: List[Dict[str, Any]] = []
        best_epoch: Optional[int] = None

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name} (Mini-Batch)",
            leave=False,
        )
        start_time = time.perf_counter()

        self._compile(["forward"])
        try:
            for epoch in pbar:
                model.train()
                total_loss = 0.0

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    if use_gnn and hasattr(batch, "edge_index"):
                        out = model(batch.x, batch.edge_index)
                    else:
                        out = model(batch.x)

                    loss = criterion(out[: batch.batch_size], batch.y[: batch.batch_size])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

                train_acc, train_f1, _, _ = self.evaluate(data.train_mask)
                val_acc, val_f1, _, _ = self.evaluate(data.val_mask)

                stop_now, f1, best_epoch, _ = early_stopper.check(
                    model, epoch=epoch, current_value=val_f1
                )
                scheduler.step(f1)

                training_history.append(
                    {
                        "epoch": epoch,
                        "train_f1": train_f1,
                        "train_accuracy": train_acc,
                        "train_loss": avg_train_loss,
                        "val_f1": val_f1,
                        "val_accuracy": val_acc,
                        "Time_per_epoch": time.perf_counter() - start_time,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

                pbar.set_postfix(
                    {"loss": f"{avg_train_loss:.4f}", "val_f1": f"{val_f1:.4f}"}
                )

                if early_stopper is not None and stop_now:
                    print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                    early_stopper.restore_best_state(model)
                    break
        finally:
            self._decompile()

        return self._final_supervised_report(
            data, best_epoch, training_history, start_time
        )

    def _fit_full_batch(self, epochs: int) -> Dict[str, Any]:
        """Loop supervisionado full-batch para GNNs com EmbeddingBag."""
        model = self.model
        data = self.datamodule.data
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        scheduler = self.scheduler
        early_stopper = self.early_stopper

        feature_indices = data.feature_indices.to(device)
        feature_offsets = data.feature_offsets.to(device)
        feature_weights = data.feature_weights.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)
        test_mask = data.test_mask.to(device)

        training_history: List[Dict[str, Any]] = []
        best_epoch: Optional[int] = None

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name} (End-to-End)",
            leave=False,
        )
        start_time = time.perf_counter()

        self._compile(["forward"])
        try:
            for epoch in pbar:
                model.train()
                optimizer.zero_grad()

                out = model(feature_indices, feature_offsets, feature_weights, edge_index)
                train_loss = criterion(out[train_mask], y[train_mask])
                train_loss.backward()
                optimizer.step()

                train_acc, train_f1, _, _ = self.evaluate(train_mask)
                val_acc, val_f1, _, _ = self.evaluate(val_mask)

                stop_now, f1, best_epoch, _ = early_stopper.check(
                    model, epoch=epoch, current_value=val_f1
                )
                scheduler.step(f1)

                training_history.append(
                    {
                        "epoch": epoch,
                        "train_f1": train_f1,
                        "train_accuracy": train_acc,
                        "train_loss": train_loss.item(),
                        "val_f1": val_f1,
                        "val_accuracy": val_acc,
                        "Time_per_epoch": time.perf_counter() - start_time,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

                pbar.set_postfix(
                    {"loss": f"{train_loss.item():.4f}", "val_f1": f"{val_f1:.4f}"}
                )

                if stop_now:
                    print(f"[EARLY STOPPING] Parando no epoch {epoch}")
                    early_stopper.restore_best_state(model)
                    break
        finally:
            self._decompile()

        return self._final_supervised_report(
            data, best_epoch, training_history, start_time
        )

    # ------------------------------------------------------------------
    # Caminho não supervisionado (GAE / VGAE)
    # ------------------------------------------------------------------
    def fit_gae(
        self,
        epochs: int,
        early_stopper,
        scheduler_metric_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Loop full-batch não supervisionado (reconstrução de arestas).

        - ``early_stopper``: ``UniversalEarlyStopper`` (interface de 2 tuplas).
        """
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device

        self.datamodule.prepare(device)
        data = self.datamodule.data

        training_history: List[Dict[str, Any]] = []
        stop_now = False

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )
        epoch_timer = DeviceTimer(str(device), disable_gc=False)

        self._compile(["encode", "decode"])
        try:
            with DeviceTimer(str(device), disable_gc=True) as total_timer:
                for epoch in pbar:
                    with epoch_timer:
                        model.train()
                        optimizer.zero_grad()
                        edge_index = data.edge_index

                        if HAS_TE:
                            with te.fp8_autocast(enabled=True):
                                z = model.encode(data)
                                total_loss = model.compute_total_loss(z, data, edge_index)
                        else:
                            with torch.autocast(
                                device_type=device.type, dtype=torch.bfloat16
                            ):
                                z = model.encode(data)
                                total_loss = model.compute_total_loss(z, data, edge_index)

                        total_loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    stop_now, report = early_stopper.check(
                        epoch=epoch,
                        model=model,
                        data=data,
                        train_mask=data.train_mask,
                        eval_mask=data.val_mask,
                    )
                    scheduler.step(
                        report[scheduler_metric_name]
                        if scheduler_metric_name
                        else total_loss.item()
                    )

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
                        early_stopper.restore_best_state(model)
                        break
        finally:
            self._decompile()

        main_metric_name = getattr(early_stopper.metrics[0], "name", None)
        best_score = (
            early_stopper.best_values.get(main_metric_name)
            if main_metric_name is not None
            else None
        )

        return {
            "total_training_time": total_timer.duration,
            "best_epoch": early_stopper.best_epoch,
            "best_score": best_score,
            "best_scores": early_stopper.best_values,
            "training_history": training_history,
        }
