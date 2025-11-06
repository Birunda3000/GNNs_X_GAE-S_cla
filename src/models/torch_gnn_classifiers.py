import src.models.base_model as basemodel
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, Any
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import Config








class PyTorchClassifier(basemodel.BaseModel, nn.Module):
    """
    Classe base para classificadores PyTorch. Contém o loop de treino completo.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        BaseClassifier.__init__(self, config)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _train_step(self, optimizer, criterion, data, use_gnn):
        self.train()
        optimizer.zero_grad()

        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)

        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        return train_loss.item()

    @torch.no_grad()
    def evaluate(self, x, y, use_gnn, train_or_test_mask, edge_index = None):
        self.eval()

        if use_gnn and edge_index is not None:
            args = [x, edge_index]
        else:
            print("[INFO]: Usando classificador sem informações de arestas.")
            args = [x]
        
        out = self(*args)
        pred = out.argmax(dim=1)

        y_true = y[train_or_test_mask]
        y_pred = pred[train_or_test_mask]

        acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average="weighted")
        report = classification_report(
            y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
        )

        return acc, f1, report


    def train_model(self,
                    data: Data,
                    use_gnn: bool,
                    epochs: int,
                    optimizer: optim.Optimizer):

        training_history = []

        print(f"\n--- Avaliando (PyTorch): {self.model_name} ---")
        device = torch.device(self.config.DEVICE)
        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )

        start_time = time.process_time()
        
        for epoch in pbar:
            train_loss = self._train_step(optimizer, criterion, data, use_gnn)

            train_acc, train_f1, _ = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.train_mask)

            test_acc, test_f1, _ = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.test_mask)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,

                "test_acc": test_acc,
                "test_f1": test_f1,

            }
            training_history.append(epoch_metrics)

            pbar.set_postfix(
                {"train_loss": f"{train_loss:.4f}"},
                {"test_f1": f"{test_f1:.4f}"},
            )


        train_time = time.process_time() - start_time# CUIDADO: time.process_time() USADO

        test_report = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.test_mask)
        train_report = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.train_mask)

        report = {
            "total_training_time": train_time,
            "test_report": test_report,
            "train_report": train_report,
            "training_history": training_history,
        }

        return test_acc, test_f1, train_time, report