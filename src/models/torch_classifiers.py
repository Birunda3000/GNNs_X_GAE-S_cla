import torch
import torch.nn as nn
import src.models.base_model as basemodel
import time
from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import Config
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


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
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Método forward() deve ser implementado na subclasse.")

    def _train_step(self, optimizer, criterion, use_gnn, x, y, edge_index=None, train_mask=None):
        self.train()
        optimizer.zero_grad()

        if use_gnn and edge_index is not None:
             args = [x, edge_index]
        else:
            print("[INFO]: Usando classificador sem informações de arestas.")
            args = [x]

        out = self(*args)

        train_loss = criterion(out[train_mask], y[train_mask])
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
    
    def verify_train_input_data(self, data: Data):
        assert data.x is not None, "Os dados de entrada devem conter atributos de nó (data.x)."
        assert data.y is not None, "Os dados de entrada devem conter rótulos de nó (data.y)."
        assert data.train_mask is not None, "Os dados de entrada devem conter uma máscara de treino (data.train_mask)."
        assert data.test_mask is not None, "Os dados de entrada devem conter uma máscara de teste (data.test_mask)."

    def internal_train_model(self, data: Data, use_gnn: bool, epochs: int, optimizer: optim.Optimizer, criterion=nn.CrossEntropyLoss()):
        self.verify_train_input_data(data)

        training_history = []

        print(f"\n--- Avaliando (PyTorch): {self.model_name} ---")
        device = torch.device(self.config.DEVICE)
        self.to(device)

        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )

        start_time = time.process_time()
        
        for epoch in pbar:
            train_loss = self._train_step(optimizer, criterion, use_gnn, x=data.x, y=data.y, edge_index=data.edge_index, train_mask=data.train_mask)

            train_acc, train_f1, _ = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.train_mask, edge_index=getattr(data, "edge_index", None))

            test_acc, test_f1, _ = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.test_mask, edge_index=getattr(data, "edge_index", None))

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
                {"train_loss": f"{train_loss:.4f}", "test_f1": f"{test_f1:.4f}"}
            )


        train_time = time.process_time() - start_time# CUIDADO: time.process_time() USADO

        _,_,test_report = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.test_mask, edge_index=getattr(data, "edge_index", None))
        _,_,train_report = self.evaluate(x=data.x, y=data.y, use_gnn=use_gnn, train_or_test_mask=data.train_mask, edge_index=getattr(data, "edge_index", None))

        report = {
            "total_training_time": train_time,
            "test_report": test_report,
            "train_report": train_report,
            "training_history": training_history,
        }

        return test_acc, test_f1, train_time, report
    
    def inference(self, x):
        NotImplementedError("Método inference() não implementado.")


# --- Implementações Específicas ---


class MLPClassifier(PyTorchClassifier):
    """Classificador MLP que opera em um tensor de features denso."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

    def train_model(self, data: Data):

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        return self.internal_train_model(
            data,
            use_gnn=False,
            epochs=self.config.EPOCHS,
            optimizer=optimizer,
            criterion=criterion
        )


class GCNClassifier(PyTorchClassifier):
    """Classificador GCN que opera em features e na estrutura do grafo."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
    
    def verify_train_input_data(self, data: Data):
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index (data.edge_index)."
        super().verify_train_input_data(data)

    def train_model(self, data: Data):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        return self.internal_train_model(
            data,
            use_gnn=True,
            epochs=self.config.EPOCHS,
            optimizer=optimizer,
            criterion=criterion
        )


class GATClassifier(PyTorchClassifier):
    """Classificador GAT que utiliza mecanismos de atenção."""

    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)
    
    def verify_train_input_data(self, data: Data):
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index (data.edge_index)."
        super().verify_train_input_data(data)

    def train_model(self, data: Data):

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        return self.internal_train_model(
            data,
            use_gnn=True,
            epochs=self.config.EPOCHS,
            optimizer=optimizer,
            criterion=criterion
        )