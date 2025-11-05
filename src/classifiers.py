"""
Importa as bibliotecas necessárias para a construção de modelos de classificação. Para os embeddings.
"""
# Standard library imports
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import xgboost as xgb

# Local imports
from src.config import Config



class BaseClassifier(ABC):
    """
    Classe base abstrata. Define que todo classificador deve saber
    como treinar e se avaliar a partir de um objeto WSG.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.__class__.__name__

    @abstractmethod
    def train_and_evaluate(
        self, data: Data
    ) -> Tuple[float, float, float, Dict]:  # <-- MUDANÇA 1: Assinatura
        """
        Orquestra o processo de treinamento e avaliação para o modelo.
        Recebe dados já processados do PyTorch Geometric.
        Deve retornar: (acurácia, f1_score, tempo_de_treino, relatório_detalhado)
        """
        pass


class SklearnClassifier(BaseClassifier):
    """Wrapper para modelos Scikit-learn que contém sua própria lógica de treino."""

    def __init__(self, config: Config, model_class, **model_params):
        super().__init__(config)
        self.model_name = model_class.__name__
        try:
            self.model = model_class(random_state=config.RANDOM_SEED, **model_params)
        except TypeError:
            self.model = model_class(**model_params)

    def train_and_evaluate(
        self, data: Data
    ) -> Tuple[float, float, float, Dict]:  # <-- MUDANÇA 2: Assinatura
        print(f"\n--- Avaliando (Sklearn): {self.model_name} ---")

        # --- REMOVIDO ---
        # pyg_data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False)
        # ---

        # Usa 'data' que veio como argumento
        pyg_data = data
        X = pyg_data.x.cpu().numpy()
        y = pyg_data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto pyg_data
        X_train, y_train = X[pyg_data.train_mask], y[pyg_data.train_mask]
        X_test, y_test = X[pyg_data.test_mask], y[pyg_data.test_mask]

        start_time = time.process_time()
        self.model.fit(X_train, y_train)
        train_time = time.process_time() - start_time

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        return acc, f1, train_time, report


class PyTorchClassifier(BaseClassifier, nn.Module):
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

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test_step(self, data, use_gnn):
        self.eval()
        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]

        acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average="weighted")
        report = classification_report(
            y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
        )

        return acc, f1, report

    def _train_and_evaluate_internal(
        self, data: Data, use_gnn: bool
    ):  # <-- MUDANÇA 3: Assinatura
        print(f"\n--- Avaliando (PyTorch): {self.model_name} ---")
        device = torch.device(self.config.DEVICE)
        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        start_time = time.process_time()

        pbar = tqdm(
            range(self.config.EPOCHS),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )
        for epoch in pbar:
            loss = self._train_step(optimizer, criterion, data, use_gnn)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        train_time = time.process_time() - start_time

        acc, f1, report = self._test_step(data, use_gnn)
        return acc, f1, train_time, report


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

    def train_and_evaluate(self, data: Data): 
        return self._train_and_evaluate_internal(data, use_gnn=False)


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

    def train_and_evaluate(self, data: Data): 
        return self._train_and_evaluate_internal(data, use_gnn=True)


class GATClassifier(PyTorchClassifier):
    """Classificador GAT que utiliza mecanismos de atenção."""

    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=8):
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

    def train_and_evaluate(self, data: Data):  # <-- MUDANÇA 6: Assinatura
        return self._train_and_evaluate_internal(data, use_gnn=True)


class XGBoostClassifier(BaseClassifier):
    """
    Implementação de um classificador XGBoost robusto com busca de hiperparâmetros.
    XGBoost geralmente oferece alto desempenho, mas pode levar mais tempo para treinar.
    """

    def __init__(self, config: Config, num_boost_round=100, **model_params):
        super().__init__(config)
        self.model_name = "XGBoostClassifier"


        # Parâmetros padrão otimizados para classificação
        self.params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",  # Método rápido para grandes datasets
            "random_state": config.RANDOM_SEED,
        }
        # Substitui ou adiciona parâmetros personalizados
        self.params.update(model_params)
        self.num_boost_round = num_boost_round
        self.model = None

    def train_and_evaluate(self, data: Data):  # <-- MUDANÇA 7: Assinatura
        print(f"\n--- Avaliando (XGBoost): {self.model_name} ---")
        print(
            "Este modelo pode levar mais tempo para treinar, mas geralmente oferece excelente desempenho."
        )

        # --- REMOVIDO ---
        # pyg_data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False)
        # ---

        # Usa 'data' que veio como argumento
        pyg_data = data
        X = pyg_data.x.cpu().numpy()
        y = pyg_data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto pyg_data
        X_train, y_train = X[pyg_data.train_mask], y[pyg_data.train_mask]
        X_test, y_test = X[pyg_data.test_mask], y[pyg_data.test_mask]

        # Obter o número de classes
        num_classes = len(set(y))
        self.params["num_class"] = num_classes

        # Preparar dados no formato DMatrix do XGBoost para maior eficiência
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Medir o tempo de treinamento
        start_time = time.process_time()

        print(f"Treinando XGBoost por {self.num_boost_round} rounds...")
        # Treinar com feedback de progresso
        eval_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            self.num_boost_round,
            evals=[(dtrain, "train"), (dtest, "eval")],
            evals_result=eval_result,
            verbose_eval=10,  # Mostrar progresso a cada 10 rounds
        )

        train_time = time.process_time() - start_time

        # Fazer predições
        y_pred_probs = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        # Apresentar algumas informações sobre características importantes
        print("\nFeature Importances:")
        feature_importance = self.model.get_score(importance_type="weight")
        sorted_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_importance[:5]:  # Top 5 features
            print(f"  Feature {feature}: {importance}")

        return acc, f1, train_time, report



def salvar_modelo_pytorch(model, dataset_name: str, timestamp: str, save_dir: str = "models") -> Tuple[str, str]:
    """
    Salva qualquer modelo PyTorch com metadados estruturais e pesos.
    Inclui:
    - Arquitetura (camadas, parâmetros customizados)
    - Tipo do modelo
    - Pesos (state_dict)
    - Timestamp e dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    model_name = getattr(model, "model_name", model.__class__.__name__)
    base_name = f"{dataset_name}__{model_name}__{timestamp}"

    weights_path = os.path.join(save_dir, f"{base_name}.pth")
    meta_path = os.path.join(save_dir, f"{base_name}_meta.json")

    # --- 1. Salvar pesos ---
    torch.save(model.state_dict(), weights_path)

    # --- 2. Extrair metadados da arquitetura ---
    meta = {
        "model_type": model.__class__.__name__,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "parameters": {
            k: v
            for k, v in model.__dict__.items()
            if isinstance(v, (int, float, str, bool, list, tuple, dict))
        },
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"✅ Modelo salvo em: {weights_path}")
    print(f"🧩 Metadados em: {meta_path}")
    return weights_path, meta_path


def carregar_modelo_pytorch(meta_path: str, weights_path: str, available_models: dict, device: str = "cpu"):
    """
    Carrega um modelo PyTorch salvo com salvar_modelo_pytorch().
    
    Args:
        meta_path (str): Caminho para o arquivo JSON com os metadados.
        weights_path (str): Caminho para o arquivo .pth com os pesos.
        available_models (dict): Dicionário com classes de modelos disponíveis, ex:
            {
                "MLPClassifier": MLPClassifier,
                "GCNClassifier": GCNClassifier,
                "GATClassifier": GATClassifier,
                ...
            }
        device (str): 'cpu' ou 'cuda'.

    Returns:
        model (nn.Module): Modelo PyTorch reconstruído com pesos carregados.
        meta (dict): Metadados do modelo.
    """
    # --- 1. Ler os metadados ---
    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_type = meta["model_type"]
    params = meta.get("parameters", {})

    # --- 2. Verificar se a classe do modelo está disponível ---
    if model_type not in available_models:
        raise ValueError(f"Modelo '{model_type}' não encontrado em available_models. Forneça a classe correspondente.")

    model_class = available_models[model_type]

    # --- 3. Recriar a instância com os mesmos parâmetros ---
    
    model = model_class(**{k: v for k, v in params.items() if not k.startswith('_')})


    # --- 4. Carregar pesos ---
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"✅ Modelo '{model_type}' carregado com sucesso do dataset '{meta['dataset']}'.")
    return model, meta




