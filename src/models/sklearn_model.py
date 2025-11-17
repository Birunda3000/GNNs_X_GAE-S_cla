import time
from typing import Any, Dict, Tuple, cast
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.config import Config
from src.models.base_model import BaseModel
import torch
from torch_geometric.data import Data


class SklearnClassifier(BaseModel):
    """Wrapper para modelos Scikit-learn que contém sua própria lógica de treino."""

    def __init__(self, config: Config, model_class, **model_params):
        super().__init__(config)
        self.model_name = model_class.__name__
        try:
            self.model = model_class(random_state=config.RANDOM_SEED, **model_params)
        except TypeError:
            self.model = model_class(**model_params)

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
            data.test_mask is not None
        ), "Os dados de entrada devem conter uma máscara de teste (data.test_mask)."

    def train_model(self, data: Data) -> Dict[str, Any]:
        print(f"\n--- Avaliando (Sklearn): {self.model_name} ---")

        assert isinstance(
            data.x, torch.Tensor
        ), f"Esperado torch.Tensor, obtido {type(data.x)}"
        assert isinstance(
            data.y, torch.Tensor
        ), f"Esperado torch.Tensor, obtido {type(data.y)}"

        X = data.x.cpu().numpy()
        y = data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto data
        X_train, y_train = X[data.train_mask], y[data.train_mask]
        X_test, y_test = X[data.test_mask], y[data.test_mask]

        # ✅ Se houver GridSearch, usar val_mask
        # Para modelos sklearn simples sem GridSearch, usamos test_mask
        # mas renomeamos as métricas para deixar claro

        start_time = time.process_time()
        self.model.fit(X_train, y_train)
        train_time = time.process_time() - start_time

        y_test_pred = self.model.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))

        train_report = cast(
            Dict[str, Any],
            classification_report(
                y_train, self.model.predict(X_train), output_dict=True, zero_division=0
            ),
        )

        test_report = cast(
            Dict[str, Any],
            classification_report(
                y_test, y_test_pred, output_dict=True, zero_division=0
            ),
        )

        return {
            "total_training_time": train_time,
            "best_test_accuracy": test_acc,
            "best_test_f1": test_f1,
            "test_report": test_report,
            "train_report": train_report,
        }

    def evaluate(self, data: Data) -> None:
        NotImplementedError(
            "Método 'evaluate' não implementado para SklearnClassifier."
        )

    def inference(self, x):
        NotImplementedError(
            "Método 'inference' não implementado para SklearnClassifier."
        )
