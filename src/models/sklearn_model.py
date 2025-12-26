import time
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch_geometric.data import Data

from src.config import Config
from src.models.base_model import BaseModel


class SklearnClassifier(BaseModel):
    """
    Wrapper genérico para modelos compatíveis com a interface Scikit-Learn.

    Suporta nativamente:
    - sklearn.linear_model.LogisticRegression
    - sklearn.neighbors.KNeighborsClassifier
    - sklearn.ensemble.RandomForestClassifier
    - sklearn.svm.SVC
    - sklearn.naive_bayes.GaussianNB
    - sklearn.neural_network.MLPClassifier
    - xgboost.XGBClassifier (interface sklearn)

    Exemplo de uso:
        model = SklearnClassifier(config, model_class=LogisticRegression, max_iter=1000)
        model = SklearnClassifier(config, model_class=XGBClassifier, n_estimators=100)
    """

    def __init__(self, config: Config, model_class, **model_params):
        super().__init__(config)
        self.model_class = model_class
        self.model_name = model_class.__name__
        self.model_params = model_params

        # Tenta passar random_state se o modelo suportar
        try:
            self.model = model_class(random_state=config.RANDOM_SEED, **model_params)
        except TypeError:
            # Alguns modelos (ex: KNeighborsClassifier, GaussianNB) não têm random_state
            self.model = model_class(**model_params)

        # Flag para identificar se é XGBoost (tem comportamento especial no fit)
        self._is_xgboost = (
            "XGB" in self.model_name or "xgboost" in str(model_class).lower()
        )

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

    def train_model(self, data: Data) -> Dict[str, Any]:
        print(f"\n--- Avaliando (Sklearn): {self.model_name} ---")
        self.verify_train_input_data(data)

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
        X_val, y_val = X[data.val_mask], y[data.val_mask]
        X_test, y_test = X[data.test_mask], y[data.test_mask]

        start_time = time.perf_counter()

        # Fit com suporte especial para XGBoost (eval_set para early stopping)
        if self._is_xgboost:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

        train_time = time.perf_counter() - start_time

        # Avaliar em validação (para early stopping/seleção)
        y_val_pred = self.model.predict(X_val)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_f1 = float(f1_score(y_val, y_val_pred, average="weighted"))

        # Avaliar em teste (para relatório final)
        y_test_pred = self.model.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))

        # Relatórios completos
        train_report = cast(
            Dict[str, Any],
            classification_report(
                y_train, self.model.predict(X_train), output_dict=True, zero_division=0
            ),
        )
        val_report = cast(
            Dict[str, Any],
            classification_report(y_val, y_val_pred, output_dict=True, zero_division=0),
        )
        test_report = cast(
            Dict[str, Any],
            classification_report(
                y_test, y_test_pred, output_dict=True, zero_division=0
            ),
        )

        return {
            "total_training_time": train_time,

            "test_accuracy": test_acc,
            "test_f1": test_f1,

            "val_accuracy": val_acc,
            "val_f1": val_f1,

            "train_report": train_report,
            "val_report": val_report,
            "test_report": test_report,
        }

    def evaluate(self, data: Data) -> Dict[str, Any]:
        """Avalia o modelo em um conjunto de dados."""
        if not hasattr(self.model, "predict"):
            raise RuntimeError("Modelo não foi treinado. Chame 'train_model' primeiro.")

        X = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else data.x
        y = data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y

        y_pred = self.model.predict(X)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
        }

    def inference(self, x) -> np.ndarray:
        """Executa inferência no modelo treinado."""
        if not hasattr(self.model, "predict"):
            raise RuntimeError("Modelo não foi treinado. Chame 'train_model' primeiro.")

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        return self.model.predict(x)

    def predict_proba(self, x) -> Optional[np.ndarray]:
        """Retorna probabilidades se o modelo suportar."""
        if not hasattr(self.model, "predict_proba"):
            return None

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        return self.model.predict_proba(x)
