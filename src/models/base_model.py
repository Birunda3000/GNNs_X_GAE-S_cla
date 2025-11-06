
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from typing import Dict, Optional, Any, Tuple
from src.config import Config
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import cast, Dict, Any
import torch
import time


class BaseModel(ABC):
    """Classe base abstrata para modelos de ML."""

    @abstractmethod
    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.__class__.__name__

    @abstractmethod
    def verify_train_input_data(self, data: Data):
        """Verifica se os dados de entrada para treino estão corretos."""
        pass

    @abstractmethod
    def train_model(self, data, train_split: Optional[Any] = None) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(self, x, y: Optional[Any] = None) -> Any:
        """Evaluate the model."""
        pass

    @abstractmethod
    def inference(self, x):
        """Run inference with the model."""
        pass






















































































class SklearnClassifier(BaseModel):
    """Wrapper para modelos Scikit-learn que contém sua própria lógica de treino."""

    def __init__(self, config: Config, model_class, **model_params):
        super().__init__(config)
        self.model_name = model_class.__name__
        try:
            self.model = model_class(random_state=config.RANDOM_SEED, **model_params)
        except TypeError:
            self.model = model_class(**model_params)

    def train_model(
        self,
        data: Data,
    ):
        """Treina o modelo Scikit-learn com os dados fornecidos."""
        
        assert isinstance(data.x, torch.Tensor), f"Esperado torch.Tensor, obtido {type(data.x)}"
        assert isinstance(data.y, torch.Tensor), f"Esperado torch.Tensor, obtido {type(data.y)}"

        X = data.x.cpu().numpy()
        y = data.y.cpu().numpy()

        X_train, y_train = X[data.train_mask], y[data.train_mask]
        X_test, y_test = X[data.test_mask], y[data.test_mask]

        start_time = time.process_time()
        self.model.fit(X_train, y_train)
        train_time = time.process_time() - start_time

        y_pred_test = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)

        test_acc = float(accuracy_score(y_test, y_pred_test))
        test_f1 = float(f1_score(y_test, y_pred_test, average="weighted"))

        report = {}
        report["test_report"] = cast(
            Dict[str, Any],
            classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
        )
        report["train_report"] = cast(
            Dict[str, Any],
            classification_report(y_train, y_pred_train, output_dict=True, zero_division=0),
        )

        return float(test_acc), float(test_f1), float(train_time), report

    def evaluate(self, input_data, y: Optional[Any] = None) -> Any:
        """Evaluate the model."""
        NotImplementedError("Use train_and_evaluate para modelos Sklearn.")

    def inference(self, input_data):
        """Run inference with the model."""
        NotImplementedError("Use train_and_evaluate para modelos Sklearn.")
