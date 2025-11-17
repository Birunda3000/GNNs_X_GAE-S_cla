
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
