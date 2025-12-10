import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch_geometric.data import Data
from typing import Dict, Optional, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import cast
from src.config import Config


# ======================================================
# 🔥 Função Global: get_activation_fn (atualizada)
# ======================================================

def get_activation_fn(activation):
    """
    Converte uma classe OU instância de ativação PyTorch em uma função funcional.

    Suporta:
      - nn.ReLU, nn.ELU, nn.LeakyReLU, nn.GELU, nn.Tanh
      - nn.SiLU (Swish)
      - nn.Mish
      - instâncias: nn.ReLU(), nn.GELU(), etc.
      - fallback seguro para ReLU
    """

    # Caso venha a classe, instanciar
    if isinstance(activation, type):
        activation = activation()

    # Mapeamento instância -> função funcional
    mapping = {
        nn.ReLU:      F.relu,
        nn.ELU:       F.elu,
        nn.LeakyReLU: F.leaky_relu,
        nn.GELU:      F.gelu,
        nn.Tanh:      torch.tanh,
        nn.SiLU:      F.silu,   # ✔️ Adicionado
        nn.Mish:      F.mish,   # ✔️ Adicionado
    }

    # Encontrar função funcional correspondente
    for cls, fn in mapping.items():
        if isinstance(activation, cls):
            return fn

    print(f"[WARNING] Activation {type(activation)} not recognized -> Using ReLU.")
    return F.relu


# ======================================================
# 🔥 Classe Base
# ======================================================

class BaseModel(ABC):
    """Classe base abstrata para modelos."""

    @abstractmethod
    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.__class__.__name__

    @abstractmethod
    def verify_train_input_data(self, data: Data):
        pass

    @abstractmethod
    def train_model(self, data, train_split: Optional[Any] = None) -> Any:
        pass
    
    @abstractmethod
    def evaluate(self, x, y: Optional[Any] = None) -> Any:
        pass

    @abstractmethod
    def inference(self, x):
        pass