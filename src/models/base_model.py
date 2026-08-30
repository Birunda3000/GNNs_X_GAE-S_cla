import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch_geometric.data import Data
from typing import Optional, Any
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

    if isinstance(activation, type):
        activation = activation()

    mapping = {
        nn.ReLU:      F.relu,
        nn.ELU:       F.elu,
        nn.LeakyReLU: F.leaky_relu,
        nn.GELU:      F.gelu,
        nn.Tanh:      torch.tanh,
        nn.SiLU:      F.silu,
        nn.Mish:      F.mish,
    }

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

    # 🔥 FASE 3: Injetor de Compilação JIT Dinâmico LIMPO
    def compile_methods(self, method_names: list, dynamic: bool = True):
        """Aplica o compilador JIT de forma limpa, sem supressão de erros."""
        if not getattr(self, "_is_compiled", False):
            import torch
            
            self._compiled_method_names = []
            print(f"\n⚡ [JIT] Compilando {self.model_name} (Métodos: {method_names} | dynamic={dynamic})...")
            
            for name in method_names:
                if hasattr(self, name):
                    original_method = getattr(self, name)
                    setattr(self, name, torch.compile(original_method, dynamic=dynamic))
                    self._compiled_method_names.append(name)
            self._is_compiled = True

    def decompile_methods(self):
        """Remove a armadura do JIT, restaurando os métodos originais para permitir o salvamento (Pickle)."""
        if getattr(self, "_is_compiled", False):
            for name in getattr(self, "_compiled_method_names", []):
                if name in self.__dict__:
                    del self.__dict__[name] # Remove o wrapper compilado da instância
            self._is_compiled = False
            self._compiled_method_names = []