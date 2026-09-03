import torch
import torch.nn as nn
import torch.nn.functional as F

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

class BaseModel:
    """Base mínima para modelos.

    Após a migração para o padrão Trainer, não impõe mais o contrato de
    ``train_model``/``evaluate``/``inference`` via ABC. Modelos PyTorch são
    ``nn.Module`` puros; esta base mantém apenas ``config`` e ``model_name``
    para os wrappers que ainda a herdam (ex.: ``SklearnClassifier``).
    """

    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.__class__.__name__