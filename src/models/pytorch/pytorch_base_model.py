import torch
import torch.nn as nn
from src.models.base_model import BaseModel
from src.config import Config


class PyTorchBaseModel(BaseModel, nn.Module):
    """
    Classe base para todos os modelos PyTorch do projeto.
    Unifica a herança do nn.Module, inicialização do BaseModel e o setup de hardware.
    """

    def __init__(self, config: Config):
        # Inicializa explicitamente ambas as classes pai
        BaseModel.__init__(self, config)
        nn.Module.__init__(self)

        # Centraliza a alocação de device para qualquer modelo PyTorch
        self.device = torch.device(self.config.DEVICE)
        self.to(self.device)
