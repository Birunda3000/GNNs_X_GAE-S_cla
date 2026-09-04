import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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

    def inference(self, input_data):
        """Template centralizado para inferência segura em PyTorch."""
        input_data = input_data.to(self.device)
        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self._predict_step(input_data)
        finally:
            if training_was:
                self.train()

    def _predict_step(self, data):
        """Deve ser implementado pelas classes filhas."""
        raise NotImplementedError

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """Calcula todas as métricas padrão a partir de tensores na CPU/GPU."""
        y_true_cpu = y_true.cpu()
        y_pred_cpu = y_pred.cpu()
        
        acc = float(accuracy_score(y_true_cpu, y_pred_cpu))
        f1 = float(f1_score(y_true_cpu, y_pred_cpu, average="weighted"))
        rep = classification_report(y_true_cpu, y_pred_cpu, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_cpu, y_pred_cpu)
        
        return acc, f1, rep, cm
