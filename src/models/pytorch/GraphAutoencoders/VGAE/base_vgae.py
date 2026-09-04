import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from src.models.pytorch.GraphAutoencoders.base_gae_common import BaseGAECommon

class BaseVGAE(BaseGAECommon):
    """
    Base para Variational Graph Autoencoders (VGAE).
    Introduz a Divergência KL e a amostragem latente.
    """
    conv1: MessagePassing
    conv_mu: MessagePassing
    conv_logstd: MessagePassing

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mu__ = self.__logstd__ = None

    def kl_loss(self) -> torch.Tensor:
        if self.__mu__ is None or self.__logstd__ is None:
            return torch.tensor(0.0)
        return -0.5 * torch.mean(
            torch.sum(
                1 + 2 * self.__logstd__ - self.__mu__.pow(2) - self.__logstd__.exp().pow(2),
                dim=1,
            )
        )

    def compute_total_loss(self, z, data, edge_index):
        assert data.num_nodes is not None, "data.num_nodes must be valid."
        # Soma a reconstrução com a divergência KL
        return self.reconstruction_loss(z, edge_index) + (1.0 / float(data.num_nodes)) * self.kl_loss()

    def inference(self, input_data) -> Tensor:
        device = next(self.parameters()).device
        input_data.to(device)
        training_was = self.training
        self.eval()
        try:
            with torch.no_grad():
                self.encode(input_data)
        finally:
            if training_was:
                self.train()
        if self.__mu__ is None:
            raise RuntimeError("O atributo `__mu__` não foi definido pelo método `encode`.")
        return self.__mu__
