import torch
from src.models.pytorch.GraphAutoencoders.base_gae_common import BaseGAECommon

class BaseVGAE(BaseGAECommon):
    """
    Base para Variational Graph Autoencoders (VGAE).
    Introduz a Divergência KL e a amostragem latente.
    """

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

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """
        Aplica o Reparameterization Trick para permitir a retropropagação.
        z = mu + epsilon * exp(logstd)
        """
        # Se o modelo estiver treinando, injeta o ruído aleatório
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(logstd)
        # Se o modelo estiver em inferência/avaliação, retorna apenas a média (determinístico)
        else:
            return mu
