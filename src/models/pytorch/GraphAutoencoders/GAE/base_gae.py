from src.models.pytorch.GraphAutoencoders.base_gae_common import BaseGAECommon

class BaseGAE(BaseGAECommon):
    """
    Base para Graph Autoencoders (GAE) tradicionais.
    """
    def compute_total_loss(self, z, data, edge_index):
        # Utiliza apenas a perda de reconstrução
        return self.reconstruction_loss(z, edge_index)