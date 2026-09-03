"""GraphDataModule: dataset + split masks + loaders, decoupled from the model.

Split masking is produced upstream in ``src.data_converters`` (the converters
already attach ``train/val/test_mask``). This class only consumes the finished
``Data`` object and yields mini-batch loaders.
"""

from torch_geometric.data import Data

try:
    from torch_geometric.loader.cugraph import CuGraphNeighborLoader
except ImportError:
    # Fallback seguro caso a imagem NGC precise de ajustes finos
    from torch_geometric.loader import NeighborLoader as CuGraphNeighborLoader


class GraphDataModule:
    """Owns the PyG ``Data`` and yields ``CuGraphNeighborLoader`` instances."""

    def __init__(
        self,
        data: Data,
        *,
        num_layers: int,
        batch_size: int = 1024,
        num_neighbors_per_layer: int = 15,
    ):
        self.data = data
        self.num_layers = num_layers
        self.batch_size = batch_size
        # [15] * 0 == [] -> loader sem vizinhos (caminho MLP)
        self.neighbors = [num_neighbors_per_layer] * num_layers
        self._prepared = False

    def prepare(self, device) -> None:
        """Valida os tensores de entrada e move o grafo completo para ``device``."""
        if self._prepared:
            return

        data = self.data
        assert data.y is not None, "Os dados de entrada devem conter rótulos (data.y)."
        assert data.train_mask is not None, "Os dados de entrada devem conter data.train_mask."
        assert data.val_mask is not None, "Os dados de entrada devem conter data.val_mask."
        assert data.test_mask is not None, "Os dados de entrada devem conter data.test_mask."
        assert (
            getattr(data, "x", None) is not None
            or getattr(data, "feature_indices", None) is not None
        ), "Os dados de entrada devem conter features (data.x ou data.feature_indices)."

        self.data = data.to(device)
        self._prepared = True

    def train_dataloader(self):
        """Mini-batch neighbor sampler sobre os nós de treino."""
        print("\n🚀 Inicializando CuGraphNeighborLoader (Amostragem em UVM)...")
        return CuGraphNeighborLoader(
            self.data,
            num_neighbors=self.neighbors,
            batch_size=self.batch_size,
            input_nodes=self.data.train_mask,
            shuffle=True,
        )

    def full_batch(self) -> Data:
        """Grafo completo no device preparado, para avaliação."""
        return self.data
