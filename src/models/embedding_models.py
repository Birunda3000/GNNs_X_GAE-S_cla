# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import List, Dict, Any, cast
import torch.optim as optim
import time
import src.models.base_model as basemodel
from src.config import Config



class VGAE(basemodel.BaseModel, nn.Module):
    """
    Implementação de um Autoencoder Variacional de Grafo (VGAE).

    Esta arquitetura aprende embeddings de nós de forma auto-supervisionada,
    tentando reconstruir a estrutura de adjacência do grafo.

    Ela é composta por:
    1. Uma camada de EmbeddingBag para processar as features esparsas de entrada.
    2. Um Encoder GNN (baseado em GCN) que gera uma distribuição latente (mu, log_std) para cada nó.
    3. Um Decoder baseado em produto escalar que reconstrói as arestas.
    """

    def __init__(
        self,
        config: Config,
        num_total_features: int,
        embedding_dim: int,
        hidden_dim: int,
        out_embedding_dim: int,
    ):
        """
        Inicializador do modelo VGAE.

        Args:
            num_total_features (int): Tamanho do vocabulário de features (de pyg_data.num_total_features).
            embedding_dim (int): Dimensão do embedding para cada feature do vocabulário.
            hidden_dim (int): Dimensão da camada GCN intermediária.
            out_embedding_dim (int): Dimensão final dos embeddings dos nós.
        """
        basemodel.BaseModel.__init__(self, config)
        nn.Module.__init__(self)

        # Camada de entrada: processa os índices/pesos e cria um vetor denso inicial
        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum",  # **MEAN NOT IMPLEMENTED**
        )

        # Encoder: duas camadas GCN para gerar os parâmetros da distribuição
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv_mu = GCNConv(
            hidden_dim, out_embedding_dim
        )  # Cabeça para a média (mu)
        self.conv_logstd = GCNConv(
            hidden_dim, out_embedding_dim
        )  # Cabeça para o log da variância



        # Variáveis para armazenar os parâmetros da última chamada do encode
        self.__mu__ = self.__logstd__ = None

    def encode(self, data: Data) -> torch.Tensor:
        """
        Executa a passagem de codificação.

        Args:
            data (Data): Objeto de dados do PyTorch Geometric.

        Returns:
            torch.Tensor: A matriz de embeddings Z, amostrada da distribuição latente.
        """
        # 1. Gera a matriz de features 'x' a partir das features esparsas ponderadas
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )

        x = F.normalize(x, p=2, dim=-1)

        # 2. Propaga pela GCN para obter os parâmetros da distribuição
        x = F.relu(self.conv1(x, data.edge_index))
        self.__mu__ = self.conv_mu(x, data.edge_index)
        self.__logstd__ = self.conv_logstd(x, data.edge_index)

        # 3. Amostragem (Reparameterization Trick)
        # Garante que o gradiente possa fluir através da operação de amostragem
        random_noise = torch.randn_like(self.__mu__)
        z = self.__mu__ + random_noise * torch.exp(self.__logstd__)

        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decodifica os embeddings Z para reconstruir as arestas.

        Args:
            z (torch.Tensor): Matriz de embeddings dos nós.
            edge_index (torch.Tensor): Arestas a serem pontuadas.

        Returns:
            torch.Tensor: A probabilidade (logit) de existência para cada aresta.
        """
        # O produto escalar entre os embeddings dos nós de uma aresta
        # mede sua similaridade, que usamos como logit para a existência da aresta.
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def kl_loss(self) -> torch.Tensor:
        """
        Calcula a divergência KL entre a distribuição latente aprendida e uma
        distribuição normal padrão. Atua como um regularizador.
        """
        if self.__mu__ is None or self.__logstd__ is None:
            # Se o encode não foi chamado, não há perda KL para calcular.
            # Isso pode acontecer em cenários de inferência ou se a lógica estiver separada.
            return torch.tensor(0.0)

        return -0.5 * torch.mean(
            torch.sum(
                1
                + 2 * self.__logstd__
                - self.__mu__.pow(2)
                - self.__logstd__.exp().pow(2),
                dim=1,
            )
        )

    def reconstruction_loss(
        self, z: torch.Tensor, pos_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula a loss de reconstrução. Compara as arestas reconstruídas
        com as arestas reais (positivas) e arestas amostradas (negativas).
        """
        pos_logits = self.decode(z, pos_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, z.new_ones(pos_edge_index.size(1))
        )

        # Amostragem de arestas negativas
        num_neg_samples = pos_edge_index.size(
            1
        )  # Amostra a mesma quantidade de arestas negativas
        neg_edge_index = torch.randint(
            0, z.size(0), (2, num_neg_samples), dtype=torch.long, device=z.device
        )
        neg_logits = self.decode(z, neg_edge_index)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, z.new_zeros(num_neg_samples)
        )

        return pos_loss + neg_loss

    def inference(self, input_data: Data) -> torch.Tensor:
        """
        Após o treinamento, gera os embeddings finais e estáveis dos nós.
        Usa apenas a média (mu) da distribuição, ignorando a variância.
        """
        with torch.no_grad():
            x = self.feature_embedder(
                input_data.feature_indices,
                input_data.feature_offsets,
                per_sample_weights=input_data.feature_weights,
            )

            x = F.normalize(x, p=2, dim=-1)

            x = F.relu(self.conv1(x, input_data.edge_index))
            mu = self.conv_mu(x, input_data.edge_index)
        return mu

    def verify_train_input_data(self, data: Data):
        """
        Validates that the required fields for unsupervised VGAE training are present.

        Requirements:
        - data.edge_index: adjacency (Tensor)
        - data.feature_indices / feature_offsets / feature_weights: sparse feature representation for EmbeddingBag
        - data.num_nodes: total number of nodes (> 0)

        Raises:
            AssertionError: if any required attribute is missing or invalid.
        """
        # VGAE builds x internally; train/test masks are not needed for unsupervised training.
        assert (
            data.edge_index is not None
        ), "Input data must contain edge_index (data.edge_index)."
        assert (
            data.feature_indices is not None
        ), "Input data must contain feature_indices."
        assert (
            data.feature_offsets is not None
        ), "Input data must contain feature_offsets."
        assert (
            data.feature_weights is not None
        ), "Input data must contain feature_weights."
        assert data.num_nodes is not None, "Input data must contain num_nodes."
        assert data.num_nodes > 0, "data.num_nodes must be greater than zero."

    def train_model(
        self,
        data: Data,
        optimizer: optim.Optimizer,
        epochs: int,
    ) -> Dict[str, Any]:
        """
        Treina o modelo VGAE no conjunto de dados fornecido.

        Args:
            data (Data): Objeto de dados do PyTorch Geometric.
            epochs (int): Número de épocas para treinar.

        Returns:
        Dict[str, Any]: Relatório com histórico e tempo total.
        """
        self.verify_train_input_data(data)
        edge_index = cast(torch.Tensor, data.edge_index)
        num_nodes = cast(int, data.num_nodes)

        training_history: List[Dict[str, float]] = []

        start_time = time.process_time()

        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            z = self.encode(data)
            recon_loss = self.reconstruction_loss(z, edge_index)

            kl_loss = (1.0 / float(num_nodes)) * self.kl_loss()
            total_loss = recon_loss + kl_loss
            total_loss.backward()
            optimizer.step()

            epoch_metrics = {
                "epoch": epoch,
                "train_total_loss": total_loss.item(),
                "train_recon_loss": recon_loss.item(),
                "train_kl_loss": kl_loss.item(),
                "test_total_loss": None,
                "test_recon_loss": None,
                "test_kl_loss": None,
            }
            training_history.append(epoch_metrics)

        total_time = time.process_time() - start_time
        return {
            "total_training_time": float(total_time),
            "epochs": epochs,
            "final_train_loss": total_loss.item(),
            "training_history": training_history,
        }

    def evaluate(self, data: Data) -> Any:
        """
        Avalia o modelo no conjunto de dados fornecido.

        Args:
            data (Data): Objeto de dados do PyTorch Geometric.

        Returns:
            Any: Resultados da avaliação (a definir conforme necessidade).
        """
        raise NotImplementedError("Função de avaliação não implementada para VGAE.")
