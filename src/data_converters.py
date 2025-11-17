# src/data_converter.py

from __future__ import annotations

# Standard library
from typing import List, Tuple

# Third-party
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Local
from src.data_format_definition import WSG
from src.config import Config


def wsg_to_edge_index(wsg: WSG) -> torch.Tensor:
    """Extracts and validates the edge_index (graph structure) from a WSG object.

    Supports two list formats:
    - [[src...], [dst...]] -> returns tensor shaped (2, num_edges)
    - [[src, dst], [src, dst], ...] -> returns tensor shaped (2, num_edges)
    If the list is empty or unexpected, returns an empty edge_index tensor.
    """
    edge_index_data = wsg.graph_structure.edge_index
    if isinstance(edge_index_data, list):
        if len(edge_index_data) > 0 and len(edge_index_data[0]) == len(
            edge_index_data[1]
        ):
            # Standard format: [[src...], [dst...]]
            edge_index = torch.tensor(edge_index_data, dtype=torch.long)
        elif (
            len(edge_index_data) > 0
            and isinstance(edge_index_data[0], list)
            and len(edge_index_data[0]) == 2
        ):
            # Format: [[src, dst], [src, dst], ...] -> transpose to (2, E)
            edge_index = torch.tensor(edge_index_data, dtype=torch.long).t()
        else:
            # Empty list or unexpected format -> return empty edge_index
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        raise TypeError(f"edge_index must be a list, got {type(edge_index_data)}")

    return edge_index


def wsg_to_labels(wsg: WSG) -> torch.Tensor:
    """Extracts node label vector (y) from a WSG object.

    None labels are replaced by -1 as a placeholder for unlabeled nodes.
    """
    num_nodes = wsg.metadata.num_nodes
    labels_list = wsg.graph_structure.y
    y = torch.tensor(
        [-1 if label is None else int(label) for label in labels_list],
        dtype=torch.long,
    )
    return y


def wsg_to_embeddingbag_features(
    wsg: WSG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts WSG sparse features into tensors suitable for nn.EmbeddingBag.

    Returns:
        (feature_indices, feature_weights, feature_offsets)
    """
    num_nodes = wsg.metadata.num_nodes
    # Build flattened arrays for EmbeddingBag
    all_indices: List[int] = []
    all_weights: List[float] = []
    offsets: List[int] = [0]  # first offset is always 0

    for i in range(num_nodes):
        node_id_str = str(i)
        node_feat = wsg.node_features[node_id_str]

        all_indices.extend(node_feat.indices)
        all_weights.extend(node_feat.weights)
        offsets.append(offsets[-1] + len(node_feat.indices))

    # Remove the last offset (it is the total length and EmbeddingBag expects offsets per sample)
    offsets.pop()

    feature_indices = torch.tensor(all_indices, dtype=torch.long)
    feature_weights = torch.tensor(all_weights, dtype=torch.float)
    feature_offsets = torch.tensor(offsets, dtype=torch.long)

    return feature_indices, feature_weights, feature_offsets


def wsg_to_multi_hot_features(wsg: WSG) -> torch.Tensor:
    """Generates a multi-hot feature matrix (x) for GCN/GAT from sparse features.

    Only supports 'sparse_binary' feature_type.
    """
    num_nodes = wsg.metadata.num_nodes
    feature_type = wsg.metadata.feature_type

    if feature_type != "sparse_binary":
        raise ValueError(
            "wsg_to_multi_hot_features only supports 'sparse_binary' features."
        )

    # Naive multi-hot matrix construction
    num_features = wsg.metadata.num_total_features
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for node_id, feature in wsg.node_features.items():
        indices = feature.indices
        node_idx = int(node_id)
        x[node_idx, indices] = 1.0

    return x


def wsg_to_dense_features(wsg: WSG) -> torch.Tensor:
    """Generates a dense feature matrix (x) from node feature weight vectors.

    Used for dense classifiers (e.g., MLP).
    """
    num_nodes = wsg.metadata.num_nodes
    # Determine feature dimensionality from first node feature weights
    feature_dim = len(next(iter(wsg.node_features.values())).weights)
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    for node_id, feature in wsg.node_features.items():
        node_idx = int(node_id)
        x[node_idx] = torch.tensor(feature.weights, dtype=torch.float)

    return x


def create_train_test_masks(
    y: torch.Tensor,
    labels_list: List,
    num_nodes: int,
    train_size: float,
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates boolean train/test masks for node classification.

    Only nodes with non-None labels are considered for stratified splitting.
    """
    valid_indices = [i for i, y_val in enumerate(labels_list) if y_val is not None]

    if not valid_indices:
        raise ValueError("No valid (non-None) labels found in the WSG object.")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_idx, temp_idx = train_test_split(
        valid_indices,
        train_size=train_size,  # 80%
        random_state=config.RANDOM_SEED,
        stratify=y[valid_indices],  # Estratifica usando todos os labels válidos
    )

    val_idx, test_idx = train_test_split(
        temp_idx,  # Divide apenas os índices temporários
        train_size=0.5,  # 50% de 'temp_idx'
        random_state=config.RANDOM_SEED,
        stratify=y[temp_idx],  # Estratifica usando APENAS os labels do conjunto temp
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# --- Conversion Functions for Different Models ---


def wsg_for_vgae(wsg: WSG, config: Config, train_size_ratio: float = 0.8) -> Data:
    """
    Converts a WSG object into a torch_geometric.data.Data object
    suitable for VGAE models.

    This includes sparse feature tensors (feature_indices, feature_weights, feature_offsets)
    required by models using nn.EmbeddingBag, plus labels and train/test masks for evaluation.
    """
    print("Converting WSG object to PyTorch Geometric format (VGAE)...")

    if wsg.metadata.feature_type != "sparse_binary":
        print(
            f"[WARNING]: Features for wsg_for_vgae are not 'sparse_binary', "
            f"but '{wsg.metadata.feature_type}'."
        )

    edge_index = wsg_to_edge_index(wsg)
    num_nodes = wsg.metadata.num_nodes
    num_total_features = wsg.metadata.num_total_features

    # Labels and masks (úteis para avaliação posterior)
    labels = wsg_to_labels(wsg)
    train_mask, val_mask, test_mask = create_train_test_masks(
        labels, wsg.graph_structure.y, num_nodes, train_size_ratio, config
    )

    feature_indices, feature_weights, feature_offsets = wsg_to_embeddingbag_features(
        wsg
    )

    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_total_features=num_total_features,
        feature_indices=feature_indices,
        feature_weights=feature_weights,
        feature_offsets=feature_offsets,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    print("wsg_for_vgae conversion completed successfully.")
    return data


def wsg_for_gcn_gat_multi_hot(
    wsg: WSG, config: Config, train_size_ratio: float = 0.8
) -> Data:
    """
    Converts a WSG object into a torch_geometric.data.Data object
    suitable for GCN or GAT models using multi-hot node features.

    Args:
        wsg (WSG): The input weighted structured graph object.
        config (Config): Configuration object with random seed.
        train_size (float, optional): Fraction of nodes for training. Defaults to 0.8.

    Returns:
        Data: A PyTorch Geometric Data object with multi-hot node features and train/test masks.
    """
    print("Converting WSG object to PyTorch Geometric format (GCN/GAT)...")

    if wsg.metadata.feature_type != "sparse_binary":
        print(
            f"WARNING: Features for wsg_for_gcn_gat_multi_hot are not 'sparse_binary', "
            f"but '{wsg.metadata.feature_type}'."
        )

    edge_index = wsg_to_edge_index(wsg)
    num_nodes = wsg.metadata.num_nodes
    labels = wsg_to_labels(wsg)
    node_features = wsg_to_multi_hot_features(wsg)

    train_mask, val_mask, test_mask = create_train_test_masks(
        labels, wsg.graph_structure.y, num_nodes, train_size_ratio, config
    )

    data = Data(
        edge_index=edge_index,
        x=node_features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    print("wsg_for_gcn_gat_multi_hot conversion completed successfully.")
    return data


def wsg_for_dense_classifier(
    wsg: WSG, config: Config, train_size_ratio: float = 0.8
) -> Data:
    """
    Converts a WSG object into a PyTorch Geometric Data object
    suitable for dense classifiers (e.g., MLP).

    Uses dense node feature matrices and creates train/test masks.

    Args:
        wsg (WSG): The input weighted structured graph object.
        config (Config): Configuration object with random seed.
        train_size (float, optional): Fraction of nodes for training. Defaults to 0.8.

    Returns:
        Data: A PyTorch Geometric Data object with dense node features and train/test masks.
    """
    print("Converting WSG object to PyTorch Geometric format (Dense Classifier)...")

    if wsg.metadata.feature_type != "dense_continuous":
        print(
            f"WARNING: Features for wsg_for_dense_classifier are not 'dense_continuous', "
            f"but '{wsg.metadata.feature_type}'."
        )

    labels = wsg_to_labels(wsg)
    node_features = wsg_to_dense_features(wsg)

    train_mask, val_mask, test_mask = create_train_test_masks(
        labels, wsg.graph_structure.y, wsg.metadata.num_nodes, train_size_ratio, config
    )

    data = Data(
        x=node_features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    print("wsg_for_dense_classifier conversion completed successfully.")
    return data


# --- Deprecated ---
# The old DataConverter class is kept here for reference only.
# Use the new modular functions instead.
'''
class DataConverter:
    """
    Converte um objeto WSG validado para um objeto torch_geometric.data.Data,
    pronto para ser usado em modelos GNN.
    
    Pode gerar dados para duas finalidades:
    1. Para camadas EmbeddingBag (esparso): (for_embedding_bag=True)
       Gera: feature_indices, feature_offsets, feature_weights
    2. Para classificadores padrão (denso): (for_embedding_bag=False)
       Gera: x, y, train_mask, test_mask
    """

    @staticmethod
    def to_pyg_data(wsg_obj: WSG, for_embedding_bag: bool = False) -> Data:
        """
        Converte um objeto WSG para um objeto torch_geometric.data.Data.

        Args:
            wsg_obj (WSG): O objeto de dados validado.
            for_embedding_bag (bool): Se True, formata os dados para um
                                      nn.EmbeddingBag (VGAE). Se False,
                                      cria uma matriz 'x' densa e máscaras
                                      (GCN, MLP, etc.).
        """
        print(f"Convertendo objeto WSG para formato PyTorch Geometric (for_embedding_bag={for_embedding_bag})...")

        # --- 1. Processar Estrutura do Grafo (Edge Index) ---
        edge_data = wsg_obj.graph_structure.edge_index
        if isinstance(edge_data, list):
            if len(edge_data) > 0 and len(edge_data[0]) == len(edge_data[1]):
                 # Formato padrão: [[src...], [dst...]]
                edge_index = torch.tensor(edge_data, dtype=torch.long)
            elif len(edge_data) > 0 and isinstance(edge_data[0], list) and len(edge_data[0]) == 2:
                # Formato: [[src, dst], [src, dst], ...]
                edge_index = torch.tensor(edge_data, dtype=torch.long).t()
            else:
                # Lista vazia ou formato inesperado
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
             raise TypeError(f"edge_index deve ser uma Lista, mas é {type(edge_data)}")


        # --- 2. Processar Labels (y) ---
        num_nodes = wsg_obj.metadata.num_nodes
        y_list = wsg_obj.graph_structure.y
        # Substitui None por -1 (um placeholder comum para nós não rotulados)
        y = torch.tensor(
            [-1 if label is None else int(label) for label in y_list],
            dtype=torch.long,
        )

        # --- 3. Montar o Objeto Data ---
        pyg_data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
        
        # --- 4. Processar Features (Lógica Condicional) ---
        
        if for_embedding_bag:
            # --- LÓGICA PARA VGAE ---
            # Gera os tensores esparsos para nn.EmbeddingBag
            all_indices = []
            all_weights = []
            offsets = [0] # O primeiro offset é sempre 0

            for i in range(num_nodes):
                node_id_str = str(i)
                node_feat = wsg_obj.node_features[node_id_str]
                
                all_indices.extend(node_feat.indices)
                all_weights.extend(node_feat.weights)
                offsets.append(offsets[-1] + len(node_feat.indices))
            
            offsets.pop() # Remove o último offset (comprimento total)

            pyg_data.feature_indices = torch.tensor(all_indices, dtype=torch.long)
            pyg_data.feature_weights = torch.tensor(all_weights, dtype=torch.float)
            pyg_data.feature_offsets = torch.tensor(offsets, dtype=torch.long)
            
            # Anexa metadados necessários para o modelo VGAE
            pyg_data.num_total_features = wsg_obj.metadata.num_total_features

        else:
            # --- LÓGICA PARA GCN/MLP ---
            # Gera a matriz 'x' densa e as máscaras
            feature_type = wsg_obj.metadata.feature_type

            if feature_type == "sparse_binary":
                # Cria matriz multi-hot (Abordagem Ingênua para GCN/GAT)
                num_features = wsg_obj.metadata.num_total_features
                x = torch.zeros((num_nodes, num_features), dtype=torch.float)
                for node_id, feature in wsg_obj.node_features.items():
                    indices = feature.indices
                    node_idx = int(node_id)
                    x[node_idx, indices] = 1.0
            else: 
                # Cria matriz densa (p/ classificar embeddings)
                feature_dim = len(next(iter(wsg_obj.node_features.values())).weights)
                x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
                for node_id, feature in wsg_obj.node_features.items():
                    node_idx = int(node_id)
                    x[node_idx] = torch.tensor(feature.weights, dtype=torch.float)
            
            pyg_data.x = x





            # Cria máscaras de treino/teste (só fazem sentido na classificação)
            valid_indices = [i for i, y_val in enumerate(y_list) if y_val is not None]
            
            if not valid_indices:
                 raise ValueError("Nenhum rótulo válido (não-None) encontrado no objeto WSG.")
                 
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_idx, test_idx = train_test_split(
                valid_indices, train_size=0.8, random_state=42, stratify=y[valid_indices]
            )

            train_mask[train_idx] = True
            test_mask[test_idx] = True
            
            pyg_data.train_mask = train_mask
            pyg_data.test_mask = test_mask

        print("Conversão concluída com sucesso.")
        return pyg_data
'''
