# src/data_converter.py

# third-party
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# local
from src.data_format_definition import WSG
from src.config import Config

def wsg_to_edge_index(wsg: WSG) -> torch.Tensor:
    """Extrai e valida o edge_index (estrutura do grafo) a partir de um objeto WSG."""
    edge_data = wsg.graph_structure.edge_index
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

    return edge_index


def wsg_to_labels(wsg: WSG) -> torch.Tensor:
    """Extrai o vetor de rótulos (y) dos nós do grafo."""
    num_nodes = wsg.metadata.num_nodes
    y_list = wsg.graph_structure.y
    # Substitui None por -1 (um placeholder comum para nós não rotulados)
    y = torch.tensor(
        [-1 if label is None else int(label) for label in y_list],
        dtype=torch.long,
    )
    return y


def wsg_to_embeddingbag_features(wsg: WSG) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converte as features do WSG em formato compatível com nn.EmbeddingBag.
    Retorna (feature_indices, feature_weights, feature_offsets).
    """
    num_nodes = wsg.metadata.num_nodes
    # Gera os tensores esparsos para nn.EmbeddingBag
    all_indices = []
    all_weights = []
    offsets = [0] # O primeiro offset é sempre 0

    for i in range(num_nodes):
        node_id_str = str(i)
        node_feat = wsg.node_features[node_id_str]
        
        all_indices.extend(node_feat.indices)
        all_weights.extend(node_feat.weights)
        offsets.append(offsets[-1] + len(node_feat.indices))
    
    offsets.pop() # Remove o último offset (comprimento total)

    feature_indices = torch.tensor(all_indices, dtype=torch.long)
    feature_weights = torch.tensor(all_weights, dtype=torch.float)
    feature_offsets = torch.tensor(offsets, dtype=torch.long)
    
    # Anexa metadados necessários para o modelo VGAE

    return feature_indices, feature_weights, feature_offsets


def wsg_to_multi_hot_features(wsg: WSG) -> torch.Tensor:
    """Gera a matriz multi-hot de features (x) para GCN/GAT a partir de features esparsas."""
    # --- LÓGICA PARA GCN/GAT ---
    # Gera a matriz 'x'
    num_nodes = wsg.metadata.num_nodes
    feature_type = wsg.metadata.feature_type

    if not feature_type == "sparse_binary":
        raise ValueError("wsg_to_multi_hot_features só suporta 'sparse_binary' features.")

    # Cria matriz multi-hot (Abordagem Ingênua para GCN/GAT)
    num_features = wsg.metadata.num_total_features
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for node_id, feature in wsg.node_features.items():
        indices = feature.indices
        node_idx = int(node_id)
        x[node_idx, indices] = 1.0

    return x


def wsg_to_dense_features(wsg: WSG) -> torch.Tensor:
    """Gera a matriz densa de features (x)"""
    num_nodes = wsg.metadata.num_nodes
    # Cria matriz densa (p/ classificar embeddings)
    feature_dim = len(next(iter(wsg.node_features.values())).weights)
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    for node_id, feature in wsg.node_features.items():
        node_idx = int(node_id)
        x[node_idx] = torch.tensor(feature.weights, dtype=torch.float)
        
    return x


def create_train_test_masks(y: torch.Tensor, y_wsg: list, num_nodes: int, train_size: float, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Cria máscaras de treino e teste para classificação de nós."""
    # Cria máscaras de treino/teste (só fazem sentido na classificação)
    #y_list = wsg_y

    valid_indices = [i for i, y_val in enumerate(y_wsg) if y_val is not None]
    
    if not valid_indices:
            raise ValueError("Nenhum rótulo válido (não-None) encontrado no objeto WSG.")
            
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_idx, test_idx = train_test_split(
        valid_indices, train_size=train_size, random_state=config.RANDOM_SEED, stratify=y[valid_indices]
    )

    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, test_mask



def wsg_for_vgae(wsg: WSG) -> Data:
    """Converte um objeto WSG para um objeto torch_geometric.data.Data para uso com VGAE."""
    print("Convertendo objeto WSG para formato PyTorch Geometric (VGAE)...")

    edge_index = wsg_to_edge_index(wsg)
    num_nodes = wsg.metadata.num_nodes
    num_total_features = wsg.metadata.num_total_features

    feature_indices, feature_weights, feature_offsets = wsg_to_embeddingbag_features(wsg)

    pyg_data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_total_features=num_total_features,
        #
        feature_indices=feature_indices,
        feature_weights=feature_weights,
        feature_offsets=feature_offsets,
    )

    print(f"Conversão \"wsg_for_vgae\" concluída com sucesso.")
    return pyg_data


def wsg_for_gcn_gat_multi_hot(wsg: WSG, config: Config, train_size: float = 0.8) -> Data:
    """Converte um objeto WSG para um objeto torch_geometric.data.Data para uso com GCN/GAT."""
    print("Convertendo objeto WSG para formato PyTorch Geometric (GCN/GAT)...")
    if not wsg.metadata.feature_type == "sparse_binary":
        print(f"WARNING: As features passadas para wsg_for_gcn_gat não são do tipo 'sparse_binary' são '{wsg.metadata.feature_type}'.")

    edge_index = wsg_to_edge_index(wsg)
    num_nodes = wsg.metadata.num_nodes #verificar se precisa
    num_total_features = wsg.metadata.num_total_features #verificar se precisa
    y = wsg_to_labels(wsg)
    
    x = wsg_to_multi_hot_features(wsg)

    train_mask, test_mask = create_train_test_masks(y, wsg.graph_structure.y, num_nodes, train_size, config)

    pyg_data = Data(
        edge_index=edge_index,
        #num_nodes=num_nodes, #verificar se precisa
        #num_total_features=num_total_features, #verificar se precisa
        x=x,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    print(f"Conversão \"wsg_for_gcn_gat\" concluída com sucesso.")
    return pyg_data


def wsg_for_dense_classifier(wsg: WSG, config: Config, train_size: float = 0.8) -> Data:
    """Converte um objeto WSG para uso com classificadores densos (MLP, etc.)."""
    print("Convertendo objeto WSG para formato PyTorch Geometric (Classificador Denso)...")

    y = wsg_to_labels(wsg)

    x = wsg_to_dense_features(wsg)

    train_mask, test_mask = create_train_test_masks(y, wsg.graph_structure.y, wsg.metadata.num_nodes, train_size, config)

    pyg_data = Data(
        x=x,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    print(f"Conversão \"wsg_for_dense_classifier\" concluída com sucesso.")
    return pyg_data





















# --- DEPRECATED ---
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
