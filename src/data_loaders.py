# src/data_loader.py

# Standard library
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Third-party
import pandas as pd
import numpy as np
import scipy.sparse as sp

# Local application
from src.data_format_definition import GraphStructure, Metadata, NodeFeaturesEntry, WSG
from src.paths import musae_facebook_paths, musae_github_paths, flickr_paths 


class BaseDatasetLoader(ABC):
    """Classe base que define o contrato para os loaders de dataset."""

    @abstractmethod
    def load(self) -> WSG:
        """
        Método de carregamento principal.

        Deve carregar os dados brutos de um dataset e transformá-los em um
        objeto que segue a especificação do formato Weighted Sparse Graph (WSG).

        Returns:
            WSG: Um objeto Pydantic representando o grafo no formato WSG.
        """
        pass


class DirectWSGLoader(BaseDatasetLoader):
    """Carrega um dataset já no formato WSG a partir de um arquivo JSON local."""

    def __init__(self, file_path: str):
        self.file_path = file_path  # define o caminho do arquivo

    @property
    def dataset_name(self) -> str:
        """Gera dinamicamente o nome do dataset a partir do arquivo carregado."""
        return f"Direct WSG Loader (from {self.file_path})"

    def load(self) -> WSG:
        """
        Carrega o arquivo JSON e o valida como um objeto WSG.

        Returns:
            WSG: Um objeto Pydantic contendo o grafo completo e validado no formato WSG.
        """
        with open(self.file_path, "r") as f:
            wsg_data: Dict[str, Any] = json.load(f)

        wsg_object = WSG(**wsg_data)
        return wsg_object


class CoraLoader(BaseDatasetLoader):
    """Carrega o dataset Cora a partir de arquivos locais."""

    dataset_name = "Cora"

    def load(self) -> WSG:
        """
        Carrega e processa o dataset Cora para o formato WSG.

        Raises:
            NotImplementedError: Esta função ainda não foi implementada.
        """
        # TODO: Implementar a lógica de carregamento e processamento para o dataset Cora.
        raise NotImplementedError(
            "O loader para o dataset Cora ainda não foi implementado."
        )


class MusaeGithubLoader(BaseDatasetLoader):
    """Carrega o dataset Musae-Github a partir de arquivos locais."""

    dataset_name = "Musae-Github"

    def load(self) -> WSG:
        """
        Carrega os dados brutos do Musae-Github e os transforma para o formato WSG.

        O processo consiste em:
        1. Carregar as arestas, alvos (labels) e features dos arquivos CSV e JSON.
        2. Construir os dicionários para metadados, estrutura do grafo e features.
        3. Instanciar o objeto Pydantic `WSG`, que valida automaticamente a estrutura e os tipos.
        4. Retornar o objeto `WSG` validado.

        Returns:
            WSG: Um objeto Pydantic contendo o grafo completo e validado no formato WSG.
        """
        edges_df = pd.read_csv(musae_github_paths.GITHUB_MUSAE_EDGES_PATH)
        target_df = pd.read_csv(musae_github_paths.GITHUB_MUSAE_TARGET_PATH)
        with open(musae_github_paths.GITHUB_MUSAE_FEATURES_PATH, "r") as f:
            features_json: Dict[str, List[int]] = json.load(f)

        print(
            "Arquivos do Github carregados. Iniciando processamento para o formato WSG..."
        )

        # --- 1. Preparar dados para os modelos Pydantic ---

        # Garante que arestas não direcionadas sejam únicas e bidirecionais
        # Cria pares (min(u,v), max(u,v)) para identificar arestas únicas
        unique_edges = set(
            tuple(sorted(edge)) for edge in edges_df.itertuples(index=False, name=None)
        )

        source_nodes = [u for u, v in unique_edges] + [v for u, v in unique_edges]
        target_nodes = [v for u, v in unique_edges] + [u for u, v in unique_edges]

        num_nodes: int = len(target_df)
        num_edges: int = len(source_nodes)

        all_indices = (idx for indices in features_json.values() for idx in indices)
        try:
            max_feature_index = max(all_indices)
            num_total_features = max_feature_index + 1
        except ValueError:
            num_total_features = 0

        tz_offset = timedelta(hours=-3)
        tz_info = timezone(tz_offset)
        processed_at: str = datetime.now(tz_info).isoformat()

        metadata_data = {
            "dataset_name": "Musae-Github",
            "feature_type": "sparse_binary",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_total_features": num_total_features,
            "processed_at": processed_at,
            "directed": False,
        }

        graph_structure_data = {
            "edge_index": [
                source_nodes,
                target_nodes,
            ],
            "y": target_df["ml_target"]
            .where(pd.notnull(target_df["ml_target"]), None)
            .tolist(),
            "node_names": target_df["name"].tolist(),
        }

        # Garante que todos os nós de 0 a num_nodes-1 tenham uma entrada de feature.
        # Se um nó não estiver em features_json, ele recebe listas vazias.
        node_features_data = {
            str(i): {
                "indices": features_json.get(str(i), []),
                "weights": [1.0] * len(features_json.get(str(i), [])),
            }
            for i in range(num_nodes)
        }

        # --- 2. Instanciar e validar o objeto WSG ---
        # A instanciação dos modelos Pydantic substitui as asserções manuais.
        # Se os dados não estiverem no formato correto, Pydantic levantará um `ValidationError`.
        wsg_object = WSG(
            metadata=Metadata(**metadata_data),
            graph_structure=GraphStructure(**graph_structure_data),
            node_features={
                k: NodeFeaturesEntry(**v) for k, v in node_features_data.items()
            },
        )

        print("Processamento e validação com Pydantic concluídos com sucesso.")
        return wsg_object


class MusaeFacebookLoader(BaseDatasetLoader):

    dataset_name = "Musae-Facebook"

    """Carrega o dataset Musae-Facebook a partir de arquivos locais."""

    def load(self) -> WSG:
        """
        Carrega os dados brutos do Musae-Facebook e os transforma para o formato WSG.

        O processo consiste em:
        1. Carregar as arestas, alvos (labels) e features dos arquivos CSV e JSON.
        2. Mapear os labels de string (ex: "tvshow") para inteiros (ex: 0).
        3. Construir os dicionários para metadados, estrutura do grafo e features.
        4. Instanciar o objeto Pydantic `WSG` para validação.
        5. Retornar o objeto `WSG` validado.

        Returns:
            WSG: Um objeto Pydantic contendo o grafo completo e validado no formato WSG.
        """
        # TODO: Verifique se os caminhos em src/config.py estão corretos
        # (ex: Config.FACEBOOK_MUSAE_EDGES_PATH)
        edges_df = pd.read_csv(musae_facebook_paths.FACEBOOK_MUSAE_EDGES_PATH)
        target_df = pd.read_csv(musae_facebook_paths.FACEBOOK_MUSAE_TARGET_PATH)
        with open(musae_facebook_paths.FACEBOOK_MUSAE_FEATURES_PATH, "r") as f:
            features_json: Dict[str, List[int]] = json.load(f)

        print(
            "Arquivos do Facebook carregados. Iniciando processamento para o formato WSG..."
        )

        # --- 1. Preparar dados para os modelos Pydantic ---

        # Trata arestas não direcionadas (idêntico ao Github)
        unique_edges = set(
            tuple(sorted(edge)) for edge in edges_df.itertuples(index=False, name=None)
        )

        source_nodes = [u for u, v in unique_edges] + [v for u, v in unique_edges]
        target_nodes = [v for u, v in unique_edges] + [u for u, v in unique_edges]

        num_nodes: int = len(target_df)
        num_edges: int = len(source_nodes)

        # Processamento de features (idêntico ao Github)
        all_indices = (idx for indices in features_json.values() for idx in indices)
        try:
            max_feature_index = max(all_indices)
            num_total_features = max_feature_index + 1
        except ValueError:
            num_total_features = 0

        tz_offset = timedelta(hours=-3)
        tz_info = timezone(tz_offset)
        processed_at: str = datetime.now(tz_info).isoformat()

        metadata_data = {
            "dataset_name": "Musae-Facebook",
            "feature_type": "sparse_binary",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_total_features": num_total_features,
            "processed_at": processed_at,
            "directed": False,  # Conforme README
        }

        # --- DIFERENÇA-CHAVE: Mapeamento de Labels ---
        # As amostras mostram "tvshow", "government", "company", "politician"
        label_mapping = {"tvshow": 0, "government": 1, "company": 2, "politician": 3}

        y_labels = target_df["page_type"].map(label_mapping)
        y_labels = y_labels.where(y_labels.notnull(), None).tolist()

        graph_structure_data = {
            "edge_index": [
                source_nodes,
                target_nodes,
            ],
            "y": y_labels,
            "node_names": target_df["page_name"].tolist(),
        }

        node_features_data = {
            str(i): {
                "indices": features_json.get(str(i), []),
                "weights": [1.0] * len(features_json.get(str(i), [])),
            }
            for i in range(num_nodes)
        }

        # --- 2. Instanciar e validar o objeto WSG ---
        wsg_object = WSG(
            metadata=Metadata(**metadata_data),
            graph_structure=GraphStructure(**graph_structure_data),
            node_features={
                k: NodeFeaturesEntry(**v) for k, v in node_features_data.items()
            },
        )

        print("Processamento e validação com Pydantic concluídos com sucesso.")
        return wsg_object


class FlickrLoader(BaseDatasetLoader):
    """Carrega o dataset Flickr a partir de arquivos locais (formato GraphSAINT)."""

    dataset_name = "Flickr"

    def load(self) -> WSG:
        """
        Carrega os dados brutos do Flickr (npz, npy, json) e transforma para WSG.
        
        O processo consiste em:
        1. Carregar a matriz de adjacência esparsa (CSR).
        2. Carregar a matriz densa de features e convertê-la para representação esparsa.
        3. Mapear o class_map para uma lista ordenada de targets.
        """
        print("Carregando arquivos do Flickr...")
        
        # 1. Carregar Estruturas de Dados
        # adj_full.npz geralmente é uma matriz CSR salva pelo scipy
        adj_matrix = sp.load_npz(flickr_paths.FLICKR_ADJ_PATH)
        feats_matrix = np.load(flickr_paths.FLICKR_FEATS_PATH)
        
        with open(flickr_paths.FLICKR_CLASS_MAP_PATH, "r") as f:
            class_map: Dict[str, int] = json.load(f)
            
        # O arquivo role.json existe para splits, mas o WSG foca na estrutura e features.
        # Os splits são gerenciados posteriormente no pipeline de treino.

        # 2. Processamento de Metadados Básicos
        num_nodes = feats_matrix.shape[0]
        num_features = feats_matrix.shape[1]
        # A matriz CSR conta arestas não-zero. Se for não-direcionado, isso conta (u,v) e (v,u).
        num_edges = adj_matrix.nnz 

        tz_offset = timedelta(hours=-3)
        tz_info = timezone(tz_offset)
        processed_at: str = datetime.now(tz_info).isoformat()

        print(f"Metadados detectados: {num_nodes} nós, {num_edges} arestas, {num_features} features.")

        # 3. Construir GraphStructure (Edge Index)
        # Converter CSR para COOrdinate format para extrair row/col facilmente
        coo_adj = adj_matrix.tocoo()
        row_indices = coo_adj.row.tolist()
        col_indices = coo_adj.col.tolist()

        # Construir lista de Targets (y)
        # O class_map é um dicionário {"id_str": label_int}. 
        # Precisamos garantir a ordem 0..N-1
        y_labels = [None] * num_nodes
        for node_id_str, label in class_map.items():
            idx = int(node_id_str)
            if 0 <= idx < num_nodes:
                y_labels[idx] = label
        
        # Flickr não tem "nomes" de usuários no dataset público, usamos o ID como nome
        node_names = [str(i) for i in range(num_nodes)]

        graph_structure_data = {
            "edge_index": [row_indices, col_indices],
            "y": y_labels,
            "node_names": node_names
        }

        # 4. Construir Node Features (Conversão Densa -> Esparsa)
        # O formato WSG exige {node_id: {indices: [], weights: []}}
        # Como iterar 89k linhas em Python puro é lento, usamos lógica vetorial onde possível,
        # mas para montar o dicionário final, iteramos.
        
        print("Convertendo features densas para formato esparso WSG (isso pode levar alguns segundos)...")
        node_features_data = {}
        
        # Otimização: Se a matriz for muito densa, isso fica grande. 
        # Mas no Flickr as features são bag-of-words (esparsas), então safe.
        for i in range(num_nodes):
            # Pega a linha i
            row_data = feats_matrix[i]
            # Acha onde não é zero
            non_zero_indices = np.nonzero(row_data)[0]
            non_zero_weights = row_data[non_zero_indices]
            
            node_features_data[str(i)] = {
                "indices": non_zero_indices.tolist(),
                "weights": non_zero_weights.tolist() # float array -> list
            }

        metadata_data = {
            "dataset_name": "Flickr",
            "feature_type": "dense_converted_to_sparse", # Original era npy denso
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_total_features": num_features,
            "processed_at": processed_at,
            "directed": False, # Flickr geralmente é tratado como não-direcionado
        }

        # 5. Instanciar e Validar
        print("Validando schema WSG...")
        wsg_object = WSG(
            metadata=Metadata(**metadata_data),
            graph_structure=GraphStructure(**graph_structure_data),
            node_features={
                k: NodeFeaturesEntry(**v) for k, v in node_features_data.items()
            },
        )

        print("Flickr processado com sucesso.")
        return wsg_object











def save_wsg(wsg_obj: WSG, file_path: str):
    """
    Salva um objeto WSG em um arquivo JSON no caminho especificado.

    Args:
        wsg_obj (WSG): O objeto Pydantic WSG a ser salvo.
        file_path (str): O caminho completo do arquivo onde o JSON será salvo.
    """
    print(f"Salvando objeto WSG em '{file_path}'...")
    with open(file_path, "w") as f:
        # O método model_dump() converte o objeto Pydantic para um dicionário Python
        json.dump(wsg_obj.model_dump(), f, indent=4)

    print(f"Objeto WSG salvo com sucesso em '{file_path}'.")

