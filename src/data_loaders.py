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
import os

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



class MusaeTwitchLoader(BaseDatasetLoader):
    """Carrega o dataset Musae-Twitch (Combinado) a partir de múltiplas regiões."""

    dataset_name = "Musae-Twitch"

    def load(self) -> WSG:
        # Caminho base validado pelo script de reconhecimento
        base_path = "data/datasets/twitch/twitch"
        
        print("Iniciando o carregamento unificado do Musae-Twitch...")

        all_source_nodes = []
        all_target_nodes = []
        all_y = []
        all_node_names = []
        all_node_features = {}
        
        global_node_offset = 0
        max_feature_index = 0

        # Lista as pastas das regiões garantindo a ordem
        pastas = sorted([p for p in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, p))])
        
        for pasta in pastas:
            folder_path = os.path.join(base_path, pasta)
            
            edges_file = target_file = features_file = None
            for f in os.listdir(folder_path):
                if "edges.csv" in f: edges_file = os.path.join(folder_path, f)
                elif "target.csv" in f: target_file = os.path.join(folder_path, f)
                elif f.endswith(".json"): features_file = os.path.join(folder_path, f)
            
            if not (edges_file and target_file and features_file):
                print(f"Aviso: Arquivos incompletos na pasta {pasta}. Pulando.")
                continue
                
            # 1. Carrega os dados da região
            edges_df = pd.read_csv(edges_file)
            target_df = pd.read_csv(target_file)
            with open(features_file, 'r') as f:
                features_json = json.load(f)
                
            num_nodes_region = len(target_df)
            
            # 2. Processa Edges (Ajustando com o offset global)
            edges_df['from'] += global_node_offset
            edges_df['to'] += global_node_offset
            
            # Garante bidirecionalidade e arestas únicas
            unique_edges = set(tuple(sorted(edge)) for edge in edges_df[['from', 'to']].itertuples(index=False, name=None))
            
            source_nodes = [u for u, v in unique_edges] + [v for u, v in unique_edges]
            target_nodes = [v for u, v in unique_edges] + [u for u, v in unique_edges]
            
            all_source_nodes.extend(source_nodes)
            all_target_nodes.extend(target_nodes)
            
            # 3. Processa Target (mature: bool -> 1/0)
            # O target é a classificação se usa linguagem explícita ou não
            y_region = target_df['mature'].astype(int).tolist()
            all_y.extend(y_region)
            
            # Usar o 'id' original da twitch + a região como identificador de nome
            names_region = (target_df['id'].astype(str) + "_" + pasta).tolist()
            all_node_names.extend(names_region)
            
            # 4. Processa Features com Offset (Protegido contra nós ausentes no JSON)
            for local_id in range(num_nodes_region):
                global_node_id = local_id + global_node_offset
                
                # Tenta pegar as features; se não existir no JSON, retorna lista vazia []
                feature_list = features_json.get(str(local_id), [])
                
                all_node_features[str(global_node_id)] = {
                    "indices": feature_list,
                    "weights": [1.0] * len(feature_list)
                }
                
                if feature_list:
                    local_max = max(feature_list)
                    if local_max > max_feature_index:
                        max_feature_index = local_max
                        
            print(f"Região {pasta} processada. {num_nodes_region} nós adicionados. (Offset atualizado para a próxima: {global_node_offset + num_nodes_region})")
            global_node_offset += num_nodes_region

        # --- Instanciar o WSG ---
        num_total_nodes = global_node_offset
        num_total_edges = len(all_source_nodes)
        num_total_features = max_feature_index + 1 if max_feature_index > 0 else 0

        tz_offset = timedelta(hours=-3)
        tz_info = timezone(tz_offset)
        processed_at = datetime.now(tz_info).isoformat()

        metadata_data = {
            "dataset_name": self.dataset_name,
            "feature_type": "sparse_binary",
            "num_nodes": num_total_nodes,
            "num_edges": num_total_edges,
            "num_total_features": num_total_features,
            "processed_at": processed_at,
            "directed": False,
        }

        graph_structure_data = {
            "edge_index": [all_source_nodes, all_target_nodes],
            "y": all_y,
            "node_names": all_node_names,
        }

        print("Validando estrutura concatenada com Pydantic...")
        wsg_object = WSG(
            metadata=Metadata(**metadata_data),
            graph_structure=GraphStructure(**graph_structure_data),
            node_features={k: NodeFeaturesEntry(**v) for k, v in all_node_features.items()}
        )

        print("Musae-Twitch (Global) processado e validado com sucesso!")
        return wsg_object



class RedditLoader(BaseDatasetLoader):
    """
    Carrega o dataset Reddit Threads transformando minigrafos isolados em um Super Grafo.
    
    Decisões Metodológicas Fundamentadas:
    1. SUPER GRAFO: As 203.088 threads isoladas são fundidas usando um offset de IDs, 
       permitindo o processamento em lote pelo VGAE.
    2. PROPAGAÇÃO DE RÓTULO: Como o dataset classifica a thread inteira e o modelo 
       classifica nós, o rótulo da thread (0 ou 1) é propagado para todos os seus participantes.
    3. CONSTANT FEATURE REPRESENTATION: Como o dataset original não possui features de nós:
       - Evitamos One-Hot Encoding para não gerar uma matriz NxN gigantesca (OOM - Out of Memory).
       - Injetamos uma feature constante estática (índice 0, peso 1.0).
       - Justificativa Matemática: Ao usar peso 1.0 uniforme, a primeira camada de Message 
         Passing da GNN somará os 1s da vizinhança, calculando o "Grau do Nó" de forma natural 
         e endógena, forçando o modelo a focar puramente na topologia da rede.
    """

    dataset_name = "Reddit"

    def load(self) -> WSG:
        edges_path = "data/datasets/reddit_threads/reddit_edges.json"
        target_path = "data/datasets/reddit_threads/reddit_target.csv"

        print("Iniciando carregamento do Reddit Threads (Metodologia Super Grafo + Feature Constante)...")

        # 1. Carrega os arquivos brutos
        with open(edges_path, 'r') as f:
            edges_json = json.load(f)
        target_df = pd.read_csv(target_path)

        # Transforma o target num dicionário O(1) para busca rápida (id_da_thread -> target)
        target_dict = dict(zip(target_df['id'].astype(str), target_df['target']))

        all_source_nodes = []
        all_target_nodes = []
        all_y = []
        all_node_names = []
        all_node_features = {}

        global_node_offset = 0

        print(f"Processando {len(edges_json)} threads... isso pode levar um minuto.")

        # 2. Constrói as "Ilhas" do Super Grafo
        for thread_id_str, edges in edges_json.items():
            thread_label = target_dict.get(thread_id_str, None)
            
            # Se a thread estiver vazia, pula para não quebrar a lógica
            if not edges:
                continue

            # Descobre a quantidade de nós nesta thread específica (maior ID + 1)
            max_local_id = max([max(u, v) for u, v in edges])
            num_nodes_here = max_local_id + 1

            # Adiciona as arestas com offset global
            # (Garante bidirecionalidade duplicando u->v e v->u, já que o grafo é não-direcionado)
            for u, v in edges:
                u_glob = u + global_node_offset
                v_glob = v + global_node_offset
                
                if u_glob != v_glob:  # Evita self-loops desnecessários
                    all_source_nodes.extend([u_glob, v_glob])
                    all_target_nodes.extend([v_glob, u_glob])

            # Cria os metadados dos nós
            for local_id in range(num_nodes_here):
                global_id = local_id + global_node_offset
                
                # Propagação do Rótulo: o usuário herda a nota da thread
                all_y.append(thread_label)
                
                # Rastreabilidade: guarda a origem exata do usuário
                all_node_names.append(f"thread_{thread_id_str}_user_{local_id}")

                # Feature Constante: Permite que o VGAE calcule o grau do nó indiretamente
                all_node_features[str(global_id)] = {
                    "indices": [0],
                    "weights": [1.0]
                }

            global_node_offset += num_nodes_here

        # 3. Limpeza estrutural profunda (Remove qualquer duplicidade)
        print("Limpando arestas duplicadas geradas pela bidirecionalidade...")
        unique_edges = set(zip(all_source_nodes, all_target_nodes))
        all_source_nodes = [e[0] for e in unique_edges]
        all_target_nodes = [e[1] for e in unique_edges]

        # 4. Empacota tudo no formato WSG Canônico
        num_total_nodes = global_node_offset
        num_total_edges = len(all_source_nodes)
        num_total_features = 1  # Única feature: a camisa branca constante

        tz_offset = timedelta(hours=-3)
        tz_info = timezone(tz_offset)
        processed_at = datetime.now(tz_info).isoformat()

        metadata_data = {
            "dataset_name": self.dataset_name,
            "feature_type": "sparse_binary",
            "num_nodes": num_total_nodes,
            "num_edges": num_total_edges,
            "num_total_features": num_total_features,
            "processed_at": processed_at,
            "directed": False,
        }

        graph_structure_data = {
            "edge_index": [all_source_nodes, all_target_nodes],
            "y": all_y,
            "node_names": all_node_names,
        }

        print("Validando estrutura final do Reddit com Pydantic...")
        wsg_object = WSG(
            metadata=Metadata(**metadata_data),
            graph_structure=GraphStructure(**graph_structure_data),
            node_features={k: NodeFeaturesEntry(**v) for k, v in all_node_features.items()}
        )

        print("Reddit Threads encapsulado com sucesso e pronto para o VGAE!")
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

