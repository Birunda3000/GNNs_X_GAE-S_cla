# src/visualization.py (versão final com abordagem de 'pontos de cor' e 'tooltip')

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree, k_hop_subgraph
from pyvis.network import Network
import networkx as nx
from typing import List
from bs4 import BeautifulSoup
import math
import os  # Importa o módulo 'os'

from config import Config

# Opções simplificadas: foco na clareza e na física
GRAPH_OPTIONS = """
var options = {
  "nodes": {
    "shape": "dot",
    "size": 15,
    "borderWidth": 1,
    "borderWidthSelected": 2
  },
  "edges": {
    "color": {
      "inherit": "from"
    },
    "smooth": {
      "type": "continuous"
    },
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 0.5
      }
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -100000,
      "centralGravity": 0.03,
      "springLength": 500,
      "springConstant": 0.05,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75
  }
}
"""


# a função get_graph_info_text permanece a mesma...
def get_graph_info_text(data: Data, num_classes: int) -> str:
    num_nodes = data.num_nodes
    num_edges = data.num_edges if data.edge_index is not None else 0
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    num_features = data.num_node_features if data.x is not None else 0
    node_degrees = degree(data.edge_index[0], num_nodes=num_nodes)
    has_isolated_nodes = (node_degrees == 0).any().item()
    has_loops = (data.edge_index[0] == data.edge_index[1]).any().item()
    info = (
        f"--- Representação do Grafo ---\n"
        f"Nós: {num_nodes:,}\n"
        f"Arestas: {num_edges:,}\n"
        f"Grau Médio: {avg_degree:.2f}\n"
        f"Número de Features por Nó: {num_features}\n"
        f"Número de Classes: {num_classes}\n"
        f"Contém Nós Isolados: {has_isolated_nodes}\n"
        f"Contém Self-Loops: {has_loops}\n"
        f"---------------------------------"
    )
    return info


def draw_interactive_graph(
    data: Data, num_classes: int, class_names: List[str], dataset_name: str
):
    print(f"Iniciando a geração do grafo interativo final para '{dataset_name}'...")

    subgraph_data = data
    if data.num_nodes > Config.VIS_SAMPLES:
        print(f"Grafo muito grande. Amostrando a vizinhança do nó mais conectado.")
        total_degree = degree(data.edge_index[0], num_nodes=data.num_nodes) + degree(
            data.edge_index[1], num_nodes=data.num_nodes
        )
        center_node_idx = torch.argmax(total_degree).item()
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=center_node_idx,
            num_hops=10,
            edge_index=data.edge_index,
            relabel_nodes=True,
        )
        subgraph_data = Data(
            edge_index=sub_edge_index, y=data.y[subset], num_nodes=subset.size(0)
        )
        print(
            f"Subgrafo criado com {subgraph_data.num_nodes} nós e {subgraph_data.num_edges} arestas."
        )

    g = to_networkx(subgraph_data)
    net = Network(
        height="800px",
        width="100%",
        notebook=True,
        cdn_resources="in_line",
        directed=True,
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    # --- MELHORIA: Tamanho do nó baseado no grau e tooltip aprimorado ---
    degrees = dict(g.degree())
    min_size, max_size = 10, 50  # Define o tamanho mínimo e máximo para os nós

    # Evita divisão por zero se não houver graus
    max_degree = max(degrees.values()) if degrees else 1
    log_max_degree = math.log(max_degree + 1)

    for node_id in g.nodes():
        node_label_idx = subgraph_data.y[node_id].item()
        color = colors[node_label_idx % len(colors)]
        degree = degrees.get(node_id, 0)

        # Mapeamento logarítmico para o tamanho do nó
        if degree > 0:
            log_degree = math.log(degree + 1)
            # Normaliza o grau logarítmico para o intervalo [0, 1] e depois mapeia para [min_size, max_size]
            node_size = min_size + (max_size - min_size) * (log_degree / log_max_degree)
        else:
            node_size = min_size

        tooltip = (
            f"ID do Nó: {node_id}<br>"
            f"Classe: {class_names[node_label_idx]}<br>"
            f"Conexões: {degree}"
        )

        net.add_node(
            node_id, label=" ", color=color, title=tooltip, size=node_size
        )  # Define o tamanho do nó

    net.add_edges(g.edges())
    net.set_options(GRAPH_OPTIONS)

    # ... (a lógica da legenda e de salvar o HTML não muda) ...
    temp_html_path = "temp_graph.html"
    net.save_graph(temp_html_path)

    legend_html = '<div style="position: absolute; top: 10px; left: 10px; background-color: rgba(255,255,255,0.8); border: 1px solid #ccc; padding: 10px; border-radius: 5px; font-family: sans-serif; font-size: 12px; z-index: 999;">'
    legend_html += "<strong>Legenda</strong><ul>"
    for i, name in enumerate(class_names):
        color = colors[i % len(colors)]
        legend_html += f'<li style="list-style-type: none; margin-bottom: 5px;"><span style="background-color:{color}; width: 15px; height: 15px; display: inline-block; margin-right: 5px; vertical-align: middle;"></span>{name}</li>'
    legend_html += "</ul></div>"

    with open(temp_html_path, "r") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    soup.body.append(BeautifulSoup(legend_html, "html.parser"))

    # --- ALTERAÇÃO: Define o caminho de saída dinamicamente ---
    if dataset_name.lower() == "cora":
        output_dir = os.path.dirname(Config.CORA_CONTENT_PATH)
    elif dataset_name.lower() == "musaegithub":
        output_dir = os.path.dirname(Config.MUSAE_EDGES_PATH)
    else:
        # Fallback para o diretório de output padrão se o dataset não for reconhecido
        output_dir = "data/output"

    output_filename = f"{dataset_name.lower()}_visualization.html"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as f:
        f.write(str(soup))

    print(
        f"Grafo interativo com legenda customizada salvo com sucesso em: '{output_path}'"
    )


# ... (o bloco if __name__ == '__main__' não muda) ...
if __name__ == "__main__":
    try:
        from src.data_loaders import get_loader
    except ImportError:
        from src.data_loaders import get_loader

    loader = get_loader(Config.DATASET_NAME)
    graph_data, num_classes, class_names = loader.load()
    text_info = get_graph_info_text(graph_data, num_classes)
    print(text_info)
    draw_interactive_graph(graph_data, num_classes, class_names, Config.DATASET_NAME)
