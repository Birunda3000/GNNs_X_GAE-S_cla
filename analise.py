import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src.data_loaders import DirectWSGLoader
from src.data_converter import DataConverter


def analisar_dados(wsg_file_path):
    """
    Função para explorar e analisar os dados de um arquivo WSG.

    Args:
        wsg_file_path (str): Caminho para o arquivo .wsg.json
    """
    print("=" * 80)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS WSG")
    print("=" * 80)
    print(f"Arquivo analisado: {wsg_file_path}")
    print()

    # Carregar os dados
    loader = DirectWSGLoader(file_path=wsg_file_path)
    wsg_obj = loader.load()

    # === METADATA ===
    print("📊 METADADOS:")
    print(f"  Dataset: {wsg_obj.metadata.dataset_name}")
    print(f"  Tipo de features: {wsg_obj.metadata.feature_type}")
    print(f"  Número total de nós: {wsg_obj.metadata.num_nodes}")
    print(f"  Número total de arestas: {wsg_obj.metadata.num_edges}")
    print(f"  Número total de features: {wsg_obj.metadata.num_total_features}")
    print(f"  Grafo direcionado: {'Sim' if wsg_obj.metadata.directed else 'Não'}")
    print(f"  Processado em: {wsg_obj.metadata.processed_at}")
    print()

    # === ESTRUTURA DO GRAFO ===
    print("🔗 ESTRUTURA DO GRAFO:")

    # Análise dos labels (classes)
    y = wsg_obj.graph_structure.y
    labels_validos = [label for label in y if label is not None]
    num_labels_validos = len(labels_validos)
    num_labels_nulos = len(y) - num_labels_validos

    print(f"  Total de nós com labels: {num_labels_validos}")
    print(f"  Total de nós sem labels: {num_labels_nulos}")

    if labels_validos:
        classes_unicas = sorted(set(labels_validos))
        num_classes = len(classes_unicas)
        print(f"  Número de classes: {num_classes}")
        print(f"  Classes encontradas: {classes_unicas}")

        # Distribuição das classes
        contagem_classes = Counter(labels_validos)
        print("  Distribuição das classes:")
        for classe, count in sorted(contagem_classes.items()):
            porcentagem = (count / num_labels_validos) * 100
            print(f"    Classe {classe}: {count} amostras ({porcentagem:.2f}%)")

        # Estatísticas de balanceamento
        valores = list(contagem_classes.values())
        media = np.mean(valores)
        desvio = np.std(valores)
        minimo = min(valores)
        maximo = max(valores)
        print(f"  Estatísticas de balanceamento:")
        print(f"    Média por classe: {media:.1f}")
        print(f"    Desvio padrão: {desvio:.1f}")
        print(f"    Classe menor: {minimo} amostras")
        print(f"    Classe maior: {maximo} amostras")

        # Plot da distribuição (opcional)
        plt.figure(figsize=(10, 6))
        plt.bar(contagem_classes.keys(), contagem_classes.values())
        plt.xlabel("Classe")
        plt.ylabel("Número de Amostras")
        plt.title("Distribuição das Classes")
        plt.xticks(classes_unicas)
        plt.grid(True, alpha=0.3)
        plt.savefig("distribuicao_classes.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("  📈 Gráfico salvo como 'distribuicao_classes.png'")
    else:
        print("  ⚠️  Nenhum label válido encontrado!")

    print()

    # Análise das arestas
    edge_index = wsg_obj.graph_structure.edge_index
    if isinstance(edge_index, list) and len(edge_index) == 2:
        sources, targets = edge_index
        print(f"  Arestas: {len(sources)} conexões")

        # Calcular graus dos nós
        graus = Counter(sources + targets)
        graus_entrada = Counter(targets)
        graus_saida = Counter(sources)

        print(f"  Nós com conexões: {len(graus)}")
        print(f"  Grau médio: {np.mean(list(graus.values())):.2f}")
        print(f"  Grau máximo: {max(graus.values())}")
        print(f"  Grau mínimo: {min(graus.values())}")

        # Plot da distribuição de graus (opcional)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(list(graus.values()), bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Grau")
        plt.ylabel("Número de Nós")
        plt.title("Distribuição de Graus (Total)")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if wsg_obj.metadata.directed:
            plt.hist(
                list(graus_entrada.values()),
                bins=50,
                alpha=0.7,
                label="Entrada",
                edgecolor="black",
            )
            plt.hist(
                list(graus_saida.values()),
                bins=50,
                alpha=0.7,
                label="Saída",
                edgecolor="black",
            )
            plt.legend()
            plt.title("Distribuição de Graus (Direcionado)")
        else:
            plt.hist(list(graus.values()), bins=50, alpha=0.7, edgecolor="black")
            plt.title("Distribuição de Graus (Não Direcionado)")
        plt.xlabel("Grau")
        plt.ylabel("Número de Nós")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("distribuicao_graus.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("  📈 Gráfico de graus salvo como 'distribuicao_graus.png'")
    else:
        print("  ⚠️  Formato de edge_index inválido!")

    print()

    # === FEATURES DOS NÓS ===
    print("🔍 FEATURES DOS NÓS:")

    node_features = wsg_obj.node_features
    feature_type = wsg_obj.metadata.feature_type

    print(f"  Tipo de features: {feature_type}")
    print(f"  Nós com features: {len(node_features)}")

    if feature_type == "dense_continuous":
        # Para embeddings densos
        pesos_todos = []
        for node_id, feature in node_features.items():
            pesos_todos.extend(feature.weights)

        if pesos_todos:
            pesos_array = np.array(pesos_todos)
            print(f"  Dimensão dos embeddings: {len(feature.weights)}")
            print(f"  Total de valores: {len(pesos_array)}")
            print(f"  Média dos pesos: {np.mean(pesos_array):.4f}")
            print(f"  Desvio padrão: {np.std(pesos_array):.4f}")
            print(f"  Valor mínimo: {np.min(pesos_array):.4f}")
            print(f"  Valor máximo: {np.max(pesos_array):.4f}")

            # Plot da distribuição dos valores dos embeddings
            plt.figure(figsize=(10, 6))
            plt.hist(pesos_array, bins=100, alpha=0.7, edgecolor="black")
            plt.xlabel("Valor do Embedding")
            plt.ylabel("Frequência")
            plt.title("Distribuição dos Valores dos Embeddings")
            plt.grid(True, alpha=0.3)
            plt.savefig("distribuicao_embeddings.png", dpi=300, bbox_inches="tight")
            plt.show()
            print("  📈 Histograma salvo como 'distribuicao_embeddings.png'")

    elif feature_type == "sparse_binary":
        # Para features esparsas
        indices_todos = []
        for node_id, feature in node_features.items():
            indices_todos.extend(feature.indices)

        indices_unicos = set(indices_todos)
        print(f"  Features distintas usadas: {len(indices_unicos)}")
        print(
            f"  Densidade média por nó: {len(indices_todos) / len(node_features):.2f}"
        )

        # Distribuição do número de features por nó
        num_features_por_no = [
            len(feature.indices) for feature in node_features.values()
        ]
        print(f"  Média de features por nó: {np.mean(num_features_por_no):.2f}")
        print(f"  Máximo de features por nó: {max(num_features_por_no)}")
        print(f"  Mínimo de features por nó: {min(num_features_por_no)}")

    print()

    # === CONVERSÃO PARA PYG E ANÁLISE ADICIONAL ===
    print("🔄 CONVERSÃO PARA PYTORCH GEOMETRIC:")
    try:
        pyg_data = DataConverter.to_pyg_data(wsg_obj)
        print("  ✅ Conversão bem-sucedida!")
        print(f"  Shape dos dados X: {pyg_data.x.shape}")
        print(f"  Shape do edge_index: {pyg_data.edge_index.shape}")
        print(f"  Shape dos labels y: {pyg_data.y.shape}")
        print(f"  Nós de treino: {pyg_data.train_mask.sum().item()}")
        print(f"  Nós de teste: {pyg_data.test_mask.sum().item()}")
    except Exception as e:
        print(f"  ❌ Erro na conversão: {e}")

    print()
    print("=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    # Caminho padrão - altere conforme necessário
    wsg_file_path = "data/output/EMBEDDING_RUNS/musae-github__loss_0_9483__emb_dim_32__01-10-2025_14-58-11/musae-github_embeddings.wsg.json"

    # Verifica se o arquivo existe
    if not os.path.exists(wsg_file_path):
        print(f"❌ Arquivo não encontrado: {wsg_file_path}")
        print("Por favor, ajuste o caminho do arquivo WSG.")
    else:
        analisar_dados(wsg_file_path)
