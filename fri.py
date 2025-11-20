import sys
import os
import numpy as np

# Adiciona o diretório atual ao path para conseguir importar 'src'
sys.path.append(os.getcwd())

from src.data_loaders import FlickrLoader
from src.data_format_definition import WSG

def analyze_flickr():
    print("==================================================")
    print("INICIANDO INSPEÇÃO DO DATASET FLICKR (WSG)")
    print("==================================================")

    # 1. Instancia o Loader e Carrega
    loader = FlickrLoader()
    try:
        wsg_data: WSG = loader.load()
    except FileNotFoundError as e:
        print(f"\nERRO CRÍTICO: Arquivos não encontrados.\n{e}")
        print("Verifique se os caminhos em src/paths.py estão apontando corretamente para a pasta 'Flickr/raw'.")
        return

    # 2. Exibir Metadados Gerais
    meta = wsg_data.metadata
    print("\n--- 1. METADADOS GERAIS ---")
    print(f"Dataset:           {meta.dataset_name}")
    print(f"Processado em:     {meta.processed_at}")
    print(f"Total de Nós:      {meta.num_nodes}")
    print(f"Total de Arestas:  {meta.num_edges}")
    print(f"Total de Features: {meta.num_total_features} (Dimensão do vocabulário)")
    print(f"Tipo de Feature:   {meta.feature_type}")

    # 3. Análise da Estrutura (Arestas)
    edges = wsg_data.graph_structure.edge_index
    sources = edges[0]
    targets = edges[1]
    
    print("\n--- 2. ESTRUTURA DO GRAFO ---")
    print(f"Formato edge_index: [2, {len(sources)}]")
    print("Amostra de 5 arestas (Source -> Target):")
    for i in range(5):
        print(f"  {sources[i]} -> {targets[i]}")
    
    # Cálculo de densidade
    density = meta.num_edges / (meta.num_nodes ** 2)
    print(f"Densidade do Grafo: {density:.6f} (Muito esparso)")

    # 4. Análise dos Rótulos (Targets/Classes)
    y = wsg_data.graph_structure.y
    unique_classes = set(c for c in y if c is not None)
    
    print("\n--- 3. TARGETS (LABELS) ---")
    print(f"Total de classes únicas encontradas: {len(unique_classes)}")
    print(f"Classes presentes: {sorted(list(unique_classes))}")
    print("Amostra dos 10 primeiros nós:")
    for i in range(10):
        print(f"  Nó {i}: Classe {y[i]}")

    # 5. Análise das Features (Esparsidade)
    print("\n--- 4. FEATURES (Conversão Densa -> Esparsa) ---")
    features = wsg_data.node_features
    
    # Pegar um nó aleatório para inspecionar
    sample_id = "0" # O nó 0 geralmente existe
    if sample_id in features:
        feat_entry = features[sample_id]
        print(f"Inspecionando Features do Nó '{sample_id}':")
        print(f"  Quantidade de features ativas: {len(feat_entry.indices)}")
        print(f"  Índices (primeiros 10): {feat_entry.indices[:10]} ...")
        print(f"  Pesos (primeiros 10):   {feat_entry.weights[:10]} ...")
        
        # Calcular a esparsidade média baseada no nó 0 (aproximação)
        sparsity_node = 1.0 - (len(feat_entry.indices) / meta.num_total_features)
        print(f"  Esparsidade estimada deste nó: {sparsity_node*100:.2f}%")


        print("\nno completo")
        d = "0"
        print(f"  {wsg_data.node_features[d]}")
    
    print("\n==================================================")
    print("INSPEÇÃO CONCLUÍDA COM SUCESSO")
    print("==================================================")

if __name__ == "__main__":
    analyze_flickr()