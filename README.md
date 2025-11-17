# TCC-GNN: Framework para Geração e Análise de Embeddings de Grafos

Framework modular para experimentação com GNNs, com foco em:
- Geração auto-supervisionada de node embeddings (GAE/VGAE).
- Classificação em embeddings gerados (Sklearn, MLP, XGBoost).
- Classificação fim-a-fim em grafos (GCN/GAT).
- Relatórios e gerenciamento de execuções com organização automática de pastas.

O pipeline usa o formato canônico Weighted Sparse Graph (WSG), definido em [wsg_definition.txt](wsg_definition.txt).

---

## Sumário
1. Visão Geral
2. Ambiente e Execução
3. Fluxos de Trabalho
4. Dados e Formato WSG
5. Relatórios e Resultados
6. Ferramentas de Apoio
7. Estrutura do Projeto

---

## 1) Visão Geral

- Formato padronizado (WSG) desacopla preparação de dados da modelagem: [src/data_format_definition.py](src/data_format_definition.py), [wsg_definition.txt](wsg_definition.txt).
- Carregadores de datasets convertidos para WSG: [src/data_loaders.py](src/data_loaders.py).
- Conversores WSG → PyTorch Geometric: [src/data_converters.py](src/data_converters.py).
- Orquestração de execuções, memória e relatórios: [src/experiment_runner.py](src/experiment_runner.py), [src/report_manager.py](src/report_manager.py), [src/directory_manager.py](src/directory_manager.py).
- Configurações centralizadas: [src/config.py](src/config.py).

---

## 2) Ambiente e Execução

Este projeto é pensado para Dev Containers (VS Code).

- Pré-requisitos:
  - Docker, VS Code, extensão Dev Containers.
- Abrir no contêiner:
  - Reabra o projeto no container via “Dev Containers: Reopen in Container”.
- CPU/GPU:
  - Altere o compose desejado em .devcontainer/devcontainer.json.

Dependências Python estão em [requirements.txt](requirements.txt). O Dev Container já prepara o ambiente.

---

## 3) Fluxos de Trabalho

As execuções criam pastas em data/output, com subpastas organizadas automaticamente por [src/directory_manager.py](src/directory_manager.py).

### Fluxo 1: Geração de Embeddings (VGAE/GAE)

Script: [run_embedding_generation.py](run_embedding_generation.py)

- O script percorre datasets (Musae-Facebook, Musae-Github) e múltiplas dimensões de embedding.
- Converte WSG → PyG para VGAE: [`wsg_for_vgae`](src/data_converters.py).
- Treina modelo GraphSAGE-GAE: [src/models/embedding_models/autoencoders_models.py](src/models/embedding_models/autoencoders_models.py).
- Early Stopping com métrica de F1 em avaliação interna: [src/early_stopper.py](src/early_stopper.py), [src/embeddings_eval.py](src/embeddings_eval.py).
- Salva:
  - Modelo .pt completo: [`salvar_modelo_pytorch_completo`](src/utils.py).
  - Embeddings no padrão WSG: [`save_embeddings_to_wsg`](src/utils.py).
  - Relatório da execução: [src/report_manager.py](src/report_manager.py).

Como executar:
```bash
python run_embedding_generation.py
```

Saída típica (por execução em data/output/EMBEDDING_RUNS):
- <Dataset>__score_<...>__emb_dim_<...>__<timestamp>/
  - <Dataset>_(k)_embeddings_<timestamp>.wsg.json
  - <Dataset>__GraphSageGAE__<timestamp>.pt
  - run_report.json

Parâmetros principais em [src/config.py](src/config.py): EPOCHS, LEARNING_RATE, OUT_EMBEDDING_DIM (o script sobrescreve durante o loop), EARLY_STOPPING_* e DEVICE.

### Fluxo 2: Classificação de Embeddings

Script: [run_feature_classification.py](run_feature_classification.py)

- Procura automaticamente todos os arquivos .wsg.json dentro de data/output/EMBEDDING_RUNS.
- Converte WSG → matriz densa (x) para classificadores: [`wsg_for_dense_classifier`](src/data_converters.py).
- Executa:
  - Sklearn: LogisticRegression, KNN, RandomForest ([src/models/sklearn_model.py](src/models/sklearn_model.py)).
  - MLP PyTorch ([src/models/pytorch_classification/classification_models.py](src/models/pytorch_classification/classification_models.py)).
  - XGBoost ([src/models/xgboost_classifier.py](src/models/xgboost_classifier.py)).
- Orquestração e memória: [src/experiment_runner.py](src/experiment_runner.py).

Como executar:
```bash
python run_feature_classification.py
```

Saída em data/output/CLASSIFICATION_RUNS:
- <Dataset>-Embeddings__best_test_<...>__model_<...>__<timestamp>/run_report.json

### Fluxo 3: Classificação de Grafo Fim-a-Fim (GCN/GAT)

Script: [run_graph_classification.py](run_graph_classification.py)

- Carrega datasets brutos e prepara WSG via loaders:
  - Musae-Github, Musae-Facebook em [src/data_loaders.py](src/data_loaders.py).
  - Cora loader ainda não implementado.
- Converte WSG → multi-hot para GCN/GAT: [`wsg_for_gcn_gat_multi_hot`](src/data_converters.py).
- Executa GCN e GAT: [src/models/pytorch_classification/classification_models.py](src/models/pytorch_classification/classification_models.py).

Como executar:
```bash
python run_graph_classification.py
```

Saída em data/output/GRAPH_CLASSIFICATION_RUNS:
- <Dataset>-GCN-GAT__best_test_<...>__model_<...>__<timestamp>/run_report.json

---

## 4) Dados e Formato WSG

O formato Weighted Sparse Graph (WSG) é definido em [wsg_definition.txt](wsg_definition.txt) e implementado em [src/data_format_definition.py](src/data_format_definition.py).

- **Atributos do Grafo:** Os grafos são representados com atributos ponderados, permitindo uma representação esparsa e eficiente.
- **Conversão para WSG:** Use os loaders em [src/data_loaders.py](src/data_loaders.py) para converter datasets brutos para o formato WSG.

---

## 5) Relatórios e Resultados

Os relatórios de execução são gerados automaticamente e salvos nas pastas de saída.

- **Estrutura do Relatório:**
  - Métricas detalhadas (Acurácia, F1-Score, etc.).
  - Parâmetros do modelo e do treinamento.
  - Informações sobre o ambiente de execução.

---

## 6) Ferramentas de Apoio

Ferramentas adicionais estão disponíveis para auxiliar na análise e visualização dos resultados.

- **Visualização de Grafos:** Scripts para visualizar os grafos originais e os embeddings aprendidos.
- **Análise de Resultados:** Ferramentas para comparar o desempenho dos modelos e gerar gráficos de desempenho.

---

## 7) Estrutura do Projeto

```
gnn_tcc/
├── .devcontainer/      # Configurações do Docker e VS Code Dev Container
├── data/
│   ├── datasets/       # Datasets brutos
│   └── output/         # Resultados dos experimentos
├── src/                # Código-fonte principal
│   ├── models/         # Definições dos modelos (GAE, GCN, Sklearn, etc.)
│   ├── config.py       # Configurações centralizadas
│   ├── data_loaders.py # Loaders para carregar datasets para o formato WSG
│   ├── data_converters.py # Conversores do formato WSG para PyTorch Geometric
│   ├── experiment_runner.py # Orquestrador dos pipelines de execução
│   └── ...
├── requirements.txt    # Dependências Python
├── run_embedding_generation.py  # Script para o Fluxo 1
├── run_feature_classification.py # Script para o Fluxo 2
└── run_graph_classification.py   # Script para o Fluxo 3
```

---

## 🧩 Extensão e Personalização

### Adicionando Novos Datasets

1.  Crie uma nova classe que herde de `BaseDatasetLoader` em [`src/data_loaders.py`](src/data_loaders.py) e implemente o método `load` para retornar um objeto `WSG`.
2.  Utilize seu novo loader nos scripts de execução.

### Adicionando Novos Classificadores

-   **Modelos scikit-learn/XGBoost:** Adicione uma nova instância de `SklearnClassifier` ou `XGBoostClassifier` à lista `models_to_run` em [`run_feature_classification.py`](run_feature_classification.py).
-   **Modelos PyTorch:** Crie uma nova classe que herde de `PyTorchClassifier` em `src/models/pytorch_classification/classification_models.py` e implemente sua arquitetura.