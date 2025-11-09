# TCC-GNN: Framework para Geração e Análise de Embeddings de Grafos

Este repositório contém o código-fonte de um framework desenvolvido para experimentação com Redes Neurais de Grafos (GNNs). O foco principal é a geração de *node embeddings* (representações vetoriais de nós) de forma auto-supervisionada usando modelos como *Graph Autoencoder* (GAE) e *Variational Graph Autoencoder* (VGAE), e a subsequente avaliação da qualidade desses embeddings em tarefas de classificação de nós.

O projeto é construído com ênfase em modularidade, reprodutibilidade e um pipeline de dados bem definido, utilizando um formato de dados customizado chamado **Weighted Sparse Graph (WSG)**.

## Tabela de Conteúdos

1.  [Principais Funcionalidades](#principais-funcionalidades)
2.  [Como Começar](#-como-começar)
    *   [Pré-requisitos](#pré-requisitos)
    *   [Configuração do Ambiente](#configuração-do-ambiente)
3.  [Fluxos de Trabalho](#-fluxos-de-trabalho)
    *   [Fluxo 1: Geração de Embeddings](#fluxo-1-geração-de-embeddings)
    *   [Fluxo 2: Avaliação de Classificadores em Embeddings](#fluxo-2-avaliação-de-classificadores-em-embeddings)
    *   [Fluxo 3: Classificação de Grafo Fim-a-Fim](#fluxo-3-classificação-de-grafo-fim-a-fim)
4.  [Estrutura do Projeto](#-estrutura-do-projeto)
5.  [Extensão e Personalização](#-extensão-e-personalização)
    *   [Adicionando Novos Datasets](#adicionando-novos-datasets)
    *   [Adicionando Novos Classificadores](#adicionando-novos-classificadores)

## Principais Funcionalidades

-   **Formato de Dados Padronizado (WSG):** Define uma especificação (`.wsg.json`) para representar grafos, desacoplando a preparação dos dados da modelagem.
-   **Ambiente Reproduzível com Docker:** Configuração completa para `dev containers` do VS Code, com suporte para ambientes **CPU** e **GPU (NVIDIA)**, garantindo consistência entre máquinas.
-   **Pipeline Modular:**
    1.  **Carregamento de Dados:** Converte datasets brutos (ex: Musae-Github, Musae-Facebook) para o formato WSG.
    2.  **Geração de Embeddings:** Treina modelos GAE/VGAE para aprender representações de nós e salva os embeddings resultantes em um novo arquivo WSG.
    3.  **Avaliação em Tarefas Downstream:** Utiliza os embeddings gerados para treinar e avaliar diversos modelos de classificação (MLP, XGBoost, Sklearn).
-   **Gerenciamento de Experimentos:** Salva automaticamente os resultados de cada execução (modelo treinado, embeddings, logs e métricas) em diretórios nomeados de forma descritiva.

## 🚀 Como Começar

Este projeto foi projetado para ser executado dentro de um ambiente de desenvolvimento em contêiner, o que simplifica a configuração.

### Pré-requisitos

-   [Docker](https://www.docker.com/get-started)
-   [Visual Studio Code](https://code.visualstudio.com/)
-   Extensão [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) para o VS Code.

### Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd gnn_tcc
    ```

2.  **Escolha o ambiente (CPU ou GPU):**
    -   Abra o arquivo `.devcontainer/devcontainer.json`.
    -   Por padrão, ele está configurado para usar o ambiente de CPU. Para usar GPU, comente a linha `docker-compose.cpu.yml` e descomente `docker-compose.gpu.yml`:
        ```json
        // "dockerComposeFile": "docker-compose.cpu.yml",
        "dockerComposeFile": "docker-compose.gpu.yml"
        ```

3.  **Abra o projeto no Dev Container:**
    -   No VS Code, pressione `F1` para abrir a paleta de comandos.
    -   Digite e selecione **"Dev Containers: Reopen in Container"**.
    -   O VS Code irá construir a imagem Docker e iniciar o contêiner. Este processo pode levar alguns minutos na primeira vez.

## ⚙️ Fluxos de Trabalho

O framework oferece três fluxos principais, implementados como scripts separados.

### Fluxo 1: Geração de Embeddings

Este fluxo treina um modelo GAE/VGAE para gerar embeddings de nós.

#### **Configuração**

-   **Dataset:** Altere a variável `WSG_DATASET` no script [`run_embedding_generation.py`](run_embedding_generation.py) para o loader desejado (ex: `MusaeGithubLoader`).
-   **Hiperparâmetros:** Ajuste os parâmetros no arquivo [`src/config.py`](src/config.py) (ex: `OUT_EMBEDDING_DIM`, `EPOCHS`, `LEARNING_RATE`).

#### **Execução**

```bash
python run_embedding_generation.py
```

#### **Saída**

Os resultados são salvos em `data/output/EMBEDDING_RUNS/` com uma estrutura similar a:
```
Musae-Github__score_0_8415__emb_dim_8__09-11-2025_16-18-50/
├── Musae-Github_(8)_embeddings_epoch_500.wsg.json  # Embeddings no formato WSG
├── Musae-Github__GraphSageGAE__09-11-2025_16-18-50.pt # Modelo PyTorch salvo
└── run_report.json                                 # Relatório completo da execução
```

### Fluxo 2: Avaliação de Classificadores em Embeddings

Este fluxo avalia a qualidade dos embeddings gerados usando múltiplos classificadores.

#### **Configuração**

-   **Arquivo de Embeddings:** Edite a variável `wsg_file_paths` em [`run_feature_classification.py`](run_feature_classification.py) para apontar para o arquivo `.wsg.json` gerado no fluxo anterior.

#### **Execução**

```bash
python run_feature_classification.py
```

#### **Saída**

Um relatório (`run_report.json`) é salvo em `data/output/CLASSIFICATION_RUNS/`, contendo métricas detalhadas (Acurácia, F1-Score, tempo de treino, uso de memória) para cada classificador (LogisticRegression, KNN, RandomForest, MLP, XGBoost).

### Fluxo 3: Classificação de Grafo Fim-a-Fim

Este fluxo treina e avalia modelos GNN (GCN, GAT) diretamente nas features originais do grafo.

#### **Configuração**

-   **Dataset:** Altere a variável `WSG_DATASET` no script [`run_graph_classification.py`](run_graph_classification.py).
-   **Hiperparâmetros:** Ajuste os parâmetros em [`src/config.py`](src/config.py).

#### **Execução**

```bash
python run_graph_classification.py
```

#### **Saída**

Os resultados são salvos em `data/output/GRAPH_CLASSIFICATION_RUNS/`, com um relatório (`run_report.json`) comparando o desempenho dos modelos GNN.

## 📂 Estrutura do Projeto

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

## 🧩 Extensão e Personalização

### Adicionando Novos Datasets

1.  Crie uma nova classe que herde de `BaseDatasetLoader` em [`src/data_loaders.py`](src/data_loaders.py) e implemente o método `load` para retornar um objeto `WSG`.
2.  Utilize seu novo loader nos scripts de execução.

### Adicionando Novos Classificadores

-   **Modelos scikit-learn/XGBoost:** Adicione uma nova instância de `SklearnClassifier` ou `XGBoostClassifier` à lista `models_to_run` em [`run_feature_classification.py`](run_feature_classification.py).
-   **Modelos PyTorch:** Crie uma nova classe que herde de `PyTorchClassifier` em `src/models/pytorch_classification/classification_models.py` e implemente sua arquitetura.