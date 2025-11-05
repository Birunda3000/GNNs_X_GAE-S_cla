# TCC-GNN: Framework para Geração e Análise de Embeddings de Grafos

Este repositório contém o código-fonte de um framework desenvolvido para experimentação com Redes Neurais de Grafos (GNNs). O foco principal é a geração de *node embeddings* (representações vetoriais de nós) de forma auto-supervisionada usando um *Variational Graph Autoencoder* (VGAE) e a subsequente avaliação da qualidade desses embeddings em tarefas de classificação de nós.

O projeto é construído com ênfase em modularidade, reprodutibilidade e um pipeline de dados bem definido, utilizando um formato de dados customizado chamado **Weighted Sparse Graph (WSG)**.

## Tabela de Conteúdos

1.  [Principais Funcionalidades](https://www.google.com/search?q=%23principais-funcionalidades)
2.  [Como Começar](https://www.google.com/search?q=%23-como-come%C3%A7ar)
      * [Pré-requisitos](https://www.google.com/search?q=%23pr%C3%A9-requisitos)
      * [Configuração do Ambiente](https://www.google.com/search?q=%23configura%C3%A7%C3%A3o-do-ambiente)
3.  [Fluxos de Trabalho](https://www.google.com/search?q=%23-fluxos-de-trabalho)
      * [Fluxo 1: Geração de Embeddings](https://www.google.com/search?q=%23fluxo-1-gera%C3%A7%C3%A3o-de-embeddings)
      * [Fluxo 2: Avaliação dos Embeddings](https://www.google.com/search?q=%23fluxo-2-avalia%C3%A7%C3%A3o-dos-embeddings)
      * [Exemplo de Experimento Completo](https://www.google.com/search?q=%23exemplo-de-experimento-completo)
4.  [Estrutura do Projeto](https://www.google.com/search?q=%23estrutura-do-projeto)
5.  [Extensão e Personalização](https://www.google.com/search?q=%23-extens%C3%A3o-e-personaliza%C3%A7%C3%A3o)
      * [Adicionando Novos Datasets](https://www.google.com/search?q=%23adicionando-novos-datasets)
      * [Adicionando Novos Classificadores](https://www.google.com/search?q=%23adicionando-novos-classificadores)
6.  [Nota sobre Anonimização](https://www.google.com/search?q=%23-nota-sobre-anonimiza%C3%A7%C3%A3o)
7.  [Contribuições](https://www.google.com/search?q=%23contribui%C3%A7%C3%B5es)
8.  [Licença](https://www.google.com/search?q=%23licen%C3%A7a)
9.  [Contato](https://www.google.com/search?q=%23contato)

## Principais Funcionalidades

  - **Formato de Dados Padronizado (WSG):** Define uma especificação (`.wsg.json`) para representar grafos de maneira universal, desacoplando a preparação dos dados da modelagem.
  - **Ambiente Reproduzível com Docker:** Configuração completa para `dev containers` do VS Code, com suporte para ambientes **CPU** e **GPU (NVIDIA)**, garantindo que o projeto funcione de forma consistente em diferentes máquinas.
  - **Pipeline Modular:**
    1.  **Carregamento de Dados:** Converte datasets brutos (ex: Musae-Github) para o formato WSG.
    2.  **Geração de Embeddings:** Treina um modelo VGAE para aprender representações de nós e salva os embeddings resultantes em um novo arquivo WSG.
    3.  **Avaliação em Tarefas Downstream:** Utiliza os embeddings gerados para treinar e avaliar modelos de classificação (MLP, Regressão Logística, etc.).
  - **Geração de Embeddings com VGAE:** Implementação de um *Variational Graph Autoencoder* em PyTorch Geometric para aprender embeddings de nós de forma auto-supervisionada.
  - **Gerenciamento de Experimentos:** Salva automaticamente os resultados de cada execução (modelo treinado, embeddings, logs e métricas) em diretórios nomeados de forma descritiva.
  - **Anonimização de Features:** Inclui um script para aplicar privacidade diferencial (mecanismo de Laplace) às features dos nós e avaliar o impacto na utilidade dos dados para tarefas de classificação.

## 🚀 Como Começar

Este projeto foi projetado para ser executado dentro de um ambiente de desenvolvimento em contêiner, o que simplifica a configuração.

### Pré-requisitos

  - [Docker](https://www.docker.com/get-started)
  - [Visual Studio Code](https://code.visualstudio.com/)
  - Extensão [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) para o VS Code.

### Configuração do Ambiente

1.  **Clone o repositório:**

    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd gnn_tcc
    ```

2.  **Escolha o ambiente (CPU ou GPU):**

      - Abra o arquivo `.devcontainer/devcontainer.json`.
      - Por padrão, ele está configurado para usar o ambiente de CPU:
        ```json
        "dockerComposeFile": "docker-compose.cpu.yml"
        ```
      - Se você possui uma GPU NVIDIA e os drivers corretos instalados, pode mudar para o ambiente de GPU, comentando uma linha e descomentando a outra:
        ```json
        // "dockerComposeFile": "docker-compose.cpu.yml",
        "dockerComposeFile": "docker-compose.gpu.yml"
        ```

3.  **Abra o projeto no Dev Container:**

      - No VS Code, pressione `F1` para abrir a paleta de comandos.
      - Digite e selecione **"Dev Containers: Reopen in Container"**.
      - O VS Code irá construir a imagem Docker e iniciar o contêiner. Este processo pode levar alguns minutos na primeira vez.

## ⚙️ Fluxos de Trabalho

O framework oferece dois fluxos principais de trabalho, implementados como scripts separados.

### Fluxo 1: Geração de Embeddings

Este fluxo treina um modelo VGAE em um dataset de grafo para gerar embeddings densos que capturam tanto as características dos nós quanto a estrutura da rede.

#### **Configuração**

1.  **Ajuste os parâmetros no arquivo `src/config.py`:**
      * `DATASET_NAME`: Define o dataset a ser usado (ex: "musae-github", "cora").
      * `EMBEDDING_DIM`: Dimensão do embedding inicial das features.
      * `HIDDEN_DIM`: Dimensão da camada oculta do GNN.
      * `OUT_EMBEDDING_DIM`: Dimensão final dos embeddings gerados.
      * `EPOCHS`: Número de épocas de treinamento.
      * `LEARNING_RATE`: Taxa de aprendizado para o otimizador.

#### **Execução**

```bash
python run_embedding_generation.py
```

#### **O que acontece:**

1.  **Carregamento dos dados:** O script carrega o dataset especificado em `DATASET_NAME`.
2.  **Conversão de formato:** Os dados são convertidos para o formato do PyTorch Geometric.
3.  **Treinamento do modelo:** Um VGAE é treinado para reconstruir as arestas do grafo.
4.  **Extração de embeddings:** Os nós são codificados no espaço latente do autoencoder.
5.  **Salvamento dos resultados:** Os embeddings, o modelo treinado e um relatório são salvos em uma pasta em `data/output/EMBEDDING_RUNS/`, seguindo o padrão:
    ```
    [dataset]__loss_[valor]__emb_dim_[dimensão]__[timestamp]/
    ├── [dataset]_embeddings.wsg.json  # Embeddings no formato WSG
    ├── vgae_model.pt                  # Modelo treinado
    └── run_summary.txt                # Relatório com métricas e tempos
    ```

### Fluxo 2: Avaliação dos Embeddings

Este fluxo avalia a qualidade dos embeddings gerados usando múltiplos classificadores e métricas.

#### **Configuração**

1.  **Especifique o arquivo de embeddings a ser avaliado:**
      * Abra o script `run_feature_classification.py`.
      * Edite a variável `wsg_file_path` para apontar para o arquivo `.wsg.json` que você deseja avaliar (gerado no fluxo anterior).

#### **Execução**

```bash
python run_feature_classification.py
```

#### **O que acontece:**

1.  **Carregamento dos embeddings:** O arquivo WSG especificado é carregado.
2.  **Treinamento de classificadores:** Quatro modelos são treinados e avaliados:
      * **LogisticRegression:** Testa a separabilidade linear das classes.
      * **KNeighborsClassifier (KNN):** Testa a coesão local dos embeddings.
      * **RandomForestClassifier:** Testa relações não-lineares complexas.
      * **MLPClassifier:** Testa a capacidade de redes neurais de utilizar os embeddings.
3.  **Geração de Relatórios:** Os resultados são compilados, exibidos no console e salvos em um arquivo.

#### **Saída:**

1.  **Relatório no console:** Uma tabela comparativa é exibida no terminal:
    ```
    RELATÓRIO DE COMPARAÇÃO FINAL
    -------------------------------------------------------------
    Fonte dos Dados: musae-github_embeddings.wsg.json
    Tipo de Feature: dense_continuous
    -------------------------------------------------------------
    Modelo                    | Acurácia     | F1-Score     | Tempo (s) 
    =================================================================
    LogisticRegression        | 0.8491       | 0.8398       | 0.43      
    KNeighborsClassifier      | 0.8354       | 0.8308       | 0.01      
    RandomForestClassifier    | 0.8481       | 0.8419       | 13.00     
    MLPClassifier             | 0.8534       | 0.8476       | 23.41     
    ```
2.  **Relatório detalhado em JSON:** Um arquivo JSON com os resultados é salvo em `data/output/CLASSIFICATION_RUNS/`.

### Exemplo de Experimento Completo

Para uma avaliação completa, o processo ideal é:

1.  **Gere embeddings com diferentes dimensões:**

      * Altere `OUT_EMBEDDING_DIM` em `src/config.py` (ex: 16, 32, 64, 128).
      * Execute `run_embedding_generation.py` para cada configuração.

2.  **Avalie cada conjunto de embeddings:**

      * Para cada arquivo de embedding gerado, atualize o `wsg_file_path` em `run_feature_classification.py`.
      * Execute o script de classificação.

3.  **Compare os resultados:**

      * Analise os relatórios para encontrar o melhor equilíbrio entre **dimensionalidade**, **acurácia** e **tempo de treinamento**.

## Estrutura do Projeto

```
/app/gnn_tcc/
├── .devcontainer/         # Configurações do Docker e VS Code Dev Container (CPU/GPU)
├── data/
│   ├── datasets/          # Datasets brutos (ex: musae-github)
│   └── output/            # Diretório para salvar os resultados dos experimentos
├── src/                   # Código-fonte principal do projeto
│   ├── anon.py            # Script para anonimização de dados
│   ├── classifiers.py     # Definição de modelos de classificação (MLP, GCN)
│   ├── config.py          # Configurações centralizadas do projeto
│   ├── data_converter.py  # Conversor do formato WSG para PyTorch Geometric
│   ├── data_format_definition.py # Definição Pydantic do formato WSG
│   ├── data_loader.py     # Loaders para carregar datasets para o formato WSG
│   ├── directory_manager.py # Gerenciador de diretórios de saída
│   ├── model.py           # Definição do modelo VGAE
│   ├── train.py           # Lógica de treinamento do VGAE
│   └── train_loop.py      # Lógica de treinamento para os classificadores
├── requirements.txt       # Dependências Python
├── run_embedding_generation.py # Script principal para gerar embeddings
└── run_feature_classification.py # Script principal para avaliar classificadores
```

## 🧩 Extensão e Personalização

### Adicionando Novos Datasets

1.  Crie uma nova classe loader em `src/data_loader.py`:
    ```python
    class NewDatasetLoader(BaseDatasetLoader):
        def load(self) -> WSG:
            # Implementação da lógica de carregamento
            ...
    ```
2.  Registre-o na função `get_loader` no mesmo arquivo:
    ```python
    def get_loader(dataset_name: str) -> BaseDatasetLoader:
        loaders = {
            # ...outros loaders
            "new-dataset": NewDatasetLoader(),
        }
    ```

### Adicionando Novos Classificadores

1.  **Modelos scikit-learn:** Basta adicionar uma nova instância à lista `models_to_run` em `run_feature_classification.py`:
    ```python
    models_to_run = [
        # ...outros modelos
        SklearnClassifier(config, model_class=NovoClassificador, **params),
    ]
    ```
2.  **Modelos PyTorch:** Crie uma nova classe que herde de `PyTorchClassifier` em `src/classifiers.py` e implemente sua arquitetura.

## 📝 Nota sobre Anonimização

O arquivo `src/anon.py` contém uma funcionalidade experimental para aplicar privacidade diferencial (mecanismo de Laplace) aos embeddings. Ele permite análises sobre o trade-off entre privacidade e utilidade, mas não está integrado ao fluxo principal de trabalho.

## Contribuições

Contribuições são bem-vindas\! Sinta-se à vontade para abrir *issues* ou enviar *pull requests*.
