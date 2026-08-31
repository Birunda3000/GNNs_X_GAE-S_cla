

### 🐳 FASE 1: Fundação, Governança e Configuração (Infraestrutura MLOps)

* [x] **1. Estabilização do Container (Dockerfile):** Inserção do comando de atualização do `pip` nativo e instalação limpa das bibliotecas de vanguarda (`mlflow`, `hydra-core`, `dvc`, `safetensors`, `ruff`). Isso previne falhas de empacotamento com os *kernels* da NVIDIA.
* [x] **2. Governança de Código com Ruff:** Criação do arquivo `pyproject.toml` para ativar o linter ultraveloz. Ele padronizará o código e removerá imports fantasmas instantaneamente ao salvar os arquivos no VS Code.
* [ ] **3. Migração Dinâmica para Hydra:** Exclusão da classe estática `Config` (`src/config.py`). Transferência de todas as variáveis de hiperparâmetros para a pasta `conf/*.yaml`, destrancando a capacidade de alterar dimensões e taxas de aprendizado direto pela linha de comando.
* [ ] **4. Orquestração Acíclica com DVC:** Criação do `dvc.yaml`. Definição de que o script de classificação só rodará se o script de embeddings finalizar com sucesso. O DVC fará o cache das matrizes, evitando reprocessar a Fase 1 à toa.

---

### 📊 FASE 2: Telemetria, Loops Otimizados e Dados Extremos

* [ ] **5. Desacoplamento e Múltiplos Critérios no EarlyStopper:** Refatoração da classe de parada antecipada. Injeção de um período de *warm-up* (ex: ignorar as 5 primeiras épocas) e suporte a checagens combinadas (parar o treino se a função F1 estagnar e a função Loss explodir simultaneamente).
* [ ] **6. Rastreamento Visual com MLflow:** Remoção do arcaico `ReportManager` e do salvamento manual de arquivos JSON/pastas longas. Implementação de `mlflow.log_metrics()` e `mlflow.log_params()`, com mapeamento da porta 5000 para visualização de gráficos e consumo de memória em tempo real no navegador.
* [ ] **7. Serialização Extrema via Safetensors / Parquet:** Remoção da função `save_embeddings_to_wsg` baseada em JSON. A matriz latente $Z$ será exportada em formato binário de memória mapeada (mmap), garantindo que um monstro topológico como o Reddit carregue instantaneamente na memória RAM sem causar estouro (OOM) na fase de inferência.

---

### 🧠 FASE 3: A Fronteira Matemática (O Cérebro do Framework)

* [ ] **8. Fábrica de Amostragem Dinâmica (Resolvendo o paradoxo das classes):** Criação de uma inteligência no orquestrador. Tarefas de previsão de conexões (Autoencoder) invocarão automaticamente o `LinkNeighborLoader` (amostrando arestas). Tarefas de classificação (End-to-End) invocarão o `CuGraphNeighborLoader` (amostrando nós com balanceamento estatístico perfeito).
* [ ] **9. Autoencoder de Reconstrução Dupla (Topologia + Features):** Refatoração da classe `BaseGAECommon`. Adição de uma segunda cabeça de rede neural (Decoder de Features) responsável por pegar os embeddings $Z$ e tentar recriar a matriz $X$ original via erro quadrático médio (MSE). A rede será punida se errar a vizinhança E se errar as características do nó.
* [ ] **10. Suporte Universal a Atributos de Arestas:** Injeção do argumento `edge_attr=None` na base de todas as propagações matemáticas (`forward` e `encode`). Isso garante que o seu *framework* estará silenciosamente preparado para ingerir grafos transacionais ou temporais no futuro, sem necessidade de reescrever a base do projeto.

---