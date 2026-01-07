# run_graph_classification_batch.py

import os
import random
import numpy as np
import psutil
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
import time

import src.data_converters as data_converters
import src.data_loaders as data_loaders
from src.config import Config
from src.experiment_runner import ExperimentRunner
from src.models.pytorch_classification.dynamic_gnn import (
    FacebookGNNClassifier,
    GitHubGNNClassifier,
    FacebookEmbeddingGNN,
    GithubEmbeddingGNN
)


def run_graph_classification(WSG_DATASET):
    """
    Executa o pipeline de classificação de grafo para um dataset específico.
    """
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    print("=" * 70)
    print(f"INICIANDO CLASSIFICAÇÃO DE GRAFO PARA DATASET: {WSG_DATASET.dataset_name}")
    print("=" * 70)

    # --- 2. Carregar Dados ---
    wsg_obj = WSG_DATASET.load()

    # --- 3. Definir Modelos ---
    input_dim = wsg_obj.metadata.num_total_features
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    dataset_name = WSG_DATASET.dataset_name.lower()  # Normaliza para minúsculas

    if "facebook" in dataset_name:
        print(
            f">> Dataset identificado: MUSAE-Facebook. Carregando parâmetros otimizados."
        )
        models_to_run = [
            #FacebookGNNClassifier(config,input_dim=input_dim,output_dim=output_dim)
            FacebookEmbeddingGNN(config=config, num_total_features=input_dim, output_dim=output_dim)
        ]
    elif "github" in dataset_name:
        print(
            f">> Dataset identificado: MUSAE-GitHub. Carregando parâmetros otimizados."
        )
        models_to_run = [
            #GitHubGNNClassifier(config, input_dim=input_dim, output_dim=output_dim)
            GithubEmbeddingGNN(config=config, num_total_features=input_dim, output_dim=output_dim)
        ]
    else:
        raise ValueError(
            f"Dataset não reconhecido para classificação de grafo: '{WSG_DATASET.dataset_name}'"
        )

    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="GRAPH_CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=WSG_DATASET.dataset_name,
        #data_converter=data_converters.wsg_for_gcn_gat_multi_hot,
        data_converter=data_converters.wsg_for_vgae,
    )

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss

    runner.run(models_to_run, process=process, mem_start=mem_start)

    print(f"\nConcluído: {WSG_DATASET.dataset_name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    datasets = [
        data_loaders.MusaeFacebookLoader(),
        data_loaders.MusaeGithubLoader(),
    ]

    for dataset in datasets:
        run_graph_classification(dataset)
        print("❄️  Pausa de 10s para resfriamento da CPU...")
        time.sleep(10)
