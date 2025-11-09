import os
import random

import numpy as np
import psutil
import torch

import src.data_converters as data_converters
import src.data_loaders as data_loaders
from src.config import Config
from src.experiment_runner import ExperimentRunner
from src.models.pytorch_classification.classification_models import GCNClassifier, GATClassifier


WSG_DATASET = data_loaders.MusaeGithubLoader()



def main():
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE GRAFO (FIM-A-FIM)")
    print(f"Dataset de entrada: {WSG_DATASET.dataset_name}\n", "=" * 65)

    # --- 2. Carregar Dados ---
    wsg_obj = WSG_DATASET.load()

    # --- 3. Definir Modelos ---
    input_dim = wsg_obj.metadata.num_total_features
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    models_to_run = [
        GCNClassifier(config, input_dim=input_dim, hidden_dim=config.HIDDEN_DIM, output_dim=output_dim),
        GATClassifier(config, input_dim=input_dim, hidden_dim=config.HIDDEN_DIM, output_dim=output_dim),
    ]

    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="GRAPH_CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=os.path.basename(WSG_DATASET.dataset_name),
        data_converter=data_converters.wsg_for_gcn_gat_multi_hot
    )

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss

    runner.run(models_to_run, process=process, mem_start=mem_start)


if __name__ == "__main__":
    main()
