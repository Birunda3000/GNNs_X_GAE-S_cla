import torch
import random
import numpy as np
import os
from src.config import Config
import src.data_loaders as data_loaders
from src.classifiers import GCNClassifier, GATClassifier
from src.runner import ExperimentRunner  # <-- Importa a nova classe

wsg_file_paths = ["/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-facebook__loss_2_2581__emb_dim_32__30-10-2025_19-32-36/musae-facebook_(32)_embeddings_epoch_200.wsg.json"]

WSG_DATASET = data_loaders.DirectWSGLoader(file_path=wsg_file_paths[0])



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
        GCNClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),
        GATClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),

    ]

    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="GRAPH_CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=os.path.basename(WSG_DATASET.file_path),
    )
    runner.run(models_to_run, for_embedding_bag=False)


if __name__ == "__main__":
    main()
