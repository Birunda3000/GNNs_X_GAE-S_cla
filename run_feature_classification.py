import torch
import glob
import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.config import Config
import src.data_loaders as data_loaders
from src.classifiers import SklearnClassifier, MLPClassifier, XGBoostClassifier
from src.runner import ExperimentRunner
import xgboost

wsg_file_paths = ["/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-facebook__loss_2_2581__emb_dim_32__30-10-2025_19-32-36/musae-facebook_(32)_embeddings_epoch_200.wsg.json"]

WSG_DATASET = data_loaders.DirectWSGLoader(file_path=wsg_file_paths[0])


def main(wsg_file_path: str):
    # --- 1. Configuração Inicial ---
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    # --- 2. Carregar Dados ---

    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE EMBEDDINGS")
    print(f"Arquivo de entrada: {wsg_file_path}\n", "=" * 65)

    wsg_obj = WSG_DATASET.load()

    # --- 3. Definir Modelos ---
    input_dim = len(wsg_obj.node_features["0"].weights)
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))
    models_to_run = [
        SklearnClassifier(config, model_class=LogisticRegression, max_iter=1000, class_weight='balanced'),
        SklearnClassifier(config, model_class=KNeighborsClassifier, n_neighbors=5),
        SklearnClassifier(config, model_class=RandomForestClassifier, class_weight='balanced'),
        MLPClassifier(
            config, input_dim=input_dim, hidden_dim=128, output_dim=output_dim
        ),
        XGBoostClassifier(config)
    ]

    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=os.path.basename(WSG_DATASET.file_path),
    )
    runner.run(models_to_run, for_embedding_bag=False)


if __name__ == "__main__":
    # Caminhos-base onde procurar os .wsg.json
    base_dirs = [
        "data/output/EMBEDDING_RUNS",
        #"data/output/EMBEDDING_RUNS_epochs(1)",
        #"data/output/t_EMBEDDING_RUNS_epochs(1)",
        #"data/output/t_EMBEDDING_RUNS_epochs(200)",
    ]

    # Encontra todos os arquivos .wsg.json nessas pastas (e subpastas)
    wsg_file_paths = []
    for base in base_dirs:
        wsg_file_paths.extend(glob.glob(os.path.join(base, "**", "*.wsg.json"), recursive=True))

    for wsg_file_path in wsg_file_paths:
        main(wsg_file_path)
