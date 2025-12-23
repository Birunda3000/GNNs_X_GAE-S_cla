"""
Script de classificação de embeddings gerados por modelos GNN (ex.: VGAE).
Executa múltiplos classificadores (Sklearn, MLP, XGBoost) sobre embeddings salvos em formato WSG.
"""

# === IMPORTS PADRÃO ===
import os
import glob
import random
from datetime import datetime
from zoneinfo import ZoneInfo

# === IMPORTS DE TERCEIROS ===
from requests import get
import torch
import numpy as np
import psutil
import gc

# Sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # <--- Otimizado (Gaussiano)
from sklearn.neural_network import MLPClassifier

# XGBoost (interface sklearn)
from xgboost import XGBClassifier

# === IMPORTS INTERNOS DO PROJETO ===
from src.config import Config
import src.data_loaders as data_loaders
import src.data_converters as data_converters
from src.models.sklearn_model import SklearnClassifier
from src.experiment_runner import ExperimentRunner
from src.model_args import get_github_models, get_facebook_models


def main(wsg_file_path: str):
    # --- 1. Configuração Inicial ---
    config = Config()
    
    # Garante reprodutibilidade
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )
    
    # Extrai parâmetros de consistência do Config
    PATIENCE = config.EARLY_STOPPING_PATIENCE
    MIN_DELTA = config.EARLY_STOPPING_MIN_DELTA

    # --- 2. Carregar Dados ---
    print("=" * 65, "\nINICIANDO TAREFA DE CLASSIFICAÇÃO DE EMBEDDINGS")
    print(f"Arquivo de entrada: {wsg_file_path}\n", "=" * 65)

    WSG_DATASET = data_loaders.DirectWSGLoader(file_path=wsg_file_path)
    wsg_obj = WSG_DATASET.load()




    # --- 3. Definir Modelos (A Lista de Ouro do TCC) ---
    # --- 3. Definir Modelos (Seleção Dinâmica) ---
    filename_lower = os.path.basename(wsg_file_path).lower()

    if "github" in filename_lower:
        print(f">> Dataset identificado: MUSAE-GitHub. Carregando parâmetros otimizados.")
        models_to_run = get_github_models(config=config)
    elif "facebook" in filename_lower:
        print(f">> Dataset identificado: MUSAE-Facebook. Carregando parâmetros otimizados.")
        models_to_run = get_facebook_models(config=config)
    else:
        # Fallback de segurança (ou lançar erro)
        print("!! AVISO: Dataset não identificado no nome. Usando GitHub como padrão.")
        models_to_run = get_github_models(config=config)




    # --- 4. Executar o Experimento ---
    runner = ExperimentRunner(
        config=config,
        run_folder_name="CLASSIFICATION_RUNS",
        wsg_obj=wsg_obj,
        data_source_name=os.path.basename(WSG_DATASET.file_path),
        data_converter=data_converters.wsg_for_dense_classifier,
    )

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss

    runner.run(models_to_run, process=process, mem_start=mem_start)


if __name__ == "__main__":
    base_path = "data/output/EMBEDDING_RUNS"
    pattern = "*.wsg.json"
    list_of_files = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)

    print(f"Encontrados {len(list_of_files)} arquivos para processar.")

    for file_path in list_of_files:
        print(f"\n=== Rodando para: {file_path} ===")
        try:
            main(file_path)
        except Exception as e:
            print(f"Erro ao processar {file_path}: {e}")
        gc.collect()