import os
import glob
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
import random
import torch

# Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Imports do Projeto
import directory_manager
from report_manager import ReportManager
from src.config import Config
from src.data_loaders import DirectWSGLoader
import src.grid_search.grid_search_params as grids
import src.data_converters as data_converters
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager

def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Converte classes e tipos em strings para serialização JSON."""
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, type):
            clean_params[k] = v.__name__
        else:
            clean_params[k] = v
    return clean_params


def run_feature_grid(WSG_DATASET: Any, config: Config):
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )
    device = torch.device(config.DEVICE)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"Dispositivo: {device}")
    print(f"Dataset: {WSG_DATASET.dataset_name}")


    print("\n[FASE 1] Carregando dados...")
    wsg_obj = WSG_DATASET.load()

    print("\n[FASE 2] Convertendo dados para classificador denso...")

    data = data_converters.wsg_for_dense_classifier(
        wsg=wsg_obj,
        config=config,
        train_size_ratio=config.TRAIN_SPLIT_RATIO,
    )
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    X_train, y_train = X[data.train_mask], y[data.train_mask]
    X_val, y_val = X[data.val_mask], y[data.val_mask]
    X_test, y_test = X[data.test_mask], y[data.test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = len(wsg_obj.node_features["0"].weights)
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    print(f"   Input Dim: {input_dim}")
    print(f"   Classes (Output): {output_dim}")

    print("\n[FASE 3] Iniciando Grid Search para Classificadores...")

    knn_grid_source = grids.CLASSIFIER_GRIDS["KNN"]
    knn_param_grid_list = list(ParameterGrid(knn_grid_source))
    rf_grid_source = grids.CLASSIFIER_GRIDS["RandomForest"]
    rf_param_grid_list = list(ParameterGrid(rf_grid_source))
    lr_grid_source = grids.CLASSIFIER_GRIDS["LogisticRegression"]
    lr_param_grid_list = list(ParameterGrid(lr_grid_source))


    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name=f"GRID_SEARCH_RUNS/MLs/{WSG_DATASET.dataset_name}"
    )
    report_manager = ReportManager(directory_manager)
    report = {}
    report["Dataset"] = WSG_DATASET.dataset_name
    report["Timestamp"] = config.TIMESTAMP
    report["Model_Type"] = "Dense_Classifiers_1"
    report["Fixed_Config"] = grids.TRAINING_CONFIG









    # continuar a partir daqui?


if __name__ == "__main__":
    config = Config()
    wsg_file_path = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Flickr__score_0_3365__emb_dim_8__19-11-2025_19-00-11/Flickr_(8)_embeddings_19-11-2025_19-00-11.wsg.json"

    WSG_DATASET = DirectWSGLoader(file_path=wsg_file_path)