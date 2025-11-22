from sqlite3 import Time
import sys
import os
import torch
import numpy as np
import random
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.model_selection import ParameterGrid
from typing import Dict, Any, List
import torch.nn as nn

from src.config import Config
from src.data_loaders import MusaeGithubLoader, MusaeFacebookLoader
from src.models.embedding_models.din_gae import DynamicGAE, DynamicVGAE
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.early_stopper import EarlyStopper
from src.embeddings_eval import evaluate_embeddings
import src.data_converters as data_converters
import src.grid_search.grid_search_params as grids
from torch.optim import lr_scheduler

def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Converte classes e tipos em strings para serialização JSON."""
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, type):
            clean_params[k] = v.__name__
        else:
            clean_params[k] = v
    return clean_params


def run_grid_search(WSG_DATASET: Any, config: Config, model_class: Any):
    # Configuração inicial
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )
    device = torch.device(config.DEVICE)

    # Seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"Dispositivo: {device}")
    print(f"Dataset: {WSG_DATASET.dataset_name}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1] Carregando dados...")
    wsg_obj = WSG_DATASET.load()

    print("\n[FASE 2] Convertendo para formato Pytorch Geometric...")
    pyg_data = data_converters.wsg_for_vgae(wsg_obj, config)
    pyg_data = pyg_data.to(device)  # Mover dados para GPU uma vez

    print("\n[FASE 3] Iniciando busca em grid...")

    # Seleciona o grid correto
    grid_source = grids.GAE_VGAE_GRID
    param_grid_list = list(ParameterGrid(grid_source))
    total_combinations = len(param_grid_list)

    print(f"Total de combinações encontradas: {total_combinations}\n")

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name=f"GRID_SEARCH_RUNS/GAEs/{model_class.__name__}/{WSG_DATASET.dataset_name}",
    )
    report_manager = ReportManager(directory_manager)
    report = {}
    report["Dataset"] = WSG_DATASET.dataset_name
    report["Timestamp"] = config.TIMESTAMP
    report["Model_Type"] = model_class.__name__
    report["Fixed_Config"] = grids.TRAINING_CONFIG

    results_list = []
    training_report_list = []

    print("\nIniciando iteração sobre as combinações de hiperparâmetros...\n")

    for i, params in enumerate(param_grid_list, 1):
        print(f"--- Combinação {i}/{total_combinations} ---")
        print(sanitize_params(params))

        train_report = {
            "params": sanitize_params(params),
            "best_score": None,
            "training_log": [],
        }

        model = model_class(
            config=config,
            num_total_features=pyg_data.num_total_features,
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["hidden_dim"],
            out_embedding_dim=params["out_embedding_dim"],
            layer_type=params["layer_type"],
            num_layers=params["num_layers"],
            activation=params["activation"],
            dropout=params["dropout"],
            normalize_embeddings=params["normalize_embeddings"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=grids.TRAINING_CONFIG["learning_rate"],
            weight_decay=grids.TRAINING_CONFIG["weight_decay"],
        )

        early_stopper = EarlyStopper(
            patience=grids.TRAINING_CONFIG["early_stopping_patience"],
            min_delta=grids.TRAINING_CONFIG["early_stopping_min_delta"],
            mode="max",  # Maximizando métrica de avaliação de embedding
            metric_name="max_val_f1",
            custom_eval=lambda m: evaluate_embeddings(m, pyg_data, device),
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
            min_lr=config.MIN_LR,
        )
        # Loop de Treinamento
        training_report = model.train_model(
            data=pyg_data,
            optimizer=optimizer,
            epochs=grids.TRAINING_CONFIG["epochs"],
            early_stopper=early_stopper,
            scheduler=scheduler,
        )

        training_report_list.append(training_report)
        results_list.append({
            "params": sanitize_params(params),
            "best_score": training_report["best_score"],
        })



        if "model" in locals():
            del model
        if "optimizer" in locals():
            del optimizer
        if "early_stopper" in locals():
            del early_stopper
        if "scheduler" in locals():
            del scheduler
        torch.cuda.empty_cache()

        print(f"-Best Score desta combinação: {training_report['best_score']}\n")


    results_list.sort(key=lambda x: x.get("best_score", 0), reverse=True)
    report["Results"] = results_list
    report["Training_Reports"] = training_report_list

    report_manager.create_report(report)
    report_manager.save_report()

    best_run = results_list[0] if results_list else {}
    
    metrics_to_name = {
        "score": best_run.get("best_score", 0),
        "best_params": best_run.get("params", "NA")
    }
    final_path = directory_manager.finalize_run_directory(
        dataset_name=WSG_DATASET.dataset_name,
        metrics=metrics_to_name,
    )

    print(f"\nGrid Search concluído. Relatório salvo em {directory_manager.get_run_path()}")


if __name__ == "__main__":
    config = Config()

    # Exemplo de execução
    dataset = MusaeGithubLoader()

    # Rodar para GAE
    run_grid_search(dataset, config, model_class=DynamicGAE)

    # Rodar para VGAE
    # run_grid_search(dataset, config, emb_dim=128, model_class=DynamicVGAE)
