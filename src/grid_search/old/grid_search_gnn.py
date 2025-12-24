import sys
import os
import torch
import numpy as np
import random
import gc
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.model_selection import ParameterGrid
from typing import Dict, Any, List

# --- Imports do Projeto ---
from src.config import Config
from src.data_loaders import MusaeGithubLoader, MusaeFacebookLoader
# Importa o modelo dinâmico que acabamos de criar
from src.models.pytorch_classification.dynamic_gnn import DynamicGNNClassifier
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.early_stopper import EarlyStopper
import src.data_converters as data_converters
import src.grid_search.grid_search_params as grids
import torch.optim as optim
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


def run_baseline_grid(WSG_DATASET: Any, config: Config):
    # Configurações Iniciais
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )
    device = torch.device(config.DEVICE)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"Dispositivo: {device}")
    print(f"Dataset: {WSG_DATASET.dataset_name}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1] Carregando dados...")
    wsg_obj = WSG_DATASET.load()

    print("\n[FASE 2] Convertendo para formato PyTorch Geometric (Supervisionado)...")

    pyg_data = data_converters.wsg_for_gcn_gat_multi_hot(wsg_obj, config)
    pyg_data = pyg_data.to(device)

    # Calcula dimensões automaticamente
    input_dim = wsg_obj.metadata.num_total_features
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))

    print(f"   Input Dim: {input_dim}")
    print(f"   Classes (Output): {output_dim}")

    print("\n[FASE 3] Iniciando Grid Search Supervisionado (Baseline)...")

    # Carrega Grid de Classificadores GNN
    grid_source = grids.GNN_CLASSIFIER_GRID
    param_grid_list = list(ParameterGrid(grid_source))
    total_combinations = len(param_grid_list)

    print(f"Total de combinações encontradas: {total_combinations}\n")

    # Gerenciadores
    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name=f"GRID_SEARCH_RUNS/GNNs/{WSG_DATASET.dataset_name}",
    )
    report_manager = ReportManager(directory_manager)
    report = {}
    report["Dataset"] = WSG_DATASET.dataset_name
    report["Timestamp"] = config.TIMESTAMP
    report["Model_Type"] = "DynamicGNNClassifier"
    report["Fixed_Config"] = grids.TRAINING_CONFIG


    results_list = []
    training_report_list = []

    print("\nIniciando iteração sobre as combinações de hiperparâmetros...\n")

    # --- Loop de Execução ---
    for i, params in enumerate(param_grid_list, 1):
        print(f"--- Combinação {i}/{total_combinations} ---")
        print(f"--- Dataset: {WSG_DATASET.dataset_name} ---")
        print(f"--- Params: {sanitize_params(params)} ---")

        # 1. Instancia o Modelo Dinâmico
        # Injeta input_dim/output_dim e desempacota params do grid
        model = DynamicGNNClassifier(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=params["layer_type"],
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            activation=params["activation"],
            heads=params.get("heads", 1),  # Padrão 1 se não fornecido
        ).to(device)

        # 2. Configura Otimizador
        optimizer = optim.Adam(
            model.parameters(),
            lr=grids.TRAINING_CONFIG["learning_rate"],
            weight_decay=grids.TRAINING_CONFIG["weight_decay"],
        )

        # 3. Configura Early Stopper
        # Aqui usamos "val_f1" que é calculado internamente pelo PyTorchClassifier.evaluate
        early_stopper = EarlyStopper(
            patience=grids.TRAINING_CONFIG["early_stopping_patience"],
            min_delta=grids.TRAINING_CONFIG["early_stopping_min_delta"],
            mode="max",
            metric_name="val_f1", 
        )

        # 4. Configura Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
            min_lr=config.MIN_LR,
        )

        # 5. Executa Treino
        # O método train_model da classe PyTorchClassifier já gerencia o loop e avaliação
        training_report = model.train_model(
            data=pyg_data,
            optimizer=optimizer,
            epochs=grids.TRAINING_CONFIG["epochs"],
            early_stopper=early_stopper,
            scheduler=scheduler,
            criterion=torch.nn.CrossEntropyLoss()
        )

        # 6. Registra Resultados
        # ✅ CORREÇÃO: Extrair métrica de validação para decisão
        val_f1 = training_report["val_report"]["weighted avg"]["f1-score"]
        
        training_report_list.append(training_report)
        results_list.append({
            "params": sanitize_params(params),
            "val_f1": round(val_f1, 6),       # Critério de escolha
            "test_f1": round(training_report["test_f1"], 6) # Apenas informativo
        })

        # Limpeza
        del model, optimizer, scheduler, early_stopper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --- Finalização ---
    # ✅ CORREÇÃO: Ordenar pela métrica de VALIDAÇÃO (Evita Overfitting no Teste)
    results_list.sort(key=lambda x: x["val_f1"], reverse=True)
    
    report["Results"] = results_list
    report["Training_Reports"] = training_report_list

    report_manager.create_report(report)
    report_manager.save_report()

    # Renomeia pasta com o melhor resultado
    best_run = results_list[0] if results_list else {}

    metrics_to_name = {
        "test_f1": best_run["test_f1"],
        "best_params": best_run["params"]
    }
    
    directory_manager.finalize_run_directory(
        dataset_name=WSG_DATASET.dataset_name,
        metrics=metrics_to_name,
    )
    print(f"\nGrid Search Baseline concluído. Relatório salvo em {directory_manager.get_run_path()}")

if __name__ == "__main__":
    config = Config()
    
    # Dataset para teste
    #dataset = MusaeGithubLoader()

    #run_baseline_grid(dataset, config)

    dataset = MusaeFacebookLoader()

    run_baseline_grid(dataset, config)
