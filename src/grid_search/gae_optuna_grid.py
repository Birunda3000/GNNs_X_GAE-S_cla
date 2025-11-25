import sys
import os
import torch
import numpy as np
import random
import gc
import optuna
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.optim import lr_scheduler
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Any

# Imports do projeto
from src.config import Config
from src.data_loaders import MusaeGithubLoader, MusaeFacebookLoader
from src.models.pytorch_classification.dynamic_gnn import DynamicGNNClassifier
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.early_stopper import EarlyStopper
import src.data_converters as data_converters
import src.grid_search.grid_search_params as grids

# ============================================================
# FUNÇÃO OBJETIVO (GNN CLASSIFIER / BASELINE)
# ============================================================

def objective(trial, pyg_data, config, input_dim, output_dim, dataset_name):
    
    # 1. DEFINIÇÃO DO ESPAÇO DE BUSCA
    # ===============================
    
    # Escolha da Arquitetura
    layer_type_str = trial.suggest_categorical("layer_type", ["SAGEConv", "GCNConv", "GATConv"])
    
    # Mapeamento String -> Classe
    if layer_type_str == "SAGEConv":
        layer_type = SAGEConv
    elif layer_type_str == "GCNConv":
        layer_type = GCNConv
    else:
        layer_type = GATConv

    # Hiperparâmetros Gerais
    num_layers = trial.suggest_int("num_layers", 2, 3) 
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
    
    # Função de Ativação
    activation_str = trial.suggest_categorical("activation", ["ReLU", "ELU", "LeakyReLU"])
    if activation_str == "ReLU": activation = nn.ReLU
    elif activation_str == "ELU": activation = nn.ELU
    else: activation = nn.LeakyReLU

    # Parâmetros Específicos (GAT) - Heads
    heads = 1
    if layer_type_str == "GATConv":
        heads = trial.suggest_int("heads", 1, 4)
    
    # Learning Rate Dinâmico
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    device = torch.device(config.DEVICE)

    # 2. GERENCIAMENTO DE DIRETÓRIOS
    # ==============================
    directory_manager = DirectoryManager(
        timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        run_folder_name=f"OPTUNA_RUNS/GNN_Classifiers/{dataset_name}/Trial_{trial.number}",
    )

    try:
        # 3. INSTANCIAÇÃO DO MODELO
        # =========================
        model = DynamicGNNClassifier(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=layer_type,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            heads=heads
        ).to(device)

        # 4. SETUP DE TREINO
        # ==================
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # EarlyStopper com PRUNING ativado
        # IMPORTANTE: Usamos "val_f1" para guiar a otimização
        early_stopper = EarlyStopper(
            patience=grids.TRAINING_CONFIG["early_stopping_patience"],
            min_delta=grids.TRAINING_CONFIG["early_stopping_min_delta"],
            mode="max",
            metric_name="val_f1", 
            trial=trial # <--- Ativa o Pruning do Optuna
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=grids.TRAINING_CONFIG["scheduler_patience"],
            factor=grids.TRAINING_CONFIG["scheduler_factor"],
            min_lr=grids.TRAINING_CONFIG["min_lr"],
        )

        # 5. LOOP DE TREINAMENTO
        # ======================
        training_report = model.train_model(
            data=pyg_data,
            optimizer=optimizer,
            # ✅ CORREÇÃO: Usar o valor do config, não deixar vazio
            epochs=grids.TRAINING_CONFIG["epochs"], 
            early_stopper=early_stopper,
            scheduler=scheduler,
            criterion=torch.nn.CrossEntropyLoss()
        )

        # Métricas Finais
        best_val_f1 = training_report["val_report"]["weighted avg"]["f1-score"]
        test_f1 = training_report["test_f1"] # Apenas informativo

        # 6. SALVAR RELATÓRIO
        # ===================
        report_manager = ReportManager(directory_manager)
        full_report = {
            "params": trial.params,
            "best_val_f1": best_val_f1,
            "test_f1": test_f1,
            "training_history": training_report["training_history"]
        }
        report_manager.create_report(full_report)
        report_manager.save_report()
        
        # Renomeia pasta final com o score de VALIDAÇÃO (critério de escolha)
        directory_manager.finalize_run_directory(
            dataset_name=dataset_name,
            metrics={"val_f1": best_val_f1, "test_f1": test_f1}
        )

        return best_val_f1

    # 7. TRATAMENTO DE PRUNING E ERROS
    # ================================
    except optuna.TrialPruned:
        print(f"[PRUNED] Trial {trial.number}")
        tmp_path = directory_manager.get_run_path()
        if os.path.exists(tmp_path) and "_tmp__" in tmp_path:
            try:
                shutil.rmtree(tmp_path)
            except Exception as e:
                print(f"Erro ao limpar pasta: {e}")
        raise

    except Exception as e:
        print(f"[ERRO] Trial {trial.number}: {e}")
        raise e
    
    finally:
        # Limpeza Segura (Idêntica ao GAE Script para evitar UnboundLocalError)
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'early_stopper' in locals(): del early_stopper
        if 'scheduler' in locals(): del scheduler
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def run_optuna_gnn(WSG_DATASET: Any, config: Config, n_trials=50):
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y_%H-%M-%S")
    device = torch.device(config.DEVICE)
    
    print(f"\n--- Iniciando Optuna (GNN Classifier) para {WSG_DATASET.dataset_name} ---")

    # 1. Carregar Dados
    wsg_obj = WSG_DATASET.load()

    # 2. Converter para PyG (Supervisionado)
    pyg_data = data_converters.wsg_for_gcn_gat_multi_hot(wsg_obj, config).to(device)

    # 3. Calcular Dimensões
    input_dim = wsg_obj.metadata.num_total_features
    output_dim = len(set(y for y in wsg_obj.graph_structure.y if y is not None))
    
    print(f"Input Dim: {input_dim} | Output Classes: {output_dim}")

    # 4. Criar Estudo
    study = optuna.create_study(
        direction="maximize",
        study_name=f"GNN_Baseline_{WSG_DATASET.dataset_name}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )

    func = lambda trial: objective(trial, pyg_data, config, input_dim, output_dim, WSG_DATASET.dataset_name)
    
    study.optimize(func, n_trials=n_trials)

    print("\n" + "="*50)
    print(f"📌 MELHOR RESULTADO GNN ({WSG_DATASET.dataset_name}):")
    print(f"Best Val F1: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    print("="*50 + "\n")


if __name__ == "__main__":
    config = Config()
    
    # Ajuste de épocas para o Optuna
    grids.TRAINING_CONFIG["epochs"] = 200 # Deixe alto, o Pruning corta o excesso
    
    dataset_github = MusaeGithubLoader()
    run_optuna_gnn(dataset_github, config, n_trials=30)
    
    # dataset_fb = MusaeFacebookLoader()
    # run_optuna_gnn(dataset_fb, config, n_trials=30)