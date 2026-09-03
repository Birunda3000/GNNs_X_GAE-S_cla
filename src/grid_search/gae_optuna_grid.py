import sys
import os
import torch
import numpy as np
import random
import gc
import optuna
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.optim import lr_scheduler
import torch.nn as nn
from typing import Any
import shutil

# Imports do projeto
from src.config import Config
from src.data_loaders import MusaeGithubLoader, MusaeFacebookLoader, MusaeTwitchLoader, RedditLiteLoader, RedditLoader
from src.models.pytorch.GraphAutoencoders.din_gae import DynamicGAE, DynamicVGAE
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.early_stopper import EarlyStopper
from src.embeddings_eval import evaluate_embeddings
import src.data_converters as data_converters
import src.grid_search.grid_search_params as grids


# ============================================================
# FUNÇÃO OBJETIVO DO OPTUNA
# ============================================================

def objective(trial, pyg_data, config, model_class, dataset_name):

    # --- 0. GARANTIR REPRODUTIBILIDADE ---
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)


    # --------- 1. ESPAÇO DE BUSCA FINAL RECOMENDADO ----------

    layer_type_str = trial.suggest_categorical("layer_type", ["SAGEConv", "GCNConv", "GATConv"])
    if layer_type_str == "SAGEConv":
        layer_type = grids.SAGEConv
    elif layer_type_str == "GCNConv":
        layer_type = grids.GCNConv
    elif layer_type_str == "GATConv":
        layer_type = grids.GATConv
    else:
        raise ValueError(f"Unsupported layer_type_str: {layer_type_str}")

    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.2, 0.5])

    activation_str = trial.suggest_categorical("activation", ["ReLU", "ELU", "LeakyReLU"])
    if activation_str == "ReLU":
        activation = nn.ReLU
    elif activation_str == "ELU":
        activation = nn.ELU
    elif activation_str == "LeakyReLU":
        activation = nn.LeakyReLU
    else:
        raise ValueError(f"Unsupported activation_str: {activation_str}")

    # Parametros de embedding
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    out_embedding_dim = trial.suggest_categorical("out_embedding_dim", [64])
    normalize_embeddings = trial.suggest_categorical("normalize_embeddings", [True, False])

    device = torch.device(config.DEVICE)

    # --------- 2. DIRETÓRIO -------------------------------
    directory_manager = DirectoryManager(
        timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        run_folder_name=f"OPTUNA_RUNS/GAEs/{dataset_name}/{model_class.__name__}/{layer_type_str}/{config.TIMESTAMP}/Trial_{trial.number}",
    )

    try:
        # --------- 3. INSTANCIAR MODELO ---------------------
        model = model_class(
            config=config,
            num_total_features=pyg_data.num_total_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_embedding_dim=out_embedding_dim,
            layer_type=layer_type,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            normalize_embeddings=normalize_embeddings,
        ).to(device)

        # --------- 4. OPTIMIZER / SCHEDULER / STOPPER --------
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=grids.TRAINING_CONFIG["learning_rate"],
            weight_decay=grids.TRAINING_CONFIG["weight_decay"],
        )

        early_stopper = EarlyStopper(
            patience=grids.TRAINING_CONFIG["early_stopping_patience"],
            min_delta=grids.TRAINING_CONFIG["early_stopping_min_delta"],
            mode="max",
            metric_name="max_val_f1",
            custom_eval=lambda m: evaluate_embeddings(m, pyg_data, device),
            trial=trial
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            patience=grids.TRAINING_CONFIG["scheduler_patience"],
            factor=grids.TRAINING_CONFIG["scheduler_factor"],
            min_lr=grids.TRAINING_CONFIG["min_lr"],
        )

        # --------- 5. TREINAMENTO ---------------------------
        training_report = model.train_model(
            data=pyg_data,
            optimizer=optimizer,
            epochs=grids.TRAINING_CONFIG["epochs"],
            early_stopper=early_stopper,
            scheduler=scheduler,
        )
        
        best_score = training_report["best_score"]

        # --------- 6. SALVAR RELATÓRIO -----------------------
        report_manager = ReportManager(directory_manager)
        full_report = {
            "random_seed": config.RANDOM_SEED,
            "params": trial.params,
            "best_score": best_score,
            "training_history": training_report["training_history"]
        }
        report_manager.create_report(full_report)
        report_manager.save_report()
        
        directory_manager.finalize_run_directory(
            dataset_name=dataset_name,
            metrics={"score": best_score}
        )

        return best_score

    except optuna.TrialPruned:
        print(f"[PRUNED] Trial {trial.number}")
        tmp_path = directory_manager.get_run_path()

        if os.path.exists(tmp_path) and "_tmp__" in tmp_path:
            try:
                shutil.rmtree(tmp_path)
            except Exception as e:
                print(f"Erro removendo pasta: {e}")
        raise

    except Exception as e:
        print(f"[ERRO] Trial {trial.number}: {e}")
        raise e

    finally:
        # Limpeza Segura
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

def run_optuna_optimization(WSG_DATASET: Any, config: Config, model_class: Any, n_trials=40):
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y_%H-%M-%S")
    device = torch.device(config.DEVICE)
    
    print(f"\n--- Iniciando Optuna para {model_class.__name__} em {WSG_DATASET.dataset_name} ---")

    # 1. Carregar WSG
    wsg_obj = WSG_DATASET.load()

    # 2. Converter para PyG
    pyg_data = data_converters.wsg_for_vgae(wsg_obj, config).to(device)

    # 3. CRIAR ESTUDO OPTUNA
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_class.__name__}_{WSG_DATASET.dataset_name}",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )

    func = lambda trial: objective(trial, pyg_data, config, model_class, WSG_DATASET.dataset_name)
    study.optimize(func, n_trials=n_trials)

    print("\n" + "="*50)
    print(f"  🏆 MELHOR RESULTADO ({model_class.__name__} - {WSG_DATASET.dataset_name}):")
    print(f"  Pasta (Trial): Trial_{study.best_trial.number}") # <--- Indica a pasta exata!
    print(f"  Score (F1):    {study.best_value:.6f}")
    print(f"  Params:        {study.best_params}")
    print("="*50 + "\n")



# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":
    config = Config()
    dataset_twitch = MusaeTwitchLoader()

    #print("=== Testando GAE para o Twitch ===")
    #run_optuna_optimization(dataset_twitch, config, DynamicGAE, n_trials=30)

    #print("=== Testando VGAE para o Twitch ===")
    #run_optuna_optimization(dataset_twitch, config, DynamicVGAE, n_trials=30)



    print("Iniciando carregamento do dataset REDDIT (Versão Lite)...")
    # threads_per_class=500 vai gerar um grafo com 1000 threads fundidas.
    # DICA: Se a memória da GPU ainda chorar com 500, diminua para 200 ou 300.
    dataset_reddit_lite = RedditLiteLoader(threads_per_class=500)
    
    #print("\n=== Testando GAE para o Reddit (Lite) ===")
    # Mantendo n_trials=30. O pruner agressivo vai agir rápido!
    #run_optuna_optimization(dataset_reddit_lite, config, DynamicGAE, n_trials=30)
    
    print("\n=== Testando VGAE para o Reddit (Lite) ===")
    run_optuna_optimization(dataset_reddit_lite, config, DynamicVGAE, n_trials=30)