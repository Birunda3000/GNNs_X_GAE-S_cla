import sys
import os
import shutil
import json
import optuna
import numpy as np
import torch
import gc
from functools import partial
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Imports do projeto
from src.config import Config
from src.data_loaders import DirectWSGLoader
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
import src.data_converters as data_converters
from src.grid_search.grid_search_params import SKLEARN_GRIDS, SKLEARN_MODEL_MAP

# Listas de Controle de Paralelismo
PARALLEL_INTERNAL_MODELS = ["RandomForest", "XGBoost", "KNN", "LogisticRegression"]
HEAVY_MEMORY_MODELS = ["KNN", "SVM", "QDA", "MLP"]

def objective(trial, X, y, config, model_name, dataset_name):
    if model_name not in SKLEARN_GRIDS:
        raise ValueError(f"Grid não definido para {model_name}")
        
    grid = SKLEARN_GRIDS[model_name]
    params = {}
    for param_name, param_values in grid.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    directory_manager = DirectoryManager(
        timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        run_folder_name=f"OPTUNA_RUNS/SKLEARN_Classifiers/{config.TIMESTAMP}/{dataset_name}/{model_name}/Trial_{trial.number}",
    )

    try:
        model_class = SKLEARN_MODEL_MAP[model_name]
        fixed_args = {}
        
        if model_name not in ["KNN", "QDA"]: 
            fixed_args["random_state"] = config.RANDOM_SEED
            
        if model_name in PARALLEL_INTERNAL_MODELS:
            fixed_args["n_jobs"] = 1

        # --- TRATAMENTOS DE CONVERGÊNCIA (CORREÇÃO DE WARNINGS) ---
        
        if model_name == "MLP":
            # Turbina o MLP: Mais iterações + Early Stopping
            fixed_args["max_iter"] = 500 
            fixed_args["early_stopping"] = True
            fixed_args["n_iter_no_change"] = 20
            fixed_args["validation_fraction"] = 0.1
        
        elif model_name == "SVM":
            fixed_args["max_iter"] = 5000 # Elimina warning Liblinear
        
        elif model_name == "LogisticRegression":
            fixed_args["max_iter"] = 5000  # Elimina warning lbfgs

        elif model_name == "XGBoost":
            fixed_args["eval_metric"] = "mlogloss"
            fixed_args["use_label_encoder"] = False
            if "n_estimators" not in params:
                fixed_args["n_estimators"] = 100
        
        model = model_class(**params, **fixed_args)

        # --- AVALIAÇÃO SEGURA (SEM CRASH DE MEMÓRIA) ---
        scorer = make_scorer(f1_score, average="weighted")
        
        # Se for pesado, roda 1 por vez. Se for leve, roda 4.
        cv_n_jobs = 1 if model_name in HEAVY_MEMORY_MODELS else 4

        scores = cross_val_score(
            model, X, y, cv=5, scoring=scorer, n_jobs=cv_n_jobs
        )

        mean_cv_score = scores.mean()
        std_cv_score = scores.std()

        report_manager = ReportManager(directory_manager)
        report_manager.create_report({
            "model_name": model_name,
            "params": params,
            "best_cv_f1_mean": mean_cv_score,
            "best_cv_f1_std": std_cv_score,
            "fold_scores": scores.tolist(),
        })
        report_manager.save_report()
        
        directory_manager.finalize_run_directory(
            dataset_name=dataset_name, metrics={"cv_f1": mean_cv_score}
        )

        return mean_cv_score

    except Exception as e:
        print(f"[ERRO] Trial {trial.number} ({model_name}): {e}")
        shutil.rmtree(directory_manager.run_dir_path, ignore_errors=True)
        raise e
    finally:
        gc.collect()

def run_sklearn_optimization(wsg_file_path: str, n_trials=30):
    config = Config()
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y_%H-%M-%S")
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    dataset_name = os.path.basename(wsg_file_path).replace(".wsg.json", "")
    
    print(f"\n🚀 OPTUNA SKLEARN (Green AI) | Dataset: {dataset_name}")
    loader = DirectWSGLoader(wsg_file_path)
    wsg = loader.load()
    pyg_data = data_converters.wsg_for_dense_classifier(wsg, config)
    
    X = pyg_data.x.numpy()
    y = pyg_data.y.numpy()
    optim_mask = pyg_data.train_mask | pyg_data.val_mask
    X_optim = X[optim_mask]
    y_optim = y[optim_mask]

    print(f"   Dados (Treino+Val): {len(y_optim)} | Teste: {pyg_data.test_mask.sum()}")

    final_summary = {}
    available_models = [m for m in SKLEARN_MODEL_MAP.keys() if m in SKLEARN_GRIDS]

    for model_name in available_models:
        print(f"\n>> Otimizando: {model_name}...")
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_{dataset_name}",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        func = partial(objective, X=X_optim, y=y_optim, config=config, model_name=model_name, dataset_name=dataset_name)
        
        # Reduz trials para modelos lentos para economizar tempo
        trials_count = n_trials // 2 if model_name in ["MLP", "SVM"] else n_trials
        
        study.optimize(func, n_trials=trials_count)
        
        print(f"   🏆 Melhor F1: {study.best_value:.4f}")
        final_summary[model_name] = {
            "best_cv_f1": study.best_value,
            "best_params": study.best_params,
        }

    summary_path = f"data/output/OPTUNA_RUNS/SKLEARN_Classifiers/{config.TIMESTAMP}/{dataset_name}/best_hyperparameters_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=4)

    print(f"\n✅ Concluído! Resumo salvo em: {summary_path}")

if __name__ == "__main__":
    # Ajuste para o seu caminho do GitHub ou Facebook
    WSG_PATH = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Musae-Github__score_1_6792__emb_dim_32__10-12-2025_15-59-55/Musae-Github_(32)_embeddings_10-12-2025_15-59-55.wsg.json"

    #WSG_PATH = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Musae-Facebook__score_1_7161__emb_dim_32__10-12-2025_15-33-25/Musae-Facebook_(32)_embeddings_10-12-2025_15-33-25.wsg.json" 
    
    if os.path.exists(WSG_PATH):
        run_sklearn_optimization(WSG_PATH, n_trials=30)
    else:
        print(f"❌ Arquivo não encontrado: {WSG_PATH}")