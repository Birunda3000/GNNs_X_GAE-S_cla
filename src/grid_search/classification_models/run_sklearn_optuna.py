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

from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight  # 🔧 FIX

from src.config import Config
from src.data_loaders import DirectWSGLoader
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
import src.data_converters as data_converters
from src.grid_search.grid_search_params import SKLEARN_GRIDS, SKLEARN_MODEL_MAP

PARALLEL_INTERNAL_MODELS = ["RandomForest", "XGBoost", "KNN", "LogisticRegression"]

# ============================================================
# FUNÇÃO OBJETIVO (TREINO E VALIDAÇÃO APENAS)
# ============================================================

def objective(trial, X, y, masks, config, model_name, dataset_name):
    # Verificação de segurança
    if model_name not in SKLEARN_GRIDS:
        raise ValueError(f"Grid não definido para {model_name}")
    
    grid = SKLEARN_GRIDS[model_name]
    params = {}
    for param_name, param_values in grid.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    directory_manager = DirectoryManager(
        timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        run_folder_name=f"OPTUNA_RUNS/SKLEARN_Classifiers-nokfold/{config.TIMESTAMP}/{dataset_name}/{model_name}/Trial_{trial.number}",
    )

    try:
        # --- PREPARAÇÃO DOS DADOS (STRICT TRAIN/VAL) ---
        # Apenas máscaras de treino e validação são usadas.
        X_train, y_train = X[masks['train']], y[masks['train']]
        X_val, y_val = X[masks['val']], y[masks['val']]

        classes = np.unique(y_train)              # 🔧 FIX
        is_binary = len(classes) == 2             # 🔧 FIX

        # Configuração do Modelo
        model_class = SKLEARN_MODEL_MAP[model_name]
        fixed_args = {}
        
        if model_name not in ["KNN", "QDA"]: 
            fixed_args["random_state"] = config.RANDOM_SEED
            
        if model_name in PARALLEL_INTERNAL_MODELS:
            fixed_args["n_jobs"] = 4 

        # Ajustes de Convergência (Igualando à GNN)
        if model_name == "MLP":
            fixed_args["max_iter"] = 1000
            fixed_args["early_stopping"] = True 
            fixed_args["n_iter_no_change"] = 20
            fixed_args["validation_fraction"] = 0.1 
        elif model_name == "SVM":
            fixed_args["max_iter"] = 2000
        elif model_name == "LogisticRegression":
            fixed_args["max_iter"] = 2000
        elif model_name == "XGBoost":
            fixed_args["eval_metric"] = "mlogloss"
            fixed_args["use_label_encoder"] = False

        model = model_class(**params, **fixed_args)

        # --- TREINO (Fit no Train) ---
        if model_name == "XGBoost" and not is_binary:   # 🔧 FIX
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        
        # --- AVALIAÇÃO (Score no Val) ---
        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="weighted")
        val_acc = accuracy_score(y_val, val_preds)

        # --- RELATÓRIO (Sem Teste!) ---
        report_manager = ReportManager(directory_manager)
        report_manager.create_report({
            "model_name": model_name,
            "params": params,
            "val_f1": val_f1, 
            "val_acc": val_acc,
            "split_type": "fixed_mask_opt_only" # Deixa claro no log
        })
        report_manager.save_report()
        
        # Nome da pasta reflete o score de validação
        directory_manager.finalize_run_directory(
            dataset_name=dataset_name, 
            metrics={"val_f1": val_f1}
        )

        return val_f1

    except Exception as e:
        print(f"[ERRO] Trial {trial.number} ({model_name}): {e}")
        tmp_path = directory_manager.get_run_path()
        if os.path.exists(tmp_path) and "_tmp__" in tmp_path:
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise e
    finally:
        gc.collect()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def run_sklearn_optimization(wsg_file_path: str, n_trials=30):
    config = Config()
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y_%H-%M-%S")
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    dataset_name = os.path.basename(wsg_file_path).replace(".wsg.json", "")
    
    print(f"\n🚀 OPTUNA SKLEARN (Search Only) | Dataset: {dataset_name}")
    print(f"   Modo: Otimização Estrita (Train/Val). Teste ignorado.")
    
    loader = DirectWSGLoader(wsg_file_path)
    wsg = loader.load()
    pyg_data = data_converters.wsg_for_dense_classifier(wsg, config)
    
    X = pyg_data.x.numpy()
    y = pyg_data.y.numpy()
    
    # Passamos as máscaras para garantir o split correto
    masks = {
        'train': pyg_data.train_mask.numpy(),
        'val': pyg_data.val_mask.numpy(),
        # 'test' nem é passado ou usado, garantindo isolamento total
    }

    print(f"   Amostras: Treino={masks['train'].sum()} | Val={masks['val'].sum()}")

    final_summary = {}
    available_models = [m for m in SKLEARN_MODEL_MAP.keys() if m in SKLEARN_GRIDS]

    for model_name in available_models:
        print(f"\n>> Otimizando: {model_name}...")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_{dataset_name}",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        func = partial(objective, X=X, y=y, masks=masks, config=config, model_name=model_name, dataset_name=dataset_name)
        
        # Roda a busca
        study.optimize(func, n_trials=n_trials)

        print(f"   ✅ Melhores Params (Val F1: {study.best_value:.4f})")
        
        # Salva apenas o resultado de validação e os parâmetros
        final_summary[model_name] = {
            "best_val_f1": study.best_value,
            "best_params": study.best_params,
        }

    # Salva o arquivo de "Campeões" (Best Hyperparameters)
    # Este arquivo será usado depois pelo script de Teste Final
    summary_path = f"data/output/OPTUNA_RUNS/SKLEARN_Classifiers-nokfold/{config.TIMESTAMP}/{dataset_name}/best_hyperparameters_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=4)

    print(f"\n✅ Busca Concluída! Hiperparâmetros salvos em: {summary_path}")
    print("➡️  Próximo passo: Use um script separado para rodar a avaliação final no Test Set.")


if __name__ == "__main__":
    # AJUSTE O CAMINHO AQUI (Facebook ou Github)
    # Ajuste para o seu caminho do GitHub ou Facebook
    WSG_PATH = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Musae-Github__score_1_6792__emb_dim_32__10-12-2025_15-59-55/Musae-Github_(32)_embeddings_10-12-2025_15-59-55.wsg.json"

    #WSG_PATH = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Musae-Facebook__score_1_7161__emb_dim_32__10-12-2025_15-33-25/Musae-Facebook_(32)_embeddings_10-12-2025_15-33-25.wsg.json" 
    
    if os.path.exists(WSG_PATH):
        run_sklearn_optimization(WSG_PATH, n_trials=30)
    else:
        print(f"Arquivo não encontrado: {WSG_PATH}")
