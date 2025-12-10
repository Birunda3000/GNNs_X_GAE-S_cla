import sys
import os
import shutil
import json
import optuna
import numpy as np
import torch
import gc
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

# Imports de terceiros (Modelos e Métricas)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier

# Imports do projeto
from src.config import Config
from src.data_loaders import DirectWSGLoader
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
import src.data_converters as data_converters
from src.grid_search.grid_search_params import SKLEARN_GRIDS, SKLEARN_MODEL_MAP

# ============================================================
# FUNÇÃO OBJETIVO (SKLEARN CLASSIFIER)
# ============================================================


def objective(trial, X, y, config, model_name, dataset_name):

    # 1. DEFINIÇÃO DO ESPAÇO DE BUSCA
    # ===============================
    grid = SKLEARN_GRIDS[model_name]
    params = {}

    # Optuna escolhe valores do grid definido no arquivo auxiliar
    for param_name, param_values in grid.items():
        params[param_name] = trial.suggest_categorical(param_name, param_values)

    # 2. GERENCIAMENTO DE DIRETÓRIOS
    # ==============================
    directory_manager = DirectoryManager(
        timestamp=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        run_folder_name=f"OPTUNA_RUNS/SKLEARN_Classifiers/{config.TIMESTAMP}/{dataset_name}/{model_name}/Trial_{trial.number}",
    )

    try:
        # 3. INSTANCIAÇÃO DO MODELO
        # =========================
        model_class = SKLEARN_MODEL_MAP[model_name]

        # --- Configurações Específicas para Consistência ---
        if model_name == "MLP":
            model = model_class(
                **params,
                max_iter=500,  # Teto fixo para CV
                random_state=config.RANDOM_SEED,
            )
        elif model_name == "XGBoost":
            model = model_class(
                **params,
                # Nota: No CV do Optuna, usamos n_estimators fixo ou do grid.
                # Early stopping por fold é complexo com cross_val_score, então confiamos no CV.
                eval_metric="mlogloss",
                use_label_encoder=False,
                n_jobs=1,  # IMPORTANTE: 1 job por fold para não conflitar com paralelismo do CV
                random_state=config.RANDOM_SEED,
            )
        elif model_name in ["KNN", "NaiveBayes"]:
            # Modelos sem random_state
            if "n_jobs" in model_class().get_params():
                model = model_class(**params, n_jobs=1)
            else:
                model = model_class(**params)
        else:
            # RF, LogReg, SVM
            extra_args = {"random_state": config.RANDOM_SEED}
            if "n_jobs" in model_class().get_params():
                extra_args["n_jobs"] = 1  # Deixa o CV controlar o paralelismo
            model = model_class(**params, **extra_args)

        # 4. AVALIAÇÃO (CROSS-VALIDATION)
        # ===============================
        # Usamos 5-Fold CV nos dados de (Treino + Validação) combinados
        # Test Set já foi removido no main() para evitar Data Leakage
        scorer = make_scorer(f1_score, average="weighted")

        scores = cross_val_score(
            model, X, y, cv=5, scoring=scorer, n_jobs=-1  # Paralelismo via CV (5 cores)
        )

        mean_cv_score = scores.mean()
        std_cv_score = scores.std()

        # 5. SALVAR RELATÓRIO
        # ===================
        report_manager = ReportManager(directory_manager)

        full_report = {
            "model_name": model_name,
            "params": params,
            "best_cv_f1_mean": mean_cv_score,
            "best_cv_f1_std": std_cv_score,
            "fold_scores": scores.tolist(),  # Histórico dos 5 folds
        }

        report_manager.create_report(full_report)
        report_manager.save_report()

        # Renomeia pasta com o score final
        directory_manager.finalize_run_directory(
            dataset_name=dataset_name, metrics={"cv_f1": mean_cv_score}
        )

        return mean_cv_score

    # 6. TRATAMENTO DE ERROS E LIMPEZA
    # ================================
    except Exception as e:
        print(f"[ERRO] Trial {trial.number} ({model_name}): {e}")

        # Limpa pasta temporária em caso de erro
        tmp_path = directory_manager.get_run_path()
        if os.path.exists(tmp_path) and "_tmp__" in tmp_path:
            try:
                shutil.rmtree(tmp_path)
            except OSError:
                pass
        raise e

    finally:
        # Garante limpeza de memória, especialmente se usar CUDA em algum momento
        gc.collect()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================


def run_sklearn_optimization(wsg_file_path: str, n_trials=30):
    # 1. Configuração Inicial
    config = Config()
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    # Sementes Globais
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    dataset_name = os.path.basename(wsg_file_path).replace(".wsg.json", "")

    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO OPTUNA (SKLEARN) | Dataset: {dataset_name}")
    print(f"📂 Arquivo: {wsg_file_path}")
    print(f"{'='*70}")

    # 2. Carregar e Converter Dados
    loader = DirectWSGLoader(wsg_file_path)
    wsg = loader.load()

    print("🔄 Convertendo WSG e gerando máscaras (wsg_for_dense_classifier)...")
    # Usa a função padronizada do projeto para garantir mesma seed de split
    pyg_data = data_converters.wsg_for_dense_classifier(wsg, config)

    # 3. Preparar Dados para Optuna (PREVENÇÃO DE LEAKAGE)
    # ----------------------------------------------------
    # Convertendo para Numpy (Sklearn)
    X = pyg_data.x.numpy()
    y = pyg_data.y.numpy()

    # CRÍTICO: Usamos APENAS (Treino | Validação) para o Optuna.
    # O Test Set é sagrado e só será usado na avaliação final (run_feature_classification.py)
    optimization_mask = pyg_data.train_mask | pyg_data.val_mask

    X_optim = X[optimization_mask]
    y_optim = y[optimization_mask]

    print(f"\n📊 Estatísticas de Dados:")
    print(f"   Total Nós: {len(y)}")
    print(f"   --------------------------------")
    print(f"   Usado no Optuna (Treino + Val): {len(y_optim)} amostras")
    print(
        f"   Reservado (Teste):              {pyg_data.test_mask.sum().item()} amostras (INTOCADO)"
    )

    # 4. Loop de Otimização por Modelo
    final_summary = {}

    for model_name in SKLEARN_MODEL_MAP.keys():
        print(f"\n>> Otimizando: {model_name}...")

        # Cria estudo único por Modelo
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_{dataset_name}",
            # Pruner median (útil se integrarmos partial_fit no futuro, por enquanto ajuda a organizar)
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Função Lambda para injetar dados estáticos
        func = lambda trial: objective(
            trial, X_optim, y_optim, config, model_name, dataset_name
        )

        # Ajuste dinâmico de trials (MLP e SVM são mais lentos)
        current_trials = n_trials // 2 if model_name in ["MLP", "SVM"] else n_trials

        study.optimize(func, n_trials=current_trials)

        print(f"   🏆 Melhor F1 (CV): {study.best_value:.4f}")
        print(f"   ⚙️ Params: {study.best_params}")

        final_summary[model_name] = {
            "best_cv_f1": study.best_value,
            "best_params": study.best_params,
        }

    # 5. Salvar Resumo Final ("Campeões")
    # Salva um JSON mestre na raiz da execução com os vencedores de todos os modelos
    summary_path = f"data/output/OPTUNA_RUNS/SKLEARN_Classifiers/{config.TIMESTAMP}/{dataset_name}/best_hyperparameters_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=4)

    print(f"\n✅ Otimização concluída! Resumo salvo em:\n{summary_path}")


if __name__ == "__main__":
    # Argumentos hardcoded
    WSG_FILE_PATH = "data/output/EMBEDDING_RUNS/facebook_embeddings.wsg.json"
    N_TRIALS = 30

    if not os.path.exists(WSG_FILE_PATH):
        print(f"Erro: Arquivo não encontrado: {WSG_FILE_PATH}")
    else:
        run_sklearn_optimization(WSG_FILE_PATH, N_TRIALS)
