# --- Configuração do Optuna (Extraído das Tabelas do TCC) ---

# Tabela tab:d2-clf-top1 (D2 — MUSAE-GitHub)
optuna_params_github = {
    "lr": {"C": 0.01, "solver": "lbfgs", "class_weight": "balanced"},
    "knn": {"n_neighbors": 21, "weights": "distance", "metric": "cosine"},
    "rf": {
        "class_weight": "balanced_subsample",
        "n_estimators": 100,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "log2",
        "criterion": "entropy",
    },
    "svc": {
        "C": 0.01,
        "penalty": "l1",
        "loss": "squared_hinge",
        "dual": False,
        # Nota: class_weight não estava explícito na Tab D2, mas geralmente se usa balanced
        "class_weight": "balanced",
    },
    "qda": {"reg_param": 0.5, "tol": 1e-4},
    "mlp": {
        "solver": "adam",
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "alpha": 0.001,
        "learning_rate_init": 0.01,
        "learning_rate": "constant",
        "batch_size": 128,
    },
    "xgb": {
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 0.5,
        "min_child_weight": 5.0,
        "reg_alpha": 1.5,
    },
}

# Tabela tab:d1-clf-top1 (D1 — MUSAE-Facebook)
optuna_params_facebook = {
    "lr": {"C": 100.0, "solver": "liblinear", "class_weight": "balanced"},
    "knn": {"n_neighbors": 3, "weights": "distance", "metric": "euclidean"},
    "rf": {
        "class_weight": "balanced_subsample",
        "n_estimators": 300,
        "min_samples_split": 2,
        "max_features": "log2",
        "criterion": "entropy",
    },
    "svc": {
        "C": 10.0,
        "class_weight": "balanced",
        "tol": 1e-5,
        "penalty": "l1",
        "loss": "squared_hinge",
        "dual": False,
    },
    "qda": {"reg_param": 0.0, "tol": 1e-4},
    "mlp": {
        "solver": "adam",
        "hidden_layer_sizes": (128, 64, 32, 16),
        "activation": "relu",
        "alpha": 0.01,
        "learning_rate_init": 0.001,
        "learning_rate": "invscaling",
        "batch_size": 64,
    },
    "xgb": {
        "learning_rate": 0.1,
        "n_estimators": 300,
        "max_depth": 12,
        "subsample": 0.6,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "min_child_weight": 5.0,
        "reg_alpha": 1.5,
    },
}


# =====================================================================
# FUNÇÕES PARA CRIAR LISTAS DE MODELOS SKLEARN (uso em run_feature_classification)
# =====================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import Config
from src.models.sklearn.sklearn_model import SklearnClassifier


def get_github_models(config: Config = None) -> list:
    """
    Retorna lista de classificadores com hiperparâmetros otimizados para MUSAE-GitHub (D2).
    """
    if config is None:
        config = Config()

    p = optuna_params_github

    return [
        # 1. Logistic Regression
        SklearnClassifier(
            config, model_class=LogisticRegression, max_iter=1000, **p["lr"]
        ),
        # 2. K-Nearest Neighbors
        SklearnClassifier(
            config, model_class=KNeighborsClassifier, n_jobs=-1, **p["knn"]
        ),
        # 3. Random Forest
        SklearnClassifier(
            config, model_class=RandomForestClassifier, n_jobs=-1, **p["rf"]
        ),
        # 4. Linear SVC
        SklearnClassifier(config, model_class=LinearSVC, max_iter=2000, **p["svc"]),
        # 5. Quadratic Discriminant Analysis
        SklearnClassifier(
            config, model_class=QuadraticDiscriminantAnalysis, **p["qda"]
        ),
        # 6. MLP Classifier
        SklearnClassifier(
            config,
            model_class=MLPClassifier,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=config.EARLY_STOPPING_PATIENCE,
            validation_fraction=0.1,
            **p["mlp"]
        ),
        # 7. XGBoost
        SklearnClassifier(
            config,
            model_class=XGBClassifier,
            n_jobs=-1,
            early_stopping_rounds=config.EARLY_STOPPING_PATIENCE,
            **p["xgb"]
        ),
    ]


def get_facebook_models(config: Config = None) -> list:
    """
    Retorna lista de classificadores com hiperparâmetros otimizados para MUSAE-Facebook (D1).
    """
    if config is None:
        config = Config()

    p = optuna_params_facebook

    return [
        # 1. Logistic Regression
        SklearnClassifier(
            config, model_class=LogisticRegression, max_iter=1000, **p["lr"]
        ),
        # 2. K-Nearest Neighbors
        SklearnClassifier(
            config, model_class=KNeighborsClassifier, n_jobs=-1, **p["knn"]
        ),
        # 3. Random Forest
        SklearnClassifier(
            config, model_class=RandomForestClassifier, n_jobs=-1, **p["rf"]
        ),
        # 4. Linear SVC
        SklearnClassifier(config, model_class=LinearSVC, max_iter=2000, **p["svc"]),
        # 5. Quadratic Discriminant Analysis
        SklearnClassifier(
            config, model_class=QuadraticDiscriminantAnalysis, **p["qda"]
        ),
        # 6. MLP Classifier
        SklearnClassifier(
            config,
            model_class=MLPClassifier,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=config.EARLY_STOPPING_PATIENCE,
            validation_fraction=0.1,
            **p["mlp"]
        ),
        # 7. XGBoost
        SklearnClassifier(
            config,
            model_class=XGBClassifier,
            n_jobs=-1,
            early_stopping_rounds=config.EARLY_STOPPING_PATIENCE,
            **p["xgb"]
        ),
    ]
