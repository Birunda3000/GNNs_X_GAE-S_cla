"""
Grid Search / Optuna Parameter Definitions
==========================================

Este módulo centraliza todos os espaços de busca para hiperparâmetros.
Focado em eficiência computacional e cobertura teórica de famílias ML.
"""

from typing import Dict, Any, List

# GNN Imports
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn as nn

# Sklearn Imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ============================================================
# 🎯 TRAINING CONFIGURATION (Global)
# ============================================================

TRAINING_CONFIG: Dict[str, Any] = {
    "epochs": 500,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "early_stopping_patience": 32,
    "early_stopping_min_delta": 1e-6,
    "scheduler_patience": 10,
    "scheduler_factor": 0.6,
    "min_lr": 1e-8,
}


# ============================================================
# 📊 EMBEDDING MODELS (GAE/VGAE)
# ============================================================

GAE_VGAE_GRID: Dict[str, List[Any]] = {
    "layer_type": ["SAGEConv", "GCNConv"],
    "num_layers": [2, 3],
    "embedding_dim": [128, 256],
    "hidden_dim": [128, 256],
    "out_embedding_dim": [32, 64, 128],
    "activation": ["ReLU", "LeakyReLU", "ELU"],
    "dropout": [0.0, 0.2, 0.5],
    "normalize_embeddings": [True, False],
}


# ============================================================
# 🧠 SUPERVISED GNN CLASSIFIERS
# ============================================================

GNN_CLASSIFIER_GRID: Dict[str, List[Any]] = {
    "layer_type": ["SAGEConv", "GCNConv", "GATConv"],
    "num_layers": [2, 3],
    "hidden_dim": [64, 128, 256],
    "activation": ["ReLU", "ELU", "LeakyReLU", "Tanh"],
    "dropout": [0.0, 0.2, 0.5],
    "heads": [1, 2, 4],
}


# ============================================================
# 🤖 CLASSICAL ML CLASSIFIERS (SKLEARN)
# ============================================================

# --- 1. Logistic Regression (Família: Linear Probabilística) ---
LOGISTIC_REGRESSION_GRID: Dict[str, List[Any]] = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "solver": ["lbfgs", "liblinear", "saga"],
    "class_weight": [None, "balanced"],
}

# --- 2. KNN (Família: Vizinhança) ---
KNN_GRID: Dict[str, List[Any]] = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 51],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"],
}

# --- 3. Random Forest (Família: Bagging/Árvores) ---
RANDOM_FOREST_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}

# --- 4. Linear SVM (Família: Linear Geométrica/Margem) ---
SVM_GRID: Dict[str, List[Any]] = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
    "penalty": ["l2", "l1"],
    "loss": ["squared_hinge"],
    "dual": [False],
    "tol": [1e-4, 1e-5],
    "class_weight": ["balanced", None],
}

# --- 5. QDA (Família: Gaussiana/Radial/Curva) ---
QDA_GRID: Dict[str, List[Any]] = {
    "reg_param": [0.0, 0.001, 0.01, 0.1, 0.5],
    "tol": [1e-4],
}

# --- 6. MLP (Família: Redes Neurais/Universal) ---
MLP_SKLEARN_GRID = {
    "hidden_layer_sizes": [
        (8,),
        (16, 8),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (16, 16, 16, 16),
        (64, 64, 32, 16),
        (256, 128, 64, 32),
    ],
    "activation": ["relu", "tanh", "identity"],  # <--- sem "logistic"
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.01],
    "solver": ["adam", "sgd"],                 # <--- muito relevante
    "learning_rate": ["constant", "adaptive"], # <--- sem "invscaling"
    "batch_size": ["auto", 64, 128],           # <--- bom e leve
}


# --- 7. XGBoost (Família: Boosting) ---
XGBOOST_GRID: Dict[str, List[Any]] = {
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [1.0, 1.5, 2.0],
    "n_estimators": [100, 200, 300],
}


# ============================================================
# 📦 COLLECTIONS
# ============================================================

SKLEARN_MODEL_MAP = {
    "LogisticRegression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "RandomForest": RandomForestClassifier,
    "SVM": LinearSVC,
    "QDA": QuadraticDiscriminantAnalysis,
    "MLP": MLPClassifier,
    "XGBoost": XGBClassifier,
}

SKLEARN_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "LogisticRegression": LOGISTIC_REGRESSION_GRID,
    "KNN": KNN_GRID,
    "RandomForest": RANDOM_FOREST_GRID,
    "SVM": SVM_GRID,
    "QDA": QDA_GRID,
    "MLP": MLP_SKLEARN_GRID,
    "XGBoost": XGBOOST_GRID,
}

# Compatibilidade legada
EMBEDDING_GRIDS = {"GAE": GAE_VGAE_GRID, "VGAE": GAE_VGAE_GRID}
GNN_GRIDS = {"GNN_Classifier": GNN_CLASSIFIER_GRID}
