"""
Grid Search Parameter Definitions
==================================

Este módulo centraliza todos os espaços de busca para hiperparâmetros
dos modelos, organizados por categoria.

Uso:
    from src.grid_search.grid_search_params import GAE_VGAE_GRID, TRAINING_CONFIG

    for params in ParameterGrid(GAE_VGAE_GRID):
        model = DynamicGAE(**params, **TRAINING_CONFIG)
"""

from typing import Dict, Any, List
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn


# ============================================================
# 🎯 TRAINING CONFIGURATION (Global, Fixed)
# ============================================================
# Hiperparâmetros de treinamento validados academicamente
# Baseados em: Kipf & Welling (GCN), Hamilton et al. (GraphSAGE)

TRAINING_CONFIG: Dict[str, Any] = {
    "epochs": 3,#500,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "early_stopping_patience": 32,
    "early_stopping_min_delta": 1e-6,
    "scheduler_patience": 10,
    "scheduler_factor": 0.6,
    "min_lr": 1e-8,
}


# ============================================================
# 📊 EMBEDDING MODELS (Unsupervised)
# ============================================================

GAE_VGAE_GRID: Dict[str, List[Any]] = {
    # Architecture
    "layer_type": [SAGEConv, GCNConv],
    "num_layers": [2, 3],
    # Dimensions
    "embedding_dim": [128, 256],  # EmbeddingBag projection
    "hidden_dim": [128, 256],  # GNN hidden layers
    "out_embedding_dim": [32, 64, 128],  # Final embedding size
    # Regularization
    "activation": [nn.ReLU, nn.LeakyReLU, nn.ELU],
    "dropout": [0.0, 0.2, 0.5],
    # Output normalization
    "normalize_embeddings": [True, False],
}

# Aliases para clareza
GAE_GRID = GAE_VGAE_GRID
VGAE_GRID = GAE_VGAE_GRID


# ============================================================
# 🧠 SUPERVISED GNN CLASSIFIERS
# ============================================================

GNN_CLASSIFIER_GRID: Dict[str, List[Any]] = {
    # Architecture
    "layer_type": [SAGEConv, GCNConv],
    "num_layers": [2], #[2, 3],
    "hidden_dim": [8], #[128, 256],
    # Regularization
    "activation": [nn.ReLU, nn.ELU],
    "dropout": [0.5] #[0.0, 0.2, 0.5],
}


# ============================================================
# 🤖 CLASSICAL ML CLASSIFIERS
# ============================================================

KNN_GRID: Dict[str, List[Any]] = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
}

RANDOM_FOREST_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 20, 40],
    "max_features": ["sqrt", "log2"],
}

LOGISTIC_REGRESSION_GRID: Dict[str, List[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs", "saga"],
    "max_iter": [1000],
}

XGBOOST_GRID: Dict[str, List[Any]] = {
    "max_depth": [3, 6, 10],
    "n_estimators": [200, 500],
    "learning_rate": [0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "gamma": [0],
    "reg_lambda": [1.0],
}

MLP_GRID: Dict[str, List[Any]] = {
    "hidden_dim": [8, 32, 64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.25, 0.5],
    "activation": [nn.ReLU],  # Fixo para MLPs simples
}


# ============================================================
# 📦 COLLECTIONS (para iteração fácil)
# ============================================================

EMBEDDING_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "GAE": GAE_GRID,
    "VGAE": VGAE_GRID,
}

CLASSIFIER_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "KNN": KNN_GRID,
    "RandomForest": RANDOM_FOREST_GRID,
    "LogisticRegression": LOGISTIC_REGRESSION_GRID,
    "XGBoost": XGBOOST_GRID,
    "MLP": MLP_GRID,
}

GNN_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "GNN_Classifier": GNN_CLASSIFIER_GRID,
}

ALL_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    **EMBEDDING_GRIDS,
    **GNN_GRIDS,
    **CLASSIFIER_GRIDS,
}


# ============================================================
# 🛠️ UTILITY FUNCTIONS
# ============================================================


def get_grid_size(grid: Dict[str, List[Any]]) -> int:
    """Calcula o número total de combinações no grid."""
    from math import prod

    return prod(len(values) for values in grid.values())


def print_grid_summary():
    """Imprime resumo de todos os grids definidos."""
    print("\n" + "=" * 70)
    print("GRID SEARCH CONFIGURATION SUMMARY".center(70))
    print("=" * 70)

    print("\n📊 EMBEDDING MODELS:")
    for name, grid in EMBEDDING_GRIDS.items():
        print(f"  • {name:20s}: {get_grid_size(grid):6,d} combinações")

    print("\n🧠 GNN CLASSIFIERS:")
    for name, grid in GNN_GRIDS.items():
        print(f"  • {name:20s}: {get_grid_size(grid):6,d} combinações")

    print("\n🤖 CLASSICAL ML:")
    for name, grid in CLASSIFIER_GRIDS.items():
        print(f"  • {name:20s}: {get_grid_size(grid):6,d} combinações")

    total = sum(get_grid_size(grid) for grid in ALL_GRIDS.values())
    print(f"\n{'TOTAL':>22s}: {total:6,d} combinações")
    print("=" * 70 + "\n")


def get_model_grid(model_name: str) -> Dict[str, List[Any]]:
    """
    Retorna o grid para um modelo específico.

    Args:
        model_name: Nome do modelo (ex: "GAE", "KNN", "MLP")

    Returns:
        Dicionário com os parâmetros do grid

    Raises:
        KeyError: Se o modelo não existir
    """
    if model_name not in ALL_GRIDS:
        available = ", ".join(ALL_GRIDS.keys())
        raise KeyError(
            f"Modelo '{model_name}' não encontrado. "
            f"Modelos disponíveis: {available}"
        )
    return ALL_GRIDS[model_name]


# ============================================================
# 🧪 QUICK TESTS (executar como script)
# ============================================================

if __name__ == "__main__":
    print_grid_summary()

    # Teste de importação
    print("\n✅ Teste de acesso:")
    print(f"  GAE embedding_dim: {GAE_VGAE_GRID['embedding_dim']}")
    print(f"  KNN n_neighbors: {KNN_GRID['n_neighbors']}")

    # Teste de função
    print(f"\n✅ Grid size GAE/VGAE: {get_grid_size(GAE_VGAE_GRID):,} combinações")
