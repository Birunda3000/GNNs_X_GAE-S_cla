from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# GLOBAL TRAINING HYPERPARAMETERS (validados academicamente)
# ============================================================
# Mantidos pois possuem forte validação teórica e empírica (GCN, GraphSAGE, VGAE papers)

EPOCHS = 500                      # suficiente p/ convergência, com early stopping
LEARNING_RATE = 1e-3              # valor padrão em GAE/GNN benchmarks
EARLY_STOPPING_PATIENCE = 32      # tolerante, evita parar cedo demais
EARLY_STOPPING_MIN_DELTA = 1e-6   # melhora mínima estável
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.6
MIN_LR = 1e-8
WEIGHT_DECAY = 5e-4               # padrão clássico em GCN e GraphSAGE


# ============================================================
# SEARCH SPACE — GAE / VGAE (UNSUPERVISED)
# ============================================================
# Usamos CLASSES de ativação (nn.ReLU, nn.ELU, etc.), pois:
# - são serializáveis
# - aparecem em logs/hyperparam tracking
# - são compatíveis com checkpointing
# - seguem explicitamente o padrão PyTorch + PyG
# NUNCA usar F.relu no grid (apenas no forward).

GAE_VGAE = {
    "layer_type": [SAGEConv, GCNConv],         # ambos amplamente usados na literatura
    "num_layers": [2, 3],                      # profundidade típica (evita oversmoothing)
    "embedding_dim": [128, 256],               # projeção densa inicial (EmbeddingBag)
    "hidden_dim": [128, 256],                  # camadas internas
    "out_embedding_dim": [32, 64, 128],        # emb de saída (variar mais tarde)
    "activation": [nn.ReLU, nn.LeakyReLU, nn.ELU],
    "dropout": [0.2, 0.5],
}


# ============================================================
# SEARCH SPACE — GNNs SUPERVISIONADAS (GCN / GraphSAGE)
# ============================================================
# Mantemos conjunto paralelo ao GAE/VGAE por justiça comparativa

GNNs = {
    "layer_type": [SAGEConv, GCNConv],
    "num_layers": [2, 3],
    "hidden_dim": [128, 256],
    "activation": [nn.ReLU, nn.LeakyReLU, nn.ELU],
    "dropout": [0.2, 0.5],
}


# ============================================================
# SEARCH SPACE — KNN
# ============================================================
# Seleção baseada em estudos de dimensionalidade alta (embeddings)
# k altos podem suavizar ruído, k baixos preservam sinal local

KNN_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
}


# ============================================================
# SEARCH SPACE — RANDOM FOREST
# ============================================================
# max_features=["sqrt", "log2"] são defaults clássicos e robustos.

RF_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 20, 40],
    "max_features": ["sqrt", "log2"],
}


# ============================================================
# SEARCH SPACE — LOGISTIC REGRESSION (ADICIONADO)
# ============================================================
# LR é crucial: modelos lineares servem como baseline interpretável.
# ‘C’ controla inverso de regularização → valores típicos
# saga/lbfgs funcionam bem para embeddings médios.

LOGREG_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs", "saga"],
    "max_iter": [1000],
}


# ============================================================
# SEARCH SPACE — XGBOOST
# ============================================================
# Hiperparâmetros padrão, todos cientificamente sustentados.
# colsample_bytree ajustado (0.5/0.8/1.0) — prática comum para embeddings.

XGB_GRID = {
    "max_depth": [3, 6, 10],
    "n_estimators": [200, 500],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],  # recomendado

    # parâmetros fixos recomendados
    "learning_rate": [0.1],
    "gamma": [0],
    "reg_lambda": [1.0],
}


# ============================================================
# SEARCH SPACE — MLP (classificador simples)
# ============================================================
# Apenas ReLU porque MLP pequeno não se beneficia de ativação variada.

MLP_GRID = {
    "hidden_dim": [8, 32, 64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.25, 0.5],

    "activation": [nn.ReLU],   # fixo
}
