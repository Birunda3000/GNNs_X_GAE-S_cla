
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn as nn


EPOCHS = 500
LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 32  # Épocas sem melhora antes de parar
EARLY_STOPPING_MIN_DELTA = 1e-6  # Melhora mínima para considerar como progresso

SCHEDULER_PATIENCE = 10  # Épocas sem melhora antes de reduzir LR
SCHEDULER_FACTOR = 0.6  # Fator de redução do LR
MIN_LR = 1e-8  # LR mínimo permitido
WEIGHT_DECAY = 5e-4



GAE_VGAE = {
    "layer_type": [SAGEConv, GCNConv],
    "num_layers": [2, 3],
    "embedding_dim": [128,256], # Dimensão do embedding das features de entrada
    "hidden_dim": [128,256], # Dimensão da camada oculta
    "out_embedding_dim": [32], # Dimensão do embedding final de saida
    "activation": [nn.ReLU, nn.LeakyReLU, nn.ELU],
    "dropout": [0.2, 0.5]
}

GNNs = {
    "layer_type": [SAGEConv, GCNConv],
    "num_layers": [2,3],
    #"embedding_dim": [64,128,256], # Dimensão do embedding das features de entrada
    "hidden_dim": [128,256], # Dimensão da camada oculta
    #"out_embedding_dim": [32], # Dimensão do embedding final de saida
    "activation": [nn.ReLU, nn.LeakyReLU, nn.ELU],
    "dropout": [0.2, 0.5]
}

KNN_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
}

RF_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 20, 40],
    "max_features": ["sqrt", "log2"],
}

XGB_GRID = {
    "max_depth": [3, 6, 10],
    "n_estimators": [200, 500],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],  # <== ajuste recomendado!
    
    # parâmetros fixos (não fazem parte do grid)
    "learning_rate": [0.1],      # fixo
    "gamma": [0],                 # fixo
    "reg_lambda": [1.0],          # fixo
}


MLP_GRID = {
    "hidden_dim": [8, 32, 64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.25, 0.5],

    # Parâmetros fixos
    "activation": [nn.ReLU],   # fixo
}





