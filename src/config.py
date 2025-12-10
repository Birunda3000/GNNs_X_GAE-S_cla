# src/config.py (VERSÃO CORRIGIDA E LIMPA)

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo


class Config:
    """
    Classe centralizada para todas as configurações globais do projeto.
    """

    # --- Timestamp da Execução ---
    TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    # --- Ambiente ---
    DEVICE = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"

    # Semente global (para reprodutibilidade)
    RANDOM_SEED = 25369  # antes: int(time.time())

    # --- Splits ---
    TRAIN_SPLIT_RATIO = 0.8  # geralmente não usado porque Musae tem split próprio

    # --- Hiperparâmetros gerais para GAE/VGAE/GNN ---
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    OUT_EMBEDDING_DIM = 3  # variar em [8, 32, 64, 128] no Optuna

    # --- Treinamento ---
    EPOCHS = 500
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4

    EARLY_STOPPING_PATIENCE = 32
    EARLY_STOPPING_MIN_DELTA = 1e-6

    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.6
    MIN_LR = 1e-8

    # --- Visualização ---
    VIS_SAMPLES = 1500


print(f"Configurações carregadas. Usando dispositivo: {Config.DEVICE}")


'''
TRAINING_CONFIG: Dict[str, Any] = {
    "epochs": 500,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "early_stopping_patience": 32,
    "early_stopping_min_delta": 1e-6,
    "scheduler_patience": 5,
    "scheduler_factor": 0.6,
    "min_lr": 1e-8,
}
'''
