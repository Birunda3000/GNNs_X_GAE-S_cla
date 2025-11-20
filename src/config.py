# src/config.py (versão atualizada)
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import time




class Config:
    """
    Classe centralizada para todas as configurações do projeto.
    """

    # --- Timestamp da Execução ---
    # Gera um timestamp único no momento da inicialização para identificar a execução.
    # Usa o fuso horário de São Paulo para consistência.
    TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )
    # --- Configurações do Ambiente ---
    DEVICE = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    RANDOM_SEED = 25369#int(time.time())

    TRAIN_SPLIT_RATIO = 0.8  # Proporção de dados para treino vs teste/validação

    # --- Hiperparâmetros do Modelo VGAE ---
    EMBEDDING_DIM = 128  # Dimensão do embedding das features de entrada
    HIDDEN_DIM = 256  # Dimensão da camada GCN oculta

    OUT_EMBEDDING_DIM: int = 3 # Dimensão do embedding final do nó variar [8,32,64,128]

    # --- Configurações de Treinamento ---
    EPOCHS = 500
    LEARNING_RATE = 1e-3

    EARLY_STOPPING_PATIENCE = 32  # Épocas sem melhora antes de parar
    EARLY_STOPPING_MIN_DELTA = 1e-6  # Melhora mínima para considerar como progresso

    SCHEDULER_PATIENCE = 10  # Épocas sem melhora antes de reduzir LR
    SCHEDULER_FACTOR = 0.6  # Fator de redução do LR
    MIN_LR = 1e-8  # LR mínimo permitido

    # --- Configurações de Visualização ---
    VIS_SAMPLES = 1500  # Número máximo de nós para incluir na visualização


print(f"Configurações carregadas. Usando dispositivo: {Config.DEVICE}")