# run_embedding_generation_batch.py

import os
import random
import time
from typing import Any, cast
from datetime import datetime
from zoneinfo import ZoneInfo
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from src.config import Config
import src.data_converters as data_converters
import src.data_loaders as data_loaders
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.models.embedding_models.autoencoders_models import GraphSageGAE, GCNGAE, GCNVGAE
from src.models.embedding_models.din_gae import GithubVGAE, FacebookVGAE, RedditVGAE, TwitchVGAE

# ✅ IMPORTAÇÕES ATUALIZADAS: Nova Infraestrutura
from src.early_stopper import UniversalEarlyStopper
from src.embeddings_eval import (
    KNNMetric, LogRegMetric, QDAMetric, CentroidMetric, DTMetric, ReconstructionLossMetric
)

from src.utils import format_bytes, run_isolated_inference, salvar_modelo_pytorch_completo, save_embeddings_to_wsg, DeviceTimer, PeakMemoryProfiler


def run_embedding_generation(WSG_DATASET, emb_dim: int):
    """
    Roda o pipeline de geração de embeddings para o dataset e dimensão especificados.
    """
    print("\n" + "=" * 80)
    print(f"INICIANDO EXECUÇÃO PARA DATASET: {WSG_DATASET.dataset_name} | EMBEDDING_DIM: {emb_dim}")
    print("=" * 80)

    # --- Configuração Inicial ---
    config = Config()
    config.OUT_EMBEDDING_DIM = emb_dim  # sobrescreve dimensão
    config.TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    # Seeds e prints
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"Dispositivo: {config.DEVICE}")
    print(f"Dataset: {WSG_DATASET.dataset_name}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1] Carregando dados...")
    wsg_obj = WSG_DATASET.load()

    print("\n[FASE 2] Convertendo para formato Pytorch Geometric...")
    pyg_data = data_converters.wsg_for_vgae(wsg_obj, config)

    # --- Modelo ---
    print("\n[FASE 3] Construindo o modelo GAE/VGAE...")

    # Seleção baseada no nome do dataset carregado
    if "facebook" in WSG_DATASET.dataset_name.lower():
        model = FacebookVGAE(
            config=config,
            num_total_features=pyg_data.num_total_features,
            out_embedding_dim=config.OUT_EMBEDDING_DIM,
        )
    elif "github" in WSG_DATASET.dataset_name.lower():
        model = GithubVGAE(
            config=config,
            num_total_features=pyg_data.num_total_features,
            out_embedding_dim=config.OUT_EMBEDDING_DIM,
        )
    elif "reddit" in WSG_DATASET.dataset_name.lower():
        model = RedditVGAE(
            config=config,
            num_total_features=pyg_data.num_total_features,
            out_embedding_dim=config.OUT_EMBEDDING_DIM,
        )
    elif "twitch" in WSG_DATASET.dataset_name.lower():
        model = TwitchVGAE(
            config=config,
            num_total_features=pyg_data.num_total_features,
            out_embedding_dim=config.OUT_EMBEDDING_DIM,
        )
    else:
        raise ValueError(f"Modelo não definido para: {WSG_DATASET.dataset_name}")

    model = model.to(config.DEVICE)

    directory_manager = DirectoryManager(timestamp=config.TIMESTAMP, run_folder_name="EMBEDDING_RUNS")
    report_manager = ReportManager(directory_manager)
    
    # =====================================================================
    # ✅ NOVA INFRAESTRUTURA DE ORQUESTRAÇÃO (Early Stopper & Scheduler)
    # =====================================================================
    
    # 1. Empacota as métricas (Dando uma folga extra de +5 épocas para a Loss se estabilizar)
    metrica_guia = 0
    metricas_ativas = [
        ReconstructionLossMetric(patience=config.EARLY_STOPPING_PATIENCE + 5),
        KNNMetric(patience=config.EARLY_STOPPING_PATIENCE),
        LogRegMetric(patience=config.EARLY_STOPPING_PATIENCE),
        QDAMetric(patience=config.EARLY_STOPPING_PATIENCE),
        CentroidMetric(patience=config.EARLY_STOPPING_PATIENCE),
        DTMetric(patience=config.EARLY_STOPPING_PATIENCE)
    ]
    
    # 2. Instancia o Universal Stopper exigindo que TODAS estagnem
    early_stopper = UniversalEarlyStopper(
        metrics=metricas_ativas,
        stop_condition="all",
        restore_best=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 3. Scheduler amarrado à Validação do KNN (F1-Score: Quanto maior, melhor -> max)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metricas_ativas[metrica_guia].mode,
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR,
        min_lr=config.MIN_LR,
    )

    # =====================================================================
    # --- FASE 4: TREINAMENTO (Medição Isolada) ---
    # =====================================================================
    print("\n[FASE 4] Treinando modelo...")
    
    train_profiler = PeakMemoryProfiler(device=config.DEVICE, step_name="Treino")
    with train_profiler:
        training_report = model.train_model(
            data=pyg_data,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            early_stopper=early_stopper,
            scheduler=scheduler,
            scheduler_metric_name=metricas_ativas[metrica_guia].name
        )

    # =====================================================================
    # --- FASE 5: INFERÊNCIA (Medição Isolada de Tempo e Memória) ---
    # =====================================================================
    print("\n[FASE 5] Gerando Embeddings (Inferência Isolada)...")
    
    # Toda a mágica do Spawn, fila e discos acontece silenciosamente aqui
    inf_metrics, final_embeddings = run_isolated_inference(
        model=model, 
        pyg_data=pyg_data, 
        config=config, 
        save_dir=directory_manager.get_run_path()
    )
    
    print(f"Inferência concluída em {inf_metrics['duration']:.4f}s")


    # --- Relatórios e Salvamentos ---
    report = {
        "Metadata": {
            "Dataset_Name": WSG_DATASET.dataset_name,
            "Model_Name": getattr(model, "model_name", model.__class__.__name__),
            "Timestamp": config.TIMESTAMP,
        },
        "Graph_Structure": {
            "Num_Nodes": wsg_obj.metadata.num_nodes,
            "Num_Edges": wsg_obj.metadata.num_edges,
            "Num_Total_Features": wsg_obj.metadata.num_total_features,
            "Directed": wsg_obj.metadata.directed,
        },
        "Hyperparameters": {
            "Random_Seed": config.RANDOM_SEED,
            "Device": str(config.DEVICE),
            "Out_Embedding_Dim": emb_dim,
            "Epochs": config.EPOCHS,
            "Learning_Rate": config.LEARNING_RATE,
            "Weight_Decay": config.WEIGHT_DECAY,
            "Early_Stopping_Patience": config.EARLY_STOPPING_PATIENCE,
            "Early_Stopping_Min_Delta": config.EARLY_STOPPING_MIN_DELTA,
            "Scheduler_Patience": config.SCHEDULER_PATIENCE,
            "Scheduler_Factor": config.SCHEDULER_FACTOR,
            "Min_LR": config.MIN_LR,
        },
        "Performance_Metrics": {
            "Training_Duration_Seconds": training_report["total_training_time"],
            "Inference_Duration_Seconds": inf_metrics["duration"],
            
            # --- MÉTRICAS DE RAM HOST (CPU) ---
            "Training_RAM_Snapshot_Inicial_MB": train_profiler.cpu_ram_start_mb,
            "Training_RAM_Incremento_Real_MB": train_profiler.cpu_diff_mb,
            "Training_RAM_Teto_Absoluto_MB": train_profiler.cpu_ram_start_mb + train_profiler.cpu_diff_mb,
            
            "Inference_RAM_Snapshot_Inicial_MB": inf_metrics["cpu_ram_start_mb"],
            "Inference_RAM_Incremento_Real_MB": inf_metrics["cpu_diff_mb"],
            
            # --- MÉTRICAS DE VRAM DEVICE (GPU) ---
            "Training_VRAM_Pico_Alocado_MB": train_profiler.gpu_alloc_mb,
            "Training_VRAM_Pico_Reservado_MB": train_profiler.gpu_reserved_mb,
            
            "Inference_VRAM_Pico_Alocado_MB": inf_metrics["gpu_alloc_mb"],
            "Inference_VRAM_Pico_Reservado_MB": inf_metrics["gpu_reserved_mb"],
        },
        "Data_Split": {
            "Train_Size_Ratio_Configured": getattr(config, 'TRAIN_SPLIT_RATIO', 0.8),
            "Train_Nodes": int(pyg_data.train_mask.sum()),
            "Val_Nodes": int(pyg_data.val_mask.sum()),
            "Test_Nodes": int(pyg_data.test_mask.sum()),
        },
        "Training_Report": training_report,
    }

    report_manager.create_report(report)
    report_manager.save_report()

    salvar_modelo_pytorch_completo(
        model=model,
        dataset_name=WSG_DATASET.dataset_name,
        timestamp=config.TIMESTAMP,
        save_dir=directory_manager.get_run_path(),
    )

    save_embeddings_to_wsg(
        final_embeddings=final_embeddings,
        wsg_obj=wsg_obj,
        config=config,
        save_path=directory_manager.get_run_path(),
    )

    # ✅ CORREÇÃO FINAL: Acessa os "best_scores" pelo nome da métrica
    # Estamos nomeando a pasta com o melhor F1-Score do KNN para fácil identificação visual!
    metrics_to_name = {
        "knn_f1": training_report["best_scores"]["KNN"], 
        "emb_dim": emb_dim,
    }
    final_path = directory_manager.finalize_run_directory(
        dataset_name=WSG_DATASET.dataset_name, metrics=metrics_to_name
    )
    print(f"Resultados salvos em: {final_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # --- CONFIGURAÇÃO ---
    datasets = [
        #data_loaders.MusaeFacebookLoader(),
        data_loaders.MusaeGithubLoader(),
        #data_loaders.MusaeTwitchLoader(),
        #data_loaders.RedditLiteLoader(threads_per_class=20), 
    ]
    
    emb_sizes = [64]

    for dataset in datasets:
        for emb in emb_sizes:
            run_embedding_generation(dataset, emb)
            gc.collect()
            print("❄️  Pausa de 5s para resfriamento...")
            time.sleep(5)