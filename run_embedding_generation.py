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
from src.early_stopper import EarlyStopper
from src.embeddings_eval import evaluate_embeddings
from src.utils import format_bytes, salvar_modelo_pytorch_completo, save_embeddings_to_wsg
from src.utils import DeviceTimer, PeakMemoryProfiler


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
    
    device = torch.device(config.DEVICE)

    # Seeds e prints
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"Dispositivo: {device}")
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

    model = model.to(device)

    directory_manager = DirectoryManager(timestamp=config.TIMESTAMP, run_folder_name="EMBEDDING_RUNS")
    report_manager = ReportManager(directory_manager)
    early_stopper = EarlyStopper(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max",
        metric_name="max_f1",
        custom_eval=lambda m: evaluate_embeddings(m, pyg_data, device),
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR,
        min_lr=config.MIN_LR,
    )

    # =====================================================================
    # --- FASE 4: TREINAMENTO (Medição Isolada) ---
    # =====================================================================
    print("\n[FASE 4] Treinando modelo...")
    
    train_profiler = PeakMemoryProfiler(device=config.DEVICE, step_name="Treino_VGAE")
    
    with train_profiler:
        training_report = model.train_model(
            data=pyg_data,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            early_stopper=early_stopper,
            scheduler=scheduler,
        )

    # =====================================================================
    # --- FASE 5: INFERÊNCIA (Medição Isolada de Tempo e Memória) ---
    # =====================================================================
    print("\n[FASE 5] Gerando Embeddings (Inferência)...")
    
    inf_profiler = PeakMemoryProfiler(device=config.DEVICE, step_name="Inferencia_Embeddings")
    
    with inf_profiler:
        with DeviceTimer(config.DEVICE) as timer:
            final_embeddings = model.inference(pyg_data)
            
    inference_duration = timer.duration

    print(f"Inferência concluída em {inference_duration:.4f}s")

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
            "Device": str(device),
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
            "Inference_Duration_Seconds": inference_duration,
            "Training_Peak_RAM_MB": train_profiler.cpu_diff_mb,
            "Training_Peak_VRAM_MB": train_profiler.gpu_peak_mb,
            "Inference_Peak_RAM_MB": inf_profiler.cpu_diff_mb,
            "Inference_Peak_VRAM_MB": inf_profiler.gpu_peak_mb,
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

    metrics_to_name = {
        "score": training_report["best_score"],
        "emb_dim": emb_dim,
    }
    final_path = directory_manager.finalize_run_directory(
        dataset_name=WSG_DATASET.dataset_name, metrics=metrics_to_name
    )
    print(f"Resultados salvos em: {final_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":

    # --- CONFIGURAÇÃO LITE PARA TESTE FIM A FIM ---
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