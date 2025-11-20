# run_embedding_generation_batch.py

import os
import random
import time
from typing import Any, cast
from datetime import datetime
from zoneinfo import ZoneInfo

from memory_profiler import memory_usage
import numpy as np
import psutil
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from src.config import Config
import src.data_converters as data_converters
import src.data_loaders as data_loaders
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.models.embedding_models.autoencoders_models import GraphSageGAE, GraphSageGAE, GCNGAE, GCNVGAE
from src.early_stopper import EarlyStopper
from src.embeddings_eval import evaluate_embeddings
from src.utils import format_bytes, salvar_modelo_pytorch_completo, save_embeddings_to_wsg


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

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss
    print(f"RAM inicial do processo: {format_bytes(mem_start)}")  # ✅

    peak_ram_overall_bytes = mem_start

    if "cuda" in device.type and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        print("VRAM (GPU) Peak Stats zeradas.")

    # Seeds e prints
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print(f"Dispositivo: {device}")
    print(f"Dataset: {WSG_DATASET.dataset_name}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1] Carregando dados...")
    wsg_obj = WSG_DATASET.load()
    mem_after_load = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_load)

    print("\n[FASE 2] Convertendo para formato Pytorch Geometric...")
    pyg_data = data_converters.wsg_for_vgae(wsg_obj, config)
    mem_after_convert = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_convert)
    print(f"RAM após conversão: {format_bytes(mem_after_convert)}")  # ✅


    # --- Modelo ---
    print("\n[FASE 3] Construindo o modelo GraphSAGE-GAE...")
    model = GraphSageGAE(
        config=config,
        num_total_features=pyg_data.num_total_features,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        out_embedding_dim=config.OUT_EMBEDDING_DIM,
    ).to(device)


    directory_manager = DirectoryManager(timestamp=config.TIMESTAMP, run_folder_name="EMBEDDING_RUNS")
    report_manager = ReportManager(directory_manager)
    early_stopper = EarlyStopper(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max",
        metric_name="max_f1",
        custom_eval=lambda m: evaluate_embeddings(m, pyg_data, device),
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR,
        min_lr=config.MIN_LR,
    )

    mem_after_model = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_model)
    print(f"RAM após instanciar modelo: {format_bytes(mem_after_model)}")  # ✅

    # --- Treinamento ---
    print("\n[FASE 4] Treinando modelo...")
    func = model.train_model
    func_kwargs = dict(
        data=pyg_data,
        optimizer=optimizer,
        epochs=config.EPOCHS,
        early_stopper=early_stopper,
        scheduler=scheduler,
    )
    proc_tuple = (func, [], func_kwargs)
    mem_usage_result, training_report = memory_usage(
        proc=cast(Any, proc_tuple),
        max_usage=True,
        retval=True,
        interval=0.1,
    )
    peak_ram_train_func_mib = mem_usage_result or 0.0
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, int(peak_ram_train_func_mib * 1024 * 1024))

    mem_after_train = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_train)

    print(f"Treino concluído. RAM final: {format_bytes(mem_after_train)}")  # ✅
    if torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        print(f"VRAM pico: {format_bytes(peak_vram_bytes)}")  # ✅
    else:
        peak_vram_bytes = 0

    # --- Inferência ---
    print("\n[FASE FINAL] Inferência...")
    t0 = time.process_time()
    final_embeddings = model.inference(pyg_data)
    t1 = time.process_time()
    inference_duration = t1 - t0
    print(f"Inferência concluída em {inference_duration:.4f}s")

    # --- Relatórios e Salvamentos ---
    report = {
        "dataset_name": WSG_DATASET.dataset_name,
        "Embedding_Dim": emb_dim,
        "Inference_Duration_Seconds": inference_duration,
        "Timestamp": config.TIMESTAMP,
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
    # Lista de datasets e tamanhos de embedding
    datasets = [
        #data_loaders.MusaeFacebookLoader(),
        #data_loaders.MusaeGithubLoader(),
        data_loaders.FlickrLoader(),
    ]
    emb_sizes = [2, 3, 8, 16, 32, 64, 128]

    for dataset in datasets:
        for emb in emb_sizes:
            run_embedding_generation(dataset, emb)
