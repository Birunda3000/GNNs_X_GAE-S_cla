# run_embedding_generation.py

# Standard library
import os
import random
import time
from typing import Any, cast

# Third-party
from memory_profiler import memory_usage
import numpy as np
import psutil
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

# Local application
from src.config import Config
import src.data_converters as data_converters
import src.data_loaders as data_loaders
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.models.embedding_models.autoencoders_models import GraphSageGAE
from src.early_stopper import EarlyStopper
from src.embeddings_eval import evaluate_embeddings
from src.utils import format_b, save_embeddings_to_wsg, salvar_modelo_pytorch_completo


WSG_DATASET = data_loaders.MusaeGithubLoader()# Ou MusaeFacebookLoader()


def main():
    """
    Função principal.
    """
    # --- Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)

    # --- INICIAR MONITORAMENTO GERAL ---
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss
    print(f"RAM inicial do processo: {format_b(mem_start)}")

    peak_ram_overall_bytes = mem_start  # Pico geral começa aqui

    if "cuda" in device.type and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        print("VRAM (GPU) Peak Stats zeradas.")

    # ... (seed, prints iniciais) ...
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    print("=" * 50)
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {WSG_DATASET.dataset_name}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1] Carregando dados do WSG...")

    wsg_obj = WSG_DATASET.load()

    mem_after_load = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_load)  # Atualiza pico

    print("\n[FASE 2] Convertendo dados para formato Pytorch Geometric...")
    pyg_data = data_converters.wsg_for_vgae(wsg_obj, config)

    mem_after_convert = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_convert)  # Atualiza pico

    print(f"RAM após converter para pyg_data (EmbeddingBag): {format_b(mem_after_convert)}")
    print("Pipeline de dados concluído.")

    # --- Instanciação do Modelo ---
    print("\n[FASE 3] Construindo o modelo VGAE...")
    model = GraphSageGAE(
        config=config,
        num_total_features=pyg_data.num_total_features,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        out_embedding_dim=config.OUT_EMBEDDING_DIM,
    ).to(device)

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name=f"EMBEDDING_RUNS",
    )
    report_manager = ReportManager(directory_manager)
    early_stopper = EarlyStopper(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max",
        metric_name="avg_f1_knn_logreg",
        custom_eval=lambda model: evaluate_embeddings(model, pyg_data, device)
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR, min_lr=config.MIN_LR)
    # ------

    mem_after_model = process.memory_info().rss
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_model)  # Atualiza pico
    print(f"RAM após instanciar modelo: {format_b(mem_after_model)}")


    print("Modelo construído com sucesso.")

    # --- Loop de Treinamento ---
    print("\n[FASE 4] Iniciando treinamento do modelo...")

    peak_ram_train_func_mib = 0.0  # Pico durante a função train_model (MiB)

    # memory_usage espera (func, args, kwargs) para perfilar uma função
    func = model.train_model
    func_args = []
    func_kwargs = {
        "data": pyg_data,
        "optimizer": optimizer,
        "epochs": config.EPOCHS,
        "early_stopper": early_stopper,
        "scheduler": scheduler,
    }
    proc_tuple = (func, func_args, func_kwargs)
    mem_usage_result, training_report = memory_usage(
        proc=cast(Any, proc_tuple),  # satisfaz o stub; runtime aceita a tupla
        max_usage=True,
        retval=True,
        interval=0.1,
    )
    peak_ram_train_func_mib = mem_usage_result or 0.0
    # Atualiza pico GERAL com o pico do treino (convertido para Bytes)
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, int(peak_ram_train_func_mib * 1024 * 1024))

    # --- FIM TREINO ---

    # --- Coleta de Métricas Finais ---
    mem_after_train = process.memory_info().rss  # RAM medida *após* o bloco de treino
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_train)

    print(f"\nRAM após treinamento (imediatamente após): {format_b(mem_after_train)}")
    print(f"RAM PICO durante a função treino: {format_b(int(peak_ram_train_func_mib * 1024 * 1024))}")

    peak_vram_bytes = 0
    if "cuda" in device.type and torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        print(f"PICO VRAM (GPU) durante treino: {format_b(peak_vram_bytes)}")

    print(f"Treinamento finalizado em {training_report['total_training_time']:.2f} segundos.")

    # --- Inferência e Salvamento ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    # run_path = directory_manager.get_run_path()

    inference_start_time = time.process_time()
    final_embeddings = model.inference(pyg_data)
    inference_end_time = time.process_time()
    inference_duration = inference_end_time - inference_start_time
    print(f"Geração de embeddings (inferência) concluída em {inference_duration:.4f} segundos.")

    mem_after_inference = process.memory_info().rss  # RAM no final de tudo
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_inference)

    # --- DICIONÁRIO DE MÉTRICAS CORRIGIDO ---
    memory_metrics = {
        # Valores em Bytes
        "ram_start_bytes": mem_start,
        "ram_after_load_bytes": mem_after_load,
        "ram_after_convert_bytes": mem_after_convert,
        "ram_after_model_bytes": mem_after_model,
        "ram_after_train_bytes": mem_after_train,  # RAM após o bloco de treino
        "ram_after_inference_bytes": mem_after_inference,  # RAM no final
        "peak_ram_overall_bytes": peak_ram_overall_bytes,  # Pico máximo geral (Bytes)
        "vram_peak_bytes": peak_vram_bytes,
        # Pico da função treino (MiB) - Chave esperada por save_report
        "peak_ram_train_func_MiB": peak_ram_train_func_mib,
        # Versões legíveis - Chaves esperadas por save_report
        "ram_start_readable": format_b(mem_start),
        "ram_after_load_readable": format_b(mem_after_load),
        "ram_after_convert_readable": format_b(mem_after_convert),
        "ram_after_model_readable": format_b(mem_after_model),
        "ram_after_train_readable": format_b(mem_after_train),  # Legível RAM após treino
        "ram_after_inference_readable": format_b(mem_after_inference),  # Legível RAM final
        "peak_ram_train_func_readable": format_b(peak_ram_train_func_mib),  # Pico da func treino formatado
        "peak_ram_overall_readable": format_b(peak_ram_overall_bytes),  # Pico GERAL formatado
        "vram_peak_readable": format_b(peak_vram_bytes),
    }

    # Salvar os artefatos
    # --- VERIFICAÇÃO E SALVAMENTO DO RELATÓRIO ---
    report = {
        "dataset_name": WSG_DATASET.dataset_name,
        "Random_Seed": config.RANDOM_SEED,
        "Timestamp": config.TIMESTAMP,
        "Train_Split_Ratio": config.TRAIN_SPLIT_RATIO,
        "Model": model.__class__.__name__,
        "Embedding_Dim": config.OUT_EMBEDDING_DIM,
        "Learning_Rate": config.LEARNING_RATE,
        "Device": config.DEVICE,
        "Training_Report": training_report,
        "Memory_Metrics": memory_metrics,
        "Inference_Duration_Seconds": inference_duration,
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

    # --- SALVAMENTO DOS RESULTADOS ---
    metrics_to_name = {
        "score": report["Training_Report"]["best_score"],
        "emb_dim": report["Embedding_Dim"],
    }
    final_path = directory_manager.finalize_run_directory(
        dataset_name=WSG_DATASET.dataset_name, metrics=metrics_to_name
    )

    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em: '{final_path}'")
    print("=" * 50)


if __name__ == "__main__":
    main()
