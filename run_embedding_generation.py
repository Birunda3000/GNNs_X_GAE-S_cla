# run_embedding_generation.py (CORRIGIDO - Cálculo Pico Geral e Chaves)

import torch
import torch.optim as optim
import os
import time
import random
import numpy as np
import psutil
from functools import partial
from memory_profiler import memory_usage

from src.config import Config
import src.data_converters as data_converters
from src.model import VGAE
import src.data_loaders as data_loaders

from src.train import save_results, save_report, format_b
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.classifiers import salvar_modelo_pytorch
from src.train import save_embeddings_to_wsg


WSG_DATASET = data_loaders.MusaeGithubLoader()  # Ou MusaeFacebookLoader()


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
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS (VGAE)")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {WSG_DATASET.dataset_name}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1/2] Executando pipeline de dados...")


    wsg_obj = WSG_DATASET.load()


    mem_after_load = process.memory_info().rss
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, mem_after_load
    )  # Atualiza pico
    print(f"RAM após carregar wsg_obj: {format_b(mem_after_load)}")


    pyg_data = data_converters.wsg_for_vgae(wsg_obj)


    mem_after_convert = process.memory_info().rss
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, mem_after_convert
    )  # Atualiza pico
    print(
        f"RAM após converter para pyg_data (EmbeddingBag): {format_b(mem_after_convert)}"
    )
    print("Pipeline de dados concluído.")


    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name=f"EMBEDDING_RUNS({config.EPOCHS})", 
    )
    report_manager = ReportManager(directory_manager)




    # --- Instanciação do Modelo ---
    print("\n[FASE 3] Construindo o modelo VGAE...")
    model = VGAE(
        num_total_features=pyg_data.num_total_features,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        out_embedding_dim=config.OUT_EMBEDDING_DIM,
    ).to(device)

    mem_after_model = process.memory_info().rss
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, mem_after_model
    )  # Atualiza pico
    print(f"RAM após instanciar modelo: {format_b(mem_after_model)}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print("Modelo construído com sucesso.")

    # --- Loop de Treinamento ---
    print("\n[FASE 4] Iniciando treinamento do modelo...")

    peak_ram_train_func_mib = 0.0  # Pico durante a função train_model (MiB)


    func_to_profile = partial(
        model.train_model,
        input_data=pyg_data,
        learning_rate=config.LEARNING_RATE,
        optimizer=optimizer,
        weight_decay=getattr(config, "WEIGHT_DECAY", 0.0),
        epochs=config.EPOCHS,
    )
    mem_usage_result, (training_report) = memory_usage(
        func_to_profile, max_usage=True, retval=True, interval=0.1
    )
    peak_ram_train_func_mib = mem_usage_result or 0.0  # Garante float
    # Atualiza pico GERAL com o pico do treino (convertido para Bytes)
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, int(peak_ram_train_func_mib * 1024 * 1024)
    )

    # --- FIM TREINO ---

    # --- Coleta de Métricas Finais ---
    mem_after_train = process.memory_info().rss  # RAM medida *após* o bloco de treino
    # Garante que o pico geral considere o valor final
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_train)

    print(f"\nRAM após treinamento (imediatamente após): {format_b(mem_after_train)}")
    print(f"RAM PICO durante a função treino: {format_b(peak_ram_train_func_mib)}")

    peak_vram_bytes = 0
    if "cuda" in device.type and torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        print(f"PICO VRAM (GPU) durante treino: {format_b(peak_vram_bytes)}")

    print(f"Treinamento finalizado em {training_report['total_training_time']:.2f} segundos.")

    # --- Inferência e Salvamento ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    run_path = directory_manager.get_run_path()

    inference_start_time = time.process_time()
    final_embeddings = model.inference(pyg_data)
    inference_end_time = time.process_time()
    inference_duration = inference_end_time - inference_start_time
    print(
        f"Geração de embeddings (inferência) concluída em {inference_duration:.4f} segundos."
    )

    mem_after_inference = process.memory_info().rss  # RAM no final de tudo
    # Garante que o pico geral considere o valor final pós-inferência
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
        "ram_after_train_readable": format_b(
            mem_after_train
        ),  # Legível RAM após treino
        "ram_after_inference_readable": format_b(
            mem_after_inference
        ),  # Legível RAM final
        "peak_ram_train_func_readable": format_b(
            peak_ram_train_func_mib
        ),  # Pico da func treino formatado
        "peak_ram_overall_readable": format_b(
            peak_ram_overall_bytes
        ),  # Pico GERAL formatado
        "vram_peak_readable": format_b(peak_vram_bytes),
    }

    # Salvar os artefatos
    # --- VERIFICAÇÃO E SALVAMENTO DO RELATÓRIO ---
    report = {
        "Timestamp": config.TIMESTAMP,
        "dataset_name": WSG_DATASET.dataset_name,
        "Seed": config.RANDOM_SEED,
        "Model": model.__class__.__name__,
        "Embedding_Dim": config.OUT_EMBEDDING_DIM,
        "Training_Report": training_report,
    }
    report_manager.create_report(report)
    report_manager.add_report_section("Memory_Metrics", memory_metrics)
    report_manager.add_report_section("Inference_Duration_Seconds", {"inference_time": inference_duration})
    report_manager.save_report()

    salvar_modelo_pytorch(model=model, dataset_name=WSG_DATASET.dataset_name, timestamp=config.TIMESTAMP, save_dir=directory_manager.get_run_path())

    save_embeddings_to_wsg(final_embeddings, wsg_obj, config, save_path=run_path)
    
    #--- SALVAMENTO DOS RESULTADOS ---

    '''
    save_results(model, final_embeddings, wsg_obj, config, save_path=run_path)
    dataset_name = WSG_DATASET.dataset_name

    save_report(
        config,
        training_report["training_history"],
        training_report["total_training_time"],
        inference_duration,
        dataset_name,
        save_path=run_path,
        memory_metrics=memory_metrics,
    )
    '''


    final_metrics = training_report["training_history"][-1]
    run_metrics = {
        "train_loss": final_metrics.get("train_total_loss", 0.0),
        "emb_dim": config.OUT_EMBEDDING_DIM,
    }
    final_path = directory_manager.finalize_run_directory(
        dataset_name=WSG_DATASET.dataset_name, metrics=run_metrics
    )

    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em: '{final_path}'")
    print("=" * 50)


if __name__ == "__main__":
    main()
