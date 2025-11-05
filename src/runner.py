import os
from typing import List, Dict
import psutil
import torch
from torch_geometric.data import Data
from functools import partial # <-- 1. IMPORTADO para ajudar a chamar o método

# --- 2. IMPORTAR memory_profiler ---
try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    print("AVISO: memory_profiler não está instalado (pip install memory_profiler).")
    print("       A medição de pico de memória por modelo será desativada.")
    MEMORY_PROFILER_AVAILABLE = False
# --- FIM DA IMPORTAÇÃO ---

from src.config import Config
from src.directory_manager import DirectoryManager
from src.data_format_definition import WSG
# Assume que classifiers.py já foi refatorado para receber 'data'
from src.classifiers import BaseClassifier
from src.data_converter import DataConverter


# --- 3. AJUSTAR format_bytes ---
def format_bytes(b):
    """Converte bytes ou MiB para um formato legível (MB ou GB)."""
    # Converte de MiB (memory_profiler) para Bytes antes de formatar, se necessário
    if isinstance(b, float): # memory_profiler retorna MiB (float)
        b = int(b * 1024 * 1024) # Converte MiB para Bytes

    # Converte Bytes para MB/GB
    if isinstance(b, int):
        if b < 1024**3:
            return f"{b / 1024**2:.2f} MB"
        return f"{b / 1024**3:.2f} GB"
    return "N/A" # Caso receba algo inesperado
# --- FIM DO AJUSTE ---


class ExperimentRunner:
    """Orquestra a execução de um experimento de classification."""

    def __init__(
        self, config: Config, run_folder_name: str, wsg_obj: WSG, data_source_name: str
    ):
        self.config = config
        self.wsg_obj = wsg_obj
        self.data_source_name = data_source_name
        self.directory_manager = DirectoryManager(config.TIMESTAMP, run_folder_name)
        self.results: Dict = {}
        self.reports: Dict = {}

        self.process = psutil.Process(os.getpid())
        self.mem_start = self.process.memory_info().rss

        self.reports["memory_summary"] = {
            "ram_start_readable": format_bytes(self.mem_start)
        }
        self.reports["memory_per_model"] = {}

        print(f"RAM inicial do processo de classificação: {format_bytes(self.mem_start)}")

        if "cuda" in config.DEVICE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(config.DEVICE)
            print("VRAM (GPU) Peak Stats zeradas.")


    def run(self, models_to_run: List[BaseClassifier], for_embedding_bag: bool):
        """Executa o pipeline."""

        print("\n[ExperimentRunner] Carregando e convertendo dados...")
        if not for_embedding_bag:
             print("!!! ESTE É O PONTO DE PICO DE MEMÓRIA PARA A ABORDAGEM INGENUA (MULTI-HOT) !!!")

        pyg_data = DataConverter.to_pyg_data(
            self.wsg_obj,
            for_embedding_bag=for_embedding_bag
        ).to(self.config.DEVICE)

        ram_after_data_load = self.process.memory_info().rss
        data_load_increase = ram_after_data_load - self.mem_start
        print(f"[ExperimentRunner] Dados carregados. Aumento de RAM: {format_bytes(data_load_increase)}")
        print(f"[ExperimentRunner] RAM atual após carregar dados: {format_bytes(ram_after_data_load)}")

        self.reports["memory_summary"]["ram_after_data_load_readable"] = format_bytes(ram_after_data_load)
        self.reports["memory_summary"]["ram_data_load_increase_readable"] = format_bytes(data_load_increase)

        peak_ram_overall = ram_after_data_load
        peak_vram_bytes = 0
        # REMOVIDO: ram_before_model não é mais necessário aqui

        for model in models_to_run:
            print(f"\n--- 📊 Executando: {model.model_name} ---")

            # --- 4. MEDIÇÃO DE PICO COM memory_profiler ---
            if MEMORY_PROFILER_AVAILABLE:
                # Cria uma função que chama 'model.train_and_evaluate' passando 'pyg_data'
                func_to_profile = partial(model.train_and_evaluate, data=pyg_data)

                # Executa a função e mede o pico (max_usage=True), obtendo o pico (MiB) e os retornos originais (retval=True)
                mem_usage_result, (acc, f1, train_time, report) = memory_usage(
                    func_to_profile,
                    max_usage=True, # Retorna apenas o pico
                    retval=True,    # Retorna os valores que a função original retornaria
                    interval=0.1    # Intervalo de checagem (pode ajustar)
                ) # mem_usage_result é um float (pico em MiB)

                peak_ram_model_mib = mem_usage_result

                # Salva o pico específico do modelo no relatório
                self.reports["memory_per_model"][model.model_name] = {
                    # Guarda o valor em MiB e a versão formatada
                    "peak_ram_MiB": peak_ram_model_mib,
                    "peak_ram_readable": format_bytes(peak_ram_model_mib)
                }
                print(f"--- PICO de RAM durante {model.model_name}: {format_bytes(peak_ram_model_mib)} ---")

                # Atualiza o pico GERAL (convertendo MiB para Bytes para comparar com psutil)
                peak_ram_overall = max(peak_ram_overall, int(peak_ram_model_mib * 1024 * 1024))

            else:
                # Fallback se memory_profiler não estiver instalado
                print("AVISO: memory_profiler não disponível. Medição de pico por modelo desativada.")
                # Executa normalmente sem medir o pico interno
                acc, f1, train_time, report = model.train_and_evaluate(pyg_data)
                # Atualiza o pico geral com a memória após a execução (menos preciso)
                peak_ram_overall = max(peak_ram_overall, self.process.memory_info().rss)
                self.reports["memory_per_model"][model.model_name] = {
                    "peak_ram_readable": "N/A (memory_profiler not installed)"
                }
            # --- FIM DA MEDIÇÃO ---

            # Checa pico de VRAM (PyTorch faz isso bem)
            if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                # É importante checar após o modelo rodar, pois o pico pode ocorrer a qualquer momento
                current_vram_peak = torch.cuda.max_memory_allocated(self.config.DEVICE)
                if current_vram_peak > peak_vram_bytes:
                    peak_vram_bytes = current_vram_peak

            # Salva resultados normais
            self.results[model.model_name] = {
                "accuracy": acc,
                "f1_score_weighted": f1,
                "training_time_seconds": train_time,
            }
            if report:
                self.reports[f"{model.model_name}_classification_report"] = report

        # --- 5. Relatório Final Atualizado ---
        mem_end_run = self.process.memory_info().rss
        self.reports["memory_summary"].update({
            "ram_end_readable": format_bytes(mem_end_run),
            "ram_peak_overall_readable": format_bytes(peak_ram_overall), # Pico MÁXIMO (dados OU treino)
            "vram_peak_readable": format_bytes(peak_vram_bytes)
        })
        print(f"\n--- Resumo do Runner ---")
        print(f"PICO de RAM (Geral - Dados OU Treino): {format_bytes(peak_ram_overall)}")
        print(f"PICO de VRAM (Geral): {format_bytes(peak_vram_bytes)}")

        # Salva o relatório (que agora contém as métricas de pico corretas)
        self.directory_manager.save_classification_report(
            input_file=self.data_source_name, results=self.results, reports=self.reports
        )
        self.directory_manager.print_summary_table(
            results=self.results,
            input_file_path=self.data_source_name,
            feature_type=self.wsg_obj.metadata.feature_type,
        )

        # Finaliza o diretório
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]["accuracy"])
            best_acc = best_model[1]["accuracy"]
            best_model_name = best_model[0].lower().replace("classifier", "")
            final_path = self.directory_manager.finalize_run_directory(
                dataset_name=self.wsg_obj.metadata.dataset_name,
                metrics={"best_acc": best_acc, "model": best_model_name},
            )
            print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")