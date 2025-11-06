import os
from typing import List, Dict, Any, cast
import psutil
import torch
from torch_geometric.data import Data
from functools import partial

from memory_profiler import memory_usage

from src.config import Config
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.data_format_definition import WSG

from src.classifiers import BaseClassifier
from src.utils import format_bytes




class ExperimentRunner:
    """Orquestra a execução de um experimento de classification."""

    def __init__(
        self, 
        data_converter,
        config: Config, 
        run_folder_name: str, 
        wsg_obj: WSG, 
        data_source_name: str
    ):
        self.config = config
        self.wsg_obj = wsg_obj
        self.data_source_name = data_source_name
        self.directory_manager = DirectoryManager(config.TIMESTAMP, run_folder_name)
        self.data_converter = data_converter


        if "cuda" in config.DEVICE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(config.DEVICE)
            print("VRAM (GPU) Peak Stats zeradas.")


    def run(self, models_to_run: List[BaseClassifier], process: psutil.Process, mem_start: int):
        """Executa o pipeline."""
        report_manager = ReportManager(self.directory_manager)

        report = {}
        report["input_wsg_file"] = self.data_source_name
        report["results_summary_per_model"] = {}
        report["detailed_results_per_model"] = {}

        report["memory_summary"] = {"ram_start_readable": format_bytes(mem_start)}
        report["memory_per_model"] = {}


        print("\n[ExperimentRunner] Carregando e convertendo dados...")

        data = self.data_converter(wsg=self.wsg_obj, config=self.config, train_size_ratio=self.config.TRAIN_SPLIT_RATIO)

        ram_after_data_load = process.memory_info().rss
        data_load_increase = ram_after_data_load - mem_start
        print(f"[ExperimentRunner] Dados carregados. Aumento de RAM: {format_bytes(data_load_increase)}")
        print(f"[ExperimentRunner] RAM atual após carregar dados: {format_bytes(ram_after_data_load)}")



        report["memory_summary"]["ram_after_data_load_readable"] = format_bytes(ram_after_data_load)
        report["memory_summary"]["ram_data_load_increase_readable"] = format_bytes(data_load_increase)

        peak_ram_overall = ram_after_data_load
        peak_vram_bytes = 0


        for model in models_to_run:
            print(f"\n--- 📊 Executando: {model.model_name} ---")

            # --- 4. MEDIÇÃO DE PICO COM memory_profiler ---
            # Forma recomendada: tupla (func, args, kwargs) + cast para agradar o type checker
            func = model.train_and_evaluate
            args = []
            kwargs = {"data": data}
            # Executa a função e mede o pico (max_usage=True). Retorna (pico_em_MiB, retval)
            mem_usage_result, (acc, f1, train_time, model_report) = memory_usage(
                proc=cast(Any, (func, args, kwargs)),
                max_usage=True,   # retorna apenas o pico
                retval=True,      # retorna os valores que a função original retornaria
                interval=0.1,     # intervalo de checagem
            )
            peak_ram_model_mib = mem_usage_result

             # Salva o pico específico do modelo no relatório
             # Guarda o valor em MiB e a versão formatada
            report["memory_per_model"][model.model_name] = {
                "peak_ram_MiB": peak_ram_model_mib,
                "peak_ram_readable": format_bytes(int(peak_ram_model_mib * 1024 * 1024)),  # converte MiB -> bytes
            }



            print(f"--- PICO de RAM durante {model.model_name}: {format_bytes(peak_ram_model_mib)} ---")

            # Atualiza o pico GERAL (convertendo MiB para Bytes para comparar com psutil)
            peak_ram_overall = max(peak_ram_overall, int(peak_ram_model_mib * 1024 * 1024))


            # Checa pico de VRAM (PyTorch faz isso bem)
            if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                # É importante checar após o modelo rodar, pois o pico pode ocorrer a qualquer momento
                current_vram_peak = torch.cuda.max_memory_allocated(self.config.DEVICE)
                if current_vram_peak > peak_vram_bytes:
                    peak_vram_bytes = current_vram_peak


            # Salva resultados normais
            report["results_summary_per_model"][model.model_name] = {
                "accuracy": acc,
                "f1_score_weighted": f1,
                "training_time_seconds": train_time,
            }
            report["detailed_results_per_model"][f"{model.model_name}_model_report"] = model_report


        # --- 5. Relatório Final Atualizado ---
        mem_end_run = process.memory_info().rss
        report["memory_summary"].update({
            "ram_end_readable": format_bytes(mem_end_run),
            "ram_peak_overall_readable": format_bytes(peak_ram_overall), # Pico MÁXIMO (dados OU treino)
            "vram_peak_readable": format_bytes(peak_vram_bytes)
        })
        print(f"\n--- Resumo do Runner ---")
        print(f"PICO de RAM (Geral - Dados OU Treino): {format_bytes(peak_ram_overall)}")
        print(f"PICO de VRAM (Geral): {format_bytes(peak_vram_bytes)}")


        metric_to_folder = "f1_score_weighted"
        best_model = max(report["results_summary_per_model"].items(), key=lambda x: x[1][metric_to_folder])

        best_metric = best_model[1][metric_to_folder]
        
        best_model_name = best_model[0].lower().replace("classifier", "")
        
        final_path = self.directory_manager.finalize_run_directory(
            dataset_name=self.wsg_obj.metadata.dataset_name,
            metrics={f"best_{metric_to_folder[:3]}": best_metric, "model": best_model_name},
        )
        report_manager.create_report(report)
        report_manager.save_report()
        print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")