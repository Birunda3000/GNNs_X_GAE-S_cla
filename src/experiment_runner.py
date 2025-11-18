import os
from typing import List, Dict, Any, cast
import psutil
import torch
from torch_geometric.data import Data

# from functools import partial  # <- opcional: remover se não usar

from memory_profiler import memory_usage

from src.config import Config
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.data_format_definition import WSG

from src.models.base_model import BaseModel

from src.utils import format_bytes, format_mib


class ExperimentRunner:
    """Orquestra a execução de um experimento de classification."""

    def __init__(
        self,
        data_converter,
        config: Config,
        run_folder_name: str,
        wsg_obj: WSG,
        data_source_name: str,
    ):
        self.config = config
        self.wsg_obj = wsg_obj
        self.data_source_name = data_source_name
        self.directory_manager = DirectoryManager(config.TIMESTAMP, run_folder_name)
        self.data_converter = data_converter

        # Padroniza device para chamadas CUDA
        self._device = torch.device(config.DEVICE)

        if "cuda" in config.DEVICE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device)
            print("VRAM (GPU) Peak Stats zeradas.")

    def run(
        self, models_to_run: List[BaseModel], process: psutil.Process, mem_start: int
    ):
        """Executa o pipeline."""
        report_manager = ReportManager(self.directory_manager)

        report = {}
        report["input_wsg_file"] = self.data_source_name

        # Timestamp de embeddings (apenas quando o nome segue o padrão)
        if "_embeddings_" in self.data_source_name:
            ts_val = self.data_source_name.split("_embeddings_")[-1].split(".wsg.json")[
                0
            ]
        else:
            ts_val = None
        report["embedding_gen_timestamp"] = ts_val

        report["Random_Seed"] = self.config.RANDOM_SEED
        report["Timestamp"] = self.config.TIMESTAMP
        report["Train_Split_Ratio"] = self.config.TRAIN_SPLIT_RATIO
        report["Device"] = self.config.DEVICE

        report["results_summary_per_model"] = {}
        report["detailed_results_per_model"] = {}

        report["memory_summary"] = {"ram_start_readable": format_bytes(mem_start)}
        report["memory_per_model"] = {}

        print("\n[ExperimentRunner] Carregando e convertendo dados...")

        data = self.data_converter(
            wsg=self.wsg_obj,
            config=self.config,
            train_size_ratio=self.config.TRAIN_SPLIT_RATIO,
        )

        ram_after_data_load = process.memory_info().rss
        data_load_increase = ram_after_data_load - mem_start
        print(
            f"[ExperimentRunner] Dados carregados. Aumento de RAM: {format_bytes(data_load_increase)}"
        )
        print(
            f"[ExperimentRunner] RAM atual após carregar dados: {format_bytes(ram_after_data_load)}"
        )

        report["memory_summary"]["ram_after_data_load_readable"] = format_bytes(
            ram_after_data_load
        )
        report["memory_summary"]["ram_data_load_increase_readable"] = format_bytes(
            data_load_increase
        )

        peak_ram_overall = ram_after_data_load
        peak_vram_bytes = 0

        for model in models_to_run:
            print(f"\n--- 📊 Executando: {model.model_name} ---")

            func = model.train_model
            args = []
            kwargs = {"data": data}

            # memory_usage retorna (float_MiB, retval) com max_usage=True e retval=True
            mem_usage_result, model_report = memory_usage(
                proc=cast(Any, (func, args, kwargs)),
                max_usage=True,
                retval=True,
                interval=0.1,
            )
            peak_ram_model_mib = mem_usage_result

            report["memory_per_model"][model.model_name] = {
                "peak_ram_MiB": peak_ram_model_mib,
                "peak_ram_readable": format_mib(peak_ram_model_mib),  # ✅ correto
            }
            print(
                f"--- PICO de RAM durante {model.model_name}: {format_mib(peak_ram_model_mib)} ---"
            )

            peak_ram_overall = max(
                peak_ram_overall, int(peak_ram_model_mib * 1024 * 1024)
            )

            if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                current_vram_peak = torch.cuda.max_memory_allocated(self._device)
                if current_vram_peak > peak_vram_bytes:
                    peak_vram_bytes = current_vram_peak

            report["results_summary_per_model"][model.model_name] = {
                "test_accuracy": model_report["best_test_accuracy"],
                "test_f1_score_weighted": model_report["best_test_f1"],
                "val_f1_score_weighted": model_report["val_f1"],  
                "training_time_seconds": model_report["total_training_time"],
            }

        mem_end_run = process.memory_info().rss
        report["memory_summary"].update(
            {
                "ram_end_readable": format_bytes(mem_end_run),
                "ram_peak_overall_readable": format_bytes(peak_ram_overall),
                "vram_peak_readable": format_bytes(peak_vram_bytes),
            }
        )
        print(f"\n--- Resumo do Runner ---")
        print(
            f"PICO de RAM (Geral - Dados OU Treino): {format_bytes(peak_ram_overall)}"
        )
        print(f"PICO de VRAM (Geral): {format_bytes(peak_vram_bytes)}")

        # ✅ Escolhe melhor modelo com base em VALIDAÇÃO
        metric_to_select = "val_f1_score_weighted"
        best_model = max(
            report["results_summary_per_model"].items(),
            key=lambda x: x[1][metric_to_select],
        )

        best_val_f1 = best_model[1][metric_to_select]
        best_test_f1 = best_model[1][
            "test_f1_score_weighted"
        ]  # ✅ reporta teste também
        best_model_name = best_model[0].lower().replace("classifier", "")

        final_path = self.directory_manager.finalize_run_directory(
            dataset_name=self.wsg_obj.metadata.dataset_name,
            metrics={
                "best_val_f1": best_val_f1,
                "test_f1": best_test_f1,
                "model": best_model_name,
            },
        )
        report_manager.create_report(report)
        report_manager.save_report()
        print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")
