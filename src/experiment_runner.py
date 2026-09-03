import os
from typing import List, Dict, Any, cast
import psutil
import torch
from torch_geometric.data import Data
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# from functools import partial  # <- opcional: remover se não usar

from memory_profiler import memory_usage

from src.config import Config
from src.directory_manager import DirectoryManager
from src.report_manager import ReportManager
from src.data_format_definition import WSG
import time

from src.models.base_model import BaseModel
from src.models.sklearn.sklearn_model import SklearnClassifier

from src.utils import format_bytes, format_mib

import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_confusion_matrix_image(cm, model_name, source_info, split, output_dir=".", manual_dim=None):
    """
    Gera e salva a matriz de confusão organizada em subpastas.
    
    Estrutura criada: output_dir/confusion_matrix/{Nome_Modelo}/Arquivo.png
    """
    
    # 1. Lógica de Extração de Nomes
    basename = os.path.basename(source_info)
    match = re.search(r"(.+)_\((\d+)\)_embeddings", basename)
    
    if match:
        dataset_name = match.group(1)
        dim_val = match.group(2)
        dim_display = f"Dim: {dim_val}"
    else:
        dataset_name = source_info
        dim_val = str(manual_dim) if manual_dim else "Raw"
        dim_display = f"Dim: {manual_dim}" if manual_dim else "Features Originais"

    # 2. Sanitização de Nomes (para pastas e arquivos)
    safe_model = model_name.replace(" ", "_").replace("+", "plus").replace("/", "-")
    safe_dataset = dataset_name.replace(" ", "_")
    safe_split = split.replace(" ", "_")

    # --- NOVIDADE: CRIAÇÃO DA ESTRUTURA DE PASTAS ---
    # Define o caminho: output_dir / confusion_matrix / Nome_do_Modelo
    target_folder = os.path.join(output_dir, "confusion_matrix", safe_model)
    
    # Cria as pastas se não existirem (sem dar erro)
    os.makedirs(target_folder, exist_ok=True)
    # ------------------------------------------------

    # 3. Definição do Título e Nome do Arquivo
    title = f"{model_name} | {dataset_name} ({dim_display}) [{split}]"
    
    # Nome do arquivo: CM_Musae-Github_dim-16_Teste.png
    # (Tirei o nome do modelo do arquivo pois já está no nome da pasta, fica mais limpo)
    output_filename = f"CM_{safe_dataset}_dim-{dim_val}_{safe_split}.png"
    
    # Caminho final completo
    output_path = os.path.join(target_folder, output_filename)

    # 4. Plotagem
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(title, fontsize=12, fontweight='bold')
    
    # 5. Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matriz salva em: {output_path}")



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

            if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                torch.cuda.synchronize()

            gc.collect()


# === 2. MEDIÇÃO UNIFICADA (TEMPO + RAM) - IGUAL AO TREINO ===
            print(f"⏱️ Medindo inferência (Tempo + RAM) para {model.model_name}...")

            # Função Wrapper: Roda a inferência e devolve apenas o TEMPO gasto.
            # (O memory_usage vai monitorar a RAM enquanto essa função roda)
            def _inference_wrapper():
                # Sincroniza GPU antes do relógio
                if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t_start = time.perf_counter()

                # Lógica de Inferência (GNN vs Sklearn)
                if getattr(model, "use_gnn", False):
                    # GNN: Pipeline B (Grafo Completo)
                    if isinstance(model, torch.nn.Module):
                        model.eval()
                    with torch.no_grad():
                        model.inference(data)
                elif isinstance(model, SklearnClassifier):
                    # Sklearn: Pipeline A (Features)
                    model.inference(data.x)
                else:
                    raise ValueError(f"Modelo não suportado: {model.model_name}")

                # Sincroniza GPU depois da execução
                if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Retorna o tempo decorrido
                return time.perf_counter() - t_start

            # EXECUTA TUDO JUNTO:
            # - proc: Chama a função _inference_wrapper
            # - max_usage=True: Retorna o pico de RAM
            # - retval=True: Retorna também o valor que a função devolveu (o tempo)
            mem_usage_inf, inference_duration = memory_usage(
                proc=_inference_wrapper,
                max_usage=True,
                retval=True,
                interval=0.01 # Frequência alta (10ms) para pegar picos rápidos de inferência
            )
            
            # Tratamento de segurança para versões antigas do memory_profiler
            if isinstance(mem_usage_inf, list):
                mem_usage_inf = max(mem_usage_inf)

            print(f"   -> Tempo: {inference_duration:.6f}s | RAM Pico: {mem_usage_inf:.2f} MiB")




            report["results_summary_per_model"][model.model_name] = {
                "test_accuracy": model_report["test_accuracy"],
                "test_f1_score_weighted": model_report["test_f1"],

                "val_accuracy": model_report["val_accuracy"],
                "val_f1_score_weighted": model_report["val_f1"],

                "best_epoch": model_report.get("best_epoch", None),

                "inference_time_seconds": inference_duration,
                "pico_ram_MiB_durante_inferencia": mem_usage_inf,
                "training_time_seconds": model_report["total_training_time"],

                "detailed_reports": {
                    "test_report": model_report["test_report"],
                    "val_report": model_report["val_report"],
                    "train_report": model_report["train_report"],
                }
            }

#def save_confusion_matrix_image(cm, model_name, source_info, split, output_dir=".", manual_dim=None):

            save_confusion_matrix_image(
                cm=model_report["test_confusion_matrix"],
                model_name=model.model_name,
                source_info=self.data_source_name,
                split="teste",
                output_dir=self.directory_manager.get_run_path(),
            )

            save_confusion_matrix_image(
                cm=model_report["val_confusion_matrix"],
                model_name=model.model_name,
                source_info=self.data_source_name,
                split="validação",
                output_dir=self.directory_manager.get_run_path(),
            )

            save_confusion_matrix_image(
                cm=model_report["train_confusion_matrix"],
                model_name=model.model_name,
                source_info=self.data_source_name,
                split="treino",
                output_dir=self.directory_manager.get_run_path(),
            )

            gc.collect()
            print("❄️  Pausa de 10s para resfriamento da CPU...")
            time.sleep(10)

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
