import contextlib
import gc
import json
import logging
import os
import time
import resource
import traceback
from datetime import datetime, timezone
from typing import Any, Optional
from src.data_format_definition import Metadata, NodeFeaturesEntry, WSG
import traceback
import multiprocessing as mp

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("TCC-GNN-Profiler")

# Classe auxiliar para as cores no terminal
class TerminalColors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ==========================================================
# FORMATAÇÃO E CONVERSÃO DE MEMÓRIA
# ==========================================================


def _coerce_to_bytes(
    value: Any,
    assume_mib: bool = False,
) -> Optional[float]:
    """Converte um valor para bytes."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        numeric_value = float(value)
    elif isinstance(value, str):
        try:
            numeric_value = float(value)
        except ValueError:
            return None
    else:
        return None

    if assume_mib:
        return numeric_value * 1024 * 1024

    return numeric_value


def _format_memory_value(
    value: Any,
    assume_mib: bool = False,
) -> str:
    """Formata bytes como MB ou GB."""
    bytes_value = _coerce_to_bytes(value, assume_mib)

    if bytes_value is None or bytes_value < 0:
        return "N/A"

    gigabytes = bytes_value / (1024**3)

    if gigabytes >= 1:
        return f"{gigabytes:.2f} GB"

    megabytes = bytes_value / (1024**2)
    return f"{megabytes:.2f} MB"


def format_bytes(value: Any) -> str:
    """Formata um valor já convertido para bytes."""
    return _format_memory_value(value)


def format_mib(value: Any) -> str:
    """Formata um valor informado em MiB."""
    return _format_memory_value(value, assume_mib=True)


def fmt(value: Any, precision: int = 6) -> str:
    """Formata números com precisão definida."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"

    return "N/A"


# ==========================================================
# SALVAMENTO DE EMBEDDINGS
# ==========================================================


def save_embeddings_to_wsg(
    final_embeddings: torch.Tensor,
    wsg_obj: WSG,
    config: Any,
    save_path: str,
    tz_info=None,
) -> str:
    """Salva embeddings em um novo arquivo WSG."""
    final_embeddings = final_embeddings.detach().cpu()

    if tz_info is None:
        tz_info = datetime.now().astimezone().tzinfo or timezone.utc

    os.makedirs(save_path, exist_ok=True)

    output_metadata = Metadata(
        dataset_name=f"{wsg_obj.metadata.dataset_name}-Embeddings",
        feature_type="dense_continuous",
        num_nodes=wsg_obj.metadata.num_nodes,
        num_edges=wsg_obj.metadata.num_edges,
        num_total_features=config.OUT_EMBEDDING_DIM,
        processed_at=datetime.now(tz_info).isoformat(),
        directed=wsg_obj.metadata.directed,
    )

    embedding_indices = list(range(config.OUT_EMBEDDING_DIM))

    output_node_features = {
        str(node_id): NodeFeaturesEntry(
            indices=embedding_indices,
            weights=[float(value) for value in final_embeddings[node_id].tolist()],
        )
        for node_id in range(wsg_obj.metadata.num_nodes)
    }

    output_wsg = WSG(
        metadata=output_metadata,
        graph_structure=wsg_obj.graph_structure,
        node_features=output_node_features,
    )

    filename = (
        f"{wsg_obj.metadata.dataset_name}"
        f"_({config.OUT_EMBEDDING_DIM})"
        f"_embeddings_{config.TIMESTAMP}.wsg.json"
    )

    output_path = os.path.join(save_path, filename)

    try:
        payload = output_wsg.model_dump()
    except AttributeError:
        payload = output_wsg.dict()

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    logger.info("Embeddings salvos em: %s", output_path)
    return output_path


# ==========================================================
# MODELOS PYTORCH
# ==========================================================


def salvar_modelo_pytorch_completo(
    model: torch.nn.Module,
    dataset_name: str,
    timestamp: str,
    save_dir: str = "models",
) -> str:
    """Salva a arquitetura completa, pesos e buffers do modelo."""
    os.makedirs(save_dir, exist_ok=True)

    model_name = getattr(
        model,
        "model_name",
        model.__class__.__name__,
    )

    filename = f"{dataset_name}__{model_name}__{timestamp}.pt"
    save_path = os.path.join(save_dir, filename)

    torch.save(model, save_path)
    logger.info("Modelo completo salvo em: %s", save_path)

    return save_path


def carregar_modelo_pytorch_completo(
    save_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """Carrega um modelo PyTorch completo."""
    model = torch.load(save_path, map_location=device)
    model.eval()

    logger.info("Modelo carregado de: %s", save_path)
    return model


# ==========================================================
# MEDIÇÃO DE TEMPO
# ==========================================================


class DeviceTimer:
    """Mede o tempo de execução em CPU ou CUDA."""

    def __init__(
        self,
        device: str,
        disable_gc: bool = False,
    ):
        self.device = device.lower()
        self.disable_gc = disable_gc
        self.duration = 0.0

        if self.device == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.disable_gc:
            gc.disable()

        if self.device == "cuda":
            torch.cuda.synchronize()
            self.start_event.record()
        elif self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            raise ValueError(f"Dispositivo não suportado: {self.device}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            self.end_event.record()
            self.end_event.synchronize()
            self.duration = self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            self.duration = time.perf_counter() - self.start_time

        if self.disable_gc:
            gc.enable()

        return False


# ==========================================================
# PROFILING DE MEMÓRIA
# ==========================================================

class PeakMemoryProfiler(contextlib.ContextDecorator):
    """
    Em ambientes Linux/Docker, audita diretamente o Cgroups v2 para abranger vazamentos de multi-processing. Em GPU, monitora Alocação vs Reserva.
    """
    def __init__(self, device: str, step_name: str = "Execução"):
        self.step_name = step_name
        self.device_type = device.lower()
        
        # Caminhos do Cgroups v2 para Docker
        self.cgroup_current = "/sys/fs/cgroup/memory.current"
        self.cgroup_peak = "/sys/fs/cgroup/memory.peak"
        
        self.cpu_ram_start_mb = 0.0
        self.cpu_diff_mb = 0.0
        self.gpu_alloc_mb = 0.0
        self.gpu_reserved_mb = 0.0

    def _read_cgroup_metric(self, path: str) -> float:
        """Lê os controladores de memória do Kernel de forma passiva (O(1))."""
        try:
            with open(path, "r") as f:
                val = f.read().strip()
                if val == "max": return -1.0
                return int(val) / (1024 ** 2)
        except (FileNotFoundError, PermissionError):
            return 0.0

    def _get_cpu_ram_mb(self, use_peak: bool = False) -> float:
        """
        Tenta usar Cgroups v2 (Padrão Ouro Docker). 
        Faz fallback para USS multiplataforma se não estiver no Linux.
        """
        cgroup_val = self._read_cgroup_metric(self.cgroup_peak if use_peak else self.cgroup_current)
        if cgroup_val > 0:
            return cgroup_val
            
        # Fallback: USS (Ignora Page Cache, mas cego para subprocessos pesados)
        return psutil.Process(os.getpid()).memory_full_info().uss / (1024 ** 2)

    def __enter__(self):
        gc.collect()
        
        if self.device_type == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        self.cpu_ram_start_mb = self._get_cpu_ram_mb(use_peak=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device_type == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()

        # O pico atingido no bloco
        cpu_peak_after = self._get_cpu_ram_mb(use_peak=True)
        
        # Se Cgroups não for suportado, usamos o histórico base de ru_maxrss como fallback de pico
        if cpu_peak_after == 0.0:
             cpu_peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        self.cpu_diff_mb = cpu_peak_after - self.cpu_ram_start_mb
        
        if self.device_type == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.gpu_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        # Print padrão
        logger.info(f"==> [AUDITORIA DE MEMÓRIA] {TerminalColors.BOLD}{self.step_name}{TerminalColors.RESET}")
        logger.info(f"  -> [RAM Host] Snapshot Inicial: {self.cpu_ram_start_mb:.2f} MB")
        logger.info(f"  -> [RAM Host] Pico Histórico:   {cpu_peak_after:.2f} MB")
        
        # LÓGICA DE CORES DOS ALERTAS (RAM)
        if self.cpu_diff_mb <= 0.0:
            logger.warning(
                f"  -> [RAM Host] Incremento:       {TerminalColors.RED}{TerminalColors.BOLD}0.00 MB{TerminalColors.RESET} (Efeito Sombra)"
            )
        else:
            logger.info(
                f"  -> [RAM Host] Incremento Real:  {TerminalColors.GREEN}{TerminalColors.BOLD}{self.cpu_diff_mb:.2f} MB{TerminalColors.RESET}"
            )

        # MÉTRICAS DA VRAM
        if self.device_type in ['cuda'] and self.gpu_alloc_mb > 0:
            fragmentation_gap = self.gpu_reserved_mb - self.gpu_alloc_mb
            
            logger.info(f"  -> [VRAM GPU] Pico Alocado:     {TerminalColors.GREEN}{self.gpu_alloc_mb:.2f} MB{TerminalColors.RESET} (Matemática)")
            logger.info(f"  -> [VRAM GPU] Pico Reservado:   {TerminalColors.CYAN}{self.gpu_reserved_mb:.2f} MB{TerminalColors.RESET} (Requisito de Hardware)")
            
            if fragmentation_gap > 200: # Alerta se a fragmentação passar de 200MB
                logger.warning(
                    f"{TerminalColors.YELLOW}  -> [ALERTA VRAM] Gap de Fragmentação alto ({fragmentation_gap:.2f} MB). "
                    f"Considere exportar PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'{TerminalColors.RESET}"
                )

        return False



def _isolated_worker(queue, model_path, pyg_data_path, device_str):
    """ O Trabalhador sujo que roda em background """
    try:
        device = torch.device(device_str)
        
        pyg_data = torch.load(pyg_data_path, weights_only=False).to(device)
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        inf_profiler = PeakMemoryProfiler(device=device_str, step_name="Inferencia_Isolada")
        timer = DeviceTimer(device_str)
        
        with inf_profiler:
            with timer:
                with torch.no_grad():
                    # USANDO O SEU MÉTODO EXATO:
                    final_embeddings = model.inference(pyg_data)
                    
        results = {
            "duration": timer.duration,
            "cpu_ram_start_mb": inf_profiler.cpu_ram_start_mb,
            "cpu_diff_mb": inf_profiler.cpu_diff_mb,
            "gpu_alloc_mb": inf_profiler.gpu_alloc_mb,
            "gpu_reserved_mb": inf_profiler.gpu_reserved_mb,
            "embeddings": final_embeddings.cpu()
        }
        queue.put({"status": "success", "data": results})
    except Exception as e:
        queue.put({"status": "error", "error": f"{str(e)}\n{traceback.format_exc()}"})


def run_isolated_inference(model, pyg_data, config):
    """ 
    Orquestrador limpo. Esconde o multiprocessing do arquivo principal. 
    Retorna o dicionário de métricas e os embeddings gerados.
    """
    temp_model = os.path.join(config.OUTPUT_DIR, f"tmp_mod_{config.TIMESTAMP}.pt")
    temp_data = os.path.join(config.OUTPUT_DIR, f"tmp_dat_{config.TIMESTAMP}.pt")
    
    # Salva temporariamente para a passagem de bastão
    torch.save(model, temp_model)
    torch.save(pyg_data.cpu(), temp_data)
    
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    
    # Aciona o processo
    p = ctx.Process(target=_isolated_worker, args=(queue, temp_model, temp_data, str(config.DEVICE)))
    p.start()
    p.join()
    
    # Limpa a sujeira
    if os.path.exists(temp_model): os.remove(temp_model)
    if os.path.exists(temp_data): os.remove(temp_data)
    
    if not queue.empty():
        res = queue.get()
        if res["status"] == "error":
            raise RuntimeError(f"Erro na Inferência Isolada:\n{res['error']}")
        return res["data"], res["data"]["embeddings"]
        
    raise RuntimeError("O Worker morreu sem devolver resposta!")