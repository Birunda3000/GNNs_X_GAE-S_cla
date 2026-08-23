import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import torch
from src.data_format_definition import WSG, Metadata, NodeFeaturesEntry

import time
import torch
import gc

import resource
import contextlib
import memray


# ==========================================================
# 💡 FUNÇÕES AUXILIARES DE MEMÓRIA (corrigidas e seguras)
# ==========================================================

def _coerce_to_bytes(value: Any, assume_mib: bool = False) -> Optional[float]:
    """Converte um valor em bytes.
    
    Args:
        value: int (bytes), float (bytes OU MiB se assume_mib=True), ou string
        assume_mib: se True, trata float como MiB; se False, como bytes
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if assume_mib:
            return float(value) * 1024 * 1024
        else:
            return float(value)  # já em bytes
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return numeric * 1024 * 1024 if assume_mib else numeric
    return None


def _format_memory_value(value: Any, assume_mib: bool = False) -> str:
    """Formata bytes para MB ou GB."""
    bytes_value = _coerce_to_bytes(value, assume_mib=assume_mib)
    if bytes_value is None or bytes_value < 0:
        return "N/A"
    gigabytes = bytes_value / (1024 ** 3)
    if gigabytes >= 1:
        return f"{gigabytes:.2f} GB"
    megabytes = bytes_value / (1024 ** 2)
    return f"{megabytes:.2f} MB"


def format_bytes(b: Any) -> str:
    """Formata valores em bytes (int ou float já em bytes)."""
    return _format_memory_value(b, assume_mib=False)


def format_mib(m: Any) -> str:
    """Formata valores em MiB (ex: do memory_profiler)."""
    return _format_memory_value(m, assume_mib=True)


def fmt(val, precision=6):
    """Formata floats de forma segura; se None ou inválido, retorna 'N/A'."""
    return f"{val:.{precision}f}" if isinstance(val, (int, float)) else "N/A"


# ==========================================================
# 💾 SALVAR EMBEDDINGS EM FORMATO WSG (corrigido)
# ==========================================================

def save_embeddings_to_wsg(
    final_embeddings: torch.Tensor,
    wsg_obj: WSG,
    config,
    save_path: str,
    tz_info=None
) -> str:
    """
    Salva os embeddings finais em um novo arquivo WSG.

    Args:
        final_embeddings (torch.Tensor): Tensor de embeddings (num_nodes x dim)
        wsg_obj (WSG): Objeto WSG original (metadados + estrutura do grafo)
        config: Objeto de configuração (precisa ter OUT_EMBEDDING_DIM e EPOCHS)
        save_path (str): Caminho onde o arquivo será salvo
        tz_info (timezone, opcional): Fuso horário para timestamps

    Returns:
        str: Caminho completo do arquivo salvo
    """
    # Garante que os embeddings estão no CPU
    final_embeddings = final_embeddings.detach().cpu()

    # Fuso horário padrão
    if tz_info is None:
        tz_info = datetime.now().astimezone().tzinfo or timezone.utc

    os.makedirs(save_path, exist_ok=True)

    # --- METADADOS ---
    output_metadata = Metadata(
        dataset_name=f"{wsg_obj.metadata.dataset_name}-Embeddings",
        feature_type="dense_continuous",
        num_nodes=wsg_obj.metadata.num_nodes,
        num_edges=wsg_obj.metadata.num_edges,
        num_total_features=config.OUT_EMBEDDING_DIM,
        processed_at=datetime.now(tz_info).isoformat(),
        directed=wsg_obj.metadata.directed,
    )

    # --- EMBEDDINGS ---
    embedding_indices = list(range(config.OUT_EMBEDDING_DIM))

    # ✅ Corrigido: campos de NodeFeaturesEntry agora estão corretos
    output_node_features = {
        str(node_id): NodeFeaturesEntry(
            indices=embedding_indices,
            weights=[float(value) for value in final_embeddings[node_id].tolist()],
        )
        for node_id in range(wsg_obj.metadata.num_nodes)
    }

    # --- CRIA O NOVO WSG ---
    output_wsg = WSG(
        metadata=output_metadata,
        graph_structure=wsg_obj.graph_structure,
        node_features=output_node_features,
    )

    # --- SALVAMENTO ---
    dataset_name = wsg_obj.metadata.dataset_name
    filename = (
        f"{dataset_name}_({config.OUT_EMBEDDING_DIM})_embeddings_{config.TIMESTAMP}.wsg.json"
    )
    output_path = os.path.join(save_path, filename, )

    # Usa método compatível com Pydantic v2+
    try:
        payload = output_wsg.model_dump()
    except AttributeError:
        payload = output_wsg.dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Embeddings salvos em: '{output_path}'")
    return output_path


# ==========================================================
# 🧠 FUNÇÕES DE MODELO PYTORCH
# ==========================================================

def salvar_modelo_pytorch_completo(
    model,
    dataset_name: str,
    timestamp: str,
    save_dir: str = "models"
):
    """Salva o modelo PyTorch completo (arquitetura + pesos + buffers)."""
    os.makedirs(save_dir, exist_ok=True)

    model_name = getattr(model, "model_name", model.__class__.__name__)
    base_name = f"{dataset_name}__{model_name}__{timestamp}"

    save_path = os.path.join(save_dir, f"{base_name}.pt")

    torch.save(model, save_path)
    print(f"✅ Modelo completo salvo em: {save_path}")
    return save_path


def carregar_modelo_pytorch_completo(save_path: str, device: str = "cpu"):
    """Carrega um modelo completo salvo com torch.save(model)."""
    model = torch.load(save_path, map_location=device)
    model.eval()
    print(f"🔁 Modelo carregado de: {save_path}")
    return model



class DeviceTimer:
    """
    Temporizador de alto rigor metodológico para HPC.
    Garante precisão de hardware (Event) para GPU e de sistema (perf_counter) para CPU.
    """
    # Adicionamos a variável de controle 'disable_gc' (padrão False por segurança)
    def __init__(self, device: str, disable_gc: bool = False):
        self.device = device
        self.duration = 0.0
        self.disable_gc = disable_gc
        
        # Pré-alocação mandatória para evitar overhead de alocação de memória pelo SO
        if self.device == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        # Controle explícito: Se você pediu, ele desliga. Se não pediu, ele nem toca no GC.
        if self.disable_gc:
            gc.disable()
        
        if self.device == "cuda":
            torch.cuda.synchronize() # Limpa pendências residuais antes de largar
            self.start_event.record()
        elif self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            raise ValueError(f"Dispositivo não suportado: {self.device}")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            self.end_event.record()
            # Sincronização direcionada exclusivamente para este evento
            self.end_event.synchronize() 
            self.duration = self.start_event.elapsed_time(self.end_event) / 1000.0
        elif self.device == "cpu":
            self.duration = time.perf_counter() - self.start_time
            
        # Controle explícito: Tudo ou nada, respeitando a flag.
        if self.disable_gc:
            gc.enable()

        return False  # Propaga exceções, se houver









class PyTorchRigorousMemoryProfiler(contextlib.ContextDecorator):
    """
    Context Manager analítico O(1) delineado para auditorias de GNNs.
    Mapeia os footprints cruzados entre Cgroups, Kernel (C/C++), e CUDA VRAM.
    """
    def __init__(self, trace_file: str = "gnn_transient_spike_trace.bin", device_id: int = 0):
        self.trace_file = trace_file
        self.device_id = device_id
        
        self.cgroup_base = "/sys/fs/cgroup"
        self.cgroup_current = os.path.join(self.cgroup_base, "memory.current")
        self.cgroup_peak = os.path.join(self.cgroup_base, "memory.peak")
        self.cgroup_max = os.path.join(self.cgroup_base, "memory.max")
        
        self.memray_tracker = memray.Tracker(self.trace_file)
        self.metrics = {}

    def _read_cgroup_metric(self, path: str) -> int:
        try:
            with open(path, "r") as f:
                value = f.read().strip()
                return int(value) if value != "max" else -1
        except (FileNotFoundError, PermissionError):
            return 0

    def __enter__(self):
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.reset_accumulated_memory_stats(self.device_id)
        
        self.start_rss_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.start_cgroup_current = self._read_cgroup_metric(self.cgroup_current)
        
        self.memray_tracker.__enter__()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.memray_tracker.__exit__(exc_type, exc_val, exc_tb)
        
        end_rss_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        end_cgroup_current = self._read_cgroup_metric(self.cgroup_current)
        peak_cgroup_historical = self._read_cgroup_metric(self.cgroup_peak)
        docker_limit = self._read_cgroup_metric(self.cgroup_max)
        
        if torch.cuda.is_available():
            peak_vram_alloc = torch.cuda.max_memory_allocated(self.device_id)
            peak_vram_reserved = torch.cuda.max_memory_reserved(self.device_id)
        else:
            peak_vram_alloc = peak_vram_reserved = 0
            
        B2MB = 1024 ** 2
        
        self.metrics = {
            "duration_secs": self.end_time - self.start_time,
            "host_process_peak_rss_mb": end_rss_ru / 1024.0, 
            "docker_cgroup_initial_mb": self.start_cgroup_current / B2MB,
            "docker_cgroup_final_mb": end_cgroup_current / B2MB,
            "docker_cgroup_peak_mb": peak_cgroup_historical / B2MB,
            "docker_memory_limit_mb": docker_limit / B2MB if docker_limit > 0 else float('inf'),
            "cuda_vram_peak_allocated_mb": peak_vram_alloc / B2MB,
            "cuda_vram_peak_reserved_mb": peak_vram_reserved / B2MB,
            "cuda_vram_fragmentation_gap_mb": (peak_vram_reserved - peak_vram_alloc) / B2MB,
        }
        return False