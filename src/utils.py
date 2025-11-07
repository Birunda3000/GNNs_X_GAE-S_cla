import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import torch
from src.data_format_definition import WSG, Metadata, NodeFeaturesEntry


# ==========================================================
# 💡 FUNÇÕES AUXILIARES DE MEMÓRIA (corrigidas e seguras)
# ==========================================================

def _coerce_to_bytes(value: Any) -> Optional[float]:
    """Converte um valor em bytes, aceitando int, float (MiB) ou string numérica."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        # Considera float vindo do memory_profiler (MiB)
        return float(value) * 1024 * 1024
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return numeric * 1024 * 1024
    return None


def _format_memory_value(value: Any) -> str:
    """Formata bytes para MB ou GB, retornando 'N/A' em casos inválidos."""
    bytes_value = _coerce_to_bytes(value)
    if bytes_value is None or bytes_value < 0:
        return "N/A"

    gigabytes = bytes_value / (1024 ** 3)
    if gigabytes >= 1:
        return f"{gigabytes:.2f} GB"

    megabytes = bytes_value / (1024 ** 2)
    return f"{megabytes:.2f} MB"


def format_b(b: Any) -> str:
    """Alias para compatibilidade retroativa com a versão antiga."""
    return _format_memory_value(b)


def format_bytes(b: Any) -> str:
    """Mantido para compatibilidade com versões antigas do código."""
    return _format_memory_value(b)


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
        f"{dataset_name}_({config.OUT_EMBEDDING_DIM})_embeddings_epoch_{config.EPOCHS}.wsg.json"
    )
    output_path = os.path.join(save_path, filename)

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
