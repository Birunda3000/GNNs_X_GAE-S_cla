# src/train.py (CORRIGIDO save_report)

from multiprocessing.util import DEBUG
import torch
import os

from datetime import datetime

# --- Importa nossos módulos customizados ---
from src.data_format_definition import WSG, Metadata, NodeFeaturesEntry
from datetime import datetime
import os
import torch
from src.data_format_definition import WSG, Metadata, NodeFeaturesEntry


# --- FUNÇÃO HELPER format_b ATUALIZADA ---
def format_b(b):
    """Converte bytes ou MiB para um formato legível (MB ou GB). Mais robusta."""
    if b is None or b == 0 or b == 0.0:
        # Retorna 0.00 MB para valores nulos ou zero
        return "0.00 MB"

    try:
        if isinstance(b, float): # memory_profiler retorna MiB (float)
            # Converte MiB para Bytes
            b_bytes = int(b * 1024 * 1024)
        elif isinstance(b, int):
            # Já está em bytes
            b_bytes = b
        else:
            # Tipo inesperado
            return "N/A (type)"

        # Formata Bytes para MB/GB
        if b_bytes < 1024**3:
            return f"{b_bytes / 1024**2:.2f} MB"
        else:
            return f"{b_bytes / 1024**3:.2f} GB"
    except Exception:
        # Captura qualquer erro de conversão/formatação
        return "N/A (error)"


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


def fmt(val, precision=6):
    """Formata floats de forma segura; se None ou inválido, retorna 'N/A'."""
    return f"{val:.{precision}f}" if isinstance(val, (int, float)) else "N/A"


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
        final_embeddings (torch.Tensor): Tensor de embeddings finais (num_nodes x dim).
        wsg_obj (WSG): Objeto WSG original (usado para copiar metadados e estrutura do grafo).
        config (Config): Configuração com parâmetros do modelo.
        save_path (str): Caminho onde o arquivo será salvo.
        tz_info (timezone, opcional): Informação de fuso horário para timestamp.

    Returns:
        str: Caminho completo do arquivo salvo.
    """
    final_embeddings = final_embeddings.cpu()

    if tz_info is None:
        from zoneinfo import ZoneInfo
        tz_info = ZoneInfo("America/Sao_Paulo")

    output_metadata = Metadata(
        dataset_name=f"{wsg_obj.metadata.dataset_name}-Embeddings",
        feature_type="dense_continuous",
        num_nodes=wsg_obj.metadata.num_nodes,
        num_edges=wsg_obj.metadata.num_edges,
        num_total_features=config.OUT_EMBEDDING_DIM,
        processed_at=datetime.now(tz_info).isoformat(),
        directed=wsg_obj.metadata.directed,
    )

    output_node_features = {}
    embedding_indices = list(range(config.OUT_EMBEDDING_DIM))

    for i in range(wsg_obj.metadata.num_nodes):
        node_embedding = final_embeddings[i].tolist()
        output_node_features[str(i)] = NodeFeaturesEntry(
            indices=embedding_indices,
            weights=node_embedding,
        )

    output_wsg = WSG(
        metadata=output_metadata,
        graph_structure=wsg_obj.graph_structure,
        node_features=output_node_features,
    )

    dataset_name = wsg_obj.metadata.dataset_name
    filename = f"{dataset_name}_({config.OUT_EMBEDDING_DIM})_embeddings_epoch_{config.EPOCHS}.wsg.json"
    output_path = os.path.join(save_path, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_wsg.model_dump_json(indent=2))

    print(f"✅ Embeddings salvos em: '{output_path}'")
    return output_path


def salvar_modelo_pytorch_completo(model, dataset_name: str, timestamp: str, save_dir: str = "models"):
    """
    Salva o modelo PyTorch completo (arquitetura + pesos + buffers).
    ⚠️ Requer que o código da classe do modelo exista no mesmo caminho
       ao carregar (ex: models.vgae.VGAE).
    """
    os.makedirs(save_dir, exist_ok=True)

    model_name = getattr(model, "model_name", model.__class__.__name__)
    base_name = f"{dataset_name}__{model_name}__{timestamp}"

    save_path = os.path.join(save_dir, f"{base_name}.pt")

    torch.save(model, save_path)

    print(f"✅ Modelo completo salvo em: {save_path}")
    return save_path


def carregar_modelo_pytorch_completo(save_path: str, device: str = "cpu"):
    """
    Carrega um modelo PyTorch completo salvo com torch.save(model).
    Requer que o código original (classe do modelo) exista.
    """
    model = torch.load(save_path, map_location=device)
    model.eval()  # modo avaliação (sem dropout/batchnorm training)
    print(f"🔁 Modelo carregado de: {save_path}")
    return model