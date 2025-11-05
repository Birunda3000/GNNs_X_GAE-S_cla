import os
import io
import torch
import numpy as np
from src.data_converter import DataConverter
from src.data_loaders import DirectWSGLoader, get_loader


facebook = [
    "/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-facebook__loss_2_2581__emb_dim_32__30-10-2025_19-32-36/musae-facebook_(32)_embeddings_epoch_200.wsg.json",
    "/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-facebook__loss_2_9193__emb_dim_64__30-10-2025_19-35-38/musae-facebook_(64)_embeddings_epoch_200.wsg.json",
    "/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-facebook__loss_3_5429__emb_dim_128__30-10-2025_19-38-39/musae-facebook_(128)_embeddings_epoch_200.wsg.json"
]

github = [
    "/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-github__loss_2_0401__emb_dim_32__30-10-2025_19-16-19/musae-github_(32)_embeddings_epoch_200.wsg.json",
    "/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-github__loss_2_5249__emb_dim_64__30-10-2025_19-20-01/musae-github_(64)_embeddings_epoch_200.wsg.json",
    "/app/gnn_tcc/data/output/EMBEDDING_RUNS/musae-github__loss_3_1906__emb_dim_128__30-10-2025_19-24-58/musae-github_(128)_embeddings_epoch_200.wsg.json"
]


# ==========================================================
# ⚙️ CONFIGURAÇÕES
# ==========================================================
DEVICE = "cpu"
DATASET_NAME = "musae-github"

WSG_FILE_PATHS = github

# ==========================================================
# 🧠 FUNÇÕES AUXILIARES
# ==========================================================
def format_bytes(num_bytes):
    """Converte bytes para formato legível."""
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.2f} MB"
    else:
        return f"{num_bytes / 1024**3:.2f} GB"

def medir_tamanho_tensor(tensor, nome):
    """Retorna o tamanho real do tensor em bytes."""
    if tensor is None:
        print(f"⚠️ {nome} está vazio ou None.")
        return 0
    tamanho_bytes = tensor.element_size() * tensor.nelement()
    print(f"  - {nome:<25}: {format_bytes(tamanho_bytes)} | shape={tuple(tensor.shape)} | dtype={tensor.dtype}")
    return tamanho_bytes

def medir_array_numpy(arr, nome):
    """Retorna o tamanho real de um array NumPy em bytes."""
    if arr is None:
        print(f"⚠️ {nome} está vazio ou None.")
        return 0
    tamanho_bytes = arr.nbytes
    print(f"  - {nome:<25}: {format_bytes(tamanho_bytes)} | shape={arr.shape} | dtype={arr.dtype}")
    return tamanho_bytes

def medir_objeto_pyg(pyg_data, nome):
    """Mede o tamanho total serializado do objeto pyg_data (RAM real)."""
    buffer = io.BytesIO()
    torch.save(pyg_data, buffer)
    tamanho_bytes = buffer.getbuffer().nbytes
    print(f"\n💡 Tamanho total de {nome}: {format_bytes(tamanho_bytes)} (objeto PyG completo)\n")
    return tamanho_bytes


# ==========================================================
# 🔹 1️⃣ VGAE
# ==========================================================
print("=" * 100)
print(f"📦 [VGAE] — Dataset: {DATASET_NAME}")
print("=" * 100)
loader = get_loader(DATASET_NAME)
wsg_obj = loader.load()
vgae_pyg_data = DataConverter.to_pyg_data(wsg_obj, for_embedding_bag=True).to(DEVICE)

vgae_tensores = [
    ("feature_indices", getattr(vgae_pyg_data, "feature_indices", None)),
    ("feature_offsets", getattr(vgae_pyg_data, "feature_offsets", None)),
    ("feature_weights", getattr(vgae_pyg_data, "feature_weights", None)),
    ("edge_index", getattr(vgae_pyg_data, "edge_index", None)),
]

total_vgae_tensores = sum(medir_tamanho_tensor(t, n) for n, t in vgae_tensores)
total_vgae_pyg = medir_objeto_pyg(vgae_pyg_data, "vgae_pyg_data")

print(f"💾 Total tensores VGAE: {format_bytes(total_vgae_tensores)}")
print(f"💾 Total pyg_data VGAE: {format_bytes(total_vgae_pyg)}\n")


# ==========================================================
# 🔹 2️⃣ GCN/GAT
# ==========================================================
print("=" * 100)
print(f"📦 [GCN/GAT] — Dataset: {DATASET_NAME}")
print("=" * 100)
loader = get_loader(DATASET_NAME)
wsg_obj = loader.load()
gat_gcn_pyg_data = DataConverter.to_pyg_data(wsg_obj, for_embedding_bag=False).to(DEVICE)

gat_gcn_tensores = [
    ("x", getattr(gat_gcn_pyg_data, "x", None)),
    ("y", getattr(gat_gcn_pyg_data, "y", None)),
    ("edge_index", getattr(gat_gcn_pyg_data, "edge_index", None)),
]

total_gcn_gat_tensores = sum(medir_tamanho_tensor(t, n) for n, t in gat_gcn_tensores)
total_gcn_gat_pyg = medir_objeto_pyg(gat_gcn_pyg_data, "gat_gcn_pyg_data")

print(f"💾 Total tensores GCN/GAT: {format_bytes(total_gcn_gat_tensores)}")
print(f"💾 Total pyg_data GCN/GAT: {format_bytes(total_gcn_gat_pyg)}\n")


# ==========================================================
# 🔹 3️⃣ SKLEARN (para múltiplos embeddings)
# ==========================================================
print("=" * 100)
print(f"📦 [SKLEARN] — Embeddings VGAE ({DATASET_NAME})")
print("=" * 100)

sklearn_results = []

for file_path in WSG_FILE_PATHS:
    print(f"📁 Analisando arquivo: {os.path.basename(file_path)}")
    loader = DirectWSGLoader(file_path=file_path)
    wsg_obj = loader.load()
    sklearn_pyg_data = DataConverter.to_pyg_data(wsg_obj, for_embedding_bag=False).to(DEVICE)

    X = sklearn_pyg_data.x.cpu().numpy()
    y = sklearn_pyg_data.y.cpu().numpy()

    sklearn_arrays = [
        ("X (embeddings)", X),
        ("y (labels)", y),
    ]

    total_arrays = sum(medir_array_numpy(arr, n) for n, arr in sklearn_arrays)
    total_pyg = medir_objeto_pyg(sklearn_pyg_data, os.path.basename(file_path))

    sklearn_results.append((os.path.basename(file_path), total_arrays, total_pyg))

    print(f"💾 Total arrays: {format_bytes(total_arrays)}")
    print(f"💾 Total pyg_data: {format_bytes(total_pyg)}\n")
    print("-" * 100)


# ==========================================================
# ✅ SUMÁRIO FINAL
# ==========================================================
print("=" * 100)
print(f"📊 RESUMO GERAL — TAMANHO DOS DADOS EFETIVAMENTE USADOS no dataset {DATASET_NAME}")
print("=" * 100)
print(f"VGAE     : {format_bytes(total_vgae_tensores)} (tensores) | {format_bytes(total_vgae_pyg)} (pyg_data)")
print(f"GCN/GAT  : {format_bytes(total_gcn_gat_tensores)} (tensores) | {format_bytes(total_gcn_gat_pyg)} (pyg_data)")

for name, arr_bytes, pyg_bytes in sklearn_results:
    print(f"SKLEARN ({name}) : {format_bytes(arr_bytes)} (arrays) | {format_bytes(pyg_bytes)} (pyg_data)")

print("=" * 100)


