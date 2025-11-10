import os
import numpy as np
import plotly.graph_objects as go
from src.data_loaders import DirectWSGLoader

# Caminho do arquivo .wsg.json (substitua pelo seu)
WSG_PATH = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Musae-Facebook__score_0_7470__emb_dim_3__10-11-2025_09-43-36/Musae-Facebook_(3)_embeddings_epoch_500.wsg.json"

# === 1. Carrega o WSG ===
loader = DirectWSGLoader(WSG_PATH)
wsg_obj = loader.load()

# === 2. Extrai embeddings e labels ===
num_nodes = wsg_obj.metadata.num_nodes
embeddings = []
labels = []

for i in range(num_nodes):
    node_id = str(i)
    feat = wsg_obj.node_features[node_id]
    embeddings.append(feat.weights)
    labels.append(wsg_obj.graph_structure.y[i])

embeddings = np.array(embeddings)
labels = np.array(labels)

# === 3. Cria visualização interativa ===
unique_classes = np.unique(labels[labels != None])
fig = go.Figure()

for cls in unique_classes:
    cls = int(cls)
    mask = labels == cls
    fig.add_trace(go.Scatter3d(
        x=embeddings[mask, 0],
        y=embeddings[mask, 1],
        z=embeddings[mask, 2],
        mode='markers',
        marker=dict(size=3),
        name=f"Classe {cls}"
    ))

fig.update_layout(
    title=f"Projeção 3D Interativa dos Embeddings ({wsg_obj.metadata.dataset_name})",
    scene=dict(
        xaxis_title="Dimensão 1",
        yaxis_title="Dimensão 2",
        zaxis_title="Dimensão 3",
    ),
    legend=dict(x=0, y=1)
)

# === 4. Salva como HTML interativo ===
output_path = os.path.join(
    os.path.dirname(WSG_PATH),
    os.path.basename(WSG_PATH).replace(".wsg.json", "_scatter3d_interativo.html")
)
fig.write_html(output_path)

print(f"✅ Visualização 3D interativa salva em: {output_path}")
print("💡 Abra o arquivo HTML no navegador para girar, dar zoom e explorar!")
