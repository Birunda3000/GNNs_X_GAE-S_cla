import json
import os
import matplotlib.pyplot as plt
from src.data_format_definition import WSG
from src.data_loaders import DirectWSGLoader

# Caminho do arquivo .wsg.json (substitua pelo seu)
WSG_PATH = "data/output/EMBEDDING_RUNS/Musae-Facebook__score_0_5457__emb_dim_2__10-11-2025_14-07-52/Musae-Facebook_(2)_embeddings_epoch_500.wsg.json"

# === 1. Carrega o arquivo como objeto WSG ===
loader = DirectWSGLoader(WSG_PATH)
wsg_obj = loader.load()

# === 2. Extrai embeddings e rótulos ===
num_nodes = wsg_obj.metadata.num_nodes
embeddings = []
labels = []

for i in range(num_nodes):
    node_id = str(i)
    feat = wsg_obj.node_features[node_id]
    embeddings.append(feat.weights)  # lista de floats [x, y]
    labels.append(wsg_obj.graph_structure.y[i])

import numpy as np
embeddings = np.array(embeddings)
labels = np.array(labels)

# === 3. Cria gráfico ===
plt.figure(figsize=(8, 6))
unique_classes = np.unique(labels[labels != None])

for cls in unique_classes:
    cls = int(cls)
    mask = labels == cls
    plt.scatter(
        embeddings[mask, 0],
        embeddings[mask, 1],
        label=f"Classe {cls}",
        s=20,
        alpha=0.7
    )

plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.title(f"Projeção 2D dos Embeddings ({wsg_obj.metadata.dataset_name})")
plt.legend()
plt.grid(True)

# === 4. Salva na mesma pasta do WSG ===
output_path = os.path.join(
    os.path.dirname(WSG_PATH),
    os.path.basename(WSG_PATH).replace(".wsg.json", "_scatter.png")
)
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Gráfico salvo em: {output_path}")
