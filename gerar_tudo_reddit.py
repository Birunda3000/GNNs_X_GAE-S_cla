import os
import torch
from src.config import Config
from src.data_loaders import RedditLoader
import src.data_converters as data_converters
from src.utils import carregar_modelo_pytorch_completo, save_embeddings_to_wsg



# 1. FORÇAR A CPU (Isso salva a sua placa de vídeo de dar OOM)
config = Config()
config.DEVICE = "cpu"
device = torch.device(config.DEVICE)

# 2. CARREGAR O DATASET COMPLETO (O Super Grafo)
print("Carregando todo o Reddit...")
dataset_completo = RedditLoader()
wsg_obj = dataset_completo.load()
pyg_data = data_converters.wsg_for_vgae(wsg_obj, config).to(device)

# 3. CARREGAR O MODELO QUE FOI TREINADO NA AMOSTRA
# (Troque pelo nome do arquivo que foi gerado no Passo 2)
caminho_modelo = "/app/gnn_tcc/data/output/EMBEDDING_RUNS/Reddit-Lite__sc_19451_emb_dim_64__04-35-11/Reddit-Lite__RedditVGAE__24-08-2026_04-35-11.pt" 
model = carregar_modelo_pytorch_completo(caminho_modelo, device="cpu")

# 4. PASSAR O GRAFO INTEIRO PELO MODELO (Inferência)
print("Gerando embeddings para milhões de nós na CPU...")
final_embeddings = model.inference(pyg_data)

# 5. SALVAR NO FORMATO WSG
config.OUT_EMBEDDING_DIM = final_embeddings.shape[1]
save_embeddings_to_wsg(
    final_embeddings=final_embeddings,
    wsg_obj=wsg_obj,
    config=config,
    save_path="/app/gnn_tcc/data/output/EMBEDDING_RUNS/REDDIT_COMPLETO"
)
print("Sucesso!")