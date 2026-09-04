import torch
from src.config import Config
from src.models.pytorch.Classifiers.mlp_models import MLPClassifier
from src.models.pytorch.Classifiers.GNN.gnn_models import FacebookGNNClassifier
from src.models.pytorch.GraphAutoencoders.VGAE.vgae_models import GithubVGAE

def test_models():
    # 1. Simula as configurações do seu projeto
    config = Config()
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Iniciando testes na arquitetura refatorada (Hardware: {config.DEVICE.upper()})...\n")

    # 2. Testa a vertente de Classificação MLP
    try:
        mlp = MLPClassifier(config=config, input_dim=128, hidden_dim=64, output_dim=2)
        print(f"✅ Sucesso: {mlp.model_name} inicializado no device {mlp.device}")
    except Exception as e:
        print(f"❌ Erro no MLP: {e}")

    # 3. Testa a vertente de Classificação GNN Dinâmica (NVIDIA RAPIDS)
    try:
        fb_gnn = FacebookGNNClassifier(config=config, input_dim=64, output_dim=4)
        print(f"✅ Sucesso: {fb_gnn.model_name} inicializado no device {fb_gnn.device}")
    except Exception as e:
        print(f"❌ Erro na GNN: {e}")

    # 4. Testa a vertente de Graph Autoencoders Variacionais
    try:
        github_vgae = GithubVGAE(config=config, num_total_features=5000, out_embedding_dim=64)
        print(f"✅ Sucesso: {github_vgae.model_name} inicializado no device {github_vgae.device}")
    except Exception as e:
        print(f"❌ Erro no VGAE: {e}")

if __name__ == "__main__":
    test_models()
