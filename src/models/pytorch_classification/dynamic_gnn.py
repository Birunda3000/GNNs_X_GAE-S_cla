import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GINConv, SAGEConv # Importar SAGEConv explicitamente para checagem
from src.models.pytorch_classification.base_classifiers import PyTorchClassifier
from src.models.base_model import get_activation_fn

class DynamicGNNClassifier(PyTorchClassifier):
    
    use_gnn = True

    def __init__(
        self,
        config,
        input_dim: int,
        output_dim: int,
        layer_type: MessagePassing,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.5,
        activation = nn.ReLU,
        heads: int = 1,
        **kwargs  # <--- Captura aggr, train_eps, etc.
    ):
        super().__init__(config, input_dim, hidden_dim, output_dim)

        self.dropout_rate = dropout
        self.activation_fn = get_activation_fn(activation)
        self.num_layers = num_layers
        
        # Identificadores de tipo
        self.is_gat = layer_type.__name__ == 'GATConv' or (isinstance(layer_type, type) and issubclass(layer_type, GATConv))
        self.is_gin = layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv))
        self.is_sage = layer_type.__name__ == 'SAGEConv' or (isinstance(layer_type, type) and issubclass(layer_type, SAGEConv))

        # Captura parâmetros específicos do kwargs
        self.sage_aggr = kwargs.get('aggr', 'mean')
        self.gin_train_eps = kwargs.get('train_eps', False)

        self.convs = nn.ModuleList()
        
        # --- Helper para instanciar camadas ---
        def build_layer(in_d, out_d, is_last=False):
            if self.is_gin:
                mlp = nn.Sequential(
                    nn.Linear(in_d, out_d),
                    nn.ReLU(),
                    nn.Linear(out_d, out_d)
                )
                return layer_type(mlp, train_eps=self.gin_train_eps) # Passa train_eps
            
            elif self.is_gat:
                # Última camada: concat=False, heads=1 (padrão p/ classificação)
                if is_last:
                    return layer_type(in_d, out_d, heads=1, concat=False, dropout=dropout)
                else:
                    return layer_type(in_d, out_d, heads=heads, dropout=dropout)
            
            elif self.is_sage:
                return layer_type(in_d, out_d, aggr=self.sage_aggr) # Passa aggr
            
            else:
                # GCN e outros
                return layer_type(in_d, out_d)

        # --- 1. Camada de Entrada ---
        self.convs.append(build_layer(input_dim, hidden_dim))
        
        # Ajuste de dimensão atual
        if self.is_gat:
            current_dim = hidden_dim * heads
        else:
            current_dim = hidden_dim

        # --- 2. Camadas Ocultas ---
        for _ in range(num_layers - 2):
            self.convs.append(build_layer(current_dim, hidden_dim))
            
            if self.is_gat:
                current_dim = hidden_dim * heads
            else:
                current_dim = hidden_dim

        # --- 3. Camada de Saída ---
        self.convs.append(build_layer(current_dim, output_dim, is_last=True))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

    def verify_train_input_data(self, data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "DynamicGNN requer edge_index."






'''Tabela 16 – D1 (MUSAE-Facebook): Top-1 por família em GNN end-to-end (baseline
supervisionado).
Arquitetura
(E2E)
F1-weighted
(Val.)
Configuração Top-1 (principais hiperparâ-
metros)
SAGEConv 0.9559 
camadas: 4; hidden: 256; ativação: ReLU;
dropout: 0.5; agregador: mean

Tabela 20 – D2 (MUSAE-GitHub): Top-1 por família em GNN end-to-end (baseline
supervisionado).
Arquitetura
(E2E)
F1-weighted
(Val.)
Configuração Top-1 (principais hiperparâ-
metros)
SAGEConv 0.8705 
camadas: 3; hidden: 256; ativação: Leaky-
ReLU; dropout: 0.5; agregador: mean'''


class FacebookGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=SAGEConv,
            num_layers=4,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.ReLU,
            aggr='mean'
        )

class GitHubGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=SAGEConv,
            num_layers=3,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.LeakyReLU,
            aggr='mean'
        )




#embeddingbag****




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GINConv, SAGEConv
from torch_geometric.data import Data
from typing import Dict, Any, List, Optional
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Imports do seu projeto
from src.models.pytorch_classification.base_classifiers import PyTorchClassifier
from src.models.base_model import get_activation_fn
from src.early_stopper import EarlyStopper

class DynamicEmbeddingGNNClassifier(PyTorchClassifier):
    """
    Classificador GNN Supervisionado que aprende embeddings a partir de features esparsas (EmbeddingBag).
    Combina a lógica do 'DynamicGNNClassifier' com a entrada do 'BaseGAE'.
    """
    
    use_gnn = True

    def __init__(
        self,
        config,
        num_total_features: int, # Tamanho do vocabulário (entrada do Bag)
        embedding_dim: int,      # Tamanho do vetor denso (saída do Bag -> entrada da GNN)
        output_dim: int,
        layer_type: MessagePassing,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.5,
        activation = nn.ReLU,
        heads: int = 1,
        **kwargs
    ):
        # Inicializa o pai (PyTorchClassifier).
        # Nota: input_dim passado para o pai é 'hidden_dim' apenas para satisfazer a assinatura,
        # pois a entrada real é gerida manualmente pelo EmbeddingBag.
        super().__init__(config, hidden_dim, hidden_dim, output_dim)

        self.dropout_rate = dropout
        self.activation_fn = get_activation_fn(activation)
        self.num_layers = num_layers
        self.heads = heads
        
        # 1. Camada de Embedding Bag (Igual ao GAE/VGAE)
        # Aprende a representação vetorial das features durante o treino supervisionado.
        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum", 
        )

        # 2. Configuração das Camadas GNN (Lógica adaptada do DynamicGNN)
        self.is_gat = layer_type.__name__ == 'GATConv' or (isinstance(layer_type, type) and issubclass(layer_type, GATConv))
        self.is_gin = layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv))
        self.is_sage = layer_type.__name__ == 'SAGEConv' or (isinstance(layer_type, type) and issubclass(layer_type, SAGEConv))

        self.sage_aggr = kwargs.get('aggr', 'mean')
        self.gin_train_eps = kwargs.get('train_eps', False)

        self.convs = nn.ModuleList()

        # Helper para construir camadas
        def build_layer(in_d, out_d, is_last=False):
            if self.is_gin:
                mlp = nn.Sequential(
                    nn.Linear(in_d, out_d),
                    nn.ReLU(),
                    nn.Linear(out_d, out_d)
                )
                return layer_type(mlp, train_eps=self.gin_train_eps)
            
            elif self.is_gat:
                # Na última camada de classificação, geralmente concatenamos=False e heads=1
                if is_last:
                    return layer_type(in_d, out_d, heads=1, concat=False, dropout=dropout)
                else:
                    return layer_type(in_d, out_d, heads=heads, dropout=dropout)
            
            elif self.is_sage:
                return layer_type(in_d, out_d, aggr=self.sage_aggr)
            
            else:
                return layer_type(in_d, out_d)

        # --- Camada 1: Conecta o EmbeddingBag (embedding_dim) à GNN (hidden_dim) ---
        self.convs.append(build_layer(embedding_dim, hidden_dim))
        
        # Ajuste de dimensão para as próximas camadas (se for GAT com multi-head)
        current_dim = hidden_dim * heads if self.is_gat else hidden_dim

        # --- Camadas Ocultas ---
        for _ in range(num_layers - 2):
            self.convs.append(build_layer(current_dim, hidden_dim))
            current_dim = hidden_dim * heads if self.is_gat else hidden_dim

        # --- Camada de Saída ---
        self.convs.append(build_layer(current_dim, output_dim, is_last=True))

    def forward(self, feature_indices, feature_offsets, feature_weights, edge_index):
        """
        Forward pass customizado que inclui o passo do EmbeddingBag.
        """
        # Passo 1: Processar features esparsas via EmbeddingBag
        # Gera vetores densos a partir dos índices
        x = self.feature_embedder(feature_indices, feature_offsets, per_sample_weights=feature_weights)
        
        # Passo 2: Pipeline padrão GNN
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Última camada (Logits)
        x = self.convs[-1](x, edge_index)
        return x

    def internal_train_model(
        self,
        data: Data,
        use_gnn: bool,
        epochs: int,
        optimizer,
        early_stopper: EarlyStopper,
        scheduler,
        criterion=nn.CrossEntropyLoss(),
    ):
        """
        Sobrescrita do loop de treino para lidar com a extração de features do Data object.
        """
        device = self.device
        
        # Extração manual dos dados necessários para o EmbeddingBag
        # Esses atributos existem no objeto Data gerado pelo WSG
        feature_indices = data.feature_indices.to(device)
        feature_offsets = data.feature_offsets.to(device)
        feature_weights = data.feature_weights.to(device)
        
        y = data.y.to(device)
        edge_index = data.edge_index.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)
        test_mask = data.test_mask.to(device)

        training_history: List[Dict[str, Any]] = []
        stop_now = False
        best_epoch = None

        pbar = tqdm(range(1, epochs + 1), desc=f"Treinando {self.model_name} (End-to-End)", leave=False)
        start_time = time.perf_counter()

        # Função auxiliar para avaliação interna sem depender do 'evaluate' da classe pai
        def local_eval(mask):
            self.eval()
            with torch.no_grad():
                out = self(feature_indices, feature_offsets, feature_weights, edge_index)
                pred = out.argmax(dim=1)
                
                y_true = y[mask]
                y_pred = pred[mask]
                
                acc = float(accuracy_score(y_true.cpu(), y_pred.cpu()))
                f1 = float(f1_score(y_true.cpu(), y_pred.cpu(), average="weighted"))
                
                rep = classification_report(
                    y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
                )
                cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
                
                return acc, f1, rep, cm

        for epoch in pbar:
            self.train()
            optimizer.zero_grad()
            
            # Chama o forward com os argumentos explícitos
            out = self(feature_indices, feature_offsets, feature_weights, edge_index)
            
            train_loss = criterion(out[train_mask], y[train_mask])
            train_loss.backward()
            optimizer.step()

            # Avaliação na validação para Early Stopping
            _, val_f1, _, _ = local_eval(val_mask)
            train_acc, train_f1, _, _ = local_eval(train_mask) # Apenas para log

            stop_now, f1, best_epoch, _ = early_stopper.check(self, epoch=epoch, current_value=val_f1)
            scheduler.step(f1)

            training_history.append({
                "epoch": epoch,
                "train_f1": train_f1,
                "train_accuracy": train_acc,
                "train_loss": train_loss.item(),
                "val_f1": val_f1,
                "Time_per_epoch": time.perf_counter() - start_time,
                "learning_rate": scheduler.get_last_lr()[0],
            })

            pbar.set_postfix({"loss": f"{train_loss.item():.4f}", "val_f1": f"{val_f1:.4f}"})

            if stop_now:
                # print(f"[EARLY STOPPING] Parando no epoch {epoch}") # Opcional
                early_stopper.restore_best_state(self)
                break

        # Gera relatórios finais usando o melhor estado do modelo
        train_acc, train_f1, train_rep, train_cm = local_eval(train_mask)
        val_acc, val_f1, val_rep, val_cm = local_eval(val_mask)
        test_acc, test_f1, test_rep, test_cm = local_eval(test_mask)

        return {
            "total_training_time": time.perf_counter() - start_time,
            
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_report": test_rep,
            "test_confusion_matrix": test_cm,

            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_report": val_rep,
            "val_confusion_matrix": val_cm,
            
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_report": train_rep,
            "train_confusion_matrix": train_cm,
            
            "best_epoch": best_epoch,
            "training_history": training_history,
        }

# ==============================================================================
# IMPLEMENTAÇÕES ESPECÍFICAS (BOLSAS DE PALAVRAS + GNN)
# ==============================================================================

class FacebookEmbeddingGNN(DynamicEmbeddingGNNClassifier):
    """
    Modelo End-to-End para Facebook (Corrigido):
    - Input: Indices -> EmbeddingBag
    - GNN: SAGEConv + ReLU (Igual ao FacebookGNNClassifier original)
    """
    def __init__(self, config, num_total_features, output_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           # Mantém dimensão do embedding
            output_dim=output_dim,
            
            # --- CONFIGURAÇÃO IDÊNTICA AO SEU SNIPPET ---
            layer_type=SAGEConv,        # Era GAT, agora é SAGE
            num_layers=4,               # Mantém 4 camadas
            hidden_dim=256,             # Mantém 256
            dropout=0.5,                # Mantém 0.5
            activation=nn.ReLU,         # Era GELU, agora é ReLU
            aggr='mean'                 # Agregador mean do SAGE
            # --------------------------------------------
        )
        self.model_name = "FacebookEmbeddingGNN"


class GithubEmbeddingGNN(DynamicEmbeddingGNNClassifier):
    """
    Modelo End-to-End para Github:
    - Input: Indices -> EmbeddingBag
    - GNN: SAGEConv + LeakyReLU (Igual ao GitHubGNNClassifier original)
    """
    def __init__(self, config, num_total_features, output_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=256,
            output_dim=output_dim,
            
            # --- CONFIGURAÇÃO IDÊNTICA AO SEU SNIPPET ---
            layer_type=SAGEConv,
            num_layers=3,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.LeakyReLU,
            aggr='mean'
            # --------------------------------------------
        )
        self.model_name = "GithubEmbeddingGNN"


'''
class FacebookGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=SAGEConv,
            num_layers=4,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.ReLU,
            aggr='mean'
        )

class GitHubGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=SAGEConv,
            num_layers=3,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.LeakyReLU,
            aggr='mean'
        )
'''