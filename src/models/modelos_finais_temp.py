#/app/gnn_tcc/src/models/embedding_models/din_gae.py


class FacebookGAE(DynamicGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,       
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.GELU,         
            dropout=0.5,                
            normalize_embeddings=True,  
            heads=1,                    
        )
        self.model_name = "FacebookGAE"

class FacebookVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,       
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.GELU,         
            dropout=0.5,                
            normalize_embeddings=True,  
            heads=1,                    
        )
        self.model_name = "FacebookVGAE"

class GithubVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=256,          
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,       
            layer_type=CuGraphSAGEConv,
            num_layers=4,               
            activation=nn.LeakyReLU,    
            dropout=0.1,                
            normalize_embeddings=True,  
            aggr='mean',                
        )
        self.model_name = "GithubVGAE"

class TwitchVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim: int):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,           
            hidden_dim=256,             
            out_embedding_dim=out_embedding_dim,  
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.ReLU,         
            dropout=0.2,                
            normalize_embeddings=False, 
            heads=1,                    
        )
        self.model_name = "TwitchVGAE"

class RedditVGAE(DynamicVGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim: int):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=128,          
            hidden_dim=64,              
            out_embedding_dim=out_embedding_dim,  
            layer_type=CuGraphGATConv,
            num_layers=4,               
            activation=nn.LeakyReLU,    
            dropout=0.2,                
            normalize_embeddings=True,  
            heads=1,                    
        )
        self.model_name = "RedditVGAE"

# ==============================================================================
# A FRONTEIRA: GRAPH TRANSFORMERS COM FLEX ATTENTION (FASE 3)
# ==============================================================================

class FlexGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        
        self.q_proj = nn.Linear(in_dim, out_dim * heads)
        self.k_proj = nn.Linear(in_dim, out_dim * heads)
        self.v_proj = nn.Linear(in_dim, out_dim * heads)
        self.out_proj = nn.Linear(out_dim * heads, out_dim)

    def forward(self, x, edge_index):
        if not HAS_FLEX:
            raise ImportError("FlexAttention requer PyTorch 2.5/2.6. Atualize o ambiente para utilizar Graph Transformers.")
            
        N = x.size(0)
        device = x.device
        
        adj = torch.zeros((N, N), dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        adj.fill_diagonal_(True)

        def graph_mask_mod(b, h, q_idx, kv_idx):
            return adj[q_idx, kv_idx]

        block_mask = create_block_mask(graph_mask_mod, B=1, H=self.heads, Q_LEN=N, KV_LEN=N)

        q = self.q_proj(x).view(1, N, self.heads, self.out_dim).transpose(1, 2)
        k = self.k_proj(x).view(1, N, self.heads, self.out_dim).transpose(1, 2)
        v = self.v_proj(x).view(1, N, self.heads, self.out_dim).transpose(1, 2)

        out = flex_attention(q, k, v, block_mask=block_mask)

        out = out.transpose(1, 2).reshape(N, self.heads * self.out_dim)
        return self.out_proj(out)


class FlexTransformerGAE(BaseGAE):
    def __init__(self, config, num_total_features: int, out_embedding_dim: int):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=128,          
            hidden_dim=128,             
            out_embedding_dim=out_embedding_dim,
        )
        self.model_name = "FlexTransformerGAE"
        self.activation_fn = nn.GELU()
        self.dropout = 0.2
        
        self.layer1 = FlexGraphAttentionLayer(128, 128, heads=4)
        self.layer2 = FlexGraphAttentionLayer(128, out_embedding_dim, heads=4)

    def encode(self, data):
        x = self.feature_embedder(data.feature_indices, data.feature_offsets, per_sample_weights=data.feature_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layer1(x, data.edge_index)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layer2(x, data.edge_index)
        
        return F.normalize(x, p=2, dim=-1)



# /app/gnn_tcc/src/models/embedding_models/autoencoders_models.py


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from src.models.embedding_models.base_graph_autoenconders_model import BaseGAE, BaseVGAE


class GCNGAE(BaseGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )
        z = self.conv2(x, data.edge_index)

        return F.normalize(z, p=2, dim=-1)


class GCNVGAE(BaseVGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, out_embedding_dim)
        self.conv_logstd = GCNConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )

        self.__mu__ = self.conv_mu(x, data.edge_index)
        self.__logstd__ = self.conv_logstd(x, data.edge_index)

        z = self.__mu__ + torch.randn_like(self.__mu__) * torch.exp(self.__logstd__)

        return F.normalize(z, p=2, dim=-1)



class GraphSageGAE(BaseGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )
        z = self.conv2(x, data.edge_index)

        return F.normalize(z, p=2, dim=-1)



class GraphSageVGAE(BaseVGAE):

    def __init__(self, config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim):
        super().__init__(config, num_total_features, embedding_dim, hidden_dim, out_embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv_mu = SAGEConv(hidden_dim, out_embedding_dim)
        self.conv_logstd = SAGEConv(hidden_dim, out_embedding_dim)

    def encode(self, data):
        x = self.feature_embedder(
            data.feature_indices,
            data.feature_offsets,
            per_sample_weights=data.feature_weights,
        )
        # ❌ REMOVIDO: x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(
            F.relu(self.conv1(x, data.edge_index)),
            p=0.5,
            training=self.training
        )

        self.__mu__ = self.conv_mu(x, data.edge_index)
        self.__logstd__ = self.conv_logstd(x, data.edge_index)

        z = self.__mu__ + torch.randn_like(self.__mu__) * torch.exp(self.__logstd__)

        return F.normalize(z, p=2, dim=-1)


# modelos


#/app/gnn_tcc/src/models/pytorch_classification/classification_models.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from src.models.pytorch_classification.base_classifiers import PyTorchClassifier
from torch_geometric.data import Data
from typing import Dict, Any
from src.early_stopper import EarlyStopper
import torch.optim as optim
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLPClassifier(PyTorchClassifier):
    """Classificador MLP que opera em um tensor de features denso."""

    use_gnn = False

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class GCNClassifier(PyTorchClassifier):
    """Classificador GCN que opera em features e na estrutura do grafo."""

    use_gnn = True

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
    
    def verify_train_input_data(self, data: Data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index (data.edge_index)."


class GATClassifier(PyTorchClassifier):
    """Classificador GAT que utiliza mecanismos de atenção."""

    use_gnn = True

    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)
    
    def verify_train_input_data(self, data: Data):
        super().verify_train_input_data(data)
        assert data.edge_index is not None, "Os dados de entrada devem conter edge_index (data.edge_index)."



#/app/gnn_tcc/src/models/pytorch_classification/dynamic_gnn.py


class FacebookGNNClassifier(DynamicGNNClassifier):
    def __init__(self, config, input_dim, output_dim):
        super().__init__(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_type=CuGraphSAGEConv, # 🔥 Drop-in Replacement NVIDIA
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
            layer_type=CuGraphSAGEConv, # 🔥 Drop-in Replacement NVIDIA
            num_layers=3,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.LeakyReLU,
            aggr='mean'
        )


# ==============================================================================
# EMBEDDING BAG + GNN (END-TO-END)
# ==============================================================================

import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from typing import Dict, Any, List, Optional
from torch_geometric.data import Data
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
        super().__init__(config, hidden_dim, hidden_dim, output_dim)

        self.dropout_rate = dropout
        self.activation_fn = get_activation_fn(activation)
        self.num_layers = num_layers
        self.heads = heads
        
        self.feature_embedder = nn.EmbeddingBag(
            num_embeddings=num_total_features,
            embedding_dim=embedding_dim,
            mode="sum", 
        )

        # 🔥 FASE 2: Identificadores de tipo atualizados para suportar o RAPIDS
        self.is_gat = layer_type.__name__ in ['GATConv', 'CuGraphGATConv'] or (isinstance(layer_type, type) and issubclass(layer_type, (GATConv, CuGraphGATConv)))
        self.is_gin = layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv))
        self.is_sage = layer_type.__name__ in ['SAGEConv', 'CuGraphSAGEConv'] or (isinstance(layer_type, type) and issubclass(layer_type, (SAGEConv, CuGraphSAGEConv)))

        self.sage_aggr = kwargs.get('aggr', 'mean')
        self.gin_train_eps = kwargs.get('train_eps', False)

        self.convs = nn.ModuleList()

        def build_layer(in_d, out_d, is_last=False):
            if self.is_gin:
                mlp = nn.Sequential(
                    nn.Linear(in_d, out_d),
                    nn.ReLU(),
                    nn.Linear(out_d, out_d)
                )
                return layer_type(mlp, train_eps=self.gin_train_eps)
            
            elif self.is_gat:
                if is_last:
                    return layer_type(in_d, out_d, heads=1, concat=False, dropout=dropout)
                else:
                    return layer_type(in_d, out_d, heads=heads, dropout=dropout)
            
            elif self.is_sage:
                return layer_type(in_d, out_d, aggr=self.sage_aggr)
            
            else:
                return layer_type(in_d, out_d)

        self.convs.append(build_layer(embedding_dim, hidden_dim))
        
        current_dim = hidden_dim * heads if self.is_gat else hidden_dim

        for _ in range(num_layers - 2):
            self.convs.append(build_layer(current_dim, hidden_dim))
            current_dim = hidden_dim * heads if self.is_gat else hidden_dim

        self.convs.append(build_layer(current_dim, output_dim, is_last=True))

    def forward(self, feature_indices, feature_offsets, feature_weights, edge_index):
        x = self.feature_embedder(feature_indices, feature_offsets, per_sample_weights=feature_weights)
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
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
        device = self.device
        
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

        # 🔥 FASE 1: Aciona a compilação JIT apenas no passe frontal
        self.compile_methods(["forward"], dynamic=True)

        for epoch in pbar:
            self.train()
            optimizer.zero_grad()
            
            out = self(feature_indices, feature_offsets, feature_weights, edge_index)
            
            train_loss = criterion(out[train_mask], y[train_mask])
            train_loss.backward()
            optimizer.step()

            _, val_f1, _, _ = local_eval(val_mask)
            train_acc, train_f1, _, _ = local_eval(train_mask) 

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
                early_stopper.restore_best_state(self)
                break

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

class FacebookEmbeddingGNN(DynamicEmbeddingGNNClassifier):
    def __init__(self, config, num_total_features, output_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=64,
            output_dim=output_dim,
            layer_type=CuGraphSAGEConv,  # 🔥 Drop-in Replacement NVIDIA
            num_layers=4,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.ReLU,
            aggr='mean'
        )
        self.model_name = "FacebookEmbeddingGNN"

class GithubEmbeddingGNN(DynamicEmbeddingGNNClassifier):
    def __init__(self, config, num_total_features, output_dim):
        super().__init__(
            config=config,
            num_total_features=num_total_features,
            embedding_dim=256,
            output_dim=output_dim,
            layer_type=CuGraphSAGEConv,  # 🔥 Drop-in Replacement NVIDIA
            num_layers=3,
            hidden_dim=256,
            dropout=0.5,
            activation=nn.LeakyReLU,
            aggr='mean'
        )
        self.model_name = "GithubEmbeddingGNN"