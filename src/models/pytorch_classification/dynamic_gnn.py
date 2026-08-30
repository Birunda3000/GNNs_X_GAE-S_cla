import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GINConv, SAGEConv 
# 🔥 FASE 2: Importação dos Kernels NVIDIA RAPIDS
from torch_geometric.nn.conv.cugraph import CuGraphSAGEConv, CuGraphGATConv

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
        
        # 🔥 FASE 2: Identificadores de tipo atualizados para suportar o RAPIDS
        self.is_gat = layer_type.__name__ in ['GATConv', 'CuGraphGATConv'] or (isinstance(layer_type, type) and issubclass(layer_type, (GATConv, CuGraphGATConv)))
        self.is_gin = layer_type.__name__ == 'GINConv' or (isinstance(layer_type, type) and issubclass(layer_type, GINConv))
        self.is_sage = layer_type.__name__ in ['SAGEConv', 'CuGraphSAGEConv'] or (isinstance(layer_type, type) and issubclass(layer_type, (SAGEConv, CuGraphSAGEConv)))

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