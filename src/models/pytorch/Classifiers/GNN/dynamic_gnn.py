import time
from tqdm import tqdm
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from src.models.pytorch.Classifiers.base_classifier import PyTorchClassifier
from src.models.base_model import get_activation_fn
from src.early_stopper import EarlyStopper

# HOISTING: Importamos as funções que agora são utilitárias globais
from src.models.pytorch.layer_utils import create_layer, apply_conv, forward_hidden_layers

# ==============================================================================
# GNN DINÂMICA PADRÃO
# ==============================================================================
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
        **kwargs
    ):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.dropout_rate = dropout
        self.activation_fn = get_activation_fn(activation)
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Constrói a primeira camada usando o utilitário isolado
        self.convs.append(create_layer(layer_type, input_dim, hidden_dim, heads=heads, **kwargs))
        current_dim = hidden_dim * heads if 'GAT' in layer_type.__name__ else hidden_dim
        
        # Constrói camadas ocultas
        for _ in range(num_layers - 2):
            self.convs.append(create_layer(layer_type, current_dim, hidden_dim, heads=heads, **kwargs))
            current_dim = hidden_dim * heads if 'GAT' in layer_type.__name__ else hidden_dim
            
        # Constrói camada de saída
        self.convs.append(create_layer(layer_type, current_dim, output_dim, heads=1, **kwargs))

    def forward(self, x, edge_index):
        # HOISTING: Usa o laço padronizado e blindado do JIT
        x = forward_hidden_layers(
            x, edge_index, self.convs[:-1], self.activation_fn, self.dropout_rate, self.training
        )
        return apply_conv(self.convs[-1], x, edge_index)


# ==============================================================================
# EMBEDDING BAG + GNN (END-TO-END)
# ==============================================================================
class DynamicEmbeddingGNNClassifier(PyTorchClassifier):
    """
    Classificador GNN Supervisionado que aprende embeddings a partir de features esparsas (EmbeddingBag).
    """
    use_gnn = True
    def __init__(
        self,
        config,
        num_total_features: int, 
        embedding_dim: int,      
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

        self.convs = nn.ModuleList()

        # Constrói a primeira camada (que agora recebe do EmbeddingBag)
        self.convs.append(create_layer(layer_type, embedding_dim, hidden_dim, heads=heads, **kwargs))
        current_dim = hidden_dim * heads if 'GAT' in layer_type.__name__ else hidden_dim
        
        # Constrói camadas ocultas
        for _ in range(num_layers - 2):
            self.convs.append(create_layer(layer_type, current_dim, hidden_dim, heads=heads, **kwargs))
            current_dim = hidden_dim * heads if 'GAT' in layer_type.__name__ else hidden_dim
            
        # Constrói camada de saída
        self.convs.append(create_layer(layer_type, current_dim, output_dim, heads=1, **kwargs))

    def forward(self, feature_indices, feature_offsets, feature_weights, edge_index):
        x = self.feature_embedder(feature_indices, feature_offsets, per_sample_weights=feature_weights)
        # HOISTING: Usa o laço padronizado e blindado do JIT
        x = forward_hidden_layers(
            x, edge_index, self.convs[:-1], self.activation_fn, self.dropout_rate, self.training
        )
        return apply_conv(self.convs[-1], x, edge_index)

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
                
                # HOISTING: Toda a extração manual saiu daqui e virou uma chamada à base
                return self.compute_metrics(y_true, y_pred)

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
