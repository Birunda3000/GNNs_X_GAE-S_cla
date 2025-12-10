import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from typing import Tuple, Dict


def evaluate_embeddings(model, data: Data, device: torch.device) -> Tuple[Dict[str, float], float]:
    """
    Avalia a qualidade dos embeddings gerados pelo modelo VGAE usando
    três perspectivas complementares.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.inference(data).cpu().numpy()

    # --- CORREÇÃO DE SEGURANÇA (FAULT TOLERANCE) ---
    # Verifica se o modelo gerou números inválidos (NaN ou Infinito)
    # Isso acontece quando os gradientes explodem durante o treino.
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("⚠️ [AVISO] Embeddings contêm NaN ou Inf. Modelo instável. Pulando avaliação.")
        # Retorna scores zerados para descartar este modelo sem quebrar o script
        dummy_scores = {
            "val_KNN_f1_weighted": 0.0,
            "val_LogisticRegression_f1_weighted": 0.0,
            "val_DecisionTree_f1_weighted": 0.0,
        }
        return dummy_scores, 0.0
    # -----------------------------------------------

    y = data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else np.array(data.y)
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()

    X_train, X_val = embeddings[train_mask], embeddings[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    # --- KNN ---
    # Envolvemos em try/except como dupla segurança
    try:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        val_y_pred_knn = knn.predict(X_val)
        val_f1_knn = float(f1_score(y_val, val_y_pred_knn, average="weighted"))
    except ValueError as e:
        print(f"⚠️ [ERRO] Falha no KNN: {e}")
        val_f1_knn = 0.0

    # --- Logistic Regression ---
    try:
        logreg = LogisticRegression(max_iter=300)
        logreg.fit(X_train, y_train)
        val_y_pred_lr = logreg.predict(X_val)
        val_f1_lr = float(f1_score(y_val, val_y_pred_lr, average="weighted"))
    except ValueError as e:
        print(f"⚠️ [ERRO] Falha na Regressão Logística: {e}")
        val_f1_lr = 0.0

    # --- Decision Tree ---
    try:
        tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree.fit(X_train, y_train)
        val_y_pred_tree = tree.predict(X_val)
        val_f1_tree = float(f1_score(y_val, val_y_pred_tree, average="weighted"))
    except ValueError as e:
        print(f"⚠️ [ERRO] Falha na Decision Tree: {e}")
        val_f1_tree = 0.0

    scores = {
        "val_KNN_f1_weighted": val_f1_knn,            
        "val_LogisticRegression_f1_weighted": val_f1_lr,  
        "val_DecisionTree_f1_weighted": val_f1_tree,     
    }

    best_score = max(val_f1_knn, val_f1_lr, val_f1_tree)

    return scores, best_score