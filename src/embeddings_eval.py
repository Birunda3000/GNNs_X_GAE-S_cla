import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from torch_geometric.data import Data


def evaluate_embeddings(model, data: Data, device: torch.device):
    """
    Avalia a qualidade dos embeddings gerados pelo modelo VGAE usando
    três perspectivas complementares:

    - KNN → Coerência local / clusterização dos embeddings
    - Regressão Logística → Separabilidade linear global
    - Decision Tree rasa → Separabilidade não-linear simples (fronteiras curvas)

    Retorna um dicionário com todos os F1-scores e o score médio combinado.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.inference(data).cpu().numpy()

    y = data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else np.array(data.y)
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    X_train, X_test = embeddings[train_mask], embeddings[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # --- KNN (estrutura local / clusters) ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    f1_knn = f1_score(y_test, y_pred_knn, average="weighted")

    # --- Regressão Logística (separação linear global) ---
    logreg = LogisticRegression(max_iter=300)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    f1_lr = f1_score(y_test, y_pred_lr, average="weighted")

    # --- Decision Tree rasa (não-linear simples) ---
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    f1_tree = f1_score(y_test, y_pred_tree, average="weighted")

    # --- Score composto ---

    scores = {
        "KNN": f1_knn,
        "LogisticRegression": f1_lr,
        "DecisionTree": f1_tree,
    }

    best_score = max(f1_knn, f1_lr, f1_tree)


    return scores, best_score