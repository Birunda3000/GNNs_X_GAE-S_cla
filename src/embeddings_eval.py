import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from torch_geometric.data import Data


def evaluate_embeddings(model, data: Data, device: torch.device):
    """
    Score Proxy otimista para GAEs:
    - Recompensa qualquer melhoria real em qualquer sonda.
    - Evita penalizar ruído (QDA, DT, etc).
    - Não trava no 'early winner' (ex: KNN parado em 0.80).
    
    Fórmula final:
        final_score = best + soft_mean
    onde:
        best = max(scores)
        soft_mean = média dos scores >= 0.75 * best
    
    Isso captura progresso lento e dá robustez ao EarlyStopping.
    """

    model.eval()
    with torch.no_grad():
        embeddings = model.inference(data).cpu().numpy()

    # Segurança
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        return {}, 0.0

    # Dados
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()

    X_train, X_val = embeddings[train_mask], embeddings[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    scores = {}

    def safe_run(name, clf):
        try:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)
            scores[name] = float(f1_score(y_val, pred, average="weighted"))
        except Exception:
            scores[name] = 0.0

    # 5 sondas leves
    safe_run("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    safe_run("LogReg", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced"))
    safe_run("QDA", QuadraticDiscriminantAnalysis(reg_param=0.01))
    safe_run("Centroid", NearestCentroid())
    safe_run("DT", DecisionTreeClassifier(max_depth=8, random_state=42))

    vals = list(scores.values())
    best = max(vals)

    # SOFT-MEAN: só valores próximos ao melhor score contam
    thr = 0.75 * best  # <= limiar flexível e otimista
    soft_vals = [v for v in vals if v >= thr]

    soft_mean = np.mean(soft_vals) if soft_vals else best

    final_score = best + soft_mean

    return scores, float(final_score)