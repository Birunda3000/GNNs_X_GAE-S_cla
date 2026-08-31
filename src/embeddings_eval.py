import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from typing import Any

# Importa a classe abstrata que construímos no early_stopper.py
from src.early_stopper import Metric


class BaseProbeMetric(Metric):
    """
    Classe intermediária que abstrai a repetição de código das sondas do scikit-learn.
    Ela cuida da separação de máscaras, treinamento e tratamento de exceções.
    """

    def __init__(self, name: str, patience: int = 15, min_delta: float = 1e-4):
        # Todas as sondas buscam maximizar o F1-Score
        super().__init__(name=name, mode="max", patience=patience, min_delta=min_delta)

    def get_classifier(self):
        """As classes filhas DEVEM retornar a instância do seu classificador aqui."""
        raise NotImplementedError

    def evaluate(self, model, z, data, train_mask, eval_mask) -> float:
        embeddings = z.cpu().numpy()

        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            return 0.0

        y = data.y.cpu().numpy()

        # Converte as máscaras dinâmicas injetadas
        train_m = train_mask.cpu().numpy()
        eval_m = eval_mask.cpu().numpy()

        # O classificador treina em uma e avalia na outra!
        X_train, y_train = embeddings[train_m], y[train_m]
        X_eval, y_eval = embeddings[eval_m], y[eval_m]

        clf = self.get_classifier()
        try:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_eval)
            return float(f1_score(y_eval, pred, average="weighted"))
        except Exception:
            return 0.0


# =====================================================================
# AS 5 SONDAS ESPECIALIZADAS (Métricas)
# =====================================================================


class KNNMetric(BaseProbeMetric):
    def __init__(self, patience: int = 15):
        super().__init__(name="KNN", patience=patience)

    def get_classifier(self):
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


class LogRegMetric(BaseProbeMetric):
    def __init__(self, patience: int = 15):
        super().__init__(name="LogReg", patience=patience)

    def get_classifier(self):
        return LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")


class QDAMetric(BaseProbeMetric):
    def __init__(self, patience: int = 15):
        super().__init__(name="QDA", patience=patience)

    def get_classifier(self):
        return QuadraticDiscriminantAnalysis(reg_param=0.01)


class CentroidMetric(BaseProbeMetric):
    def __init__(self, patience: int = 15):
        super().__init__(name="Centroid", patience=patience)

    def get_classifier(self):
        return NearestCentroid()


class DTMetric(BaseProbeMetric):
    def __init__(self, patience: int = 15):
        super().__init__(name="DT", patience=patience)

    def get_classifier(self):
        return DecisionTreeClassifier(max_depth=8, random_state=42)


# =====================================================================
# Métrica de Loss Integrada
# =====================================================================
class ReconstructionLossMetric(Metric):
    """Métrica para monitorar o erro de reconstrução do Autoencoder."""

    def __init__(self, patience: int = 10):
        super().__init__(name="Recon_Loss", mode="min", patience=patience)

    def evaluate(self, model: torch.nn.Module, z: torch.Tensor, data: Any) -> float:
        # Requer que o modelo implemente 'compute_total_loss'
        return float(model.compute_total_loss(z, data, data.edge_index).item())
