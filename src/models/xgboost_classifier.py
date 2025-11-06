import time
import numpy as np
import torch
import xgboost as xgb
from typing import Any, Dict, Tuple, cast, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch_geometric.data import Data

from src.config import Config
from src.models.base_model import BaseModel

class XGBoostClassifier(BaseModel):
    """
    Implementação de um classificador XGBoost que segue a interface BaseModel.
    """

    def __init__(self, config: Config, num_boost_round=100, **model_params):
        # Chama o __init__ da classe base
        super().__init__(config)
        self.model_name = "XGBoostClassifier"

        # Parâmetros padrão otimizados para classificação
        self.params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",  # Método rápido para grandes datasets
            "random_state": config.RANDOM_SEED,
        }
        # Substitui ou adiciona parâmetros personalizados
        self.params.update(model_params)
        self.num_boost_round = num_boost_round
        self.model = None

    def verify_train_input_data(self, data: Data):
        """Verifica se os dados de entrada para treino estão corretos."""
        assert data.x is not None, "Os dados de entrada devem conter atributos de nó (data.x)."
        assert data.y is not None, "Os dados de entrada devem conter rótulos de nó (data.y)."
        assert data.train_mask is not None, "Os dados de entrada devem conter uma máscara de treino (data.train_mask)."
        assert data.test_mask is not None, "Os dados de entrada devem conter uma máscara de teste (data.test_mask)."

    def train_model(
        self, data: Data
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Treina o modelo XGBoost e retorna as métricas de acordo com
        o esperado pelo ExperimentRunner.
        """
        print(f"\n--- Avaliando (XGBoost): {self.model_name} ---")
        self.verify_train_input_data(data) # Verifica os dados de entrada

        assert isinstance(data.x, torch.Tensor), f"Esperado torch.Tensor, obtido {type(data.x)}"
        assert isinstance(data.y, torch.Tensor), f"Esperado torch.Tensor, obtido {type(data.y)}"
        
        X = data.x.cpu().numpy()
        y = data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto data
        X_train, y_train = X[data.train_mask], y[data.train_mask]
        X_test, y_test = X[data.test_mask], y[data.test_mask]

        # Obter o número de classes (considerando todos os labels)
        num_classes = len(np.unique(y[y != -1])) # Ignora -1 se for usado para nós não rotulados
        self.params["num_class"] = num_classes

        # Preparar dados no formato DMatrix do XGBoost para maior eficiência
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Medir o tempo de treinamento
        start_time = time.process_time()

        print(f"Treinando XGBoost por {self.num_boost_round} rounds...")
        eval_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            self.num_boost_round,
            evals=[(dtrain, "train"), (dtest, "eval")],
            evals_result=eval_result,
            verbose_eval=False, # Desligado para não poluir o log do runner
        )
        train_time = time.process_time() - start_time

        # --- Fazer predições de TESTE ---
        y_pred_test_probs = self.model.predict(dtest)
        y_pred_test = np.argmax(y_pred_test_probs, axis=1)

        # --- Fazer predições de TREINO ---
        y_pred_train_probs = self.model.predict(dtrain)
        y_pred_train = np.argmax(y_pred_train_probs, axis=1)

        # --- Calcular métricas ---
        test_acc = float(accuracy_score(y_test, y_pred_test))
        test_f1 = float(f1_score(y_test, y_pred_test, average="weighted"))

        # --- Montar Relatório (igual ao SklearnClassifier) ---
        test_report = cast(
            Dict[str, Any],
            classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
        )
        train_report = cast(
            Dict[str, Any],
            classification_report(y_train, y_pred_train, output_dict=True, zero_division=0),
        )

        report = {
            "total_training_time": train_time,
            "test_report": test_report,
            "train_report": train_report
        }

        # Retorna o tuple esperado pelo ExperimentRunner
        return test_acc, test_f1, train_time, report

    def evaluate(self, x, y: Optional[Any] = None) -> Any:
        """Evaluate the model."""
        raise NotImplementedError("Use 'train_model' para XGBoostClassifier, pois ele treina e avalia de uma vez.")
    
    def inference(self, x):
        """Run inference with the model."""
        if self.model is None:
            raise RuntimeError("Modelo não foi treinado. Chame 'train_model' primeiro.")
        
        # Converte para DMatrix se não for
        if not isinstance(x, xgb.DMatrix):
            x = xgb.DMatrix(x)
            
        y_pred_probs = self.model.predict(x)
        y_pred = np.argmax(y_pred_probs, axis=1)
        return y_pred