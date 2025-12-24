import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

from src.config import Config


class SklearnObjective:
    """
    Classe callable para encapsular a lógica de otimização do Optuna
    para modelos Scikit-Learn.

    Usar classe ao invés de lambda evita o bug de captura de variável por referência.
    """

    def __init__(self, X, y, model_name, grid, model_class, config: Config):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.grid = grid
        self.model_class = model_class
        self.config = config

    def __call__(self, trial):
        # 1. Sugerir Parâmetros
        params = {}
        for param_name, param_values in self.grid.items():
            params[param_name] = trial.suggest_categorical(param_name, param_values)

        # 2. Configuração Específica por Modelo
        model = self._create_model(params)

        # 3. Cross-Validation (5-Fold)
        try:
            scorer = make_scorer(f1_score, average="weighted")
            scores = cross_val_score(
                model, self.X, self.y, cv=5, scoring=scorer, n_jobs=-1
            )
            return scores.mean()

        except Exception as e:
            print(f"[ERRO] Trial {trial.number} ({self.model_name}): {e}")
            return 0.0

    def _create_model(self, params):
        """Instancia o modelo com configurações específicas por tipo."""

        if self.model_name == "MLP":
            return self.model_class(
                **params,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=self.config.RANDOM_SEED,
            )

        elif self.model_name == "XGBoost":
            return self.model_class(
                **params,
                n_estimators=200,
                #eval_metric="mlogloss",
                use_label_encoder=False,
                n_jobs=1,
                random_state=self.config.RANDOM_SEED,
            )

        elif self.model_name == "KNN":
            return self.model_class(**params, n_jobs=1)

        elif self.model_name == "QDA":
            # QDA não tem random_state nem n_jobs
            return self.model_class(**params)

        else:
            # Padrão: LogReg, RF, SVM
            extra_args = {"random_state": self.config.RANDOM_SEED}

            # Verifica se modelo suporta n_jobs
            try:
                test_model = self.model_class()
                if "n_jobs" in test_model.get_params():
                    extra_args["n_jobs"] = 1
            except Exception:
                pass

            return self.model_class(**params, **extra_args)
