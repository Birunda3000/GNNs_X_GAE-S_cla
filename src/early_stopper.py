import torch
import optuna
from typing import Optional, Callable, Tuple, Dict


class EarlyStopper:
    """
    Controle genérico de Early Stopping para modelos de aprendizado.
    """

    def __init__(
        self,
        custom_eval: Optional[
            Callable[[torch.nn.Module], Tuple[Dict[str, float], float]]
        ] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        metric_name: str = "metric",
        restore_best: bool = True,
        trial: Optional[optuna.trial.Trial] = None,
    ):
        assert mode in ["min", "max"], "mode deve ser 'min' ou 'max'."

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.restore_best = restore_best
        self.custom_eval = custom_eval

        self.trial = trial

        self.best_value = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = 0
        self.best_state_dict = None
        self.epochs_no_improve = 0

    def check(
        self, 
        model: torch.nn.Module, 
        epoch: int, 
        current_value: Optional[float] = None,
    ) -> Tuple[bool, float, int, Optional[Dict[str, float]]]:
        """
        Avalia se deve parar o treinamento com base na métrica atual.

        Retorna:
            (stop_now, current_value)
        """

        # Permite usar função customizada para calcular a métrica
        report = None
        if self.custom_eval is not None:
            report, current_value = self.custom_eval(model)

        if current_value is None:
            raise ValueError(
                "current_value não pode ser None se custom_eval não estiver definido."
            )

        # === NOVO: Integração com Optuna Pruning ===
        if self.trial is not None:
            # Reporta o valor atual para o Optuna
            self.trial.report(current_value, epoch)

            # Verifica se deve podar (mas nunca podar na época 0 ou se estivermos melhorando muito rápido)
            if self.trial.should_prune():
                print(f"[OPTUNA PRUNING] Podando trial na epoch {epoch} (Valor: {current_value:.4f})")
                raise optuna.TrialPruned()
        # ===========================================

        # Calcula melhora
        improvement = (
            self.best_value - current_value
            if self.mode == "min"
            else current_value - self.best_value
        )

        # Se houve melhora significativa
        if improvement > self.min_delta:
            self.best_value = current_value
            self.best_epoch = epoch
            self.epochs_no_improve = 0

            if self.restore_best:
                self.best_state_dict = {
                    k: v.cpu().clone().detach() for k, v in model.state_dict().items()
                }

        else:
            self.epochs_no_improve += 1

        # Verifica parada
        if self.epochs_no_improve >= self.patience:
            print(
                f"[EARLY STOPPING] {self.metric_name} não melhorou por {self.patience} épocas. "
                f"Melhor valor: {self.best_value:.6f} (epoch {self.best_epoch})"
            )
            # ✅ Retorna True para parar, mas ainda retorna o valor da métrica ATUAL
            return True, current_value, self.best_epoch, report

        # ✅ Retorno consistente em ambos os casos
        return False, current_value, self.best_epoch, report

    def restore_best_state(self, model: torch.nn.Module):
        """Restaura o melhor estado salvo do modelo, se aplicável."""
        if self.restore_best and self.best_state_dict is not None:
            print(
                f"Restaurando modelo para o melhor estado (epoch {self.best_epoch}, {self.metric_name}: {self.best_value:.6f})."
            )
            model.load_state_dict(self.best_state_dict)
        elif self.restore_best and self.best_state_dict is None:
            print("[AVISO] restore_best=True, mas nenhum estado foi salvo para restaurar.")
