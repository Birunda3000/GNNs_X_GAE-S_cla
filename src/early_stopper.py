import torch
import optuna
from typing import Optional, Tuple, Dict, List, Any
from abc import ABC, abstractmethod
import torch
from typing import Any


class Metric(ABC):
    """
    Classe base abstrata para todas as métricas do Early Stopper.
    Força as classes filhas a implementarem o método 'evaluate'.
    """

    def __init__(self, name: str, mode: str, patience: int, min_delta: float = 1e-4):
        if mode not in ["max", "min"]:
            raise ValueError(f"O mode deve ser 'max' ou 'min'. Recebido: {mode}")

        self.name = name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

    @abstractmethod
    def evaluate(self, model: torch.nn.Module, z: torch.Tensor, data: Any) -> float:
        """As classes filhas DEVEM obrigatoriamente escrever o código desta função."""
        pass


class UniversalEarlyStopper:
    """
    Orquestrador de Early Stopping 100% Desacoplado e Stateless.

    Não armazena os dados de validação nem o modelo, recebendo-os apenas no momento da checagem.
    Avalia múltiplas métricas simultaneamente de forma independente.
    """

    def __init__(
        self,
        metrics: List[Metric],
        stop_condition: str = "all",
        restore_best: bool = True,
        trial: Optional[optuna.trial.Trial] = None,
    ):
        """
        Inicializa o orquestrador de parada antecipada.

        Args:
            metrics (List[Metric]): Lista contendo as cápsulas de métricas a serem monitoradas.
            stop_condition (str): Regra geral de parada.
                - 'all': Interrompe apenas se TODAS as métricas atingirem a paciência.
                - 'any': Interrompe se QUALQUER métrica atingir a paciência.
            restore_best (bool): Se True, salva os pesos da melhor época na memória RAM.
            trial (optuna.trial.Trial, opcional): Objeto Trial do Optuna para aplicar Pruning automático.
        """
        if stop_condition not in ["all", "any"]:
            raise ValueError("stop_condition deve ser rigorosamente 'all' ou 'any'.")
        if not metrics:
            raise ValueError("A lista de métricas (metrics) não pode estar vazia.")

        self.metrics = metrics
        self.stop_condition = stop_condition
        self.restore_best = restore_best
        self.trial = trial

        # Dicionários para rastrear a vida útil isolada de cada métrica
        self.counters: Dict[str, int] = {m.name: 0 for m in metrics}
        self.best_values: Dict[str, float] = {m.name: float("inf") if m.mode == "min" else -float("inf") for m in metrics}

        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def check(self, epoch: int, model: torch.nn.Module, data: Any, **kwargs) -> Tuple[bool, Dict[str, float]]:
        """
        Executa uma avaliação completa do modelo contra todas as métricas configuradas.

        Passos:
        1. Executa um único forward pass no modelo para gerar os embeddings latentes (Z).
        2. Delega para cada objeto Metric a responsabilidade de se autoavaliar.
        3. Atualiza os contadores de paciência individuais.
        4. Avalia a condição global de parada.

        Args:
            epoch (int): A época atual do loop de treinamento.
            model (torch.nn.Module): O modelo PyTorch a ser avaliado.
            data (Any): Os dados de validação (ex: objeto Data do PyG ou DataLoader).

        Returns:
            Tuple[bool, Dict[str, float]]:
                - stop_now (bool): Sinal verde/vermelho para interromper o treinamento.
                - report (Dict): O relatório completo com os valores calculados para o MLflow.
        """
        # =================================================================
        # 1. GERAÇÃO ÚNICA (Evita desperdício de GPU reavaliando o modelo)
        # =================================================================
        z = model.inference(data)

        report: Dict[str, float] = {}
        improved_any: bool = False

        # =================================================================
        # 2. DELEGAÇÃO DA MATEMÁTICA E RASTREAMENTO INDEPENDENTE
        # =================================================================
        for metric in self.metrics:
            # A métrica resolve sua própria matemática usando a função que lhe foi injetada
            current_val = metric.evaluate(model, z, data, **kwargs)
            report[metric.name] = current_val
            best_val = self.best_values[metric.name]

            # Lógica para métricas de maximização (ex: F1, Accuracy)
            if metric.mode == "max" and current_val > (best_val + metric.min_delta):
                self.best_values[metric.name] = current_val
                self.counters[metric.name] = 0
                improved_any = True

            # Lógica para métricas de minimização (ex: Loss, MSE)
            elif metric.mode == "min" and current_val < (best_val - metric.min_delta):
                self.best_values[metric.name] = current_val
                self.counters[metric.name] = 0
                improved_any = True

            # Se não superou o delta, perde um ponto de paciência
            else:
                self.counters[metric.name] += 1

        # =================================================================
        # 3. INTEGRAÇÃO OPTUNA (Pruning) E SALVAMENTO DE CHECKPOINT
        # =================================================================
        if self.trial is not None:
            # Elege a primeira métrica da lista como guia principal para o Optuna
            main_metric_name = self.metrics[0].name
            self.trial.report(report[main_metric_name], epoch)
            if self.trial.should_prune():
                print(f"[OPTUNA PRUNING] Trial podada na época {epoch} devido à estagnação.")
                raise optuna.TrialPruned()

        if improved_any:
            self.best_epoch = epoch
            if self.restore_best:
                # Salva copiando para a CPU para evitar vazamento de memória da GPU (OOM)
                self.best_state_dict = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}

        # =================================================================
        # 4. VOTAÇÃO FINAL DA CONDIÇÃO DE PARADA
        # =================================================================
        if self.stop_condition == "all":
            # Exige unanimidade: Todos os contadores devem estar estourados
            stop_now = all(self.counters[m.name] >= m.patience for m in self.metrics)
        else:
            # Qualquer estagnação encerra a execução
            stop_now = any(self.counters[m.name] >= m.patience for m in self.metrics)

        if stop_now:
            print(f"\n[EARLY STOPPING] Condição '{self.stop_condition}' atingida.")
            print(f"Estado dos contadores de paciência: {self.counters}")

        return stop_now, report

    def restore_best_state(self, model: torch.nn.Module) -> None:
        """
        Restaura o modelo para o estado de pesos da época com melhor avaliação global.

        Args:
            model (torch.nn.Module): O modelo que receberá os pesos restaurados.
        """
        if self.restore_best and self.best_state_dict is not None:
            print(f"[RESTAURAÇÃO] Retornando modelo para os pesos da melhor época: {self.best_epoch}")
            model.load_state_dict(self.best_state_dict)
        elif self.restore_best and self.best_state_dict is None:
            print("[AVISO] Nenhuma época melhor foi registrada para restauração.")
