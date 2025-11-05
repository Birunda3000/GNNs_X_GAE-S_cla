import os
import shutil
from typing import Dict, Optional, List, Union, Any
import json

from src.config import Config


class DirectoryManager:
    """
    Gerencia a criação e o nomeação de diretórios de saída para cada execução.

    Cria um diretório temporário no início da execução e o renomeia
    no final com base nos resultados obtidos, garantindo uma organização
    clara e sem conflitos.
    """

    def __init__(
        self, timestamp: str, run_folder_name: str, base_path: Optional[str] = None
    ):
        """
        Inicializa o gerenciador e cria o diretório de execução temporário.

        Args:
            timestamp (str): O timestamp único da execução.
            run_folder_name (str): O nome do subdiretório para este tipo de execução
                                   (ex: "EMBEDDING_RUNS", "CLASSIFICATION_RUNS").
            base_path (Optional[str]): O caminho base para criar o diretório de execuções.
                                       Se None, o padrão é 'data/output/'.
        """
        if base_path is None:
            base_path = Config.OUTPUT_PATH

        self.base_path = os.path.join(base_path, run_folder_name)

        self.timestamp = timestamp
        self.temp_dir_name = f"_tmp__{self.timestamp}"
        self.run_dir_path = os.path.join(self.base_path, self.temp_dir_name)
        self.final_dir_path: Optional[str] = None

        # Cria o diretório temporário, se não existir
        os.makedirs(self.run_dir_path, exist_ok=True)
        print(f"Diretório de execução temporário criado em: '{self.run_dir_path}'")

    def get_run_path(self) -> str:
        """Retorna o caminho do diretório da execução atual (seja temporário ou final)."""
        return self.final_dir_path if self.final_dir_path else self.run_dir_path

    def save_classification_report(
        self, input_file: str, results: Dict[str, Any], reports: Dict[str, Any]
    ):
        """Salva um relatório consolidado em formato JSON dentro do diretório da execução."""
        summary = {
            "input_wsg_file": input_file,
            "classification_results": results,
            "detailed_reports": reports,
        }
        report_path = os.path.join(self.get_run_path(), "classification_summary.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nRelatório de classificação salvo em: '{report_path}'")

    def print_summary_table(
        self, results: Dict[str, Any], input_file_path: str, feature_type: str
    ):
        """Imprime a tabela de resumo dos resultados no console."""
        print("\n" + "=" * 65)
        print("RELATÓRIO DE COMPARAÇÃO FINAL".center(65))
        print("-" * 65)
        print(f"Fonte dos Dados: {os.path.basename(input_file_path)}")
        print(f"Tipo de Feature: {feature_type}")
        print("-" * 65)
        print(
            f"{'Modelo':<25} | {'Acurácia':<12} | {'F1-Score':<12} | {'Tempo (s)':<10}"
        )
        print("=" * 65)
        for name, metrics in results.items():
            print(
                f"{name:<25} | {metrics['accuracy']:<12.4f} | {metrics['f1_score_weighted']:<12.4f} | {metrics['training_time_seconds']:<10.2f}"
            )
        print("=" * 65)

    def finalize_run_directory(
        self,
        dataset_name: str,
        metrics: Dict[str, Union[float, int, str]],  # <-- Alterado para aceitar string
    ) -> str:
        """
        Renomeia o diretório temporário para um nome final descritivo e informativo.

        Exemplo de nome final: 'Cora__loss_2.5123__emb_dim_8__08-09-2025_16-44-08'

        Args:
            dataset_name (str): Nome do dataset utilizado (ex: 'Cora').
            metrics (Dict[str, Union[float, int]]): Dicionário com métricas e parâmetros.
                                                    Floats são formatados, inteiros não.

        Returns:
            str: O caminho completo para o diretório final renomeado.
        """
        if not os.path.exists(self.run_dir_path):
            print(
                f"Aviso: O diretório temporário '{self.run_dir_path}' não foi encontrado para renomear."
            )
            return ""

        metrics_str_parts: List[str] = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str_parts.append(f"{key}_{value:.4f}".replace(".", "_"))
            else:
                metrics_str_parts.append(
                    f"{key}_{value}"
                )  # <-- Funciona para int e str

        metrics_str = "__".join(metrics_str_parts)

        # Constrói o nome final, omitindo a parte das métricas se estiver vazia
        if metrics_str:
            final_dir_name = f"{dataset_name}__{metrics_str}__{self.timestamp}"
        else:
            final_dir_name = f"{dataset_name}__{self.timestamp}"

        final_path = os.path.join(self.base_path, final_dir_name)

        # Renomeia o diretório
        shutil.move(self.run_dir_path, final_path)

        self.final_dir_path = final_path
        self.run_dir_path = final_path  # Atualiza o caminho principal

        print(f"Diretório da execução finalizado e renomeado para: '{final_path}'")
        return final_path
