import json
import os
import src.directory_manager as directory_manager

class ReportManager:
    """Gerencia a criação e salvamento de relatórios de execução do modelo."""

    def __init__(self, directory_manager: directory_manager.DirectoryManager):
        self.directory_manager = directory_manager
        self.report = None

    def create_report(self, report: dict):
        """Inclui o relatório gerado na execução."""
        self.report = report
    
    def add_report_section(self, section_name: str, section_content: dict):
        """Adiciona uma seção ao relatório existente."""
        if self.report is None:
            self.report = {}
        if section_name in self.report:
            print(f"⚠️ Seção '{section_name}' já existe — sobrescrevendo.")

        self.report[section_name] = section_content
    
    def save_report(self):
        """Salva o relatório em um arquivo JSON no diretório da execução."""
        if self.report is None:
            print("⚠️ Nenhum relatório para salvar.")
            return
        
        # Caminho do diretório atual de execução
        report_path = os.path.join(self.directory_manager.get_run_path(), "run_report.json")
        
        # Cria arquivo JSON formatado
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=4, ensure_ascii=False)

        print(f"📄 Relatório salvo em: {report_path}")


