import json
import pandas as pd
import os
import glob

def main(json_path: str, csv_name: str = "result.csv") -> pd.DataFrame:
    """
    Lê um arquivo JSON de resultados e gera um DataFrame com os campos principais,
    salvando automaticamente em CSV na mesma pasta.
    
    Estrutura das colunas:
    - Modelo
    - Dataset
    - Carga de dados (RAM)
    - Pico de RAM (Treino)
    - Tempo de Treino (seg)
    - F1-Score (Ponderado)
    - Acurácia
    """
    
    # Lê o JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = data.get("input_wsg_file", "")
    carga_dados_ram = (
        data.get("detailed_reports", {})
        .get("memory_summary", {})
        .get("ram_data_load_increase_readable", "")
    )
    
    classification_results = data.get("classification_results", {})
    memory_per_model = data.get("detailed_reports", {}).get("memory_per_model", {})
    
    # Monta as linhas do DataFrame
    linhas = []
    for modelo, resultados in classification_results.items():
        linha = {
            "Modelo": modelo,
            "Dataset": dataset,
            "Carga de dados (RAM)": carga_dados_ram,
            "Pico de RAM (Treino)": memory_per_model.get(modelo, {}).get("peak_ram_readable", ""),
            "Tempo de Treino (seg)": resultados.get("training_time_seconds", None),
            "F1-Score (Ponderado)": resultados.get("f1_score_weighted", None),
            "Acurácia": resultados.get("accuracy", None)
        }
        linhas.append(linha)
    
    df = pd.DataFrame(linhas)
    
    # Gera caminho completo do CSV
    pasta = os.path.dirname(os.path.abspath(json_path))
    csv_path = os.path.join(pasta, csv_name)
    
    # Salva CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ CSV salvo em: {csv_path}")
    return df



if __name__ == "__main__":
    base_dir = os.path.join("data", "output", "t_CLASSIFICATION_RUNS")

    # Lista todos os arquivos classification_summary.json dentro da pasta base
    json_files = glob.glob(os.path.join(base_dir, "**", "classification_summary.json"), recursive=True)

    print(f"🔍 {len(json_files)} arquivos encontrados.\n")

    # Executa a função main() para cada arquivo encontrado
    for i, json_path in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processando: {json_path}")
        try:
            main(json_path)
        except Exception as e:
            print(f"❌ Erro ao processar {json_path}: {e}")
