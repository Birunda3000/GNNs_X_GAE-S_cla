import os
import json
import csv

def flatten_json(data, parent_key='', sep='.'):
    """
    Função recursiva para achatar dicionários JSON.
    Exemplo: {"a": {"b": 1}} → {"a.b": 1}
    """
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if k == "training_history":
            continue  # Ignora campo grande

        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Transforma listas em colunas indexadas
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

def collect_run_reports(base_dir):
    """
    Percorre todas as subpastas e coleta os dados de cada run_report.json
    """
    data_rows = []
    for root, dirs, files in os.walk(base_dir):
        if "run_report.json" in files:
            json_path = os.path.join(root, "run_report.json")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                flat_data = flatten_json(data)
                flat_data["run_folder"] = os.path.basename(root)
                data_rows.append(flat_data)
            except Exception as e:
                print(f"Erro ao ler {json_path}: {e}")
    return data_rows

def write_csv(data_rows, output_csv):
    """
    Salva os resultados em CSV (todas as colunas encontradas).
    """
    if not data_rows:
        print("Nenhum relatório encontrado.")
        return

    # Gera um conjunto de todas as chaves (colunas)
    fieldnames = sorted({key for row in data_rows for key in row.keys()})

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)

    print(f"✅ CSV gerado com sucesso: {output_csv}")
    print(f"Total de relatórios: {len(data_rows)}")

if __name__ == "__main__":
    run_folders = ["EMBEDDING_RUNS", "CLASSIFICATION_RUNS", "GRAPH_CLASSIFICATION_RUNS"]

    for run_folder in run_folders:
        BASE_DIR = os.path.join("data", "output", run_folder)
        OUTPUT_CSV = os.path.join(BASE_DIR, f"{run_folder}_summary_report.csv")

        reports = collect_run_reports(BASE_DIR)
        write_csv(reports, OUTPUT_CSV)

