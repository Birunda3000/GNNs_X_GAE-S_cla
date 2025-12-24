import os
import json
import glob
import pandas as pd
import re

# Configuração dos Caminhos
BASE_OUTPUT_PATH = "data/output"
CSV_DEST_PATH = "data/results_compiled"

# Cria pasta de destino se não existir
os.makedirs(CSV_DEST_PATH, exist_ok=True)

def extract_emb_dim_from_filename(filename):
    """Tenta extrair a dimensão do embedding do nome do arquivo ou caminho."""
    match = re.search(r'emb_dim_(\d+)', filename)
    if match: return int(match.group(1))
    
    match = re.search(r'\((\d+)\)', filename)
    if match: return int(match.group(1))
    return None

def flatten_json(y):
    """Achata JSONs aninhados, ignorando listas gigantes."""
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                if a in ['training_history', 'detailed_results_per_model', 'feature_weights']:
                    continue
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            pass
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def get_training_metrics(report_data):
    """
    Extrai métricas de treinamento:
    1. Tempo Total
    2. Tempo Até Melhor Época
    3. Melhor Época (Best Epoch)
    """
    # 1. Tempo Total
    total_time = report_data.get("total_training_time") or \
                 report_data.get("training_time_seconds") or \
                 report_data.get("Training_Report", {}).get("total_training_time")

    # 2. Dados de Histórico (Embeddings / GNNs com Early Stopping)
    train_report = report_data.get("Training_Report", {})
    history = train_report.get("training_history", [])
    
    # Tenta pegar best_epoch do report aninhado OU da raiz
    best_epoch = train_report.get("best_epoch") or report_data.get("best_epoch")
    
    time_to_best = None
    
    # Se temos histórico e sabemos a melhor época, buscamos o tempo exato
    if history and best_epoch is not None:
        for entry in history:
            if entry.get("epoch") == best_epoch:
                time_to_best = entry.get("Time_per_epoch")
                break
    
    # Fallback: Se o modelo já salvou o tempo direto
    if time_to_best is None:
        time_to_best = report_data.get("time_to_best_epoch")

    return total_time, time_to_best, best_epoch

def process_embedding_runs():
    print("🔄 Processando EMBEDDING RUNS...")
    files = glob.glob(os.path.join(BASE_OUTPUT_PATH, "EMBEDDING_RUNS", "**", "run_report.json"), recursive=True)
    
    data_list = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            flat_data = flatten_json(data)
            
            # Extração
            total, to_best, epoch = get_training_metrics(data)
            flat_data['Total_Training_Time_Seconds'] = total
            flat_data['Time_Until_Best_Epoch_Seconds'] = to_best
            flat_data['Best_Epoch'] = epoch
            
            flat_data['source_file'] = os.path.basename(os.path.dirname(file_path))
            data_list.append(flat_data)
        except Exception as e:
            print(f"⚠️ Erro ao ler {file_path}: {e}")

    if data_list:
        df = pd.DataFrame(data_list)
        # Ordenação Prioritária
        priority_cols = ['dataset_name', 'Embedding_Dim', 'Training_Report_best_score', 
                         'Total_Training_Time_Seconds', 'Time_Until_Best_Epoch_Seconds', 'Best_Epoch']
        cols = [c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]
        
        output_file = os.path.join(CSV_DEST_PATH, "embeddings_runs.csv")
        df[cols].to_csv(output_file, index=False)
        print(f"✅ Salvo: {output_file} ({len(df)} linhas)")

def process_graph_classification():
    print("\n🔄 Processando GRAPH CLASSIFICATION (End-to-End)...")
    files = glob.glob(os.path.join(BASE_OUTPUT_PATH, "GRAPH_CLASSIFICATION_RUNS", "**", "run_report.json"), recursive=True)
    
    data_list = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            common_data = flatten_json({k: v for k, v in data.items() if k not in ['results_summary_per_model', 'memory_per_model']})
            results = data.get('results_summary_per_model', {})
            memory = data.get('memory_per_model', {})

            for model_name, metrics in results.items():
                row = common_data.copy()
                row['Model_Name'] = model_name
                
                for k, v in metrics.items():
                    row[f'metric_{k}'] = v
                
                # Extração
                total, to_best, epoch = get_training_metrics(metrics)
                row['Total_Training_Time_Seconds'] = total
                row['Time_Until_Best_Epoch_Seconds'] = to_best
                row['Best_Epoch'] = epoch

                if model_name in memory:
                    for k, v in memory[model_name].items():
                        row[f'memory_{k}'] = v
                
                row['source_file'] = os.path.basename(os.path.dirname(file_path))
                data_list.append(row)

        except Exception as e:
            print(f"⚠️ Erro ao ler {file_path}: {e}")

    if data_list:
        df = pd.DataFrame(data_list)
        output_file = os.path.join(CSV_DEST_PATH, "graph_classification.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ Salvo: {output_file} ({len(df)} linhas)")

def process_feature_classification():
    print("\n🔄 Processando FEATURE CLASSIFICATION (Pipeline A)...")
    files = glob.glob(os.path.join(BASE_OUTPUT_PATH, "CLASSIFICATION_RUNS", "**", "run_report.json"), recursive=True)
    
    data_list = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            common_data = flatten_json({k: v for k, v in data.items() if k not in ['results_summary_per_model', 'memory_per_model']})
            
            emb_dim = extract_emb_dim_from_filename(data.get('input_wsg_file', '')) or \
                      extract_emb_dim_from_filename(file_path)
            common_data['Derived_Embedding_Dim'] = emb_dim

            results = data.get('results_summary_per_model', {})
            memory = data.get('memory_per_model', {})

            for model_name, metrics in results.items():
                row = common_data.copy()
                row['Classifier_Name'] = model_name
                
                for k, v in metrics.items():
                    row[f'metric_{k}'] = v
                
                # Extração
                total, to_best, epoch = get_training_metrics(metrics)
                row['Total_Training_Time_Seconds'] = total
                row['Time_Until_Best_Epoch_Seconds'] = to_best
                row['Best_Epoch'] = epoch
                
                if model_name in memory:
                    for k, v in memory[model_name].items():
                        row[f'memory_{k}'] = v
                
                row['source_file'] = os.path.basename(os.path.dirname(file_path))
                data_list.append(row)

        except Exception as e:
            print(f"⚠️ Erro ao ler {file_path}: {e}")

    if data_list:
        df = pd.DataFrame(data_list)
        
        # Colunas prioritárias
        priority = ['Dataset', 'Derived_Embedding_Dim', 'Classifier_Name', 'metric_test_f1_score_weighted', 
                    'Total_Training_Time_Seconds', 'Time_Until_Best_Epoch_Seconds', 'Best_Epoch']
        cols = [c for c in priority if c in df.columns] + [c for c in df.columns if c not in priority]
        
        output_file = os.path.join(CSV_DEST_PATH, "feature_classification.csv")
        df[cols].to_csv(output_file, index=False)
        print(f"✅ Salvo: {output_file} ({len(df)} linhas)")

if __name__ == "__main__":
    print("🚀 Iniciando compilação (Com Best Epoch e Timings)...")
    process_embedding_runs()
    process_graph_classification()
    process_feature_classification()
    print("\n🏁 Compilação finalizada! Verifique a pasta 'data/results_compiled'.")