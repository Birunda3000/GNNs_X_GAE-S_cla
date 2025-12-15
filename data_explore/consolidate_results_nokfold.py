import os
import json
import glob
import pandas as pd
import re
from typing import Dict, Any

# Caminhos
BASE_SEARCH_PATH = "data/output/OPTUNA_RUNS/SKLEARN_Classifiers-nokfold"
OUTPUT_CSV_PATH = "data_explore/consolidated_sklearn_results_nokfold.csv"

def extract_metadata_from_path(file_path: str) -> Dict[str, str]:
    """
    Tenta extrair informações úteis (Dataset, Modelo, Trial) baseando-se na estrutura de pastas.
    Estrutura esperada:
    .../{TIMESTAMP}/{DATASET_NAME}/{MODEL_NAME}/Trial_{ID}/{RUN_DIR}/run_report.json
    """
    parts = file_path.split(os.sep)
    metadata = {
        "timestamp_group": "Unknown",
        "dataset_name": "Unknown",
        "model_type": "Unknown",
        "trial_id": -1
    }
    
    try:
        # Percorre o caminho de trás para frente procurando padrões
        # parts[-1] = run_report.json
        # parts[-2] = Pasta da Run (com score no nome, ex: ...__f1_08953__...)
        # parts[-3] = Trial_X
        # parts[-4] = Nome do Modelo (KNN, MLP, etc)
        # parts[-5] = Nome do Dataset (Musae-Facebook...)
        # parts[-6] = Timestamp da Execução Geral
        
        if "Trial_" in parts[-3]:
            metadata["trial_id"] = int(parts[-3].replace("Trial_", ""))
            metadata["model_type"] = parts[-4]
            metadata["dataset_name"] = parts[-5]
            metadata["timestamp_group"] = parts[-6]
            
    except Exception:
        # Fallback simples se a estrutura mudar, tenta achar pelo nome da pasta
        for i, part in enumerate(parts):
             if "Trial_" in part:
                 try:
                     metadata["trial_id"] = int(part.replace("Trial_", ""))
                     if i > 0: metadata["model_type"] = parts[i-1]
                     if i > 1: metadata["dataset_name"] = parts[i-2]
                 except:
                     pass
        pass
        
    return metadata

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Achata dicionários aninhados. Ex: {'params': {'C': 1}} -> {'params_C': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # Converte listas para string para caber no CSV
            if isinstance(v, list):
                v = str(v)
            items.append((new_key, v))
    return dict(items)

def consolidate():
    print(f"🚀 Iniciando consolidação de resultados (No K-Fold) em: {BASE_SEARCH_PATH}")
    
    if not os.path.exists(BASE_SEARCH_PATH):
        print(f"❌ Erro: Diretório {BASE_SEARCH_PATH} não encontrado.")
        return

    # Encontrar todos os run_report.json
    files = glob.glob(os.path.join(BASE_SEARCH_PATH, "**", "run_report.json"), recursive=True)
    print(f"📂 Encontrados {len(files)} relatórios de execução.")
    
    data_rows = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            # 1. Extrair Metadados do Path
            meta = extract_metadata_from_path(file_path)
            
            # 2. Extrair e Achatar o Conteúdo do JSON
            flat_content = flatten_dict(content)
            
            # 3. Combinar tudo
            row = {**meta, **flat_content}
            
            # Adiciona caminho relativo para referência futura
            row["source_path"] = os.path.relpath(file_path, start=os.getcwd())
            
            data_rows.append(row)
            
        except Exception as e:
            print(f"⚠️ Erro ao processar {file_path}: {e}")

    if not data_rows:
        print("Nenhum dado válido extraído.")
        return

    # Criar DataFrame
    df = pd.DataFrame(data_rows)
    
    # Limpeza de Colunas (Opcional)
    # Remove prefixo 'params_' para deixar o CSV mais limpo
    df.columns = [c.replace("params_", "param_") for c in df.columns]

    # Ordenar colunas para facilitar leitura
    # Prioriza métricas de validação (val_f1, val_acc)
    cols = list(df.columns)
    priority_cols = ["dataset_name", "model_type", "trial_id", "val_f1", "val_acc", "split_type"]
    
    # Organiza: Prioridade + (Resto - Prioridade)
    ordered_cols = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
    df = df[ordered_cols]

    # Salvar
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("="*60)
    print(f"✅ SUCESSO! Resultados consolidados em:")
    print(f"   📄 {OUTPUT_CSV_PATH}")
    print(f"   📊 Total de Linhas: {len(df)}")
    print(f"   🧠 Modelos encontrados: {df['model_type'].unique() if not df.empty else 'Nenhum'}")
    print("="*60)

    # Preview se houver dados
    if not df.empty:
        print("\nTop 5 Melhores Resultados (Validação F1):")
        # Mostra colunas relevantes se existirem
        preview_cols = ["dataset_name", "model_type", "val_f1", "param_C", "param_n_neighbors", "param_n_estimators"]
        preview_cols = [c for c in preview_cols if c in df.columns]
        print(df.sort_values(by="val_f1", ascending=False).head(5)[preview_cols].to_string())

if __name__ == "__main__":
    consolidate()