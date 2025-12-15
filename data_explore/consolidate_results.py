import os
import json
import glob
import pandas as pd
import re
from typing import Dict, Any

# Caminhos
BASE_SEARCH_PATH = "data/output/OPTUNA_RUNS/SKLEARN_Classifiers"
OUTPUT_CSV_PATH = "data_explore/consolidated_sklearn_results.csv"

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
        # parts[-2] = Pasta da Run (com score no nome)
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
        # Fallback simples se a estrutura mudar
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
    print(f"🚀 Iniciando consolidação de resultados em: {BASE_SEARCH_PATH}")
    
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
            # O JSON tem chaves como 'model_name', 'params', 'best_cv_f1_mean'
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
    # Remove prefixo 'params_' para deixar o CSV mais limpo, se desejar
    df.columns = [c.replace("params_", "param_") for c in df.columns]

    # Ordenar colunas para facilitar leitura
    # Coloca metadados primeiro, depois métricas, depois params
    cols = list(df.columns)
    priority_cols = ["dataset_name", "model_type", "trial_id", "best_cv_f1_mean", "best_cv_f1_std"]
    
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
    print(f"   🧠 Modelos encontrados: {df['model_type'].unique()}")
    print("="*60)

    # Preview
    print("\nTop 5 Melhores Resultados:")
    print(df.sort_values(by="best_cv_f1_mean", ascending=False).head(5)[["dataset_name", "model_type", "best_cv_f1_mean", "param_C", "param_n_neighbors", "param_n_estimators"]].to_string())

if __name__ == "__main__":
    consolidate()