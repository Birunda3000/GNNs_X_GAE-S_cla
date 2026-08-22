import os
import json
import pandas as pd

def explorar_reddit_raw():
    base_path = "data/datasets/reddit_threads"
    edges_path = os.path.join(base_path, "reddit_edges.json")
    target_path = os.path.join(base_path, "reddit_target.csv")

    print("="*80)
    print("🕵️  EXPLORAÇÃO DE DADOS BRUTOS: REDDIT THREADS")
    print("="*80)

    # --- 1. EXPLORANDO O TARGET (CSV) ---
    print("\n📄 LENDO ARQUIVO: reddit_target.csv")
    if os.path.exists(target_path):
        df = pd.read_csv(target_path)
        print(f"  ↳ Linhas: {len(df)} | Colunas: {len(df.columns)}")
        print(f"  ↳ Nomes das Colunas: {list(df.columns)}")
        print(f"  ↳ Tipos de Dados:\n{df.dtypes.to_string()}")
        print(f"  ↳ Valores Nulos:\n{df.isnull().sum().to_string()}")
        print(f"  ↳ Distribuição de Classes (target):")
        print(df['target'].value_counts(normalize=True).to_string())
        print(f"\n  ↳ Amostra das 3 primeiras linhas:\n{df.head(3).to_string(index=False)}")
    else:
        print("  ❌ ARQUIVO target.csv NÃO ENCONTRADO!")

    # --- 2. EXPLORANDO AS ARESTAS (JSON) ---
    print("\n" + "="*80)
    print("📄 LENDO ARQUIVO: reddit_edges.json")
    if os.path.exists(edges_path):
        with open(edges_path, 'r') as f:
            edges_json = json.load(f)
        
        keys = list(edges_json.keys())
        print(f"  ↳ Total de Chaves (Threads) no JSON: {len(keys)}")
        
        # Cruzamento rápido
        if os.path.exists(target_path):
            chaves_no_csv = set(df['id'].astype(str))
            chaves_no_json = set(keys)
            print(f"  ↳ Threads no JSON que não estão no CSV: {len(chaves_no_json - chaves_no_csv)}")
            print(f"  ↳ Threads no CSV que não estão no JSON: {len(chaves_no_csv - chaves_no_json)}")

        print("\n🔬 RADIOGRAFIA DE 3 THREADS DE AMOSTRA:")
        for k in keys[:3]:
            arestas = edges_json[k]
            
            # Extraindo os nós únicos desta thread
            nos_unicos = set()
            for u, v in arestas:
                nos_unicos.add(u)
                nos_unicos.add(v)
                
            n_arestas = len(arestas)
            n_nos = len(nos_unicos)
            
            print(f"\n  ➤ Thread ID '{k}':")
            print(f"      - Total de arestas: {n_arestas}")
            print(f"      - Total de nós únicos: {n_nos}")
            if n_nos > 0:
                print(f"      - IDs dos nós vão de: {min(nos_unicos)} até {max(nos_unicos)}")
            print(f"      - Amostra das arestas: {arestas[:5]}{'...' if n_arestas > 5 else ''}")
            
    else:
        print("  ❌ ARQUIVO edges.json NÃO ENCONTRADO!")
        
    print("\n" + "="*80)

if __name__ == "__main__":
    explorar_reddit_raw()