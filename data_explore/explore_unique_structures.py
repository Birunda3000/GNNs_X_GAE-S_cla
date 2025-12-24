import os
import json
import glob
from typing import Any

BASE_SEARCH_PATH = "data/output"

def get_structure_signature(data: Any) -> Any:
    """
    Transforma dados em representação de tipos.
    Agora com detecção inteligente de Dicionários Homogêneos (Maps) para evitar spam.
    """
    if isinstance(data, dict):
        if not data:
            return "dict{}"
        
        # --- NOVIDADE: Lógica de Colapso para Dicionários Grandes ---
        # Se o dicionário é grande, provavelmente é um mapeamento de IDs (ex: nós do grafo)
        # e não uma configuração com chaves fixas.
        if len(data) > 10:
            # Pega uma amostra de valores (primeiros 20 para performance)
            sample_values = list(data.values())[:20]
            sample_sigs = [get_structure_signature(v) for v in sample_values]
            
            # Verifica se todos na amostra são iguais
            # (Usamos string dump para comparar dicts/listas complexas)
            first_sig_str = json.dumps(sample_sigs[0], sort_keys=True)
            is_homogeneous = all(json.dumps(s, sort_keys=True) == first_sig_str for s in sample_sigs)
            
            if is_homogeneous:
                # Retorna uma representação colapsada
                return {
                    "__DYNAMIC_KEY__ (IDs collapse)": sample_sigs[0]
                }
        # ------------------------------------------------------------

        # Se for pequeno ou heterogêneo, mantém a estrutura explícita (útil para configs)
        return {k: get_structure_signature(v) for k, v in sorted(data.items())}
    
    elif isinstance(data, list):
        if not data:
            return "list[]"
        
        # Pega assinaturas únicas (mesma lógica anterior)
        # Limitamos a amostra em listas gigantes também para performance
        sample_data = data[:50] if len(data) > 50 else data
        item_signatures = [get_structure_signature(item) for item in sample_data]
        
        unique_signatures_set = set()
        unique_signatures_list = []
        
        for sig in item_signatures:
            sig_str = json.dumps(sig, sort_keys=True)
            if sig_str not in unique_signatures_set:
                unique_signatures_set.add(sig_str)
                unique_signatures_list.append(sig)
        
        if len(unique_signatures_list) == 1:
            return [unique_signatures_list[0]] # Representação limpa: [Tipo]
        else:
            return f"list{unique_signatures_list}" 
    
    else:
        return type(data).__name__

def explore_unique_jsons():
    print(f"🔍 Varrendo estruturas únicas em: {BASE_SEARCH_PATH}...\n")
    
    if not os.path.exists(BASE_SEARCH_PATH):
        print(f"❌ Erro: Diretório '{BASE_SEARCH_PATH}' não encontrado.")
        return

    json_files = glob.glob(os.path.join(BASE_SEARCH_PATH, "**", "*.json"), recursive=True)
    
    if not json_files:
        print("⚠️ Nenhum arquivo JSON encontrado.")
        return

    unique_structures = {}
    structure_sources = {}

    print(f"📂 Total de arquivos encontrados: {len(json_files)}")
    print("Analisando estruturas...\n")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            signature = get_structure_signature(content)
            signature_key = json.dumps(signature, sort_keys=True)
            
            if signature_key not in unique_structures:
                unique_structures[signature_key] = signature
                path_parts = file_path.split(os.sep)
                short_path = os.path.join("...", *path_parts[-4:])
                structure_sources[signature_key] = short_path

        except Exception as e:
            print(f"❌ Erro ao ler {file_path}: {e}")

    print("="*80)
    print(f"🚀 RELATÓRIO DE ESTRUTURAS ÚNICAS ({len(unique_structures)} Tipos Encontrados)")
    print("="*80)

    for i, (sig_key, structure) in enumerate(unique_structures.items(), 1):
        source = structure_sources[sig_key]
        
        label = "DESCONHECIDO"
        source_str = str(source).lower()
        struct_str = str(structure)

        # Tentativa de rotular automaticamente
        if "best_hyperparameters" in source_str: label = "RESUMO (Best Params)"
        elif "run_report" in source_str: label = "RELATÓRIO DE EXECUÇÃO"
        elif "embeddings" in source_str: label = "ARQUIVO DE EMBEDDINGS (WSG)"
        elif "dynamic_key" in struct_str.lower(): label = "DADOS DE NÓS/GRAFO (Compactado)"

        print(f"\n🔹 ESTRUTURA #{i}: {label}")
        print(f"   Fonte Exemplo: {source}")
        print("-" * 40)
        
        # Formatação limpa
        json_str = json.dumps(structure, indent=4)
        print(json_str.replace('"', '').replace('__DYNAMIC_KEY__ (IDs collapse)', '<DYNAMIC_ID>')) 
        print("-" * 80)

if __name__ == "__main__":
    explore_unique_jsons()