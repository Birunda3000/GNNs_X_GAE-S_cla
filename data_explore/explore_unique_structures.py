import os
import json
import glob
from typing import Any

# Define o caminho relativo à raiz do projeto
# Nota: Assumimos que o script será executado da raiz do projeto
BASE_SEARCH_PATH = "data/output/OPTUNA_RUNS/SKLEARN_Classifiers-nokfold"

def get_structure_signature(data: Any) -> Any:
    """
    Função recursiva que transforma os dados em sua representação de Tipos.
    Retorna uma estrutura espelhada onde os valores são os nomes dos tipos.
    """
    if isinstance(data, dict):
        # Para dicionários, recursivamente pega a estrutura dos valores
        # Ordena chaves para garantir que {a:1, b:2} seja igual a {b:2, a:1} na comparação
        return {k: get_structure_signature(v) for k, v in sorted(data.items())}
    
    elif isinstance(data, list):
        if not data:
            return "list[]"
        
        # Pega as assinaturas únicas dos itens da lista
        item_signatures = [get_structure_signature(item) for item in data]
        
        # Para simplificar a visualização (assumindo listas homogêneas ou estruturas repetidas)
        # Convertemos para string para usar 'set' e remover duplicatas visuais
        unique_signatures_set = set()
        unique_signatures_list = []
        
        for sig in item_signatures:
            # Truque para tornar dicts "hashable" para o set
            sig_str = json.dumps(sig, sort_keys=True)
            if sig_str not in unique_signatures_set:
                unique_signatures_set.add(sig_str)
                unique_signatures_list.append(sig)
        
        # Se todos os itens têm a mesma estrutura (comum), retorna apenas um exemplo
        if len(unique_signatures_list) == 1:
            return f"list[{unique_signatures_list[0]}]"
        else:
            # Se a lista tem tipos misturados (ex: int e float), mostra todos
            return f"list{unique_signatures_list}" 
    
    else:
        # Retorna o nome do tipo primitivo (int, float, str, bool, NoneType)
        return type(data).__name__

def explore_unique_jsons():
    print(f"🔍 Varrendo estruturas únicas em: {BASE_SEARCH_PATH}...\n")
    
    # Verificação de segurança para o caminho
    if not os.path.exists(BASE_SEARCH_PATH):
        print(f"❌ Erro: Diretório '{BASE_SEARCH_PATH}' não encontrado.")
        print("💡 DICA: Execute este script a partir da raiz do projeto: 'python data_explore/explore_unique_structures.py'")
        return

    # Procura todos os JSONs recursivamente
    json_files = glob.glob(os.path.join(BASE_SEARCH_PATH, "**", "*.json"), recursive=True)
    
    if not json_files:
        print("⚠️ Nenhum arquivo JSON encontrado.")
        return

    # Dicionário para armazenar estruturas únicas encontradas
    # Chave: String JSON da estrutura (para garantir unicidade)
    # Valor: Exemplo da estrutura (objeto Python)
    unique_structures = {}
    
    # Mapeia qual arquivo gerou qual estrutura (para sabermos se é KNN, MLP, etc.)
    structure_sources = {}

    print(f"📂 Total de arquivos encontrados: {len(json_files)}")
    print("Analisando estruturas (isso pode levar alguns segundos)...\n")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            
            # Obtém a assinatura (Schema)
            signature = get_structure_signature(content)
            
            # Cria uma "chave" string para deduplicar
            signature_key = json.dumps(signature, sort_keys=True)
            
            if signature_key not in unique_structures:
                unique_structures[signature_key] = signature
                # Guarda o caminho encurtado como referência
                # Pega apenas os ultimos 4 niveis do path para caber na tela
                path_parts = file_path.split(os.sep)
                short_path = os.path.join("...", *path_parts[-5:])
                structure_sources[signature_key] = short_path

        except Exception as e:
            print(f"❌ Erro ao ler {file_path}: {e}")

    # --- IMPRESSÃO DOS RESULTADOS ---
    
    print("="*80)
    print(f"🚀 RELATÓRIO DE ESTRUTURAS ÚNICAS ({len(unique_structures)} Tipos Encontrados)")
    print("="*80)

    # Ordena para manter consistência na visualização
    for i, (sig_key, structure) in enumerate(unique_structures.items(), 1):
        source = structure_sources[sig_key]
        
        # Tenta adivinhar o tipo de arquivo pelo nome ou conteúdo para dar um título bonito
        label = "DESCONHECIDO"
        if "best_hyperparameters" in source:
            label = "RESUMO GERAL (Best Params)"
        elif "run_report" in source:
            if isinstance(structure, dict) and "params" in structure:
                params = structure["params"]
                # Converte params para string para checagem segura
                params_str = str(params)
                if "n_neighbors" in params_str: label = "RELATÓRIO DE EXECUÇÃO (KNN)"
                elif "hidden_layer_sizes" in params_str: label = "RELATÓRIO DE EXECUÇÃO (MLP)"
                elif "n_estimators" in params_str or "max_depth" in params_str: label = "RELATÓRIO DE EXECUÇÃO (RF/XGB)"
                elif "C" in params_str: label = "RELATÓRIO DE EXECUÇÃO (Linear/SVM)"
                else: label = "RELATÓRIO DE EXECUÇÃO (Genérico)"
            else:
                label = "RELATÓRIO DE EXECUÇÃO"

        print(f"\n🔹 ESTRUTURA #{i}: {label}")
        print(f"   Fonte Exemplo: {source}")
        print("-" * 40)
        
        # Imprime o JSON formatado, removendo aspas dos tipos para ficar mais limpo visualmente
        # Ex: "int" vira int
        json_str = json.dumps(structure, indent=4)
        print(json_str.replace('"', '')) 
        print("-" * 80)

if __name__ == "__main__":
    explore_unique_jsons()