#!/usr/bin/env python3
import os
import json
from typing import Any

# ---------- CONFIGURAÇÃO ----------
# Lista de pastas onde o script vai procurar arquivos JSON
FOLDERS = [
    "data/output/EMBEDDING_RUNS",
    "data/output/CLASSIFICATION_RUNS",
    "data/output/GRAPH_CLASSIFICATION_RUNS",
]

# Extensões que serão consideradas “JSON”
EXTENSIONS = (".json",)

# Limite de amostras para cada pasta (evita ler milhares de arquivos acidentalmente)
MAX_FILES = 500
# ----------------------------------


def guess_type(value: Any):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if not value:
            return "list[empty]"
        subtypes = {guess_type(v) for v in value[:5]}
        return f"list[{', '.join(sorted(subtypes))}]"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def describe_json_structure(obj: Any, indent: int = 0):
    spacer = "  " * indent
    if isinstance(obj, dict):
        lines = ["{"]
        for k, v in obj.items():
            t = guess_type(v)
            if isinstance(v, (dict, list)):
                lines.append(f"{spacer}  {k}: {t}")
                if indent < 2:
                    sub = describe_json_structure(v, indent + 1)
                    for line in sub.splitlines():
                        lines.append(f"{spacer}    {line}")
            else:
                lines.append(f"{spacer}  {k}: {t}")
        lines.append(f"{spacer}}}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = [f"{spacer}[{guess_type(obj)}]"]
        if obj and isinstance(obj[0], (dict, list)):
            sub = describe_json_structure(obj[0], indent + 1)
            for line in sub.splitlines():
                lines.append(f"{spacer}  {line}")
        return "\n".join(lines)
    else:
        return f"{spacer}{guess_type(obj)}"


def has_exact_json_extension(filename: str) -> bool:
    """Retorna True apenas se o arquivo tiver extensão exata '.json' (um único ponto)."""
    base = os.path.basename(filename)
    if not base.lower().endswith(".json"):
        return False
    name_part, ext = os.path.splitext(base)
    # só aceita se houver apenas um ponto no nome completo
    return ext == ".json" and "." not in name_part


def find_json_files(folders):
    """Busca arquivos JSON exatamente com extensão '.json' (ignora .wsg.json etc)."""
    files = []
    for folder in folders:
        for root, _, filenames in os.walk(folder):
            for name in filenames:
                if has_exact_json_extension(name):
                    files.append(os.path.join(root, name))
    return files


def analyze_files(files):
    structures = {}
    for fpath in files[:MAX_FILES]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text.startswith("{") or text.startswith("["):
                    obj = json.loads(text)
                else:
                    continue
            struct = describe_json_structure(obj)
            structures.setdefault(struct, []).append(fpath)
        except Exception as e:
            print(f"⚠️ Erro lendo {fpath}: {e}")
    return structures


def main():
    print("🔍 Buscando arquivos JSON...")
    files = find_json_files(FOLDERS)
    print(f"Encontrados {len(files)} arquivos JSON possíveis.")

    structures = analyze_files(files)

    print("\n📊 Estruturas distintas encontradas:")
    for idx, (structure, paths) in enumerate(structures.items(), start=1):
        print("\n" + "=" * 80)
        print(f"Tipo {idx}: ({len(paths)} arquivo(s))")
        print("-" * 80)
        print(structure)
        print("-" * 80)
        for p in paths[:5]:
            print("Exemplo:", p)
        if len(paths) > 5:
            print(f"... +{len(paths) - 5} mais")
    print("\n✅ Análise concluída.")


if __name__ == "__main__":
    main()
