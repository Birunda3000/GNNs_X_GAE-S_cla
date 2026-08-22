# -*- coding: utf-8 -*-
import os
import re

def natural_sort_key(s):
    """Ordenação natural: A1 < A2 < A10 < A11"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def gerar_arvore_diretorios(path_raiz, nome_arquivo_saida, limite_itens, ignorar=None):
    if ignorar is None:
        ignorar = []

    caminho_completo_saida = os.path.join(path_raiz, nome_arquivo_saida)

    with open(caminho_completo_saida, "w", encoding="utf-8") as f:
        nome_raiz = os.path.basename(path_raiz.rstrip("/"))
        f.write(f"{nome_raiz}/\n")

        _listar(f, path_raiz, "", limite_itens, ignorar)

    print(f"Arquivo gerado: {caminho_completo_saida}")


def _listar(arquivo_saida, pasta_atual, prefixo, limite, ignorar):
    itens = []

    try:
        for item in os.scandir(pasta_atual):
            if item.name in ignorar:
                continue
            itens.append((item.name, item.is_dir()))
    except PermissionError:
        return

    # Ordenar: pastas primeiro, depois arquivos (ordem natural)
    pastas = sorted([n for n, is_d in itens if is_d], key=natural_sort_key)
    arquivos = sorted([n for n, is_d in itens if not is_d], key=natural_sort_key)

    itens_ordenados = pastas + arquivos

    total_itens = len(itens_ordenados)
    itens_visiveis = itens_ordenados[:limite]
    itens_ocultos = total_itens - len(itens_visiveis)

    if itens_ocultos > 0:
        itens_visiveis.append(f"... (mais {itens_ocultos} itens)")

    for i, nome in enumerate(itens_visiveis):
        caminho_item = os.path.join(pasta_atual, nome)
        is_pasta = os.path.isdir(caminho_item)
        ultimo = (i == len(itens_visiveis) - 1)

        conector = "└── " if ultimo else "├── "
        arquivo_saida.write(f"{prefixo}{conector}{nome}\n")

        if is_pasta:
            novo_prefixo = prefixo + ("    " if ultimo else "│   ")
            _listar(
                arquivo_saida,
                caminho_item,
                novo_prefixo,
                limite,
                ignorar
            )


# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
if __name__ == "__main__":
    LIMITE_POR_PASTA = 10
    ARQUIVO_SAIDA = "arvore_de_diretorios.txt"

    IGNORAR = [
        # Git e IDE
        ".git",
        ".github",
        ".vscode",
        ".idea",

        # Ambientes
        ".venv",
        "venv",
        ".devcontainer",
        ".certificates",

        # Cache
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",

        # Dependências
        "node_modules",

        # Arquivos de infra
        ".gitignore",
        ".gitattributes",
        ".dockerignore",
        "Dockerfile",
        "docker-compose.yaml",
        ".python-version",
        ".env",
        ".env.example",
        ".env.docker",
        "uv.lock",

        # Outros
        ".temp",
        "static",

        # Gerados pelo script
        ARQUIVO_SAIDA,
        os.path.basename(__file__),
    ]

    caminho = os.path.dirname(os.path.abspath(__file__))

    gerar_arvore_diretorios(
        path_raiz=caminho,
        nome_arquivo_saida=ARQUIVO_SAIDA,
        limite_itens=LIMITE_POR_PASTA,
        ignorar=IGNORAR
    )