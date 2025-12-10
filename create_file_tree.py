# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import re

def natural_sort_key(s):
    """Ordenação natural: A1 < A2 < A10 < A11"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def gerar_arvore_diretorios(path_raiz, nome_arquivo_saida, limite_itens, ignorar=None):
    if ignorar is None:
        ignorar = []

    caminho_completo_saida = os.path.join(path_raiz, nome_arquivo_saida)
    
    # Conta quantas pastas existem (para a barra de progresso)
    total = sum(1 for _ in os.walk(path_raiz))

    with open(caminho_completo_saida, "w", encoding="utf-8") as f, tqdm(
        total=total, desc="Gerando árvore", unit="pasta", ncols=100
    ) as pbar:

        nome_raiz = os.path.basename(path_raiz.rstrip("/"))
        f.write(f"{nome_raiz}/\n")

        _listar(f, path_raiz, "", limite_itens, ignorar, pbar)

    print(f"\nArquivo gerado: {caminho_completo_saida}")


def _listar(arquivo_saida, pasta_atual, prefixo, limite, ignorar, pbar):
    pbar.update(1)

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

    # Adicionar “... (mais X itens)” se houver ocultos
    if itens_ocultos > 0:
        itens_visiveis.append(f"... (mais {itens_ocultos} itens)")

    # Processar cada item visual
    for i, nome in enumerate(itens_visiveis):
        is_pasta = os.path.isdir(os.path.join(pasta_atual, nome))
        ultimo = (i == len(itens_visiveis) - 1)

        conector = "└── " if ultimo else "├── "
        arquivo_saida.write(f"{prefixo}{conector}{nome}\n")

        # Só recursão se for pasta real (ignorar o item "...")
        if is_pasta:
            novo_prefixo = prefixo + ("    " if ultimo else "│   ")
            _listar(
                arquivo_saida,
                os.path.join(pasta_atual, nome),
                novo_prefixo,
                limite,
                ignorar,
                pbar
            )


# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
if __name__ == "__main__":
    LIMITE_POR_PASTA = 7
    ARQUIVO_SAIDA = "arvore_de_diretorios.txt"

    IGNORAR = [
        ".git",
        ".vscode",
        "__pycache__",
        "node_modules",
        ARQUIVO_SAIDA,
        os.path.basename(__file__)
    ]

    caminho = os.path.dirname(os.path.abspath(__file__))

    gerar_arvore_diretorios(
        path_raiz=caminho,
        nome_arquivo_saida=ARQUIVO_SAIDA,
        limite_itens=LIMITE_POR_PASTA,
        ignorar=IGNORAR
    )
