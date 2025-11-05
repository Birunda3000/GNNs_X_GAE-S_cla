# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

def gerar_arvore_diretorios(path_raiz, nome_arquivo_saida, limite_arquivos, ignorar=None):
    """
    Gera uma representação em árvore da estrutura de diretórios, com barra de progresso e otimização.

    Args:
        path_raiz (str): O caminho do diretório raiz a ser mapeado.
        nome_arquivo_saida (str): O nome do arquivo .txt que será gerado.
        limite_arquivos (int): O número máximo de arquivos a serem listados por pasta.
        ignorar (list): Lista de nomes de pastas ou arquivos a serem ignorados.
    """
    if ignorar is None:
        ignorar = []

    caminho_completo_saida = os.path.join(path_raiz, nome_arquivo_saida)
    
    try:
        # Primeira passagem: contar o número total de diretórios para a barra de progresso
        total_diretorios = 0
        for root, dirs, _ in os.walk(path_raiz):
            # Exclui as pastas da lista 'ignorar' da contagem
            dirs[:] = [d for d in dirs if d not in ignorar]
            total_diretorios += 1

        with open(caminho_completo_saida, "w", encoding="utf-8") as f, \
             tqdm(total=total_diretorios, desc="Mapeando pastas", unit="pasta", ncols=100) as pbar:
            
            nome_pasta_pai = os.path.basename(os.path.abspath(path_raiz))
            f.write(f"{nome_pasta_pai}/\n")
            
            _percorrer_e_escrever_otimizado(f, path_raiz, "", limite_arquivos, pbar, ignorar)
            
        print(f"\nSucesso! A árvore de diretórios foi salva em: {caminho_completo_saida}")

    except IOError as e:
        print(f"Erro ao escrever o arquivo: {e}")

def _percorrer_e_escrever_otimizado(arquivo_saida, diretorio_atual, prefixo, limite_arquivos, pbar, ignorar):
    """
    Função auxiliar recursiva otimizada com os.scandir() e atualização da barra de progresso.
    """
    pbar.update(1)
    
    try:
        pastas = []
        arquivos = []
        
        # Usa os.scandir() para performance
        for item in os.scandir(diretorio_atual):
            if item.name in ignorar:
                continue
            
            if item.is_dir():
                pastas.append(item.name)
            elif item.is_file():
                # Otimização: para de listar arquivos após atingir o limite + 1
                if len(arquivos) <= limite_arquivos:
                    arquivos.append(item.name)
        
        pastas.sort()
        arquivos.sort()

        excedeu_limite = len(arquivos) > limite_arquivos
        arquivos_para_mostrar = arquivos[:limite_arquivos]
            
        elementos = pastas + arquivos_para_mostrar
        if excedeu_limite:
            elementos.append("...")

        for i, nome_elemento in enumerate(elementos):
            conector = "├── " if i < len(elementos) - 1 else "└── "
            arquivo_saida.write(f"{prefixo}{conector}{nome_elemento}\n")
            
            caminho_completo = os.path.join(diretorio_atual, nome_elemento)
            if os.path.isdir(caminho_completo):
                extensao_prefixo = "│   " if i < len(elementos) - 1 else "    "
                _percorrer_e_escrever_otimizado(arquivo_saida, caminho_completo, prefixo + extensao_prefixo, limite_arquivos, pbar, ignorar)

    except (PermissionError, FileNotFoundError):
        # Em caso de erro, apenas ignora a pasta e continua
        pass

if __name__ == "__main__":
    # --- CONFIGURAÇÕES ---
    LIMITE_DE_ARQUIVOS_POR_PASTA = 20
    NOME_DO_ARQUIVO_DE_SAIDA = "arvore_de_diretorios.txt"
    
    # Lista de pastas e arquivos a ignorar no mapeamento
    LISTA_IGNORAR = [
        '.git', 
        '.vscode', 
        '__pycache__', 
        '.ipynb_checkpoints',
        'env', 
        'venv',
        'node_modules',
        # Adicione o nome do próprio script para não aparecer na árvore
        os.path.basename(__file__),
        NOME_DO_ARQUIVO_DE_SAIDA
    ]
    # --- FIM DAS CONFIGURAÇÕES ---

    caminho_do_script = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Mapeando a estrutura de diretórios a partir de: {caminho_do_script}")
    print(f"Limite de arquivos por pasta: {LIMITE_DE_ARQUIVOS_POR_PASTA}")
    print(f"Ignorando: {LISTA_IGNORAR}")
    
    gerar_arvore_diretorios(
        path_raiz=caminho_do_script,
        nome_arquivo_saida=NOME_DO_ARQUIVO_DE_SAIDA,
        limite_arquivos=LIMITE_DE_ARQUIVOS_POR_PASTA,
        ignorar=LISTA_IGNORAR
    )