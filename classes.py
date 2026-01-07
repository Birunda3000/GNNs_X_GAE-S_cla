import torch
import src.data_loaders as data_loaders
from collections import Counter

def check_balance():
    print("=== ANÁLISE DE CLASSES ===")
    
    # 1. GitHub
    try:
        loader_gh = data_loaders.MusaeGithubLoader()
        wsg_gh = loader_gh.load()
        # O y pode ser lista ou tensor dependendo de como carregou
        y_gh = wsg_gh.graph_structure.y
        if isinstance(y_gh, torch.Tensor): y_gh = y_gh.tolist()
        
        counts = Counter(y_gh)
        total = sum(counts.values())
        print(f"\nGITHUB (Total: {total}):")
        for cls, count in sorted(counts.items()):
            print(f"  Classe {cls}: {count} ({count/total:.1%})")
        
        major = max(counts.values())
        minor = min(counts.values())
        print(f"  Ratio: {major/minor:.2f}:1")
    except Exception as e:
        print(f"Erro ao ler Github: {e}")

    # 2. Facebook
    try:
        loader_fb = data_loaders.MusaeFacebookLoader()
        wsg_fb = loader_fb.load()
        y_fb = wsg_fb.graph_structure.y
        if isinstance(y_fb, torch.Tensor): y_fb = y_fb.tolist()
        
        counts = Counter(y_fb)
        total = sum(counts.values())
        print(f"\nFACEBOOK (Total: {total}):")
        for cls, count in sorted(counts.items()):
            print(f"  Classe {cls}: {count} ({count/total:.1%})")
    except Exception as e:
        print(f"Erro ao ler Facebook: {e}")

if __name__ == "__main__":
    check_balance()