import json
import os
import shutil
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Union
import src.paths as paths

class DirectoryManager:
    """
    Gerencia a criação e o nomeação de diretórios de saída.
    Versão BLINDADA (220 chars): Protege contra erro de nome longo no Docker.
    """

    def __init__(
        self, timestamp: str, run_folder_name: str, base_path: Optional[str] = None
    ):
        if base_path is None:
            base_path = paths.OUTPUT_PATH

        self.base_path = os.path.join(base_path, run_folder_name)
        self.timestamp = timestamp
        
        # Nome temporário curto
        self.temp_dir_name = f"_tmp__{self.timestamp}"
        self.run_dir_path = os.path.join(self.base_path, self.temp_dir_name)
        self.final_dir_path: Optional[str] = None

        os.makedirs(self.run_dir_path, exist_ok=True)

    def get_run_path(self) -> str:
        return self.final_dir_path if self.final_dir_path else self.run_dir_path

    def finalize_run_directory(
        self,
        dataset_name: str,
        metrics: Dict[str, Union[float, int, str]],
    ) -> str:
        """
        Renomeia o diretório. Proteção ativa se > 220 caracteres.
        """
        if not os.path.exists(self.run_dir_path):
            print(f"Aviso: Diretório '{self.run_dir_path}' não encontrado.")
            return ""

        # 1. Compactação de Nomes
        metrics_parts = []
        for key, value in metrics.items():
            # Abrevia chaves para economizar espaço
            k = (key.replace("val_", "").replace("test_", "")
                    .replace("score", "sc").replace("weighted", "")
                    .replace("accuracy", "acc").replace("cv_f1", "cv")
                    .replace("best_params", "p")
                    .strip("_"))
            
            if not k: k = "m"

            if isinstance(value, float):
                v = f"{value:.4f}".replace(".", "")
            else:
                # Corta valores de string muito longos (ex: params do MLP)
                v = str(value)
                if len(v) > 20: v = v[:20] + ".." 
            
            metrics_parts.append(f"{k}_{v}")

        metrics_str = "_".join(metrics_parts)
        final_name = f"{dataset_name}__{metrics_str}"

        # 2. TRAVA DE SEGURANÇA (Limite 220)
        MAX_LEN = 220
        
        if len(final_name) > MAX_LEN:
            print(f"⚠️ Nome longo ({len(final_name)} chars). Aplicando corte seguro.")
            # Gera hash do nome completo para garantir identidade única
            hash_suffix = hashlib.md5(final_name.encode()).hexdigest()[:8]
            # Corta deixando espaço para o hash e timestamp
            final_name = f"{final_name[:MAX_LEN-15]}..._{hash_suffix}"

        # Adiciona timestamp curto
        time_suffix = self.timestamp.split("_")[-1] if "_" in self.timestamp else "000"
        final_name = f"{final_name}__{time_suffix}"

        final_path = os.path.join(self.base_path, final_name)

        try:
            shutil.move(self.run_dir_path, final_path)
            self.final_dir_path = final_path
            self.run_dir_path = final_path
            print(f"✅ Run salva: .../{final_name}")
            return final_path
        
        except OSError as e:
            # 3. FALLBACK (Se falhar mesmo assim)
            print(f"⚠️ Erro crítico de nome ({e}). Salvando com UUID.")
            safe_name = f"RUN_{uuid.uuid4().hex[:8]}_{time_suffix}"
            safe_path = os.path.join(self.base_path, safe_name)
            try:
                shutil.move(self.run_dir_path, safe_path)
                return safe_path
            except Exception:
                return self.run_dir_path