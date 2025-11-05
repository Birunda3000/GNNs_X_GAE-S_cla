import os

def read_first_existing(paths):
    """Tenta ler o primeiro arquivo existente de uma lista."""
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    return None


def get_memory_info():
    info = {}

    # --- Detecta cgroup v1 ou v2 ---
    cgroup_v2 = os.path.exists("/sys/fs/cgroup/memory.max")

    # --- Total de memória física do host ---
    with open("/proc/meminfo") as f:
        mem_total_kb = next(int(line.split()[1]) for line in f if line.startswith("MemTotal:"))
    info["host_total_mb"] = mem_total_kb / 1024

    # --- Limite de memória do container ---
    if cgroup_v2:
        limit_str = read_first_existing(["/sys/fs/cgroup/memory.max"])
        if limit_str == "max":
            info["limit_mb"] = None
        else:
            info["limit_mb"] = int(limit_str) / 1024 / 1024

        usage_bytes = read_first_existing(["/sys/fs/cgroup/memory.current"])
        info["usage_mb"] = int(usage_bytes) / 1024 / 1024 if usage_bytes else None

    else:  # cgroup v1
        limit_bytes = read_first_existing(["/sys/fs/cgroup/memory/memory.limit_in_bytes"])
        if limit_bytes:
            limit_int = int(limit_bytes)
            # valor "infinito" típico: ~9.22e18
            info["limit_mb"] = None if limit_int > 1e15 else limit_int / 1024 / 1024

        usage_bytes = read_first_existing(["/sys/fs/cgroup/memory/memory.usage_in_bytes"])
        info["usage_mb"] = int(usage_bytes) / 1024 / 1024 if usage_bytes else None

    return info


if __name__ == "__main__":
    mem_info = get_memory_info()
    print("\n=== Informação de Memória (Container Docker) ===")
    print(f"Memória total do host: {mem_info['host_total_mb']:.0f} MB")
    if mem_info['limit_mb']:
        print(f"Limite imposto ao container: {mem_info['limit_mb']:.0f} MB")
    else:
        print("Limite imposto ao container: sem limite definido (usa toda a RAM do host)")
    if mem_info['usage_mb']:
        print(f"Uso atual de memória: {mem_info['usage_mb']:.1f} MB")
    print("=================================================\n")
