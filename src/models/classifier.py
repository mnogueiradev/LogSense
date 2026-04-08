SERVICE_PORTS_BRUTE_FORCE = {21, 22, 23, 25, 110, 143, 445, 3389}


def classify_attack(features: dict) -> str:
    """
    Classifica heurísticamente o tipo provável de ataque com base
    em regras simples derivadas de comportamento de tráfego.

    Tipos atualmente suportados:
        - ddos
        - port_scan
        - brute_force
        - normal

    A função não substitui os modelos de IA. Seu papel é complementar
    a interpretação do resultado final, tornando a saída da API mais
    explicável para o usuário.

    Args:
        features: Dicionário contendo as métricas de tráfego analisadas.

    Returns:
        Uma string representando o tipo provável de tráfego:
            - "ddos"
            - "port_scan"
            - "brute_force"
            - "normal"
    """
    connections = float(features.get("connections", 0) or 0)
    bytes_ = float(features.get("bytes", 0) or 0)
    packets = float(features.get("packets", 0) or 0)
    packet_size = float(features.get("packet_size", 0) or 0)
    ports = float(features.get("ports", 0) or 0)

    # DDoS:
    # caracteriza-se por volume muito alto de pacotes, bytes ou conexões.
    if (
        packets >= 5000
        or bytes_ >= 500000
        or (connections >= 400 and bytes_ >= 200000)
    ):
        return "ddos"

    # Port scan:
    # tende a apresentar muitas conexões, uso de portas altas
    # e tráfego relativamente baixo por tentativa.
    if connections >= 5000 and ports >= 10000 and packets <= 1500:
        return "port_scan"

    # Brute force:
    # tende a apresentar muitas tentativas, baixo volume de bytes
    # e foco em portas de serviço conhecidas.
    if (
        connections >= 3000
        and packets >= 200
        and bytes_ <= 15000
        and int(ports) in SERVICE_PORTS_BRUTE_FORCE
    ):
        return "brute_force"

    return "normal"