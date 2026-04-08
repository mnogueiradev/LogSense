from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.models.ensemble import load_all_models, predict_all
from src.models.detection import predict as predict_if
from src.models.autoencoder import predict_autoencoder
from src.models.classifier import classify_attack

app = FastAPI(title="CyberIA - Detecção Inteligente de Ataques")


@app.on_event("startup")
def startup() -> None:
    """
    Carrega os modelos de IA quando a API é iniciada.

    Isso evita recarregamento a cada requisição e melhora o tempo de resposta.
    """
    print("Carregando modelos...")
    load_all_models()
    print("Modelos carregados com sucesso.")


class NetworkInput(BaseModel):
    """
    Estrutura esperada para entrada da API.

    Cada campo representa uma feature já padronizada do tráfego de rede.
    """
    connections: float
    bytes: float
    packets: float
    packet_size: float
    ports: float


@app.get("/")
def home() -> dict:
    """
    Endpoint simples de verificação de funcionamento da API.

    Returns:
        Dicionário com status e mensagem de disponibilidade.
    """
    return {
        "status": "online",
        "message": "CyberIA API rodando"
    }


@app.post("/predict")
def predict_endpoint(data: NetworkInput) -> dict:
    """
    Executa a análise completa de uma amostra de tráfego de rede.

    Fluxo:
        1. Converte os dados recebidos em vetor numérico
        2. Executa os modelos individuais:
           - Isolation Forest
           - Autoencoder
        3. Executa o ensemble
        4. Aplica classificação heurística complementar
        5. Combina os resultados em uma decisão final híbrida
        6. Calcula nível de risco e confiança

    Args:
        data: Objeto validado pelo Pydantic contendo as features de entrada.

    Returns:
        Um dicionário contendo:
            - resultado final da análise
            - saída dos modelos individuais
            - saída do ensemble
            - classificação heurística complementar
    """
    payload = data.model_dump()

    # Montagem do vetor numérico de entrada no formato esperado pelos modelos.
    X = np.array(
        [[
            payload["connections"],
            payload["bytes"],
            payload["packets"],
            payload["packet_size"],
            payload["ports"]
        ]],
        dtype=float
    )

    # Predição dos modelos individuais.
    predictions_if, scores_if = predict_if(X)
    predictions_ae, scores_ae = predict_autoencoder(X)

    # Predição do ensemble.
    predictions_ensemble, scores_ensemble = predict_all(X)

    ml_detected_attack = bool(predictions_ensemble[0])

    # Classificação heurística complementar.
    # Essa etapa torna a saída mais interpretável, associando padrões
    # conhecidos a tipos específicos de ataque.
    heuristic_attack_type = classify_attack(payload)
    heuristic_detected_attack = heuristic_attack_type != "normal"

    # Decisão final híbrida:
    # o sistema considera tanto a decisão do ensemble quanto a heurística.
    is_attack = ml_detected_attack or heuristic_detected_attack

    if heuristic_detected_attack:
        attack_type = heuristic_attack_type
        decision_source = "ml+heuristica"
    elif ml_detected_attack:
        attack_type = "anomaly"
        decision_source = "ml"
    else:
        attack_type = "normal"
        decision_source = "ml"

    # Score bruto do ensemble.
    raw_score = float(scores_ensemble[0])

    # Normalização do score para intervalo aproximado [0, 1].
    # Isso melhora a apresentação da confiança no resultado final.
    normalized_score = 1 - np.exp(-max(raw_score, 0) / 10)
    normalized_score = min(max(normalized_score, 0), 1)

    # Caso a heurística reconheça um ataque específico, é definida uma
    # confiança mínima para manter coerência com a classificação exibida.
    heuristic_minimum_confidence = {
        "ddos": 0.90,
        "port_scan": 0.82,
        "brute_force": 0.78,
    }

    if heuristic_detected_attack:
        normalized_score = max(
            normalized_score,
            heuristic_minimum_confidence.get(heuristic_attack_type, 0.75)
        )
    elif ml_detected_attack:
        normalized_score = max(normalized_score, 0.60)

    # Classificação textual do nível de risco.
    if normalized_score >= 0.7:
        risk_level = "alto 🔴"
    elif normalized_score >= 0.4:
        risk_level = "medio 🟡"
    else:
        risk_level = "baixo 🟢"

    return {
        "resultado_final": {
            "is_attack": is_attack,
            "attack_type": attack_type,
            "risk_level": risk_level,
            "confidence": round(normalized_score, 4),
            "decision_source": decision_source
        },
        "modelos": {
            "isolation_forest": {
                "pred": int(predictions_if[0]),
                "score": float(scores_if[0])
            },
            "autoencoder": {
                "pred": int(predictions_ae[0]),
                "score": float(scores_ae[0])
            },
            "ensemble": {
                "pred": int(predictions_ensemble[0]),
                "score": raw_score
            }
        },
        "classificacao_heuristica": {
            "attack_type": heuristic_attack_type,
            "matched": heuristic_detected_attack
        }
    }