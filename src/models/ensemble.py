import numpy as np
from typing import Tuple

from src.models.autoencoder import (
    load_autoencoder,
    predict_autoencoder,
    train_autoencoder,
)
from src.models.detection import load_model, predict, train_model


def train_all(X) -> None:
    """
    Treina os dois modelos do sistema:
        1. Isolation Forest
        2. Autoencoder

    Ambos recebem a mesma matriz de features, garantindo consistência
    entre as etapas do ensemble.

    Args:
        X: Matriz de entrada contendo as features numéricas de treinamento.
    """
    train_model(X)
    train_autoencoder(X)


def load_all_models() -> None:
    """
    Carrega em memória todos os modelos utilizados pelo ensemble.
    """
    load_model()
    load_autoencoder()


def normalize(values) -> np.ndarray:
    """
    Normaliza um vetor numérico para o intervalo [0, 1].

    Regras:
        - Se existir apenas um valor, ele é retornado sem normalização.
          Isso evita divisão por zero em inferência unitária.
        - Se todos os valores forem iguais, retorna vetor de zeros.

    Args:
        values: Vetor ou lista de valores numéricos.

    Returns:
        Vetor normalizado em formato numpy.ndarray.
    """
    values = np.array(values, dtype=float)

    # Em cenários com uma única amostra, não há base estatística
    # para normalização min-max. Nesse caso, o valor é mantido.
    if len(values) == 1:
        return values

    value_range = values.max() - values.min()

    if value_range == 0:
        return np.zeros_like(values)

    return (values - values.min()) / value_range


def predict_all(X) -> Tuple[list, list]:
    """
    Executa a predição combinada dos dois modelos do sistema.

    Fluxo:
        1. Obtém predições e scores do Isolation Forest
        2. Obtém predições e scores do Autoencoder
        3. Normaliza os scores
        4. Combina os scores com pesos definidos
        5. Gera uma decisão final do ensemble

    Estratégia atual:
        - 70% de peso para o Isolation Forest
        - 30% de peso para o Autoencoder
        - Threshold final de 0.45 para classificar ataque

    Convenção de saída:
        - 0 = normal
        - 1 = ataque/anomalia

    Args:
        X: Matriz de entrada contendo as features a serem analisadas.

    Returns:
        Uma tupla contendo:
            - lista de predições finais do ensemble
            - lista de scores finais ponderados
    """
    predictions_if, scores_if = predict(X)
    predictions_ae, scores_ae = predict_autoencoder(X)

    normalized_scores_if = normalize(scores_if)
    normalized_scores_ae = normalize(scores_ae)

    final_predictions = []
    final_scores = []

    for index in range(len(X)):
        # Combinação ponderada dos scores dos dois modelos.
        # O Isolation Forest recebe maior peso por sua sensibilidade
        # a padrões anômalos globais.
        weighted_score = (
            0.7 * normalized_scores_if[index]
            + 0.3 * normalized_scores_ae[index]
        )

        # Decisão final do ensemble.
        final_prediction = 1 if weighted_score >= 0.45 else 0

        final_predictions.append(final_prediction)
        final_scores.append(float(weighted_score))

    return final_predictions, final_scores