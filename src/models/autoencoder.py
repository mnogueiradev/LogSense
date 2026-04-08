import joblib
import numpy as np
from typing import Optional, Tuple

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model

from src.core.paths import (
    AUTOENCODER_DIR,
    AUTOENCODER_MODEL_FILE,
    AUTOENCODER_THRESHOLD_FILE,
    SCALER_FILE,
)


_scaler = None
_autoencoder: Optional[Model] = None
_threshold: Optional[float] = None


def train_autoencoder(X) -> None:
    """
    Treina o modelo Autoencoder utilizando as features já normalizadas pelo scaler.

    Etapas executadas:
        1. Carrega o scaler previamente salvo
        2. Normaliza os dados de entrada
        3. Constrói a arquitetura do autoencoder
        4. Treina a rede para reconstruir os próprios dados
        5. Calcula o erro de reconstrução (MSE)
        6. Define um threshold com base no percentil 80
        7. Salva o modelo e o threshold em disco

    Args:
        X: Matriz de entrada contendo as features numéricas usadas no treinamento.
    """
    global _autoencoder, _threshold, _scaler

    AUTOENCODER_DIR.mkdir(parents=True, exist_ok=True)

    # O autoencoder utiliza o mesmo scaler empregado no treinamento
    # do Isolation Forest, garantindo consistência entre os modelos.
    _scaler = joblib.load(SCALER_FILE)
    X_scaled = _scaler.transform(X)

    input_dimension = X_scaled.shape[1]

    # Arquitetura simétrica simples:
    # compressão -> gargalo -> reconstrução
    input_layer = Input(shape=(input_dimension,))
    x = Dense(16, activation="relu")(input_layer)
    x = Dense(8, activation="relu")(x)
    x = Dense(4, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    output_layer = Dense(input_dimension, activation="linear")(x)

    _autoencoder = Model(inputs=input_layer, outputs=output_layer)
    _autoencoder.compile(optimizer="adam", loss="mse")

    _autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=20,
        batch_size=256,
        shuffle=True,
        verbose=0,
    )

    # O erro de reconstrução é usado como indicador de anomalia.
    reconstructed_data = _autoencoder.predict(X_scaled, verbose=0)
    reconstruction_error = np.mean(np.square(X_scaled - reconstructed_data), axis=1)

    # Threshold mais agressivo para aumentar a sensibilidade na detecção de ataques.
    _threshold = float(np.percentile(reconstruction_error, 80))

    _autoencoder.save(AUTOENCODER_MODEL_FILE)
    joblib.dump(_threshold, AUTOENCODER_THRESHOLD_FILE)


def load_autoencoder() -> None:
    """
    Carrega em memória o modelo Autoencoder, o threshold de decisão
    e o scaler previamente salvos em disco.
    """
    global _autoencoder, _threshold, _scaler

    _autoencoder = load_model(AUTOENCODER_MODEL_FILE)
    _threshold = joblib.load(AUTOENCODER_THRESHOLD_FILE)
    _scaler = joblib.load(SCALER_FILE)


def predict_autoencoder(X) -> Tuple[list, np.ndarray]:
    """
    Executa a predição com o Autoencoder.

    Fluxo:
        1. Normaliza os dados de entrada
        2. Reconstrói as amostras usando a rede treinada
        3. Calcula o erro de reconstrução (MSE)
        4. Marca como ataque/anomalia toda amostra cujo erro
           ultrapasse o threshold definido no treinamento

    Convenção de saída do projeto:
        - 0 = normal
        - 1 = ataque/anomalia

    Args:
        X: Matriz de entrada contendo as features a serem analisadas.

    Returns:
        Uma tupla contendo:
            - lista de predições binárias (0 = normal, 1 = ataque)
            - vetor com os erros de reconstrução (MSE)

    Raises:
        RuntimeError: Caso o modelo, o scaler ou o threshold
        ainda não tenham sido carregados.
    """
    global _autoencoder, _threshold, _scaler

    if _autoencoder is None or _threshold is None or _scaler is None:
        raise RuntimeError(
            "O Autoencoder não foi carregado corretamente. "
            "Execute load_autoencoder() antes de chamar predict_autoencoder()."
        )

    X_scaled = _scaler.transform(X)
    reconstructed_data = _autoencoder.predict(X_scaled, verbose=0)
    reconstruction_error = np.mean(np.square(X_scaled - reconstructed_data), axis=1)

    predictions = [1 if error > _threshold else 0 for error in reconstruction_error]

    return predictions, reconstruction_error