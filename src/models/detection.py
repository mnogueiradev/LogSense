import joblib
from typing import Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.core.paths import ISOLATION_MODEL_FILE, SCALER_FILE


_model: Optional[IsolationForest] = None
_scaler: Optional[StandardScaler] = None


def train_model(X) -> None:
    """
    Treina o modelo Isolation Forest a partir da matriz de features informada.

    Etapas executadas:
        1. Ajuste do StandardScaler
        2. Normalização dos dados
        3. Treinamento do Isolation Forest
        4. Persistência do modelo e do scaler em disco

    Args:
        X: Matriz de entrada contendo as features numéricas usadas no treinamento.
    """
    global _model, _scaler

    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _model = IsolationForest(
        n_estimators=300,
        contamination=0.25,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    _model.fit(X_scaled)

    joblib.dump(_model, ISOLATION_MODEL_FILE)
    joblib.dump(_scaler, SCALER_FILE)


def load_model() -> None:
    """
    Carrega em memória o modelo Isolation Forest e o scaler previamente salvos em disco.
    """
    global _model, _scaler

    _model = joblib.load(ISOLATION_MODEL_FILE)
    _scaler = joblib.load(SCALER_FILE)


def predict(X) -> Tuple[list, object]:
    """
    Executa a predição com o Isolation Forest.

    O modelo retorna:
        - 1  para amostra normal
        - -1 para amostra anômala

    Neste projeto, a saída é convertida para:
        - 0 = normal
        - 1 = ataque/anomalia

    Args:
        X: Matriz de entrada contendo as features a serem analisadas.

    Returns:
        Uma tupla contendo:
            - lista de predições binárias (0 = normal, 1 = ataque)
            - scores gerados pela função decision_function do modelo

    Raises:
        RuntimeError: Caso o modelo ou o scaler ainda não tenham sido carregados.
    """
    global _model, _scaler

    if _model is None or _scaler is None:
        raise RuntimeError(
            "O modelo Isolation Forest não foi carregado. "
            "Execute load_model() antes de chamar predict()."
        )

    X_scaled = _scaler.transform(X)

    raw_predictions = _model.predict(X_scaled)
    anomaly_scores = _model.decision_function(X_scaled)

    # Conversão do padrão do Isolation Forest:
    #   1  -> normal
    #  -1  -> anomalia
    # Para o padrão do projeto:
    #   0  -> normal
    #   1  -> ataque/anomalia
    predictions = [1 if prediction == -1 else 0 for prediction in raw_predictions]

    return predictions, anomaly_scores