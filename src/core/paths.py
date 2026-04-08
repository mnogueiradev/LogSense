from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

AUTOENCODER_DIR = MODELS_DIR / "autoencoder"

FEATURES_FILE = PROCESSED_DIR / "features.csv"

ISOLATION_MODEL_FILE = MODELS_DIR / "isolation.pkl"
SCALER_FILE = MODELS_DIR / "scaler.joblib"

AUTOENCODER_MODEL_FILE = AUTOENCODER_DIR / "model.keras"
AUTOENCODER_THRESHOLD_FILE = AUTOENCODER_DIR / "threshold.pkl"

VALIDATION_PREDICTIONS_FILE = PROCESSED_DIR / "validation_predictions.csv"
VALIDATION_REPORT_FILE = MODELS_DIR / "validation_report.json"