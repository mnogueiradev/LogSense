from __future__ import annotations

import json
import shutil
from typing import Any

import joblib
import tensorflow as tf

from src.core.paths import (
    AUTOENCODER_DIR,
    AUTOENCODER_MODEL_PATH,
    AUTOENCODER_THRESHOLD_PATH,
    ISOLATION_PATH,
    MANIFEST_PATH,
    SCALER_PATH,
    ensure_project_dirs,
)

def clear_model_artifacts() -> None:
    ensure_project_dirs()

    for path in [SCALER_PATH, ISOLATION_PATH, MANIFEST_PATH]:
        if path.exists():
            path.unlink()

    if AUTOENCODER_DIR.exists():
        shutil.rmtree(AUTOENCODER_DIR)
    AUTOENCODER_DIR.mkdir(parents=True, exist_ok=True)

def save_manifest(payload: dict[str, Any]) -> None:
    ensure_project_dirs()
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado em {MANIFEST_PATH}. Rode o pipeline de treino antes."
        )
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def models_available() -> bool:
    return (
        SCALER_PATH.exists()
        and ISOLATION_PATH.exists()
        and AUTOENCODER_MODEL_PATH.exists()
        and AUTOENCODER_THRESHOLD_PATH.exists()
        and MANIFEST_PATH.exists()
    )

def load_artifact_bundle() -> dict[str, Any]:
    if not models_available():
        raise FileNotFoundError(
            "Artefatos do modelo não encontrados. Rode `python -m src.pipelines.train --input ...`."
        )

    manifest = load_manifest()
    scaler = joblib.load(SCALER_PATH)
    isolation_payload = joblib.load(ISOLATION_PATH)
    autoencoder_model = tf.keras.models.load_model(AUTOENCODER_MODEL_PATH, compile=False)
    autoencoder_threshold = float(joblib.load(AUTOENCODER_THRESHOLD_PATH))

    return {
        "manifest": manifest,
        "scaler": scaler,
        "isolation_model": isolation_payload["model"],
        "isolation_threshold": float(isolation_payload["threshold"]),
        "autoencoder_model": autoencoder_model,
        "autoencoder_threshold": autoencoder_threshold,
    }
