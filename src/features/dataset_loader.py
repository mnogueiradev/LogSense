from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.constants import (
    BENIGN_TEXT_LABELS,
    FEATURE_COLUMNS,
    KNOWN_LABEL_COLUMNS,
    RAW_TO_FEATURE_MAPPING,
)

def load_features_csv(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def _normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for col in output.columns:
        output[col] = pd.to_numeric(output[col], errors="coerce")
    return output.fillna(0.0)

def _extract_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    working = df.copy()

    for label_col in KNOWN_LABEL_COLUMNS:
        if label_col not in working.columns:
            continue

        series = working[label_col]
        working = working.drop(columns=[label_col])

        if pd.api.types.is_numeric_dtype(series):
            y = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        else:
            y = (~series.astype(str).str.upper().str.strip().isin(BENIGN_TEXT_LABELS)).astype(int)

        return working, y

    return working, None

def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if set(FEATURE_COLUMNS).issubset(df.columns):
        features_df = df[FEATURE_COLUMNS].copy()
        return _normalize_numeric(features_df)

    raw_columns = list(RAW_TO_FEATURE_MAPPING.keys())
    if set(raw_columns).issubset(df.columns):
        renamed = df[raw_columns].rename(columns=RAW_TO_FEATURE_MAPPING)
        return _normalize_numeric(renamed[FEATURE_COLUMNS])

    raise ValueError(
        "Não foi possível montar as features. "
        f"O arquivo precisa conter {FEATURE_COLUMNS} "
        f"ou as colunas brutas {raw_columns}."
    )

def load_training_dataframe(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(csv_path)
    df, y = _extract_label(df)
    X = _build_feature_frame(df)
    return X, y

def prepare_inference_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise ValueError("A lista de registros está vazia.")
    df = pd.DataFrame(records)
    return _build_feature_frame(df)

def load_inference_csv_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(content))
    df, _ = _extract_label(df)
    return _build_feature_frame(df)

def validate_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes: {missing}")
    return df[FEATURE_COLUMNS].copy()

