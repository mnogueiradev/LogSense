import argparse
import json
import pandas as pd
from sklearn.metrics import classification_report

from src.core.paths import VALIDATION_PREDICTIONS_FILE, VALIDATION_REPORT_FILE
from src.features.constants import FEATURE_COLUMNS
from src.models.autoencoder import predict_autoencoder
from src.models.detection import predict
from src.models.ensemble import load_all_models, predict_all


def report_to_dict(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def print_report(title, y_true, y_pred):
    print(f"\n[{title}]")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()

    X = df[FEATURE_COLUMNS].fillna(0).values
    y_true = df["target"].astype(int).tolist()

    load_all_models()

    preds_if, scores_if = predict(X)
    preds_ae, scores_ae = predict_autoencoder(X)
    preds_en, scores_en = predict_all(X)

    VALIDATION_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["pred_if"] = preds_if
    df_out["pred_ae"] = preds_ae
    df_out["pred_ensemble"] = preds_en
    df_out.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)

    report = {
        "isolation_forest": report_to_dict(y_true, preds_if),
        "autoencoder": report_to_dict(y_true, preds_ae),
        "ensemble": report_to_dict(y_true, preds_en),
    }

    VALIDATION_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[ok] Previsões salvas em: {VALIDATION_PREDICTIONS_FILE}")
    print(f"[ok] Relatório salvo em: {VALIDATION_REPORT_FILE}")

    print_report("Isolation Forest", y_true, preds_if)
    print_report("Autoencoder", y_true, preds_ae)
    print_report("Ensemble", y_true, preds_en)


if __name__ == "__main__":
    main()