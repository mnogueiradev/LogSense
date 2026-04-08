import argparse
import pandas as pd

from src.features.constants import FEATURE_COLUMNS
from src.models.ensemble import train_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    print("🔹 Carregando dataset...")
    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()

    X = df[FEATURE_COLUMNS].fillna(0).values

    print("🔹 Treinando modelos...")
    train_all(X)

    print("✅ Treino finalizado!")


if __name__ == "__main__":
    main()