from pathlib import Path

from src.data_loader import load_dataset
from src.preprocessing import split_data
from src.training import train_and_evaluate

if __name__ == "__main__":
    print("Loading dataset from Kaggle...")
    df = load_dataset()

    print(f"Dataset shape after cleaning: {df.shape}")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Training five models...")
    metrics, _ = train_and_evaluate(
        X_train, X_test, y_train, y_test, output_dir="models"
    )

    print("\nModel comparison:")
    print(metrics.to_string(index=False))

    best = metrics.iloc[0]["Model"]
    print(f"\nBest model by weighted F1: {best}")
