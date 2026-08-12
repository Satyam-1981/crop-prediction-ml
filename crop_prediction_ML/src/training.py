import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .config import RANDOM_STATE
from .models import build_models


def build_pipelines():
    models = build_models()
    pipelines = {}

    for name, model in models.items():
        if name == "KNN":
            pipelines[name] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])
        else:
            pipelines[name] = Pipeline([
                ("model", model),
            ])

    return pipelines


def train_and_evaluate(X_train, X_test, y_train, y_test, output_dir="models"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # XGBoost requires integer class labels.
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Used by Streamlit to convert predictions back to crop names.
    joblib.dump(label_encoder, output / "label_encoder.joblib")

    pipelines = build_pipelines()

    rows = []
    reports = {}
    matrices = {}
    trained = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for name, pipeline in pipelines.items():
        start = time.perf_counter()

        pipeline.fit(X_train, y_train_encoded)

        elapsed = time.perf_counter() - start

        pred_encoded = pipeline.predict(X_test)
        pred = label_encoder.inverse_transform(pred_encoded.astype(int))

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test, pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, pred, average="weighted", zero_division=0
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train_encoded,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        rows.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "CV Accuracy Mean": cv_scores.mean(),
            "CV Accuracy Std": cv_scores.std(),
            "Training Time (sec)": elapsed,
        })

        reports[name] = classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        )

        matrices[name] = confusion_matrix(
            y_test,
            pred,
            labels=label_encoder.classes_,
        ).tolist()

        trained[name] = pipeline

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, output / filename)

    metrics_df = pd.DataFrame(rows).sort_values(
        ["F1 Score", "Accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    metrics_df.to_csv(output / "model_metrics.csv", index=False)

    with open(output / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    with open(output / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(matrices, f)

    return metrics_df, trained
