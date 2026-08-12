from collections import Counter

import numpy as np
import pandas as pd


def predict_all(models, values, feature_names, label_encoder):
    X = pd.DataFrame([values], columns=feature_names)

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        pred_encoded = model.predict(X)[0]

        crop = label_encoder.inverse_transform([int(pred_encoded)])[0]
        predictions[name] = str(crop)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = model.classes_
            order = np.argsort(probs)[::-1]

            probabilities[name] = [
                (
                    str(label_encoder.inverse_transform([int(classes[i])])[0]),
                    float(probs[i]),
                )
                for i in order[:5]
            ]
        else:
            probabilities[name] = []

    votes = Counter(predictions.values())
    final_crop, vote_count = votes.most_common(1)[0]
    agreement = vote_count / len(predictions)

    return predictions, probabilities, final_crop, vote_count, agreement
