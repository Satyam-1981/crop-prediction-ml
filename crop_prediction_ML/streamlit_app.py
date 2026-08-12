import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import FEATURES, MODEL_NAMES
from src.data_loader import load_dataset
from src.preprocessing import split_data
from src.training import train_and_evaluate
from src.prediction import predict_all


st.set_page_config(
    page_title="Crop Cultivation Recommendation",
    page_icon="🌱",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


@st.cache_resource(show_spinner=True)
def train_models():
    df = get_data()
    X_train, X_test, y_train, y_test = split_data(df)
    metrics, trained = train_and_evaluate(
        X_train, X_test, y_train, y_test, output_dir="models"
    )
    return metrics, trained, len(df)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 18px;
        opacity: .75;
        margin-top: 4px;
        margin-bottom: 28px;
    }
    .recommendation {
    padding: 24px;
    border-radius: 18px;
    background: #1f2937;
    border: 1px solid #374151;
    margin-top: 18px;
    color: #f9fafb;
}

.recommendation .small-note {
    color: #9ca3af !important;
    opacity: 1;
    font-size: 13px;
    font-weight: 600;
}

.recommendation .crop-name {
    color: #ffffff !important;
    font-size: 36px;
    font-weight: 850;
    margin: 0;
}

.recommendation p {
    color: #e5e7eb !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🌱 Crop Cultivation Recommendation</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Machine-learning recommendation from soil and environmental conditions</p>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading the Kaggle dataset and preparing the models..."):
        df = get_data()
        metrics, models, sample_count = train_models()
except Exception as exc:
    st.error("The application could not load/train the project.")
    st.exception(exc)
    st.info(
        "If Kaggle authentication is required in your environment, configure "
        "Kaggle credentials before running Streamlit."
    )
    st.stop()

best_model = metrics.iloc[0]["Model"]

with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Prediction model",
        ["Ensemble (majority vote)"] + MODEL_NAMES,
        index=0,
    )
    st.caption(f"Best evaluated model: **{best_model}**")
    st.caption(f"Dataset records used: **{sample_count:,}**")

st.subheader("Enter field conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)

values = [N, P, K, temperature, humidity, ph, rainfall]

if st.button("🌾 Recommend Crop", type="primary", use_container_width=True):
    label_encoder = joblib.load("models/label_encoder.joblib")

    predictions, probabilities, final_crop, vote_count, agreement = predict_all(
        models, values, FEATURES, label_encoder
    )

    if selected_model == "Ensemble (majority vote)":
        recommended = final_crop
        recommendation_reason = (
            f"{vote_count} out of {len(models)} models recommend this crop."
        )
    else:
        recommended = predictions[selected_model]
        recommendation_reason = f"Prediction from the {selected_model} model."

    st.markdown(
        f"""
        <div class="recommendation">
            <p class="small-note">RECOMMENDED CROP TO CULTIVATE</p>
            <p class="crop-name">🌾 {recommended.title()}</p>
            <p>{recommendation_reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model predictions")

    result_df = pd.DataFrame(
        [
            {"Model": model_name, "Predicted Crop": crop.title()}
            for model_name, crop in predictions.items()
        ]
    )
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    if selected_model == "Ensemble (majority vote)":
        st.success(
            f"Final recommendation: **{final_crop.title()}** "
            f"with {agreement:.0%} model agreement."
        )

    with st.expander("Model confidence details"):
        for model_name, items in probabilities.items():
            if items:
                st.write(f"**{model_name}**")
                prob_df = pd.DataFrame(items, columns=["Crop", "Probability"])
                prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)
                st.dataframe(
                    prob_df,
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()
st.subheader("Model performance")

display_metrics = metrics.copy()
for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "CV Accuracy Mean",
]:
    display_metrics[col] = (display_metrics[col] * 100).round(2).astype(str) + "%"

display_metrics["CV Accuracy Std"] = display_metrics["CV Accuracy Std"].round(4)
display_metrics["Training Time (sec)"] = display_metrics["Training Time (sec)"].round(2)

st.dataframe(
    display_metrics,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "EDA and exploratory visualizations are intentionally kept out of the UI."
)
st.caption(
    "This is a machine-learning recommendation system, not a guarantee of yield, "
    "profitability, or agricultural success. Validate recommendations with local "
    "agronomic knowledge before cultivation."
)
