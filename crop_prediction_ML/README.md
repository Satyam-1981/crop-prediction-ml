# 🌱 Crop Cultivation Recommendation System

End-to-end ML project that recommends a crop to cultivate from soil and environmental conditions.

## Models
- Random Forest
- Decision Tree
- XGBoost
- KNN
- Gradient Boosting

## Data
The public Kaggle Crop Recommendation Dataset is fetched at runtime through **KaggleHub**. No CSV is manually bundled with the project.

Dataset:
`atharvaingle/crop-recommendation-dataset`

File:
`Crop_recommendation.csv`

## Inputs
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

## Output
The application shows:
- Prediction from all five models
- Model probabilities where supported
- Model performance metrics
- Majority-vote result
- **Recommended crop to cultivate**

## 🚀 Live Demo

https://crop-recommendation-aiml.streamlit.app/

EDA/visualization code is not included in the final runtime project because it is not needed in the UI. Crop labels are encoded with LabelEncoder for XGBoost and converted back to crop names for display.

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run streamlit_app.py
```

You can also train from the command line:

```bash
python train.py
```

## Kaggle authentication

Configure Kaggle/KaggleHub credentials using the normal Kaggle credential mechanism or your deployment platform's secrets. Never commit API tokens to GitHub.

## Project structure

```text
crop_prediction/
├── streamlit_app.py
├── train.py
├── requirements.txt
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── models.py
    ├── training.py
    └── prediction.py
```

## Evaluation

The training pipeline calculates:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- 5-fold cross-validation accuracy
- Training time
- Classification reports
- Confusion matrices

The UI focuses on prediction rather than EDA.

## Important

This is a machine-learning recommendation system, not a guarantee of yield or profit. Real cultivation decisions should also consider local soil tests, season, irrigation, weather, crop variety, pests/diseases and market conditions.
