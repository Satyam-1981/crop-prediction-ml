FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

TARGET = "label"

# Public Kaggle dataset used by the project.
KAGGLE_DATASET = "atharvaingle/crop-recommendation-dataset"
KAGGLE_FILE = "Crop_recommendation.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_NAMES = [
    "Random Forest",
    "Decision Tree",
    "XGBoost",
    "KNN",
    "Gradient Boosting",
]
