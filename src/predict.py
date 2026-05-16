import joblib
import pandas as pd

from src.feature_engineering import feature_engineering
from src.preprocess import preprocess_data


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "models/student_model.pkl"
)


# =====================================================
# LOAD TRAINING FEATURES
# =====================================================

X_train, X_test, y_train, y_test = preprocess_data()

model_features = X_train.columns.tolist()


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_student(input_df):

    df = input_df.copy()

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df = feature_engineering(df)

    # =====================================================
    # REMOVE TARGET IF EXISTS
    # =====================================================

    if "performance_level" in df.columns:

        df.drop(
            "performance_level",
            axis=1,
            inplace=True
        )

    # =====================================================
    # ENCODE CATEGORICAL COLUMNS
    # =====================================================

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        df[col] = pd.factorize(df[col])[0]

    # =====================================================
    # KEEP ONLY MODEL FEATURES
    # =====================================================

    df = df[model_features]

    # =====================================================
    # PREDICTIONS
    # =====================================================

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)

    confidence = probabilities.max(axis=1)

    # =====================================================
    # OUTPUT
    # =====================================================

    results = pd.DataFrame({

        "Predicted Performance": predictions,

        "AI Confidence": confidence
    })

    return results