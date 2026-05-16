import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import feature_engineering


def preprocess_data():

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(
        "data/students.csv"
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df = feature_engineering(df)

    # =====================================================
    # LABEL ENCODING
    # =====================================================

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    encoder = LabelEncoder()

    for col in categorical_cols:

        df[col] = encoder.fit_transform(
            df[col]
        )

    # =====================================================
    # FEATURES / TARGET
    # =====================================================

    X = df.drop(
        "performance_level",
        axis=1
    )

    y = df["performance_level"]

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    return X_train, X_test, y_train, y_test