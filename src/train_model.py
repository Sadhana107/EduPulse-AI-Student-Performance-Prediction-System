import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import plotly.express as px

from src.preprocess import preprocess_data


# =====================================================
# LOAD DATA
# =====================================================

X_train, X_test, y_train, y_test = preprocess_data()


# =====================================================
# RANDOM FOREST MODEL
# =====================================================

rf_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_preds = rf_model.predict(X_test)


# =====================================================
# XGBOOST MODEL
# =====================================================

xgb_model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42,

    eval_metric="mlogloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_preds = xgb_model.predict(X_test)


# =====================================================
# EVALUATION
# =====================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_preds
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_preds
)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nXGBoost Accuracy:")
print(xgb_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        xgb_preds
    )
)


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    xgb_preds
)

cm_df = pd.DataFrame(cm)

fig = px.imshow(

    cm_df,

    text_auto=True,

    color_continuous_scale=[
        "#050816",
        "#8B5CF6",
        "#EC4899"
    ],

    title="Student Performance Confusion Matrix"
)

fig.update_layout(

    template="plotly_dark",

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    font=dict(size=18),

    title_font_size=28
)

fig.write_image(
    "outputs/confusion_matrix.png"
)


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": xgb_model.feature_importances_
})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False
)

fig2 = px.bar(

    importance_df,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale=[
        "#06B6D4",
        "#8B5CF6",
        "#EC4899"
    ],

    title="Feature Importance Analysis"
)

fig2.update_layout(

    template="plotly_dark",

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    font=dict(size=18),

    title_font_size=28
)

fig2.write_image(
    "outputs/feature_importance.png"
)


# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(

    xgb_model,

    "models/student_model.pkl"
)

print("\nMODEL TRAINED & SAVED SUCCESSFULLY")