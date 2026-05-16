import pandas as pd


def feature_engineering(df):

    df = df.copy()

    # =====================================================
    # AVERAGE SCORE
    # =====================================================

    df["average_score"] = (

        df["math score"] +
        df["reading score"] +
        df["writing score"]

    ) / 3

    # =====================================================
    # TOTAL SCORE
    # =====================================================

    df["total_score"] = (

        df["math score"] +
        df["reading score"] +
        df["writing score"]
    )

    # =====================================================
    # PERFORMANCE LABEL
    # =====================================================

    def performance(score):

        if score >= 85:
            return "Excellent"

        elif score >= 70:
            return "Good"

        elif score >= 50:
            return "Average"

        else:
            return "At Risk"

    df["performance_level"] = df[
        "average_score"
    ].apply(performance)

    # =====================================================
    # RISK SCORE
    # =====================================================

    df["risk_score"] = 100 - df["average_score"]

    return df