import pandas as pd
import plotly.express as px

from src.data_loader import load_data


# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


# =====================================================
# CREATE PERFORMANCE LABEL
# =====================================================

df["average_score"] = (

    df["math score"] +
    df["reading score"] +
    df["writing score"]

) / 3


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


print("\nPerformance Distribution:")
print(df["performance_level"].value_counts())


# =====================================================
# VISUALIZATION 1
# =====================================================

fig1 = px.histogram(

    df,

    x="average_score",

    nbins=40,

    color="performance_level",

    title="Student Performance Distribution",

    color_discrete_sequence=[
        "#8B5CF6",
        "#EC4899",
        "#06B6D4",
        "#EF4444"
    ]
)

fig1.update_layout(

    template="plotly_dark",

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    font=dict(size=18),

    title_font_size=28
)

fig1.write_image(
    "outputs/performance_distribution.png"
)


# =====================================================
# VISUALIZATION 2
# =====================================================

fig2 = px.box(

    df,

    x="gender",

    y="average_score",

    color="gender",

    title="Gender vs Student Performance",

    color_discrete_sequence=[
        "#8B5CF6",
        "#EC4899"
    ]
)

fig2.update_layout(

    template="plotly_dark",

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    font=dict(size=18),

    title_font_size=28
)

fig2.write_image(
    "outputs/gender_performance.png"
)


# =====================================================
# VISUALIZATION 3
# =====================================================

fig3 = px.bar(

    df.groupby(
        "parental level of education"
    )["average_score"].mean().reset_index(),

    x="parental level of education",

    y="average_score",

    color="average_score",

    title="Parental Education Impact",

    color_continuous_scale=[
        "#06B6D4",
        "#8B5CF6",
        "#EC4899"
    ]
)

fig3.update_layout(

    template="plotly_dark",

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    font=dict(size=16),

    title_font_size=28
)

fig3.write_image(
    "outputs/parent_education_impact.png"
)

print("\nEDA COMPLETED SUCCESSFULLY")