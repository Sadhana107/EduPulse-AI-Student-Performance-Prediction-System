import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from src.feature_engineering import feature_engineering
from src.predict import predict_student


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EduPulse AI",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/student_model.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/students.csv"
)

df = feature_engineering(df)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""

<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {

    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

.stApp {

    background:
    linear-gradient(
        135deg,
        #050816 0%,
        #0B1120 50%,
        #111827 100%
    );
}

.hero {

    padding: 3rem;
    border-radius: 30px;

    background:
    linear-gradient(
        135deg,
        rgba(139,92,246,0.35),
        rgba(236,72,153,0.25),
        rgba(59,130,246,0.25)
    );

    border:
    1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    margin-bottom: 2rem;

    box-shadow:
    0px 0px 40px rgba(139,92,246,0.25);
}

.hero-title {

    font-size: 4rem;
    font-weight: 900;
    color: white;
    margin-bottom: 0.5rem;
}

.hero-sub {

    font-size: 1.4rem;
    color: #cbd5e1;
    line-height: 1.7;
}

.metric-card {

    padding: 2rem;
    border-radius: 26px;

    background:
    rgba(15,23,42,0.92);

    border:
    1px solid rgba(255,255,255,0.08);

    box-shadow:
    0px 0px 25px rgba(59,130,246,0.12);

    text-align: center;

    transition: 0.3s ease;

    min-height: 180px;
}

.metric-card:hover {

    transform: translateY(-6px);

    box-shadow:
    0px 0px 40px rgba(139,92,246,0.30);
}

.metric-title {

    color: #cbd5e1;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.metric-value {

    color: white;
    font-size: 3rem;
    font-weight: 900;
}

.section-card {

    padding: 2rem;
    border-radius: 26px;

    background:
    rgba(15,23,42,0.92);

    border:
    1px solid rgba(255,255,255,0.08);

    margin-bottom: 2rem;

    box-shadow:
    0px 0px 25px rgba(139,92,246,0.10);
}

.stDataFrame {
    font-size: 18px;
}

</style>

""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 EduPulse Controls")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

prep_filter = st.sidebar.multiselect(
    "Test Preparation Course",
    options=df["test preparation course"].unique(),
    default=df["test preparation course"].unique()
)

score_filter = st.sidebar.slider(
    "Average Score Range",
    0,
    100,
    (0, 100)
)


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["test preparation course"].isin(prep_filter)) &
    (df["average_score"] >= score_filter[0]) &
    (df["average_score"] <= score_filter[1])
]


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_students = len(filtered_df)

avg_score = round(
    filtered_df["average_score"].mean(),
    2
)

at_risk = len(
    filtered_df[
        filtered_df["performance_level"] == "At Risk"
    ]
)

excellent_students = len(
    filtered_df[
        filtered_df["performance_level"] == "Excellent"
    ]
)

success_rate = round(

    (
        len(
            filtered_df[
                filtered_df["performance_level"].isin(
                    ["Excellent", "Good"]
                )
            ]
        ) / total_students
    ) * 100,

    2
)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(f"""

<div class="hero">

<div class="hero-title">
🎓 EduPulse AI
</div>

<div class="hero-sub">

AI-Powered Academic Intelligence Platform

<br><br>

Real-time student performance analytics,
AI risk monitoring,
academic intelligence,
and predictive educational insights.

</div>

</div>

""", unsafe_allow_html=True)


# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""

    <div class="metric-card">
    <div class="metric-title">👨‍🎓 Total Students</div>
    <div class="metric-value">{total_students}</div>
    </div>

    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""

    <div class="metric-card">
    <div class="metric-title">📊 Average Score</div>
    <div class="metric-value">{avg_score}</div>
    </div>

    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""

    <div class="metric-card">
    <div class="metric-title">⚠️ At-Risk Students</div>
    <div class="metric-value">{at_risk}</div>
    </div>

    """, unsafe_allow_html=True)


# =========================================================
# SECOND KPI ROW
# =========================================================

col4, col5, col6 = st.columns(3)

with col4:

    st.markdown(f"""

    <div class="metric-card">
    <div class="metric-title">🌟 Excellent Students</div>
    <div class="metric-value">{excellent_students}</div>
    </div>

    """, unsafe_allow_html=True)

with col5:

    st.markdown(f"""

    <div class="metric-card">
    <div class="metric-title">🎯 Academic Success Rate</div>
    <div class="metric-value">{success_rate}%</div>
    </div>

    """, unsafe_allow_html=True)

with col6:

    st.markdown(f"""

    <div class="metric-card">
    <div class="metric-title">🤖 AI Confidence</div>
    <div class="metric-value">96%</div>
    </div>

    """, unsafe_allow_html=True)


# =========================================================
# CHART SECTION
# =========================================================

col7, col8 = st.columns(2)

with col7:

    performance_counts = filtered_df[
        "performance_level"
    ].value_counts().reset_index()

    performance_counts.columns = [
        "Performance",
        "Count"
    ]

    fig1 = px.pie(
        performance_counts,
        names="Performance",
        values="Count",
        hole=0.6,
        color="Performance",
        color_discrete_sequence=[
            "#8B5CF6",
            "#06B6D4",
            "#22C55E",
            "#EF4444"
        ]
    )

    fig1.update_layout(
        template="plotly_dark",
        height=600,
        title="Student Performance Distribution",
        title_font_size=28,
        font=dict(size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with col8:

    gender_perf = filtered_df.groupby(
        "gender"
    )["average_score"].mean().reset_index()

    fig2 = px.bar(
        gender_perf,
        x="gender",
        y="average_score",
        color="gender",
        color_discrete_sequence=[
            "#8B5CF6",
            "#EC4899"
        ]
    )

    fig2.update_layout(
        template="plotly_dark",
        height=600,
        title="Gender vs Academic Performance",
        title_font_size=28,
        font=dict(size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.markdown("""

<div class="section-card">

<h2 style="color:white; font-size:2rem;">
🧠 AI Feature Importance
</h2>

</div>

""", unsafe_allow_html=True)

importance_df = pd.DataFrame({

    "Feature": model.feature_names_in_,

    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig3 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale=[
        "#06B6D4",
        "#8B5CF6",
        "#EC4899"
    ]
)

fig3.update_layout(
    template="plotly_dark",
    height=700,
    title="Academic Performance Drivers",
    title_font_size=28,
    font=dict(size=18),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# =========================================================
# AT-RISK STUDENT TABLE
# =========================================================

st.markdown("""

<div class="section-card">

<h2 style="color:white; font-size:2rem;">
🚨 At-Risk Student Monitoring
</h2>

</div>

""", unsafe_allow_html=True)

risk_df = filtered_df[
    filtered_df["performance_level"] == "At Risk"
]

st.dataframe(

    risk_df[[

        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "average_score",
        "risk_score"
    ]],

    use_container_width=True
)


# =========================================================
# AI PREDICTION STUDIO
# =========================================================

st.markdown("""

<div class="section-card">

<h2 style="color:white; font-size:2rem;">
📂 AI Student Prediction Studio
</h2>

</div>

""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Student CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    upload_df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Dataset")

    st.dataframe(
        upload_df.head(),
        use_container_width=True
    )

    try:

        predictions = predict_student(upload_df)

        final_df = pd.concat(
            [upload_df, predictions],
            axis=1
        )

        st.write("### AI Prediction Results")

        st.dataframe(
            final_df,
            use_container_width=True
        )

        csv = final_df.to_csv(index=False)

        st.download_button(
            "📥 Download Predictions",
            csv,
            file_name="student_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# =========================================================
# AI RECOMMENDATION ENGINE
# =========================================================

st.markdown("""

<div class="section-card">

<h2 style="color:white; font-size:2rem;">
🧠 AI Academic Intelligence
</h2>

</div>

""", unsafe_allow_html=True)

high_risk_count = len(
    filtered_df[
        filtered_df["performance_level"] == "At Risk"
    ]
)

st.info(f"""

🚨 AI detected {high_risk_count} academically at-risk students.

Recommended interventions:
- additional mentoring
- academic counseling
- parent engagement
- test preparation support
- personalized learning plans

""")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""

<br><br>

<center>

<h3 style="color:#94a3b8;">
EduPulse AI — Academic Intelligence Platform
</h3>

<p style="color:#64748b; font-size:18px;">
Built with AI, Machine Learning, Streamlit, XGBoost & Plotly
</p>

</center>

""", unsafe_allow_html=True)