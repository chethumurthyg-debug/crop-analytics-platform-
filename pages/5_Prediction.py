import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Crop Prediction",
    page_icon="🌱",
    layout="wide"
)

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/Crop_recommendation.csv"
    )

df = load_data()

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

@st.cache_resource
def train_model():

    X = df.drop(
        "label",
        axis=1
    )

    y = df["label"]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )

    return (
        model,
        encoder,
        accuracy,
        X.columns
    )

model, encoder, accuracy, feature_names = train_model()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        🌱 Smart Crop Recommendation Engine
    </div>

    <div class='sub-title'>
        AI Powered Crop Prediction & Decision Support
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# MODEL KPI
# ---------------------------------------------------

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col2:
    st.metric(
        "Training Records",
        len(df)
    )

with col3:
    st.metric(
        "Crop Classes",
        df["label"].nunique()
    )

st.divider()

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

st.subheader("🌾 Enter Soil & Climate Parameters")

c1,c2,c3 = st.columns(3)

with c1:

    N = st.slider(
        "Nitrogen (N)",
        0,
        150,
        90
    )

    P = st.slider(
        "Phosphorus (P)",
        0,
        150,
        42
    )

    K = st.slider(
        "Potassium (K)",
        0,
        250,
        43
    )

with c2:

    temperature = st.slider(
        "Temperature",
        0.0,
        50.0,
        25.0
    )

    humidity = st.slider(
        "Humidity",
        0.0,
        100.0,
        80.0
    )

with c3:

    ph = st.slider(
        "Soil pH",
        0.0,
        14.0,
        6.5
    )

    rainfall = st.slider(
        "Rainfall",
        0.0,
        350.0,
        150.0
    )

# ---------------------------------------------------
# SOIL HEALTH SCORE
# ---------------------------------------------------

soil_score = (
    (
        N/140
    ) +
    (
        P/145
    ) +
    (
        K/205
    ) +
    (
        ph/14
    )
) / 4 * 100

st.progress(
    int(min(soil_score,100))
)

st.write(
    f"🌱 Soil Health Score: {soil_score:.2f}%"
)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

if st.button(
    "🚀 Predict Best Crop",
    use_container_width=True
):

    input_data = np.array([
        [
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]
    ])

    prediction = model.predict(
        input_data
    )

    probabilities = model.predict_proba(
        input_data
    )[0]

    predicted_crop = encoder.inverse_transform(
        prediction
    )[0]

    confidence = (
        np.max(probabilities)
        * 100
    )

    # ---------------------------------------
    # RESULT
    # ---------------------------------------

    st.markdown(
        f"""
        <div class='prediction-card'>
            <h2>
            Recommended Crop
            </h2>

            <h1>
            {predicted_crop.upper()}
            </h1>

            <p>
            Confidence: {confidence:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------------------------------
    # TOP 3 CROPS
    # ---------------------------------------

    st.subheader(
        "🏆 Top Crop Recommendations"
    )

    top_idx = np.argsort(
        probabilities
    )[-3:][::-1]

    top_crops = encoder.inverse_transform(
        top_idx
    )

    top_scores = probabilities[top_idx] * 100

    top_df = pd.DataFrame({
        "Crop":top_crops,
        "Probability (%)":top_scores
    })

    st.dataframe(
        top_df,
        use_container_width=True
    )

    # ---------------------------------------
    # PROBABILITY CHART
    # ---------------------------------------

    fig = px.bar(
        top_df,
        x="Crop",
        y="Probability (%)",
        color="Probability (%)",
        title="Top Recommendation Scores"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------

    st.subheader(
        "📊 Model Feature Importance"
    )

    importance_df = pd.DataFrame({
        "Feature":feature_names,
        "Importance":model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    fig2 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ---------------------------------------
    # AI INSIGHTS
    # ---------------------------------------

    st.subheader(
        "🤖 AI Recommendation Insights"
    )

    insights = []

    if ph < 6:
        insights.append(
            "Soil is acidic."
        )

    elif ph > 7.5:
        insights.append(
            "Soil is alkaline."
        )

    else:
        insights.append(
            "Soil pH is suitable."
        )

    if rainfall > 200:
        insights.append(
            "High rainfall conditions detected."
        )

    if humidity > 80:
        insights.append(
            "High humidity environment."
        )

    if temperature > 35:
        insights.append(
            "Hot climate condition."
        )

    if N > 100:
        insights.append(
            "Nitrogen-rich soil."
        )

    if P > 80:
        insights.append(
            "High phosphorus availability."
        )

    if K > 100:
        insights.append(
            "Potassium-rich soil."
        )

    insights.append(
        f"Best crop recommendation is {predicted_crop}."
    )

    for item in insights:

        st.markdown(
            f"""
            <div class='insight-card'>
                ✅ {item}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------
    # REPORT
    # ---------------------------------------

    report = pd.DataFrame({
        "Parameter":[
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "pH",
            "Rainfall"
        ],
        "Value":[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]
    })

    csv = report.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Prediction Report",
        csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
    """
    <div class='footer'>
        Smart Crop Recommendation Platform |
        AI Prediction Engine
    </div>
    """,
    unsafe_allow_html=True
)
