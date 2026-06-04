import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Smart Crop Analytics Platform",
    page_icon="🌾",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 8px rgba(0,0,0,0.1);
}

h1,h2,h3{
color:#14532d;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Crop_recommendation.csv")

df = load_data()

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🌾 Smart Crop Recommendation & Analytics Platform")

st.markdown("""
AI Powered Agricultural Intelligence Dashboard

- Crop Recommendation
- Soil Analytics
- Climate Intelligence
- Machine Learning
- Deep Insights
""")

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------

st.subheader("📊 Executive Dashboard")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Total Records",
        len(df)
    )

with c2:
    st.metric(
        "Crop Types",
        df['label'].nunique()
    )

with c3:
    st.metric(
        "Avg Temperature",
        round(df['temperature'].mean(),2)
    )

with c4:
    st.metric(
        "Avg Rainfall",
        round(df['rainfall'].mean(),2)
    )

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dataset Explorer",
        "Crop Analytics",
        "Soil Analytics",
        "Climate Analytics",
        "Machine Learning",
        "Crop Recommendation"
    ]
)

# -------------------------------------------------
# DATASET EXPLORER
# -------------------------------------------------

if page == "Dataset Explorer":

    st.header("📁 Dataset Explorer")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistics")

    st.dataframe(df.describe())

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Dataset",
        csv,
        "crop_data.csv",
        "text/csv"
    )

# -------------------------------------------------
# CROP ANALYTICS
# -------------------------------------------------

elif page == "Crop Analytics":

    st.header("🌾 Crop Analytics")

    crop_count = df['label'].value_counts().reset_index()

    crop_count.columns = ['Crop','Count']

    fig = px.bar(
        crop_count,
        x='Crop',
        y='Count',
        color='Count',
        title='Crop Distribution'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Rainfall Distribution By Crop")

    fig2 = px.box(
        df,
        x='label',
        y='rainfall',
        color='label'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    highest_crop = (
        df.groupby('label')['rainfall']
        .mean()
        .idxmax()
    )

    st.success(
        f"Highest Rainfall Requirement Crop: {highest_crop}"
    )

# -------------------------------------------------
# SOIL ANALYTICS
# -------------------------------------------------

elif page == "Soil Analytics":

    st.header("🧪 Soil Analytics")

    soil_cols = ['N','P','K','ph']

    fig = px.box(
        df[soil_cols],
        title="Soil Nutrient Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    nutrient_avg = pd.DataFrame({
        "Parameter":soil_cols,
        "Average":[
            df['N'].mean(),
            df['P'].mean(),
            df['K'].mean(),
            df['ph'].mean()
        ]
    })

    fig2 = px.bar(
        nutrient_avg,
        x='Parameter',
        y='Average',
        color='Average'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# -------------------------------------------------
# CLIMATE ANALYTICS
# -------------------------------------------------

elif page == "Climate Analytics":

    st.header("🌦 Climate Analytics")

    fig = px.scatter(
        df,
        x='temperature',
        y='humidity',
        color='label',
        title='Temperature vs Humidity'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.scatter(
        df,
        x='rainfall',
        y='temperature',
        color='label',
        title='Rainfall vs Temperature'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    corr = df.drop(columns=['label']).corr()

    heatmap = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------

elif page == "Machine Learning":

    st.header("🤖 Machine Learning Insights")

    X = df.drop("label", axis=1)
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
        n_estimators=300,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    # Feature Importance

    feature_importance = pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # PCA

    scaler = StandardScaler()

    scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1","PC2"]
    )

    pca_df["Crop"] = y

    fig2 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Crop",
        title="Crop Clustering (PCA)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

elif page == "Crop Recommendation":

    st.header("🌱 Crop Recommendation System")

    X = df.drop("label", axis=1)
    y = df["label"]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(
        X,
        y_encoded
    )

    c1,c2,c3 = st.columns(3)

    with c1:
        N = st.number_input("Nitrogen",0,150,90)

        P = st.number_input("Phosphorus",0,150,42)

        K = st.number_input("Potassium",0,250,43)

    with c2:
        temp = st.number_input(
            "Temperature",
            value=25.0
        )

        humidity = st.number_input(
            "Humidity",
            value=80.0
        )

    with c3:
        ph = st.number_input(
            "pH",
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall",
            value=150.0
        )

    if st.button("Recommend Crop"):

        input_data = np.array([
            [
                N,
                P,
                K,
                temp,
                humidity,
                ph,
                rainfall
            ]
        ])

        pred = model.predict(input_data)

        crop = encoder.inverse_transform(pred)[0]

        confidence = np.max(
            model.predict_proba(input_data)
        ) * 100

        st.success(
            f"Recommended Crop: {crop}"
        )

        st.info(
            f"Prediction Confidence: {confidence:.2f}%"
        )

        st.subheader("💡 AI Insights")

        if rainfall > 200:
            st.write("High rainfall detected.")

        if ph < 6:
            st.write("Soil is acidic.")

        if ph > 7:
            st.write("Soil is alkaline.")

        if temp > 35:
            st.write("High temperature condition.")

        if humidity > 80:
            st.write("Suitable for moisture-loving crops.")
