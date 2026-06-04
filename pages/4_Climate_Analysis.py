import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------

st.set_page_config(
    page_title="Climate Analysis",
    page_icon="🌦️",
    layout="wide"
)

# ------------------------------------------------------
# LOAD CSS
# ------------------------------------------------------

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/Crop_recommendation.csv"
    )

df = load_data()

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        🌦️ Climate Intelligence Dashboard
    </div>

    <div class='sub-title'>
        Advanced Climate Analytics For Smart Agriculture
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# KPIs
# ------------------------------------------------------

avg_temp = round(df["temperature"].mean(),2)
avg_humidity = round(df["humidity"].mean(),2)
avg_rainfall = round(df["rainfall"].mean(),2)

max_temp = round(df["temperature"].max(),2)
max_rainfall = round(df["rainfall"].max(),2)

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("Avg Temp (°C)", avg_temp)
c2.metric("Avg Humidity (%)", avg_humidity)
c3.metric("Avg Rainfall", avg_rainfall)
c4.metric("Max Temp", max_temp)
c5.metric("Max Rainfall", max_rainfall)

st.divider()

# ------------------------------------------------------
# TEMPERATURE ANALYSIS
# ------------------------------------------------------

st.subheader("🌡 Temperature Distribution")

fig = px.histogram(
    df,
    x="temperature",
    nbins=30,
    title="Temperature Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# HUMIDITY ANALYSIS
# ------------------------------------------------------

st.subheader("💧 Humidity Distribution")

fig = px.histogram(
    df,
    x="humidity",
    nbins=30,
    title="Humidity Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# RAINFALL ANALYSIS
# ------------------------------------------------------

st.subheader("🌧 Rainfall Distribution")

fig = px.histogram(
    df,
    x="rainfall",
    nbins=30,
    title="Rainfall Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# CLIMATE CORRELATION
# ------------------------------------------------------

st.subheader("📊 Climate Correlation Matrix")

corr = df[
    [
        "temperature",
        "humidity",
        "rainfall"
    ]
].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# TEMP VS HUMIDITY
# ------------------------------------------------------

st.subheader("🌡️ vs 💧 Analysis")

fig = px.scatter(
    df,
    x="temperature",
    y="humidity",
    color="label",
    title="Temperature vs Humidity"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# RAINFALL VS TEMP
# ------------------------------------------------------

st.subheader("🌧 vs 🌡 Analysis")

fig = px.scatter(
    df,
    x="rainfall",
    y="temperature",
    color="label",
    title="Rainfall vs Temperature"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# CROP CLIMATE REQUIREMENTS
# ------------------------------------------------------

st.subheader("🌾 Crop Climate Requirements")

crop_summary = (
    df.groupby("label")
      .agg({
          "temperature":"mean",
          "humidity":"mean",
          "rainfall":"mean"
      })
      .round(2)
      .reset_index()
)

st.dataframe(
    crop_summary,
    use_container_width=True
)

# ------------------------------------------------------
# TEMPERATURE ZONES
# ------------------------------------------------------

st.subheader("🌡 Temperature Zones")

df_temp = df.copy()

df_temp["Temp Zone"] = pd.cut(
    df_temp["temperature"],
    bins=[0,20,30,50],
    labels=[
        "Cool",
        "Moderate",
        "Hot"
    ]
)

temp_zone = (
    df_temp["Temp Zone"]
    .value_counts()
    .reset_index()
)

temp_zone.columns = [
    "Zone",
    "Count"
]

fig = px.pie(
    temp_zone,
    names="Zone",
    values="Count",
    hole=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# RAINFALL ZONES
# ------------------------------------------------------

st.subheader("🌧 Rainfall Zones")

df_rain = df.copy()

df_rain["Rain Zone"] = pd.cut(
    df_rain["rainfall"],
    bins=[0,100,200,400],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

rain_zone = (
    df_rain["Rain Zone"]
    .value_counts()
    .reset_index()
)

rain_zone.columns = [
    "Zone",
    "Count"
]

fig = px.pie(
    rain_zone,
    names="Zone",
    values="Count",
    hole=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# CLIMATE RADAR
# ------------------------------------------------------

st.subheader("🎯 Climate Radar")

selected_crop = st.selectbox(
    "Select Crop",
    sorted(df["label"].unique())
)

crop = (
    df[df["label"] == selected_crop]
    .mean(numeric_only=True)
)

fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=[
            crop["temperature"],
            crop["humidity"],
            crop["rainfall"]/5
        ],
        theta=[
            "Temperature",
            "Humidity",
            "Rainfall"
        ],
        fill="toself",
        name=selected_crop
    )
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True
        )
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# CLIMATE CLUSTERING
# ------------------------------------------------------

st.subheader("🤖 Climate Segmentation")

features = df[
    [
        "temperature",
        "humidity",
        "rainfall"
    ]
]

scaler = StandardScaler()

scaled = scaler.fit_transform(
    features
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    scaled
)

cluster_df = df.copy()

cluster_df["Cluster"] = clusters

fig = px.scatter_3d(
    cluster_df,
    x="temperature",
    y="humidity",
    z="rainfall",
    color="Cluster",
    title="Climate Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# TOP CROPS BY CLIMATE
# ------------------------------------------------------

st.subheader("🏆 Climate Leaders")

highest_temp = (
    df.groupby("label")["temperature"]
      .mean()
      .idxmax()
)

highest_humidity = (
    df.groupby("label")["humidity"]
      .mean()
      .idxmax()
)

highest_rainfall = (
    df.groupby("label")["rainfall"]
      .mean()
      .idxmax()
)

insights = pd.DataFrame({
    "Metric":[
        "Highest Temperature",
        "Highest Humidity",
        "Highest Rainfall"
    ],
    "Crop":[
        highest_temp,
        highest_humidity,
        highest_rainfall
    ]
})

st.dataframe(
    insights,
    use_container_width=True
)

# ------------------------------------------------------
# AI INSIGHTS
# ------------------------------------------------------

st.subheader("🤖 Climate Intelligence Insights")

ai_insights = []

if avg_temp > 25:
    ai_insights.append(
        "Average temperature indicates warm crop conditions."
    )

if avg_humidity > 65:
    ai_insights.append(
        "Humidity levels support moisture-sensitive crops."
    )

if avg_rainfall > 100:
    ai_insights.append(
        "Rainfall is generally sufficient for most crops."
    )

ai_insights.append(
    f"Highest rainfall demanding crop is {highest_rainfall}."
)

ai_insights.append(
    f"Highest temperature tolerant crop is {highest_temp}."
)

for insight in ai_insights:
    st.markdown(
        f"""
        <div class='insight-card'>
            ✅ {insight}
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------
# CLIMATE SUMMARY
# ------------------------------------------------------

st.subheader("📋 Climate Summary")

summary = pd.DataFrame({
    "Metric":[
        "Temperature",
        "Humidity",
        "Rainfall"
    ],
    "Average":[
        df["temperature"].mean(),
        df["humidity"].mean(),
        df["rainfall"].mean()
    ],
    "Minimum":[
        df["temperature"].min(),
        df["humidity"].min(),
        df["rainfall"].min()
    ],
    "Maximum":[
        df["temperature"].max(),
        df["humidity"].max(),
        df["rainfall"].max()
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

# ------------------------------------------------------
# DOWNLOAD REPORT
# ------------------------------------------------------

csv = summary.to_csv(index=False)

st.download_button(
    label="⬇ Download Climate Report",
    data=csv,
    file_name="climate_report.csv",
    mime="text/csv"
)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------

st.markdown(
    """
    <div class='footer'>
        Smart Crop Recommendation Platform |
        Climate Intelligence Module
    </div>
    """,
    unsafe_allow_html=True
)
