import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Crop Analytics",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------------------------------
# LOAD CSS
# -----------------------------------------------------

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Crop_recommendation.csv")

df = load_data()

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        🌾 Crop Analytics
    </div>

    <div class='sub-title'>
        Comprehensive Crop Intelligence & Performance Analysis
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# SIDEBAR FILTER
# -----------------------------------------------------

st.sidebar.header("Filters")

selected_crop = st.sidebar.selectbox(
    "Select Crop",
    ["All"] + sorted(df["label"].unique().tolist())
)

if selected_crop != "All":
    filtered_df = df[df["label"] == selected_crop]
else:
    filtered_df = df.copy()

# -----------------------------------------------------
# KPI SECTION
# -----------------------------------------------------

total_records = len(filtered_df)

avg_temp = round(
    filtered_df["temperature"].mean(),
    2
)

avg_humidity = round(
    filtered_df["humidity"].mean(),
    2
)

avg_rainfall = round(
    filtered_df["rainfall"].mean(),
    2
)

avg_ph = round(
    filtered_df["ph"].mean(),
    2
)

k1,k2,k3,k4,k5 = st.columns(5)

with k1:
    st.metric(
        "Records",
        total_records
    )

with k2:
    st.metric(
        "Avg Temp",
        avg_temp
    )

with k3:
    st.metric(
        "Avg Humidity",
        avg_humidity
    )

with k4:
    st.metric(
        "Avg Rainfall",
        avg_rainfall
    )

with k5:
    st.metric(
        "Avg pH",
        avg_ph
    )

st.divider()

# -----------------------------------------------------
# CROP DISTRIBUTION
# -----------------------------------------------------

st.subheader("📊 Crop Distribution")

crop_dist = (
    df["label"]
    .value_counts()
    .reset_index()
)

crop_dist.columns = [
    "Crop",
    "Count"
]

fig = px.bar(
    crop_dist,
    x="Crop",
    y="Count",
    color="Count",
    title="Crop Frequency Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# TEMPERATURE ANALYSIS
# -----------------------------------------------------

st.subheader("🌡 Temperature Analysis")

fig = px.box(
    filtered_df,
    x="label",
    y="temperature",
    color="label",
    title="Temperature Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# RAINFALL ANALYSIS
# -----------------------------------------------------

st.subheader("🌧 Rainfall Analysis")

fig = px.box(
    filtered_df,
    x="label",
    y="rainfall",
    color="label",
    title="Rainfall Requirement Analysis"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# HUMIDITY ANALYSIS
# -----------------------------------------------------

st.subheader("💧 Humidity Analysis")

fig = px.box(
    filtered_df,
    x="label",
    y="humidity",
    color="label",
    title="Humidity Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# PH ANALYSIS
# -----------------------------------------------------

st.subheader("🧪 Soil pH Analysis")

fig = px.box(
    filtered_df,
    x="label",
    y="ph",
    color="label",
    title="pH Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# NPK ANALYSIS
# -----------------------------------------------------

st.subheader("🌱 NPK Nutrient Analysis")

npk_summary = pd.DataFrame({
    "Nutrient": ["Nitrogen","Phosphorus","Potassium"],
    "Average": [
        filtered_df["N"].mean(),
        filtered_df["P"].mean(),
        filtered_df["K"].mean()
    ]
})

fig = px.bar(
    npk_summary,
    x="Nutrient",
    y="Average",
    color="Average",
    title="Average Nutrient Levels"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# TOP CROPS COMPARISON
# -----------------------------------------------------

st.subheader("🏆 Crop Comparison Matrix")

summary = (
    df.groupby("label")
      .agg({
          "temperature":"mean",
          "humidity":"mean",
          "rainfall":"mean",
          "ph":"mean",
          "N":"mean",
          "P":"mean",
          "K":"mean"
      })
      .round(2)
      .reset_index()
)

st.dataframe(
    summary,
    use_container_width=True
)

# -----------------------------------------------------
# RADAR CHART
# -----------------------------------------------------

st.subheader("🎯 Crop Radar Analysis")

crop_choice = st.selectbox(
    "Select Crop for Radar View",
    sorted(df["label"].unique())
)

crop_data = (
    df[df["label"] == crop_choice]
    .mean(numeric_only=True)
)

categories = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=[crop_data[c] for c in categories],
        theta=categories,
        fill="toself",
        name=crop_choice
    )
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True
        )
    ),
    showlegend=True,
    title=f"{crop_choice} Profile"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# PARALLEL COORDINATE ANALYSIS
# -----------------------------------------------------

st.subheader("🔍 Parallel Coordinates")

sample = df.sample(
    min(500, len(df)),
    random_state=42
)

fig = px.parallel_coordinates(
    sample,
    dimensions=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ],
    color="temperature"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# SCATTER MATRIX
# -----------------------------------------------------

st.subheader("📈 Feature Relationship Matrix")

fig = px.scatter_matrix(
    sample,
    dimensions=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity"
    ],
    color="label"
)

st.plotly_chart(
    fig,
    use_container_width=True)

# -----------------------------------------------------
# INSIGHTS ENGINE
# -----------------------------------------------------

st.subheader("🤖 Crop Intelligence Insights")

highest_rain_crop = (
    df.groupby("label")["rainfall"]
    .mean()
    .idxmax()
)

highest_temp_crop = (
    df.groupby("label")["temperature"]
    .mean()
    .idxmax()
)

highest_humidity_crop = (
    df.groupby("label")["humidity"]
    .mean()
    .idxmax()
)

lowest_ph_crop = (
    df.groupby("label")["ph"]
    .mean()
    .idxmin()
)

insights = [
    f"🌧 Highest rainfall demanding crop: {highest_rain_crop}",
    f"🌡 Highest temperature crop: {highest_temp_crop}",
    f"💧 Highest humidity crop: {highest_humidity_crop}",
    f"🧪 Most acidic soil crop: {lowest_ph_crop}",
    f"🌾 Dataset contains {df['label'].nunique()} crop categories"
]

for item in insights:
    st.markdown(
        f"""
        <div class='insight-card'>
            {item}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------
# DOWNLOAD SUMMARY
# -----------------------------------------------------

st.subheader("⬇ Export Summary")

csv = summary.to_csv(index=False)

st.download_button(
    label="Download Crop Summary",
    data=csv,
    file_name="crop_summary.csv",
    mime="text/csv"
)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown(
    """
    <div class='footer'>
        Smart Crop Recommendation Platform | Crop Analytics Module
    </div>
    """,
    unsafe_allow_html=True
)
