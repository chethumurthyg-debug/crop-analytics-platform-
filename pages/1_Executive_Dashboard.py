import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD CSS
# --------------------------------------------------

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Crop_recommendation.csv")

df = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        🌾 Executive Agriculture Dashboard
    </div>

    <div class='sub-title'>
        AI Powered Crop Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

total_records = len(df)
crop_count = df["label"].nunique()

avg_temp = round(
    df["temperature"].mean(),
    2
)

avg_rainfall = round(
    df["rainfall"].mean(),
    2
)

avg_ph = round(
    df["ph"].mean(),
    2
)

k1,k2,k3,k4,k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                Total Records
            </div>
            <div class='metric-value'>
                {total_records:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                Crop Types
            </div>
            <div class='metric-value'>
                {crop_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                Avg Temperature
            </div>
            <div class='metric-value'>
                {avg_temp}°C
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                Avg Rainfall
            </div>
            <div class='metric-value'>
                {avg_rainfall}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>
                Avg Soil pH
            </div>
            <div class='metric-value'>
                {avg_ph}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# TOP ROW CHARTS
# --------------------------------------------------

col1,col2 = st.columns(2)

with col1:

    st.markdown(
        "<div class='chart-card'>",
        unsafe_allow_html=True
    )

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
        title="Crop Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        "<div class='chart-card'>",
        unsafe_allow_html=True
    )

    fig2 = px.box(
        df,
        x="label",
        y="rainfall",
        color="label",
        title="Rainfall Distribution by Crop"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# CORRELATION MATRIX
# --------------------------------------------------

st.markdown("## 🔍 Correlation Analysis")

corr = (
    df.drop(columns=["label"])
    .corr()
)

heatmap = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Feature Correlation Matrix"
)

st.plotly_chart(
    heatmap,
    use_container_width=True
)

# --------------------------------------------------
# TEMPERATURE ANALYSIS
# --------------------------------------------------

col3,col4 = st.columns(2)

with col3:

    fig3 = px.histogram(
        df,
        x="temperature",
        nbins=30,
        title="Temperature Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

with col4:

    fig4 = px.scatter(
        df,
        x="temperature",
        y="humidity",
        color="label",
        title="Temperature vs Humidity"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# --------------------------------------------------
# SOIL HEALTH
# --------------------------------------------------

st.markdown("## 🌱 Soil Health Analytics")

soil_data = pd.DataFrame({
    "Metric":[
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH"
    ],
    "Average":[
        df["N"].mean(),
        df["P"].mean(),
        df["K"].mean(),
        df["ph"].mean()
    ]
})

soil_chart = px.bar(
    soil_data,
    x="Metric",
    y="Average",
    color="Average",
    title="Average Soil Parameters"
)

st.plotly_chart(
    soil_chart,
    use_container_width=True
)

# --------------------------------------------------
# CROP SUMMARY TABLE
# --------------------------------------------------

st.markdown("## 📋 Crop Summary")

summary = (
    df.groupby("label")
    .agg(
        Avg_Temperature=("temperature","mean"),
        Avg_Humidity=("humidity","mean"),
        Avg_Rainfall=("rainfall","mean"),
        Avg_pH=("ph","mean")
    )
    .reset_index()
)

st.dataframe(
    summary,
    use_container_width=True
)

# --------------------------------------------------
# INSIGHTS ENGINE
# --------------------------------------------------

st.markdown("## 🤖 AI Generated Insights")

highest_rainfall_crop = (
    df.groupby("label")["rainfall"]
    .mean()
    .idxmax()
)

lowest_rainfall_crop = (
    df.groupby("label")["rainfall"]
    .mean()
    .idxmin()
)

highest_temp_crop = (
    df.groupby("label")["temperature"]
    .mean()
    .idxmax()
)

insights = [
    f"Highest rainfall demanding crop is {highest_rainfall_crop}.",
    f"Lowest rainfall demanding crop is {lowest_rainfall_crop}.",
    f"Highest temperature tolerant crop is {highest_temp_crop}.",
    f"Dataset contains {crop_count} crop categories.",
    f"Average soil pH is {avg_ph}.",
]

for insight in insights:

    st.markdown(
        f"""
        <div class='insight-card'>
            ✅ {insight}
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='footer'>
        Smart Crop Recommendation Platform • Executive Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
