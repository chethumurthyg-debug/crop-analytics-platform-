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
    page_title="Soil Analysis",
    page_icon="🌱",
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
        🌱 Soil Intelligence Dashboard
    </div>

    <div class='sub-title'>
        Deep Soil Analytics & Fertility Assessment
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# SOIL HEALTH SCORE
# ------------------------------------------------------

soil_score = (
    (
        df["N"].mean()/140
    ) +
    (
        df["P"].mean()/145
    ) +
    (
        df["K"].mean()/205
    ) +
    (
        df["ph"].mean()/14
    )
) / 4 * 100

# ------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------

k1,k2,k3,k4,k5 = st.columns(5)

with k1:
    st.metric(
        "Avg Nitrogen",
        round(df["N"].mean(),2)
    )

with k2:
    st.metric(
        "Avg Phosphorus",
        round(df["P"].mean(),2)
    )

with k3:
    st.metric(
        "Avg Potassium",
        round(df["K"].mean(),2)
    )

with k4:
    st.metric(
        "Avg pH",
        round(df["ph"].mean(),2)
    )

with k5:
    st.metric(
        "Soil Health %",
        round(soil_score,2)
    )

st.divider()

# ------------------------------------------------------
# SOIL HEALTH GAUGE
# ------------------------------------------------------

st.subheader("🌱 Soil Health Index")

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=soil_score,
        title={"text":"Soil Health Score"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"green"},
            "steps":[
                {"range":[0,40],"color":"red"},
                {"range":[40,70],"color":"orange"},
                {"range":[70,100],"color":"lightgreen"}
            ]
        }
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# NPK DISTRIBUTION
# ------------------------------------------------------

st.subheader("📊 Nutrient Distribution")

col1,col2,col3 = st.columns(3)

with col1:

    fig_n = px.histogram(
        df,
        x="N",
        nbins=30,
        title="Nitrogen Distribution"
    )

    st.plotly_chart(
        fig_n,
        use_container_width=True
    )

with col2:

    fig_p = px.histogram(
        df,
        x="P",
        nbins=30,
        title="Phosphorus Distribution"
    )

    st.plotly_chart(
        fig_p,
        use_container_width=True
    )

with col3:

    fig_k = px.histogram(
        df,
        x="K",
        nbins=30,
        title="Potassium Distribution"
    )

    st.plotly_chart(
        fig_k,
        use_container_width=True
    )

# ------------------------------------------------------
# NPK COMPARISON
# ------------------------------------------------------

st.subheader("🌾 Average NPK Levels")

npk_df = pd.DataFrame({
    "Nutrient":[
        "Nitrogen",
        "Phosphorus",
        "Potassium"
    ],
    "Value":[
        df["N"].mean(),
        df["P"].mean(),
        df["K"].mean()
    ]
})

fig = px.bar(
    npk_df,
    x="Nutrient",
    y="Value",
    color="Value"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# BOX PLOTS
# ------------------------------------------------------

st.subheader("📦 Nutrient Spread")

fig = px.box(
    df[["N","P","K"]],
    title="NPK Variability"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# PH ANALYSIS
# ------------------------------------------------------

st.subheader("🧪 Soil pH Analytics")

fig = px.histogram(
    df,
    x="ph",
    nbins=30,
    title="pH Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# PH CLASSIFICATION
# ------------------------------------------------------

st.subheader("🧪 Soil Type Classification")

acidic = len(df[df["ph"] < 6])

neutral = len(
    df[
        (df["ph"] >= 6) &
        (df["ph"] <= 7.5)
    ]
)

alkaline = len(df[df["ph"] > 7.5])

soil_type = pd.DataFrame({
    "Type":[
        "Acidic",
        "Neutral",
        "Alkaline"
    ],
    "Count":[
        acidic,
        neutral,
        alkaline
    ]
})

fig = px.pie(
    soil_type,
    names="Type",
    values="Count",
    hole=0.5,
    title="Soil Classification"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# CORRELATION MATRIX
# ------------------------------------------------------

st.subheader("🔍 Soil Correlation Matrix")

corr_cols = [
    "N",
    "P",
    "K",
    "ph"
]

corr = df[corr_cols].corr()

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
# RADAR CHART
# ------------------------------------------------------

st.subheader("🎯 Soil Nutrient Radar")

radar_values = [
    df["N"].mean(),
    df["P"].mean(),
    df["K"].mean(),
    df["ph"].mean()*10
]

fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=radar_values,
        theta=[
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "pH x10"
        ],
        fill="toself",
        name="Average Soil Profile"
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
# SOIL CLUSTERING
# ------------------------------------------------------

st.subheader("🤖 Soil Segmentation")

features = df[
    [
        "N",
        "P",
        "K",
        "ph"
    ]
]

scaler = StandardScaler()

scaled = scaler.fit_transform(features)

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
    x="N",
    y="P",
    z="K",
    color="Cluster",
    title="Soil Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# DEFICIENCY DETECTION
# ------------------------------------------------------

st.subheader("⚠ Nutrient Deficiency Detection")

n_low = len(df[df["N"] < df["N"].mean()])
p_low = len(df[df["P"] < df["P"].mean()])
k_low = len(df[df["K"] < df["K"].mean()])

deficiency = pd.DataFrame({
    "Nutrient":[
        "Nitrogen",
        "Phosphorus",
        "Potassium"
    ],
    "Low Records":[
        n_low,
        p_low,
        k_low
    ]
})

fig = px.bar(
    deficiency,
    x="Nutrient",
    y="Low Records",
    color="Low Records"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------------
# SOIL INSIGHTS
# ------------------------------------------------------

st.subheader("🤖 AI Soil Insights")

insights = []

if soil_score > 70:
    insights.append(
        "Excellent overall soil health detected."
    )

if acidic > alkaline:
    insights.append(
        "Most samples fall under acidic soil conditions."
    )

if df["N"].mean() > df["P"].mean():
    insights.append(
        "Nitrogen concentration exceeds phosphorus."
    )

if df["K"].mean() > 40:
    insights.append(
        "Potassium levels are favorable for crop growth."
    )

insights.append(
    f"Average soil pH is {round(df['ph'].mean(),2)}."
)

for insight in insights:

    st.markdown(
        f"""
        <div class='insight-card'>
        ✅ {insight}
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------
# SOIL SUMMARY TABLE
# ------------------------------------------------------

st.subheader("📋 Soil Summary")

summary = pd.DataFrame({
    "Metric":[
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH"
    ],
    "Mean":[
        df["N"].mean(),
        df["P"].mean(),
        df["K"].mean(),
        df["ph"].mean()
    ],
    "Min":[
        df["N"].min(),
        df["P"].min(),
        df["K"].min(),
        df["ph"].min()
    ],
    "Max":[
        df["N"].max(),
        df["P"].max(),
        df["K"].max(),
        df["ph"].max()
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

# ------------------------------------------------------
# EXPORT
# ------------------------------------------------------

csv = summary.to_csv(index=False)

st.download_button(
    "⬇ Download Soil Report",
    csv,
    file_name="soil_report.csv",
    mime="text/csv"
)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------

st.markdown(
    """
    <div class='footer'>
        Smart Crop Recommendation Platform |
        Soil Intelligence Module
    </div>
    """,
    unsafe_allow_html=True
)
