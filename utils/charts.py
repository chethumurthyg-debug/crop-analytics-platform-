import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# CROP DISTRIBUTION
# ==========================================================

def crop_distribution_chart(df):

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

    fig.update_layout(
        height=500
    )

    return fig

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap(df):

    corr = (
        df.drop(
            columns=["label"],
            errors="ignore"
        )
        .corr()
    )

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    return fig

# ==========================================================
# TEMPERATURE HISTOGRAM
# ==========================================================

def temperature_distribution(df):

    fig = px.histogram(
        df,
        x="temperature",
        nbins=30,
        title="Temperature Distribution"
    )

    return fig

# ==========================================================
# HUMIDITY HISTOGRAM
# ==========================================================

def humidity_distribution(df):

    fig = px.histogram(
        df,
        x="humidity",
        nbins=30,
        title="Humidity Distribution"
    )

    return fig

# ==========================================================
# RAINFALL HISTOGRAM
# ==========================================================

def rainfall_distribution(df):

    fig = px.histogram(
        df,
        x="rainfall",
        nbins=30,
        title="Rainfall Distribution"
    )

    return fig

# ==========================================================
# NPK BAR CHART
# ==========================================================

def npk_chart(df):

    npk = pd.DataFrame({
        "Nutrient":[
            "Nitrogen",
            "Phosphorus",
            "Potassium"
        ],
        "Average":[
            df["N"].mean(),
            df["P"].mean(),
            df["K"].mean()
        ]
    })

    fig = px.bar(
        npk,
        x="Nutrient",
        y="Average",
        color="Average",
        title="Average NPK Levels"
    )

    return fig

# ==========================================================
# SOIL PH DISTRIBUTION
# ==========================================================

def ph_distribution(df):

    fig = px.histogram(
        df,
        x="ph",
        nbins=30,
        title="Soil pH Distribution"
    )

    return fig

# ==========================================================
# RAINFALL VS TEMPERATURE
# ==========================================================

def rainfall_temperature_scatter(df):

    fig = px.scatter(
        df,
        x="rainfall",
        y="temperature",
        color="label",
        title="Rainfall vs Temperature"
    )

    return fig

# ==========================================================
# TEMPERATURE VS HUMIDITY
# ==========================================================

def temperature_humidity_scatter(df):

    fig = px.scatter(
        df,
        x="temperature",
        y="humidity",
        color="label",
        title="Temperature vs Humidity"
    )

    return fig

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance_chart(importance_df):

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Feature Importance"
    )

    return fig

# ==========================================================
# MODEL PROBABILITIES
# ==========================================================

def probability_chart(prob_df):

    fig = px.bar(
        prob_df,
        x="Crop",
        y="Probability (%)",
        color="Probability (%)",
        title="Prediction Confidence"
    )

    return fig

# ==========================================================
# SOIL HEALTH GAUGE
# ==========================================================

def soil_health_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={
                "text":"Soil Health Score"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                },
                "bar":{
                    "color":"green"
                },
                "steps":[
                    {
                        "range":[0,40],
                        "color":"red"
                    },
                    {
                        "range":[40,70],
                        "color":"orange"
                    },
                    {
                        "range":[70,100],
                        "color":"lightgreen"
                    }
                ]
            }
        )
    )

    return fig

# ==========================================================
# RADAR CHART
# ==========================================================

def crop_radar_chart(crop_data):

    categories = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    values = [
        crop_data["N"],
        crop_data["P"],
        crop_data["K"],
        crop_data["temperature"],
        crop_data["humidity"],
        crop_data["ph"],
        crop_data["rainfall"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Crop Profile"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        )
    )

    return fig

# ==========================================================
# PIE CHART
# ==========================================================

def pie_chart(df, column):

    chart_data = (
        df[column]
        .value_counts()
        .reset_index()
    )

    chart_data.columns = [
        column,
        "Count"
    ]

    fig = px.pie(
        chart_data,
        names=column,
        values="Count",
        hole=0.4
    )

    return fig

# ==========================================================
# BOXPLOT
# ==========================================================

def nutrient_boxplot(df):

    fig = px.box(
        df[
            [
                "N",
                "P",
                "K"
            ]
        ],
        title="NPK Distribution"
    )

    return fig

# ==========================================================
# CLUSTER VISUALIZATION
# ==========================================================

def cluster_3d_chart(
    df,
    x_col,
    y_col,
    z_col,
    cluster_col
):

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=cluster_col,
        title="Cluster Analysis"
    )

    return fig

# ==========================================================
# LINE CHART
# ==========================================================

def line_chart(
    data,
    x_col,
    y_col,
    title
):

    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        markers=True,
        title=title
    )

    return fig

# ==========================================================
# LEARNING CURVE
# ==========================================================

def learning_curve_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Training Size"],
            y=df["Train Score"],
            mode="lines+markers",
            name="Train"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Training Size"],
            y=df["Validation Score"],
            mode="lines+markers",
            name="Validation"
        )
    )

    fig.update_layout(
        title="Learning Curve"
    )

    return fig

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def confusion_matrix_chart(cm):

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    return fig
