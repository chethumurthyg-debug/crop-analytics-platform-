import pandas as pd
import numpy as np

# =====================================================
# EXECUTIVE DASHBOARD INSIGHTS
# =====================================================

def generate_executive_insights(df):

    insights = []

    total_records = len(df)
    total_crops = df["label"].nunique()

    top_crop = (
        df["label"]
        .value_counts()
        .idxmax()
    )

    avg_temp = round(
        df["temperature"].mean(),
        2
    )

    avg_humidity = round(
        df["humidity"].mean(),
        2
    )

    avg_rainfall = round(
        df["rainfall"].mean(),
        2
    )

    insights.append(
        f"Dataset contains {total_records:,} records."
    )

    insights.append(
        f"Dataset covers {total_crops} crop categories."
    )

    insights.append(
        f"Most frequent crop is {top_crop}."
    )

    insights.append(
        f"Average temperature is {avg_temp}°C."
    )

    insights.append(
        f"Average humidity is {avg_humidity}%."
    )

    insights.append(
        f"Average rainfall is {avg_rainfall:.2f} mm."
    )

    return insights

# =====================================================
# SOIL HEALTH SCORE
# =====================================================

def calculate_soil_health(
    N,
    P,
    K,
    ph
):

    score = (
        (
            N / 140
        ) +
        (
            P / 145
        ) +
        (
            K / 205
        ) +
        (
            ph / 14
        )
    ) / 4 * 100

    return round(score, 2)

# =====================================================
# SOIL CLASSIFICATION
# =====================================================

def classify_soil(ph):

    if ph < 6:
        return "Acidic"

    elif ph > 7.5:
        return "Alkaline"

    return "Neutral"

# =====================================================
# SOIL ANALYSIS INSIGHTS
# =====================================================

def generate_soil_insights(df):

    insights = []

    avg_n = df["N"].mean()
    avg_p = df["P"].mean()
    avg_k = df["K"].mean()
    avg_ph = df["ph"].mean()

    acidic = len(df[df["ph"] < 6])

    neutral = len(
        df[
            (df["ph"] >= 6)
            &
            (df["ph"] <= 7.5)
        ]
    )

    alkaline = len(
        df[df["ph"] > 7.5]
    )

    insights.append(
        f"Average Nitrogen level is {avg_n:.2f}."
    )

    insights.append(
        f"Average Phosphorus level is {avg_p:.2f}."
    )

    insights.append(
        f"Average Potassium level is {avg_k:.2f}."
    )

    insights.append(
        f"Average soil pH is {avg_ph:.2f}."
    )

    if neutral > acidic and neutral > alkaline:
        insights.append(
            "Most samples belong to neutral soil conditions."
        )

    if avg_n > avg_p:
        insights.append(
            "Nitrogen levels exceed phosphorus levels."
        )

    if avg_k > 40:
        insights.append(
            "Potassium levels are favorable for crop growth."
        )

    return insights

# =====================================================
# CLIMATE ZONE
# =====================================================

def climate_zone(
    temperature,
    rainfall
):

    if temperature < 20:
        temp_zone = "Cool"

    elif temperature < 30:
        temp_zone = "Moderate"

    else:
        temp_zone = "Hot"

    if rainfall < 100:
        rain_zone = "Low Rainfall"

    elif rainfall < 200:
        rain_zone = "Moderate Rainfall"

    else:
        rain_zone = "High Rainfall"

    return {
        "temperature_zone": temp_zone,
        "rainfall_zone": rain_zone
    }

# =====================================================
# CLIMATE INSIGHTS
# =====================================================

def generate_climate_insights(df):

    insights = []

    avg_temp = df["temperature"].mean()

    avg_humidity = df["humidity"].mean()

    avg_rainfall = df["rainfall"].mean()

    insights.append(
        f"Average temperature is {avg_temp:.2f}°C."
    )

    insights.append(
        f"Average humidity is {avg_humidity:.2f}%."
    )

    insights.append(
        f"Average rainfall is {avg_rainfall:.2f} mm."
    )

    if avg_temp > 25:
        insights.append(
            "Warm climatic conditions dominate the dataset."
        )

    if avg_humidity > 65:
        insights.append(
            "Humidity levels support moisture-demanding crops."
        )

    if avg_rainfall > 100:
        insights.append(
            "Rainfall levels are generally sufficient."
        )

    return insights

# =====================================================
# CROP PROFILE
# =====================================================

def crop_profile(df, crop_name):

    crop_df = df[
        df["label"] == crop_name
    ]

    if crop_df.empty:
        return None

    profile = {
        "crop": crop_name,
        "avg_nitrogen":
            round(crop_df["N"].mean(), 2),

        "avg_phosphorus":
            round(crop_df["P"].mean(), 2),

        "avg_potassium":
            round(crop_df["K"].mean(), 2),

        "avg_temperature":
            round(crop_df["temperature"].mean(), 2),

        "avg_humidity":
            round(crop_df["humidity"].mean(), 2),

        "avg_ph":
            round(crop_df["ph"].mean(), 2),

        "avg_rainfall":
            round(crop_df["rainfall"].mean(), 2)
    }

    return profile

# =====================================================
# MODEL PERFORMANCE INSIGHTS
# =====================================================

def model_insights(
    accuracy,
    precision,
    recall,
    f1
):

    insights = []

    insights.append(
        f"Model Accuracy: {accuracy*100:.2f}%"
    )

    insights.append(
        f"Precision: {precision*100:.2f}%"
    )

    insights.append(
        f"Recall: {recall*100:.2f}%"
    )

    insights.append(
        f"F1 Score: {f1*100:.2f}%"
    )

    if accuracy > 0.95:
        insights.append(
            "Model demonstrates excellent predictive performance."
        )

    elif accuracy > 0.85:
        insights.append(
            "Model performance is strong."
        )

    else:
        insights.append(
            "Additional model tuning may improve results."
        )

    return insights

# =====================================================
# FEATURE IMPORTANCE INSIGHTS
# =====================================================

def feature_importance_insights(
    importance_df
):

    insights = []

    top_feature = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .iloc[0]
    )

    insights.append(
        f"Most influential feature is {top_feature['Feature']}."
    )

    insights.append(
        f"Importance Score: {top_feature['Importance']:.4f}"
    )

    return insights

# =====================================================
# PREDICTION INSIGHTS
# =====================================================

def prediction_insights(
    crop,
    confidence,
    N,
    P,
    K,
    ph,
    temperature,
    humidity,
    rainfall
):

    insights = []

    insights.append(
        f"Recommended crop is {crop}."
    )

    insights.append(
        f"Prediction confidence is {confidence:.2f}%."
    )

    if ph < 6:
        insights.append(
            "Soil condition is acidic."
        )

    elif ph > 7.5:
        insights.append(
            "Soil condition is alkaline."
        )

    else:
        insights.append(
            "Soil pH is within the optimal range."
        )

    if rainfall > 200:
        insights.append(
            "High rainfall environment detected."
        )

    if humidity > 80:
        insights.append(
            "Humidity is very high."
        )

    if temperature > 35:
        insights.append(
            "High-temperature conditions detected."
        )

    if N > 100:
        insights.append(
            "Nitrogen concentration is high."
        )

    if P > 100:
        insights.append(
            "Phosphorus concentration is high."
        )

    if K > 100:
        insights.append(
            "Potassium concentration is high."
        )

    return insights

# =====================================================
# SUMMARY REPORT GENERATOR
# =====================================================

def generate_summary_report(df):

    report = {
        "Total Records":
            len(df),

        "Total Crops":
            df["label"].nunique(),

        "Average Temperature":
            round(df["temperature"].mean(), 2),

        "Average Humidity":
            round(df["humidity"].mean(), 2),

        "Average Rainfall":
            round(df["rainfall"].mean(), 2),

        "Average Nitrogen":
            round(df["N"].mean(), 2),

        "Average Phosphorus":
            round(df["P"].mean(), 2),

        "Average Potassium":
            round(df["K"].mean(), 2),

        "Average pH":
            round(df["ph"].mean(), 2)
    }

    return pd.DataFrame(
        report.items(),
        columns=[
            "Metric",
            "Value"
        ]
    )

# =====================================================
# INSIGHT CARD HELPER
# =====================================================

def render_insight_html(text):

    return f"""
    <div class='insight-card'>
        ✅ {text}
    </div>
    """
