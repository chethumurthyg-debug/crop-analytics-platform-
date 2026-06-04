import joblib
import pandas as pd
import numpy as np

# =====================================================
# MODEL PATHS
# =====================================================

MODEL_PATH = "models/crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# =====================================================
# LOAD ARTIFACTS
# =====================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# =====================================================
# FEATURE LIST
# =====================================================

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

def create_input_dataframe(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    return pd.DataFrame(
        [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=FEATURES
    )

# =====================================================
# SCALE INPUT
# =====================================================

def scale_input(input_df):

    return scaler.transform(
        input_df
    )

# =====================================================
# SINGLE RECOMMENDATION
# =====================================================

def recommend_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    input_df = create_input_dataframe(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    scaled = scale_input(
        input_df
    )

    prediction = model.predict(
        scaled
    )

    probabilities = model.predict_proba(
        scaled
    )[0]

    crop = encoder.inverse_transform(
        prediction
    )[0]

    confidence = (
        np.max(probabilities)
        * 100
    )

    return {
        "crop": crop,
        "confidence": round(
            confidence,
            2
        )
    }

# =====================================================
# TOP N RECOMMENDATIONS
# =====================================================

def top_recommendations(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    top_n=5
):

    input_df = create_input_dataframe(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    scaled = scale_input(
        input_df
    )

    probs = model.predict_proba(
        scaled
    )[0]

    indices = np.argsort(
        probs
    )[-top_n:][::-1]

    crops = encoder.inverse_transform(
        indices
    )

    scores = probs[
        indices
    ] * 100

    result = pd.DataFrame({
        "Crop": crops,
        "Confidence (%)":
            np.round(
                scores,
                2
            )
    })

    return result

# =====================================================
# FULL PROBABILITY TABLE
# =====================================================

def probability_table(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    input_df = create_input_dataframe(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    scaled = scale_input(
        input_df
    )

    probabilities = model.predict_proba(
        scaled
    )[0]

    crops = encoder.inverse_transform(
        np.arange(
            len(probabilities)
        )
    )

    prob_df = pd.DataFrame({
        "Crop": crops,
        "Probability (%)":
            probabilities * 100
    })

    prob_df = prob_df.sort_values(
        by="Probability (%)",
        ascending=False
    )

    return prob_df

# =====================================================
# SOIL HEALTH SCORE
# =====================================================

def soil_health_score(
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
    ) / 4

    return round(
        score * 100,
        2
    )

# =====================================================
# SOIL TYPE
# =====================================================

def soil_type(ph):

    if ph < 6:
        return "Acidic"

    elif ph > 7.5:
        return "Alkaline"

    return "Neutral"

# =====================================================
# TEMPERATURE CATEGORY
# =====================================================

def temperature_category(
    temperature
):

    if temperature < 20:
        return "Cool"

    elif temperature < 30:
        return "Moderate"

    return "Hot"

# =====================================================
# RAINFALL CATEGORY
# =====================================================

def rainfall_category(
    rainfall
):

    if rainfall < 100:
        return "Low"

    elif rainfall < 200:
        return "Moderate"

    return "High"

# =====================================================
# CLIMATE PROFILE
# =====================================================

def climate_profile(
    temperature,
    humidity,
    rainfall
):

    return {
        "Temperature Zone":
            temperature_category(
                temperature
            ),

        "Humidity":
            round(
                humidity,
                2
            ),

        "Rainfall Zone":
            rainfall_category(
                rainfall
            )
    }

# =====================================================
# SMART ADVISORY
# =====================================================

def generate_advisory(
    N,
    P,
    K,
    ph,
    temperature,
    humidity,
    rainfall
):

    advice = []

    if ph < 6:
        advice.append(
            "Soil is acidic. Consider liming practices."
        )

    elif ph > 7.5:
        advice.append(
            "Soil is alkaline. Organic matter amendments may help."
        )

    else:
        advice.append(
            "Soil pH is within the optimal range."
        )

    if N < 50:
        advice.append(
            "Nitrogen levels are relatively low."
        )

    if P < 30:
        advice.append(
            "Phosphorus supplementation may improve yield."
        )

    if K < 30:
        advice.append(
            "Potassium levels are below ideal range."
        )

    if rainfall > 250:
        advice.append(
            "Heavy rainfall conditions detected."
        )

    if temperature > 35:
        advice.append(
            "High temperature stress may impact crops."
        )

    if humidity > 85:
        advice.append(
            "High humidity may increase disease pressure."
        )

    return advice

# =====================================================
# COMPLETE ANALYSIS
# =====================================================

def complete_recommendation(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    recommendation = recommend_crop(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    soil_score = soil_health_score(
        N,
        P,
        K,
        ph
    )

    climate = climate_profile(
        temperature,
        humidity,
        rainfall
    )

    advisory = generate_advisory(
        N,
        P,
        K,
        ph,
        temperature,
        humidity,
        rainfall
    )

    return {
        "recommended_crop":
            recommendation["crop"],

        "confidence":
            recommendation["confidence"],

        "soil_health_score":
            soil_score,

        "soil_type":
            soil_type(ph),

        "climate":
            climate,

        "advisory":
            advisory
    }

# =====================================================
# RECOMMENDATION REPORT
# =====================================================

def generate_report(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    result = complete_recommendation(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    report = pd.DataFrame({
        "Metric": [
            "Recommended Crop",
            "Confidence",
            "Soil Health Score",
            "Soil Type",
            "Temperature Zone",
            "Rainfall Zone"
        ],
        "Value": [
            result["recommended_crop"],
            result["confidence"],
            result["soil_health_score"],
            result["soil_type"],
            result["climate"]["Temperature Zone"],
            result["climate"]["Rainfall Zone"]
        ]
    })

    return report

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    output = complete_recommendation(
        N=90,
        P=42,
        K=43,
        temperature=25,
        humidity=80,
        ph=6.5,
        rainfall=150
    )

    print("\nRecommendation")
    print("=" * 50)

    for k, v in output.items():
        print(k, ":", v)

    print("\nTop Recommendations")
    print(
        top_recommendations(
            90,
            42,
            43,
            25,
            80,
            6.5,
            150
        )
    )
