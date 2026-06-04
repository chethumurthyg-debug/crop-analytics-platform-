import joblib
import numpy as np
import pandas as pd

# =====================================================
# LOAD TRAINED FILES
# =====================================================

MODEL_PATH = "models/crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# =====================================================
# FEATURE NAMES
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
# SINGLE PREDICTION
# =====================================================

def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Predict best crop
    """

    input_data = pd.DataFrame(
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

    scaled_data = scaler.transform(
        input_data
    )

    prediction = model.predict(
        scaled_data
    )

    crop = encoder.inverse_transform(
        prediction
    )[0]

    confidence = (
        np.max(
            model.predict_proba(
                scaled_data
            )
        ) * 100
    )

    return {
        "crop": crop,
        "confidence": round(confidence, 2)
    }

# =====================================================
# TOP 3 RECOMMENDATIONS
# =====================================================

def get_top_recommendations(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    top_n=3
):

    input_data = pd.DataFrame(
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

    scaled_data = scaler.transform(
        input_data
    )

    probabilities = model.predict_proba(
        scaled_data
    )[0]

    top_indices = np.argsort(
        probabilities
    )[-top_n:][::-1]

    crops = encoder.inverse_transform(
        top_indices
    )

    scores = probabilities[
        top_indices
    ] * 100

    result = pd.DataFrame({
        "Crop": crops,
        "Confidence (%)": np.round(
            scores,
            2
        )
    })

    return result

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def get_feature_importance():

    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance
    })

    df = df.sort_values(
        by="Importance",
        ascending=False
    )

    return df

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
# SOIL CLASSIFICATION
# =====================================================

def classify_soil(ph):

    if ph < 6:
        return "Acidic"

    elif ph > 7.5:
        return "Alkaline"

    else:
        return "Neutral"

# =====================================================
# CLIMATE CLASSIFICATION
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
# COMPLETE ANALYSIS
# =====================================================

def full_analysis(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    prediction = predict_crop(
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

    soil_type = classify_soil(
        ph
    )

    climate = climate_zone(
        temperature,
        rainfall
    )

    return {
        "recommended_crop":
            prediction["crop"],

        "confidence":
            prediction["confidence"],

        "soil_health_score":
            soil_score,

        "soil_type":
            soil_type,

        "temperature_zone":
            climate["temperature_zone"],

        "rainfall_zone":
            climate["rainfall_zone"]
    }

# =====================================================
# TEST RUN
# =====================================================

if __name__ == "__main__":

    result = full_analysis(
        N=90,
        P=42,
        K=43,
        temperature=25,
        humidity=80,
        ph=6.5,
        rainfall=150
    )

    print("\nCrop Recommendation")
    print("-------------------")

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("\nTop Recommendations")

    print(
        get_top_recommendations(
            90,
            42,
            43,
            25,
            80,
            6.5,
            150
        )
    )
