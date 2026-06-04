import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =====================================================
# PATHS
# =====================================================

SCALER_PATH = "models/scaler.pkl"

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
# LOAD SCALER
# =====================================================

def load_scaler():
    """
    Load trained scaler from disk
    """

    return joblib.load(SCALER_PATH)

# =====================================================
# SCALE DATAFRAME
# =====================================================

def scale_dataframe(df):

    scaler = load_scaler()

    scaled = scaler.transform(df)

    return pd.DataFrame(
        scaled,
        columns=df.columns
    )

# =====================================================
# SCALE SINGLE RECORD
# =====================================================

def scale_input(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    scaler = load_scaler()

    data = pd.DataFrame(
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

    scaled = scaler.transform(data)

    return scaled

# =====================================================
# FIT NEW SCALER
# =====================================================

def train_scaler(df):

    scaler = StandardScaler()

    scaler.fit(df)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    return scaler

# =====================================================
# INVERSE TRANSFORM
# =====================================================

def inverse_scale(scaled_data):

    scaler = load_scaler()

    return scaler.inverse_transform(
        scaled_data
    )

# =====================================================
# FEATURE STATISTICS
# =====================================================

def get_scaler_stats():

    scaler = load_scaler()

    stats = pd.DataFrame({
        "Feature": FEATURES,
        "Mean": scaler.mean_,
        "Scale": scaler.scale_
    })

    return stats

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    scaled = scale_input(
        N=90,
        P=42,
        K=43,
        temperature=25,
        humidity=80,
        ph=6.5,
        rainfall=150
    )

    print("Scaled Values:")
    print(scaled)

    print("\nScaler Statistics:")
    print(get_scaler_stats())
