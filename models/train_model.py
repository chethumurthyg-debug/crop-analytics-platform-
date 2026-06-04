import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# LOAD DATA
# =====================================================

DATA_PATH = "../data/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print(f"Shape: {df.shape}")

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop("label", axis=1)

y = df["label"]

# =====================================================
# LABEL ENCODING
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================================
# MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("Training Model...")

model.fit(
    X_train,
    y_train
)

# =====================================================
# EVALUATION
# =====================================================

train_score = model.score(
    X_train,
    y_train
)

test_score = model.score(
    X_test,
    y_test
)

print("\nTraining Accuracy:", round(train_score,4))
print("Testing Accuracy :", round(test_score,4))

# =====================================================
# SAVE FILES
# =====================================================

joblib.dump(
    model,
    "crop_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("\nFiles Saved Successfully")

print("crop_model.pkl")
print("scaler.pkl")
print("label_encoder.pkl")
