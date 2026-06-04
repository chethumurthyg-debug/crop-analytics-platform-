# utils/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load crop recommendation dataset.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate dataset structure.
    """
    required_columns = FEATURE_COLUMNS + ["label"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by handling duplicates and missing values.
    """
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Drop rows with missing labels
    if "label" in df.columns:
        df.dropna(subset=["label"], inplace=True)

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Split dataframe into features and target.
    """
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    return X, y


def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """
    Train-test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def fit_scaler(X_train):
    """
    Fit StandardScaler on training data.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler


def scale_features(
    scaler,
    X_train,
    X_test=None
):
    """
    Scale train and test features.
    """
    X_train_scaled = scaler.transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    return X_train_scaled


def save_scaler(
    scaler,
    save_path="models/scaler.pkl"
):
    """
    Save scaler object.
    """
    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(scaler, save_path)


def load_scaler(
    scaler_path="models/scaler.pkl"
):
    """
    Load scaler object.
    """
    return joblib.load(scaler_path)


def preprocess_training_data(
    csv_path: str,
    scaler_save_path="models/scaler.pkl"
):
    """
    Complete preprocessing pipeline for training.
    """

    df = load_dataset(csv_path)

    validate_dataset(df)

    df = clean_data(df)

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    scaler = fit_scaler(X_train)

    X_train_scaled, X_test_scaled = scale_features(
        scaler,
        X_train,
        X_test
    )

    save_scaler(
        scaler,
        scaler_save_path
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


def preprocess_user_input(
    n,
    p,
    k,
    temperature,
    humidity,
    ph,
    rainfall,
    scaler
):
    """
    Preprocess user input before prediction.
    """

    input_df = pd.DataFrame(
        [[
            n,
            p,
            k,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=FEATURE_COLUMNS
    )

    scaled_input = scaler.transform(input_df)

    return scaled_input


def get_feature_ranges(df):
    """
    Useful for Streamlit sliders.
    """

    ranges = {}

    for col in FEATURE_COLUMNS:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean())
        }

    return ranges


def dataset_summary(df):
    """
    Dataset statistics.
    """

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "crops": df["label"].nunique(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }

    return summary


if __name__ == "__main__":

    DATA_PATH = "data/Crop_recommendation.csv"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_training_data(DATA_PATH)

    print("Preprocessing completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
