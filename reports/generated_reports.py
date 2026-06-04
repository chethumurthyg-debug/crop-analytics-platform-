import os
import pandas as pd
from datetime import datetime

# =====================================================
# REPORT DIRECTORY
# =====================================================

REPORT_DIR = "generated_reports"

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

# =====================================================
# TIMESTAMP
# =====================================================

def timestamp():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

# =====================================================
# SAVE CSV REPORT
# =====================================================

def save_csv_report(
    dataframe,
    report_name
):

    file_path = os.path.join(
        REPORT_DIR,
        f"{report_name}_{timestamp()}.csv"
    )

    dataframe.to_csv(
        file_path,
        index=False
    )

    return file_path

# =====================================================
# SAVE EXCEL REPORT
# =====================================================

def save_excel_report(
    sheets,
    report_name
):

    file_path = os.path.join(
        REPORT_DIR,
        f"{report_name}_{timestamp()}.xlsx"
    )

    with pd.ExcelWriter(
        file_path,
        engine="openpyxl"
    ) as writer:

        for sheet_name, df in sheets.items():

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    return file_path

# =====================================================
# EXECUTIVE REPORT
# =====================================================

def executive_report(df):

    report = pd.DataFrame({
        "Metric":[
            "Total Records",
            "Total Crops",
            "Average Temperature",
            "Average Humidity",
            "Average Rainfall"
        ],
        "Value":[
            len(df),
            df["label"].nunique(),
            round(df["temperature"].mean(),2),
            round(df["humidity"].mean(),2),
            round(df["rainfall"].mean(),2)
        ]
    })

    return save_csv_report(
        report,
        "executive_summary"
    )

# =====================================================
# SOIL REPORT
# =====================================================

def soil_report(df):

    report = pd.DataFrame({
        "Metric":[
            "Average N",
            "Average P",
            "Average K",
            "Average pH"
        ],
        "Value":[
            round(df["N"].mean(),2),
            round(df["P"].mean(),2),
            round(df["K"].mean(),2),
            round(df["ph"].mean(),2)
        ]
    })

    return save_csv_report(
        report,
        "soil_analysis"
    )

# =====================================================
# CLIMATE REPORT
# =====================================================

def climate_report(df):

    report = pd.DataFrame({
        "Metric":[
            "Temperature",
            "Humidity",
            "Rainfall"
        ],
        "Average":[
            round(df["temperature"].mean(),2),
            round(df["humidity"].mean(),2),
            round(df["rainfall"].mean(),2)
        ],
        "Minimum":[
            round(df["temperature"].min(),2),
            round(df["humidity"].min(),2),
            round(df["rainfall"].min(),2)
        ],
        "Maximum":[
            round(df["temperature"].max(),2),
            round(df["humidity"].max(),2),
            round(df["rainfall"].max(),2)
        ]
    })

    return save_csv_report(
        report,
        "climate_analysis"
    )

# =====================================================
# PREDICTION REPORT
# =====================================================

def prediction_report(
    recommendation,
    confidence,
    soil_score,
    soil_type
):

    report = pd.DataFrame({
        "Metric":[
            "Recommended Crop",
            "Confidence %",
            "Soil Health Score",
            "Soil Type"
        ],
        "Value":[
            recommendation,
            confidence,
            soil_score,
            soil_type
        ]
    })

    return save_csv_report(
        report,
        "crop_prediction"
    )

# =====================================================
# MODEL REPORT
# =====================================================

def model_report(
    accuracy,
    precision,
    recall,
    f1
):

    report = pd.DataFrame({
        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value":[
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    return save_csv_report(
        report,
        "model_insights"
    )

# =====================================================
# COMPLETE EXCEL REPORT
# =====================================================

def complete_excel_report(
    executive_df,
    soil_df,
    climate_df,
    prediction_df,
    model_df
):

    sheets = {
        "Executive": executive_df,
        "Soil": soil_df,
        "Climate": climate_df,
        "Prediction": prediction_df,
        "Model": model_df
    }

    return save_excel_report(
        sheets,
        "complete_agriculture_report"
    )

# =====================================================
# LIST REPORTS
# =====================================================

def list_reports():

    files = []

    for file in os.listdir(
        REPORT_DIR
    ):
        files.append(file)

    return sorted(files)

# =====================================================
# DELETE REPORTS
# =====================================================

def delete_reports():

    for file in os.listdir(
        REPORT_DIR
    ):

        path = os.path.join(
            REPORT_DIR,
            file
        )

        os.remove(path)

    return True

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print(
        "Report Generator Ready"
    )
