import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.decomposition import PCA

import shap

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Model Insights",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------------------
# LOAD CSS
# -----------------------------------------------------

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/Crop_recommendation.csv"
    )

df = load_data()

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
        🤖 Model Intelligence Dashboard
    </div>

    <div class='sub-title'>
        Explainable AI, Model Evaluation & Insights
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# PREPARE DATA
# -----------------------------------------------------

X = df.drop(
    "label",
    axis=1
)

y = df["label"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------

@st.cache_resource
def train_model():

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model

model = train_model()

pred = model.predict(X_test)

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------

accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    pred,
    average="weighted"
)

# -----------------------------------------------------
# KPI SECTION
# -----------------------------------------------------

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

c2.metric(
    "Precision",
    f"{precision*100:.2f}%"
)

c3.metric(
    "Recall",
    f"{recall*100:.2f}%"
)

c4.metric(
    "F1 Score",
    f"{f1*100:.2f}%"
)

st.divider()

# -----------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------

st.subheader(
    "📊 Feature Importance"
)

importance_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# CONFUSION MATRIX
# -----------------------------------------------------

st.subheader(
    "🎯 Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    pred
)

fig = px.imshow(
    cm,
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# CROSS VALIDATION
# -----------------------------------------------------

st.subheader(
    "🔄 Cross Validation"
)

cv_scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=5
)

cv_df = pd.DataFrame({
    "Fold":[1,2,3,4,5],
    "Score":cv_scores
})

fig = px.line(
    cv_df,
    x="Fold",
    y="Score",
    markers=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.success(
    f"Average CV Score: {cv_scores.mean():.4f}"
)

# -----------------------------------------------------
# LEARNING CURVE
# -----------------------------------------------------

st.subheader(
    "📈 Learning Curve"
)

sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y_encoded,
    cv=5
)

learning_df = pd.DataFrame({
    "Training Size":sizes,
    "Train Score":train_scores.mean(axis=1),
    "Validation Score":val_scores.mean(axis=1)
})

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=learning_df["Training Size"],
        y=learning_df["Train Score"],
        mode="lines+markers",
        name="Train"
    )
)

fig.add_trace(
    go.Scatter(
        x=learning_df["Training Size"],
        y=learning_df["Validation Score"],
        mode="lines+markers",
        name="Validation"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# PCA ANALYSIS
# -----------------------------------------------------

st.subheader(
    "🔍 PCA Crop Clusters"
)

scaler = StandardScaler()

scaled = scaler.fit_transform(X)

pca = PCA(
    n_components=2
)

components = pca.fit_transform(
    scaled
)

pca_df = pd.DataFrame(
    components,
    columns=["PC1","PC2"]
)

pca_df["Crop"] = y

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Crop",
    title="Crop Clustering"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# CLASS DISTRIBUTION
# -----------------------------------------------------

st.subheader(
    "🌾 Crop Class Distribution"
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
    color="Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# SHAP ANALYSIS
# -----------------------------------------------------

st.subheader(
    "🧠 SHAP Explainability"
)

sample_size = min(
    200,
    len(X_test)
)

sample_X = X_test.iloc[
    :sample_size
]

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    sample_X
)

if isinstance(shap_values, list):

    shap_importance = np.abs(
        shap_values[0]
    ).mean(axis=0)

else:

    shap_importance = np.abs(
        shap_values
    ).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":shap_importance
})

shap_df = (
    shap_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

fig = px.bar(
    shap_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="SHAP Feature Impact"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------------------------
# CLASSIFICATION REPORT
# -----------------------------------------------------

st.subheader(
    "📋 Classification Report"
)

report = classification_report(
    y_test,
    pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# -----------------------------------------------------
# AI INSIGHTS
# -----------------------------------------------------

st.subheader(
    "🤖 AI Model Insights"
)

top_feature = (
    importance_df.iloc[0]["Feature"]
)

insights = [
    f"Most influential feature is {top_feature}.",
    f"Model accuracy is {accuracy*100:.2f}%.",
    f"Average cross-validation score is {cv_scores.mean()*100:.2f}%.",
    "Model demonstrates strong crop classification capability.",
    "Feature importance indicates environmental conditions strongly impact recommendations."
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

# -----------------------------------------------------
# EXPORT REPORT
# -----------------------------------------------------

st.subheader(
    "⬇ Download Model Report"
)

report_export = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "CV Score"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        cv_scores.mean()
    ]
})

csv = report_export.to_csv(
    index=False
)

st.download_button(
    label="Download Report",
    data=csv,
    file_name="model_insights_report.csv",
    mime="text/csv"
)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown(
    """
    <div class='footer'>
        Smart Crop Recommendation Platform |
        Model Intelligence Module
    </div>
    """,
    unsafe_allow_html=True
)
