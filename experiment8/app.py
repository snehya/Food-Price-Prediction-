
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Food Price Forecasting",
    page_icon="🍅",
    layout="wide"
)

# -----------------------------
# Load model and dataset
# -----------------------------
MODEL_PATH = "experiment4/best_model.pkl"
DATA_PATH = "experiment2/cleaned_food_price_dataset.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

df["Dates"] = pd.to_datetime(df["Dates"])
df = df.sort_values("Dates").reset_index(drop=True)

# Forecasting-safe rolling feature
df["Tomato_PastRollingMean3"] = (
    df["Tomato"].shift(1).rolling(window=3).mean()
)

# Model features
features = [
    "Tomato_Lag1",
    "Tomato_Lag2",
    "Tomato_PastRollingMean3",
    "Month",
    "Quarter",
    "Week",
    "Month_Sin",
    "Month_Cos"
]

# -----------------------------
# Header
# -----------------------------
st.title("🍅 Essential Food Price Forecasting")
st.markdown("### Maharashtra Tomato Price Forecasting Dashboard")

st.write(
    "Forecast Tomato retail prices using the tuned Random Forest "
    "model developed in Experiment 4."
)

# -----------------------------
# Project Overview
# -----------------------------
st.subheader("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset Rows", "85")

with col2:
    st.metric("Target", "Tomato")

with col3:
    st.metric("Model", "Random Forest")

with col4:
    st.metric("Test RMSE", "2.7064")

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline RF RMSE", "2.8525")

with col2:
    st.metric("Tuned RF RMSE", "2.7064")

with col3:
    st.metric("RMSE Improvement", "5.12%")

st.write(
    "The tuned Random Forest achieved a lower test RMSE than "
    "the baseline Random Forest."
)

# -----------------------------
# Tomato Price Trend
# -----------------------------
st.subheader("📊 Tomato Price Trend")

chart_data = df[["Dates", "Tomato"]].set_index("Dates")

st.line_chart(chart_data)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔮 Tomato Price Prediction")

st.write(
    "Enter historical price and calendar information to generate "
    "a Tomato price prediction."
)

col1, col2 = st.columns(2)

with col1:
    lag1 = st.number_input(
        "Tomato Price — Previous Observation",
        min_value=0.0,
        value=30.0,
        step=0.5
    )

    lag2 = st.number_input(
        "Tomato Price — Two Observations Ago",
        min_value=0.0,
        value=30.0,
        step=0.5
    )

    rolling_mean = st.number_input(
        "Past 3-Observation Average",
        min_value=0.0,
        value=30.0,
        step=0.5
    )

with col2:
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=4,
        step=1
    )

    quarter = ((month - 1) // 3) + 1

    week = st.number_input(
        "Week of Year",
        min_value=1,
        max_value=53,
        value=15,
        step=1
    )

    st.write(f"**Calculated Quarter:** {quarter}")

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

if st.button("Predict Tomato Price", type="primary"):

    input_data = pd.DataFrame([{
        "Tomato_Lag1": lag1,
        "Tomato_Lag2": lag2,
        "Tomato_PastRollingMean3": rolling_mean,
        "Month": month,
        "Quarter": quarter,
        "Week": week,
        "Month_Sin": month_sin,
        "Month_Cos": month_cos
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Tomato Price: ₹{prediction:.2f}"
    )

# -----------------------------
# XAI / SHAP
# -----------------------------
st.subheader("🧠 Explainable AI — SHAP")

st.write(
    "SHAP shows how much each feature contributes to the "
    "Random Forest model's predictions. Larger absolute SHAP "
    "values indicate greater influence on the model output."
)

# Remove rows with missing forecasting features
shap_data = df[features].dropna()

if len(shap_data) > 0:

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_data)

    shap_importance = pd.DataFrame({
        "Feature": features,
        "Mean Absolute SHAP": np.abs(shap_values).mean(axis=0)
    }).sort_values(
        "Mean Absolute SHAP",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        shap_importance["Feature"],
        shap_importance["Mean Absolute SHAP"]
    )

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title("Global Feature Importance")

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        "The chart shows which features have the strongest "
        "overall influence on the model's predictions."
    )


# -----------------------------
# Indicative Data Drift Check
# -----------------------------
st.subheader("📉 Indicative Data Drift Check")

st.write(
    "This simple check compares the average Tomato price in the "
    "training period with the test period. It is an indicative "
    "check and is not a production drift-monitoring system."
)

# Same chronological split used in Experiment 4
drift_df = df[["Dates", "Tomato"]].dropna().reset_index(drop=True)

split_index = int(len(drift_df) * 0.8)

train_period = drift_df.iloc[:split_index]
test_period = drift_df.iloc[split_index:]

train_mean = train_period["Tomato"].mean()
test_mean = test_period["Tomato"].mean()

mean_difference = test_mean - train_mean

if train_mean != 0:
    mean_difference_pct = (
        abs(mean_difference) / train_mean
    ) * 100
else:
    mean_difference_pct = 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Mean",
        f"₹{train_mean:.2f}"
    )

with col2:
    st.metric(
        "Test Mean",
        f"₹{test_mean:.2f}"
    )

with col3:
    st.metric(
        "Mean Difference",
        f"{mean_difference_pct:.2f}%"
    )

st.write(
    "A change in the average price distribution can indicate "
    "that the test period differs from the training period. "
    "Further monitoring would be required in a production system."
)

# -----------------------------
# Recent Prices
# -----------------------------
st.subheader("🔎 Recent Tomato Prices")

recent_data = df[
    ["Dates", "Tomato"]
].tail(10).sort_values(
    "Dates",
    ascending=False
)

st.dataframe(
    recent_data,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Model Information
# -----------------------------
st.subheader("🤖 Model Information")

st.write(
    "The final model is a tuned Random Forest Regressor "
    "trained using historical Tomato prices and calendar-based "
    "features."
)

st.write("**Model Features:**")

st.code("""
Tomato_Lag1
Tomato_Lag2
Tomato_PastRollingMean3
Month
Quarter
Week
Month_Sin
Month_Cos
""")

st.info(
    "This dashboard is an experimental decision-support tool. "
    "Predictions are not guaranteed future prices."
)
