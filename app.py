# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from heart_model import predict, models, scaler, imputer  # your trained models

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
body { background-color: #f0f2f6; font-family: 'Segoe UI', sans-serif; }
h1 { color: #d62828; text-align: center; font-size: 3rem; }
h2 { color: #003049; }
.stButton>button { background-color: #d62828; color: white; font-weight: bold; border-radius: 10px; padding: 10px 20px; font-size: 16px; }
.card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.12); margin-bottom: 20px; }
main .block-container { max-height: 90vh; overflow-y: auto; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Title
# ------------------------------
st.markdown("<h1>❤️ Heart Disease Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#555;'>Predict heart disease risk with multiple ML models</p>", unsafe_allow_html=True)

# ------------------------------
# Sidebar for input
# ------------------------------
st.sidebar.markdown("<h2>Patient Details</h2>", unsafe_allow_html=True)

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting BP (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1])
restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0,1,2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0,1,2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.sidebar.selectbox("Thalassemia (1=Normal,2=Fixed,3=Reversible)", [1,2,3])

selected_models = st.sidebar.multiselect(
    "Select Models for Risk Prediction",
    list(models.keys()),
    default=list(models.keys())
)

input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# ------------------------------
# Risk Prediction
# ------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2>Risk Prediction</h2>", unsafe_allow_html=True)

if st.sidebar.button("Predict Risk"):

    # Prepare input
    input_array = np.array(input_data).reshape(1, -1)
    input_array = np.where(input_array == '?', np.nan, input_array).astype(float)
    input_array = imputer.transform(input_array)
    input_scaled = scaler.transform(input_array)

    # Compute average probability across selected models
    probs = []
    for name in selected_models:
        model = models[name]
        if name == "LightGBM":
            prob = model.predict(input_scaled)[0]
        else:
            prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else model.predict(input_scaled)[0]
        probs.append(float(prob))

    avg_risk = np.mean(probs)
    # Color coding
    if avg_risk >= 0.7:
        color = "#d62828"  # red
        risk_level = "High Risk"
    elif avg_risk >= 0.4:
        color = "#f77f00"  # orange
        risk_level = "Medium Risk"
    else:
        color = "#06d6a0"  # green
        risk_level = "Low Risk"

    # Display Risk Card
    st.markdown(f"""
        <div style="text-align:center; padding:30px; border-radius:15px; background:#fff; box-shadow:0 4px 12px rgba(0,0,0,0.12); margin-bottom:20px;">
            <h2 style="color:{color}; font-size:2.5rem;">{risk_level}</h2>
            <p style="font-size:1.5rem; color:#555;">Predicted Risk: {avg_risk*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

    # Pie chart / visual summary
    fig = go.Figure(go.Pie(
        values=[avg_risk, 1-avg_risk],
        labels=["Risk", "No Risk"],
        marker=dict(colors=[color, "#ccc"]),
        hole=0.5,
        sort=False
    ))
    fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=350, width=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Input summary
# ------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2>Input Summary</h2>", unsafe_allow_html=True)

feature_names = ['Age','Sex','CP','RestBP','Chol','FBS','RestECG','MaxHR','ExAng','Oldpeak','Slope','CA','Thal']
input_df = pd.DataFrame([input_data], columns=feature_names)
st.dataframe(input_df, height=200)  # scrollable table

st.markdown('</div>', unsafe_allow_html=True)
