import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import io
import plotly.express as px  # ✅ Works perfectly in Streamlit Cloud

# Page setup
st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️", layout="wide")

st.warning("⚠️ Disclaimer: Educational use only. Consult a doctor.")

# Load model (same as yours)
@st.cache_resource
def load_model():
    try:
        with open("heart_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ Model file missing")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        with open("scaler.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model = load_model()
scaler = load_scaler()

# Your original features
EXPECTED_FEATURES = ["Age", "Cholesterol", "MaxHR", "BP_Systolic", "BP_Diastolic", "FastingBS", "ExerciseAngina"]

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("❤️ Heart Failure Prediction")

# Inputs (same as yours)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 50)
    cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    max_hr = st.number_input("Max Heart Rate", 50, 250, 150)

with col2:
    bp_systolic = st.number_input("Systolic BP", 80, 250, 120)
    bp_diastolic = st.number_input("Diastolic BP", 40, 150, 80)
    fasting_bs = st.selectbox("FastingBS >120?", ["No", "Yes"])

exercise_angina = st.selectbox("Exercise Angina?", ["No", "Yes"])

if st.button("🔍 Predict", use_container_width=True):
    # Convert categorical
    input_data = pd.DataFrame({
        "Age": [age], "Cholesterol": [cholesterol], "MaxHR": [max_hr],
        "BP_Systolic": [bp_systolic], "BP_Diastolic": [bp_diastolic],
        "FastingBS": [1 if fasting_bs=="Yes" else 0],
        "ExerciseAngina": [1 if exercise_angina=="Yes" else 0]
    })
    
    # Scale if available
    if scaler:
        input_data = pd.DataFrame(scaler.transform(input_data), columns=EXPECTED_FEATURES)
    
    # Predict
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]*100 if hasattr(model, 'predict_proba') else 0
    
    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk", f"{prob:.1f}%")
        color = "🟢 LOW" if pred == 0 else "🔴 HIGH"
        st.subheader(color)
    
    # ✅ PLOTLY CHART (No matplotlib!)
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Feature': EXPECTED_FEATURES,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(imp_df, y='Feature', x='Importance', orientation='h',
                    title="Feature Importance", color='Importance')
        st.plotly_chart(fig, use_container_width=True)
    
    # Save history
    st.session_state.history.append({
        'Time': datetime.now().strftime("%H:%M"),
        'Risk': f"{prob:.1f}%", 'Result': 'HIGH' if pred else 'LOW'
    })

# History table
if st.session_state.history:
    st.subheader("📋 History")
    st.dataframe(pd.DataFrame(st.session_state.history))