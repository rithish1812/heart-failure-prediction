import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Heart Prediction", layout="wide")

st.warning("⚠️ Educational tool only")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("heart_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ Need heart_model.pkl file")
        st.stop()

model = load_model()

st.title("❤️ Heart Failure Prediction")

# Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 80, 50)
    cholesterol = st.number_input("Cholesterol mg/dL", 100, 400, 200)
    max_hr = st.number_input("Max HR bpm", 70, 200, 150)

with col2:
    systolic = st.number_input("Systolic BP", 90, 200, 120)
    diastolic = st.number_input("Diastolic BP", 60, 120, 80)
    fasting_bs = 1 if st.selectbox("Fasting BS >120?", ["No", "Yes"]) == "Yes" else 0
    angina = 1 if st.selectbox("Exercise Angina?", ["No", "Yes"]) == "Yes" else 0

if st.button("🔍 PREDICT", use_container_width=True):
    # Create data
    data = pd.DataFrame({
        "Age": [age], "Cholesterol": [cholesterol], "MaxHR": [max_hr],
        "BP_Systolic": [systolic], "BP_Diastolic": [diastolic],
        "FastingBS": [fasting_bs], "ExerciseAngina": [angina]
    })
    
    # Predict
    risk = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100 if hasattr(model, 'predict_proba') else 50
    
    # Results
    st.markdown("---")
    col1, col2 = st.columns([2,1])
    with col1:
        st.metric("Risk Score", f"{prob:.1f}%")
        if risk == 1:
            st.error("🚨 HIGH RISK - See doctor immediately")
        else:
            st.success("✅ LOW RISK")
    
    # Simple text chart (No plotting libraries!)
    st.subheader("📊 Top Risk Factors")
    factors = ["Age", "Cholesterol", "Max HR", "Blood Pressure"]
    for i, f in enumerate(factors[:3], 1):
        st.write(f"{i}. **{f}** - Monitor regularly")
    
    # History
    st.session_state.history = st.session_state.get('history', [])
    st.session_state.history.append({
        'Time': datetime.now().strftime("%H:%M:%S"),
        'Age': age, 'Risk': f"{prob:.1f}%"
    })
    st.rerun()

# History
if 'history' in st.session_state and st.session_state.history:
    st.subheader("📋 Prediction History")
    df = pd.DataFrame(st.session_state.history[-10:])  # Last 10
    st.dataframe(df)