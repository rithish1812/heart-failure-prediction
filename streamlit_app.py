import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap  # pip install shap

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Heart Failure Prediction - Complete Edition",
    page_icon="❤️",
    layout="wide"
)

# ==================== DISCLAIMER ====================
st.warning(
    "⚠️ Disclaimer: This tool is for educational purposes only. "
    "It is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider."
)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    try:
        with open("heart_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'heart_model.pkl' not found. Please place it in the same folder as this app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_scaler():
    """Load the fitted scaler for normalization"""
    try:
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.warning("⚠️ Scaler file 'scaler.pkl' not found. Predictions may be inaccurate without normalization.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading scaler: {e}")
        return None

model = load_model()
scaler = load_scaler()

# ==================== EXPECTED FEATURES (Extended) ====================
EXPECTED_FEATURES = [
    "Age", "Cholesterol", "MaxHR", "BP_Systolic", "BP_Diastolic", 
    "FastingBS", "ExerciseAngina", "Oldpeak", "RestingECG_Normal", 
    "RestingECG_ST", "ST_Slope_Up", "ST_Slope_Flat"
]  # Adjust based on your model's actual features

# ==================== SESSION STATE ====================
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# ==================== TITLE ====================
st.title("❤️ Heart Failure Prediction System - Complete Edition")
st.markdown("**மேம்படுத்தப்பட்ட version - Plotly charts, SHAP explanations, Extra features**")

# ==================== INPUT FIELDS ====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Metrics")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
    max_hr = st.number_input("Maximum Heart Rate (bpm)", min_value=50, max_value=250, value=150)

with col2:
    st.subheader("Blood Pressure")
    bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
    bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])

st.subheader("Advanced Metrics")
col3, col4, col5 = st.columns(3)
with col3:
    exercise_angina = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
with col4:
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
with col5:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ==================== PREPROCESSING ====================
fasting_bs_binary = 1 if fasting_bs == "Yes" else 0
exercise_angina_binary = 1 if exercise_angina == "Yes" else 0

# One-hot encoding for categoricals
resting_ecg_normal = 1 if resting_ecg == "Normal" else 0
resting_ecg_st = 1 if resting_ecg == "ST" else 0
st_slope_up = 1 if st_slope == "Up" else 0
st_slope_flat = 1 if st_slope == "Flat" else 0

# ==================== VALIDATION ====================
def validate_inputs(age, bp_systolic, bp_diastolic, cholesterol, max_hr, oldpeak):
    errors = []
    if bp_systolic < bp_diastolic:
        errors.append("❌ Systolic > Diastolic BP")
    if max_hr > (220 - age):
        errors.append(f"⚠️ Max HR high for age {age}")
    if oldpeak > 4.0:
        errors.append("⚠️ Oldpeak unusually high")
    return errors

# ==================== PREDICTION ====================
if st.button("🔍 Predict Risk", use_container_width=True):
    errors = validate_inputs(age, bp_systolic, bp_diastolic, cholesterol, max_hr, oldpeak)
    
    if errors:
        st.warning("⚠️ Issues:")
        for error in errors:
            st.write(error)
    
    # Create full feature dataframe (adjust columns to match your model)
    input_data = pd.DataFrame({
        "Age": [age], "Cholesterol": [cholesterol], "MaxHR": [max_hr],
        "BP_Systolic": [bp_systolic], "BP_Diastolic": [bp_diastolic],
        "FastingBS": [fasting_bs_binary], "ExerciseAngina": [exercise_angina_binary],
        "Oldpeak": [oldpeak], "RestingECG_Normal": [resting_ecg_normal],
        "RestingECG_ST": [resting_ecg_st], "ST_Slope_Up": [st_slope_up],
        "ST_Slope_Flat": [st_slope_flat]
    })
    
    # Scale if available
    if scaler:
        try:
            input_scaled = scaler.transform(input_data[EXPECTED_FEATURES])
            input_data = pd.DataFrame(input_scaled, columns=EXPECTED_FEATURES)
        except:
            st.warning("⚠️ Scaling failed, using raw data")
    
    # Predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100 if hasattr(model, 'predict_proba') else 0
    
    # Results
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction == 1:
            st.error("🚨 HIGH RISK")
        else:
            st.success("✅ LOW RISK")
    
    with col2:
        st.metric("Risk %", f"{prob:.1f}%")
    
    with col3:
        st.info("Consult doctor!")
    
    # Interactive Feature Importance (Plotly)
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Feature': EXPECTED_FEATURES,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(imp_df.head(10), y='Feature', x='Importance', 
                    orientation='h', title="Feature Importance",
                    color='Importance', color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Explanation
    if st.checkbox("Show SHAP Explanation"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            st.shap(shap.summary_plot(shap_values[1], input_data, show=False), height=400)
        except:
            st.warning("SHAP not available")
    
    # History
    record = {
        'Time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Age': age, 'Risk': f"{prob:.1f}%", 'Result': "HIGH" if prediction else "LOW"
    }
    st.session_state.predictions_history.append(record)

# ==================== HISTORY ====================
if st.session_state.predictions_history:
    st.subheader("📋 History")
    hist_df = pd.DataFrame(st.session_state.predictions_history)
    st.dataframe(hist_df, use_container_width=True)
    
    csv = hist_df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, 
                      f"heart_history_{datetime.now().strftime('%Y%m%d')}.csv")

# ==================== INFO ====================
with st.expander("ℹ️ Model Info"):
    st.write("""
    **Features Used:** Age, Cholesterol, MaxHR, BP, FastingBS, ExerciseAngina + Advanced (Oldpeak, ECG, ST Slope)
    **Install extras:** `pip install plotly shap`
    **Model Files:** heart_model.pkl, scaler.pkl
    """)