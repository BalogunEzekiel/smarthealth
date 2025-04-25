import streamlit as st 
import pandas as pd
import joblib
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import os

# === CACHED RESOURCES ===
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_logo():
    return Image.open("logo.png")

model = load_model()
logo = load_logo()

# === APP HEADER ===
st.image(logo, width=200)
st.title("Welcome to SmartHealth!")

with st.sidebar:
    st.info("""
    **Supported Conditions:**
    - Asthma
    - Cancer
    - COVID-19
    - Diabetes
    - Heart Disease
    - Hypertension
    - Kidney Disease
    - Liver Disease
    """)

# Sidebar note
st.sidebar.markdown("## ℹ️ About SmartHealth")
st.sidebar.info("This app uses a machine learning model to predict possible diagnoses based on symptoms provided. It is not a substitute for professional medical advice.")

st.write("Enter patient data to predict diagnosis:")

# === Symptom Grouping ===
def symptom_group(title, symptoms):
    with st.expander(f"{title} Symptoms", expanded=False):
        return {symptom: st.selectbox(symptom.replace('_', ' ').title(), ["No", "Yes"], key=symptom)
                for symptom in symptoms}

respiratory_symptoms = ['cough', 'wheezing', 'chest_tightness', 'shortness_of_breath', 'sore_throat']
cardiovascular_symptoms = ['irregular_heartbeat', 'chest_pain', 'fatigue', 'swelling_in_legs']
digestive_symptoms = ['nausea', 'abdominal_pain', 'dark_urine', 'jaundice']
metabolic_symptoms = ['increased_thirst', 'frequent_urination', 'blurred_vision', 'slow_healing_wounds']
neurological_symptoms = ['dizziness', 'headache', 'trouble_sleeping']
general_symptoms = ['fever', 'body_pain', 'loss_of_
                   ]
