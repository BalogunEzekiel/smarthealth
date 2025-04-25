import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import os

# Cache the model loading
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Cache image loading
@st.cache_data
def load_logo():
    return Image.open("logo.png")

model = load_model()
logo = load_logo()

# Page title and logo
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
    st.markdown("## ℹ️ About SmartHealth")
    st.info("This app uses a machine learning model to predict possible diagnoses based on symptoms provided. It is not a substitute for professional medical advice.")

st.write("Enter patient data to predict diagnosis:")

# --- Symptom input helper ---
def symptom_group(title, symptoms):
    with st.expander(f"{title} Symptoms", expanded=False):
        return {symptom: st.selectbox(symptom.replace('_', ' ').title(), ["No", "Yes"], key=symptom)
                for symptom in symptoms}

# --- Symptom categories ---
respiratory_symptoms = ['cough', 'wheezing', 'chest_tightness', 'shortness_of_breath', 'sore_throat']
cardiovascular_symptoms = ['irregular_heartbeat', 'chest_pain', 'fatigue', 'swelling_in_legs']
digestive_symptoms = ['nausea', 'abdominal_pain', 'dark_urine', 'jaundice']
metabolic_symptoms = ['increased_thirst', 'frequent_urination', 'blurred_vision', 'slow_healing_wounds']
neurological_symptoms = ['dizziness', 'headache', 'trouble_sleeping']
general_symptoms = ['fever', 'body_pain', 'loss_of_taste', 'unexplained_weight_loss', 'skin_changes', 'muscle_cramps']

# --- Collect inputs ---
input_data = {}
input_data.update(symptom_group("Respiratory", respiratory_symptoms))
input_data.update(symptom_group("Cardiovascular", cardiovascular_symptoms))
input_data.update(symptom_group("Digestive & Hepatic", digestive_symptoms))
input_data.update(symptom_group("Metabolic", metabolic_symptoms))
input_data.update(symptom_group("Neurological", neurological_symptoms))
input_data.update(symptom_group("General", general_symptoms))

# --- Convert to DataFrame ---
input_df = pd.DataFrame([[
    1 if val == "Yes" else 0 for val in input_data.values()
]], columns=input_data.keys())

# --- Feature alignment ---
feature_columns = [
    'irregular_heartbeat', 'sore_throat', 'dark_urine', 'slow_healing_wounds',
    'unexplained_weight_loss', 'muscle_cramps', 'fatigue', 'nausea', 'fever',
    'chest_pain', 'jaundice', 'shortness_of_breath', 'skin_changes',
    'wheezing', 'chest_tightness', 'body_pain', 'cough', 'loss_of_taste',
    'abdominal_pain', 'trouble_sleeping', 'frequent_urination', 'headache',
    'swelling_in_legs', 'increased_thirst', 'blurred_vision', 'dizziness'
]
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# --- Diagnosis map ---
diagnosis_map = {
    0: "Asthma",
    1: "COVID-19",
    2: "Cancer",
    3: "Diabetes",
    4: "Heart Disease",
    5: "Hypertension",
    6: "Kidney Disease",
    7: "Liver Disease"
}

# --- PDF Generator ---
class PDF(FPDF):
    def header(self):
        if os.path.exists("logo.png"):
            self.image("logo.png", 10, 8, 33)
        self.set_font("Arial", 'B', 18)
        self.cell(0, 10, "SmartHealth Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 9)
        self.cell(0, 10, f"Verified by SmartHealth | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')

def generate_pdf(name, df, diagnosis):
    grouped_symptoms = {
        "Respiratory": respiratory_symptoms,
        "Cardiovascular": cardiovascular_symptoms,
        "Digestive & Hepatic": digestive_symptoms,
        "Metabolic": metabolic_symptoms,
        "Neurological": neurological_symptoms,
        "General": general_symptoms
    }

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 8, f"Predicted Diagnosis: {diagnosis}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Symptom Summary", ln=True, align="C")
    pdf.ln(2)

    for group, symptoms in grouped_symptoms.items():
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{group} Symptoms", ln=True)
        pdf.set_font("Arial", size=10)
        for i in range(0, len(symptoms), 2):
            for j in range(2):
                if i + j < len(symptoms):
                    sym = symptoms[i + j]
                    val = "Yes" if df[sym].values[0] == 1 else "No"
                    label = sym.replace("_", " ").title() + ":"
                    pdf.cell(55, 8, label, border=1, align="L")
                    pdf.cell(25, 8, val, border=1, align="C")
                else:
                    pdf.cell(55, 8, "", border=1)
                    pdf.cell(
