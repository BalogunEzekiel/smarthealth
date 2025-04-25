import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import os
from io import BytesIO
import numpy as np

# --- Load model once ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --- Logo and Title ---
st.image("logo.png", width=200)
st.title("🤖 Welcome to SmartHealth")

# --- Sidebar Info ---
with st.sidebar:
    st.info("**Supported Conditions:**\n- Asthma\n- Cancer\n- COVID-19\n- Diabetes\n- Heart Disease\n- Hypertension\n- Kidney Disease\n- Liver Disease")
    st.markdown("## ℹ️ About SmartHealth")
    st.info("This ML-powered app predicts possible diagnoses based on your symptoms. Not a substitute for professional medical advice.")

# --- Symptom Input Groups ---
symptom_groups = {
    "Respiratory": ['cough', 'wheezing', 'chest_tightness', 'shortness_of_breath', 'sore_throat'],
    "Cardiovascular": ['irregular_heartbeat', 'chest_pain', 'fatigue', 'swelling_in_legs'],
    "Digestive & Hepatic": ['nausea', 'abdominal_pain', 'dark_urine', 'jaundice'],
    "Metabolic": ['increased_thirst', 'frequent_urination', 'blurred_vision', 'slow_healing_wounds'],
    "Neurological": ['dizziness', 'headache', 'trouble_sleeping'],
    "General": ['fever', 'body_pain', 'loss_of_taste', 'unexplained_weight_loss', 'skin_changes', 'muscle_cramps']
}

feature_columns = [
    'irregular_heartbeat', 'sore_throat', 'dark_urine', 'slow_healing_wounds',
    'unexplained_weight_loss', 'muscle_cramps', 'fatigue', 'nausea', 'fever',
    'chest_pain', 'jaundice', 'shortness_of_breath', 'skin_changes',
    'wheezing', 'chest_tightness', 'body_pain', 'cough', 'loss_of_taste',
    'abdominal_pain', 'trouble_sleeping', 'frequent_urination', 'headache',
    'swelling_in_legs', 'increased_thirst', 'blurred_vision', 'dizziness'
]

diagnosis_map = {
    0: "Asthma", 1: "COVID-19", 2: "Cancer", 3: "Diabetes",
    4: "Heart Disease", 5: "Hypertension", 6: "Kidney Disease", 7: "Liver Disease"
}

# --- User Input Form ---
st.subheader("📋 Enter patient data:")
with st.form("diagnosis_form"):
    patient_name = st.text_input("Patient's Full Name")

    symptom_data = {}
    for group, symptoms in symptom_groups.items():
        with st.expander(f"{group} Symptoms"):
            for symptom in symptoms:
                label = symptom.replace('_', ' ').title()
                symptom_data[symptom] = st.selectbox(label, ["No", "Yes"], key=symptom)

    submitted = st.form_submit_button("Predict Diagnosis")

# --- Convert Input & Predict ---
def convert_input(symptom_data):
    return np.array([[1 if val == "Yes" else 0 for val in symptom_data.values()]])

if submitted:
    if not patient_name.strip():
        st.warning("Please enter the patient's name.")
    else:
        with st.spinner("Analyzing symptoms..."):
            try:
                input_array = convert_input(symptom_data)
                input_df = pd.DataFrame(input_array, columns=symptom_data.keys())
                input_df = input_df[feature_columns]
                prediction = model.predict(input_df)[0]
                diagnosis = diagnosis_map.get(prediction, "Unknown")
                st.success(f"🩺 Predicted Diagnosis for **{patient_name}**: **{diagnosis}**")

                # --- PDF Report in Memory ---
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

                pdf = PDF()
                pdf.add_page()
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
                pdf.cell(0, 8, f"Predicted Diagnosis: {diagnosis}", ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Symptom Summary", ln=True, align="C")
                pdf.set_font("Arial", size=10)
                for group, symptoms in symptom_groups.items():
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 8, group, ln=True)
                    pdf.set_font("Arial", size=10)
                    for symptom in symptoms:
                        val = "Yes" if input_df[symptom].values[0] == 1 else "No"
                        label = symptom.replace("_", " ").title()
                        pdf.cell(55, 8, f"{label}:", border=1)
                        pdf.cell(25, 8, val, border=1)
                        pdf.ln()

                pdf.ln(6)
                pdf.set_font("Arial", "I", 9)
                pdf.multi_cell(0, 8, "Disclaimer: This is a preliminary diagnostic report based on machine learning predictions. Always consult a medical professional for proper diagnosis.")

                # Output PDF in memory
                pdf_buffer = BytesIO()
                pdf.output(pdf_buffer)
                pdf_buffer.seek(0)

                st.download_button(
                    label="📄 Download Diagnosis Report",
                    data=pdf_buffer,
                    file_name=f"{patient_name.replace(' ', '_')}_SmartHealth_Report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- Patient Engagement ---
st.markdown("---")
st.subheader("💡 General Health Tips")
st.info("""
- Stay hydrated and eat a balanced diet.
- Get regular checkups even if you're feeling fine.
- Avoid self-medication. Seek expert advice.
- Track chronic symptoms.
- Get enough sleep and exercise regularly.
""")

# --- Feedback Section ---
st.markdown("---")
st.subheader("💬 Tell us about your experience")
feedback_name = st.text_input("What's your name?")
if feedback_name:
    st.success(f"Thank you for using SmartHealth, {feedback_name}! Stay healthy. 💖")

st.markdown("### 📝 Feedback")
feedback = st.text_area("Do you have suggestions or comments?")
if st.button("Submit Feedback"):
    if feedback.strip():
        st.success("✅ Thank you for your feedback!")
    else:
        st.warning("Please enter your feedback before submitting.")
