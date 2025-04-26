import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import os
import openai
from openai import OpenAI

# Load model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Logo and title
try:
    logo = Image.open("logo.png")
    st.image(logo, width=200)
except Exception as e:
    st.warning(f"Logo could not be loaded: {e}")
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

# Get user input
# --- Helper function to create grouped inputs ---
def symptom_group(title, symptoms):
    with st.expander(f"{title} Symptoms", expanded=False):
        return {symptom: st.selectbox(symptom.replace('_', ' ').title(), ["No", "Yes"], key=symptom)
                for symptom in symptoms}

# --- Symptom groups ---
respiratory_symptoms = [
    'cough', 'wheezing', 'chest_tightness', 'shortness_of_breath', 'sore_throat'
]

cardiovascular_symptoms = [
    'irregular_heartbeat', 'chest_pain', 'fatigue', 'swelling_in_legs'
]

digestive_symptoms = [
    'nausea', 'abdominal_pain', 'dark_urine', 'jaundice'
]

metabolic_symptoms = [
    'increased_thirst', 'frequent_urination', 'blurred_vision', 'slow_healing_wounds'
]

neurological_symptoms = [
    'dizziness', 'headache', 'trouble_sleeping'
]

general_symptoms = [
    'fever', 'body_pain', 'loss_of_taste', 'unexplained_weight_loss', 'skin_changes', 'muscle_cramps'
]

# --- Collect all symptom inputs grouped ---
input_data = {}
input_data.update(symptom_group("Respiratory", respiratory_symptoms))
input_data.update(symptom_group("Cardiovascular", cardiovascular_symptoms))
input_data.update(symptom_group("Digestive & Hepatic", digestive_symptoms))
input_data.update(symptom_group("Metabolic", metabolic_symptoms))
input_data.update(symptom_group("Neurological", neurological_symptoms))
input_data.update(symptom_group("General", general_symptoms))

# --- Convert inputs to numeric DataFrame ---
def convert_input(val): return 1 if val == "Yes" else 0

input_df = pd.DataFrame([[
    convert_input(input_data[symptom]) for symptom in input_data
]], columns=input_data.keys())


class PDF(FPDF):
    def header(self):
    try:
        if os.path.exists("logo.png"):
            self.image("logo.png", 10, 8, 33)
    except Exception as e:
        pass  # Ignore image errors
        self.set_font("Arial", 'B', 18)
        self.cell(0, 12, "SmartHealth Report", ln=True, align="C")
        self.ln(11)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 9)
        self.cell(0, 10, f"Verified by SmartHealth | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')

def generate_pdf(name, symptoms_df, diagnosis):
    # Grouped symptoms by category
    grouped_symptoms = {
        "Respiratory Symptoms": [
            'cough', 'wheezing', 'chest_tightness', 'shortness_of_breath', 'sore_throat'
        ],
        "Cardiovascular Symptoms": [
            'irregular_heartbeat', 'chest_pain', 'fatigue', 'swelling_in_legs'
        ],
        "Digestive & Hepatic Symptoms": [
            'nausea', 'abdominal_pain', 'dark_urine', 'jaundice'
        ],
        "Metabolic Symptoms": [
            'increased_thirst', 'frequent_urination', 'blurred_vision', 'slow_healing_wounds'
        ],
        "Neurological Symptoms": [
            'dizziness', 'headache', 'trouble_sleeping'
        ],
        "General Symptoms": [
            'fever', 'body_pain', 'loss_of_taste', 'unexplained_weight_loss', 'skin_changes', 'muscle_cramps'
        ]
    }

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Patient Info
    pdf.cell(0, 8, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 8, f"Predicted Diagnosis: {diagnosis}", ln=True)
    pdf.ln(4)

    # Symptom Summary Title
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Symptom Summary", ln=True, align="C")
    pdf.ln(2)

    # Grouped Symptom Grid (2 columns)
    for group, symptoms in grouped_symptoms.items():
        # Group Header
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, group, ln=True)
        pdf.set_font("Arial", size=10)

        for i in range(0, len(symptoms), 2):
            for j in range(2):
                if i + j < len(symptoms):
                    sym = symptoms[i + j]
                    val = "Yes" if symptoms_df[sym].values[0] == 1 else "No"
                    label = sym.replace("_", " ").title() + ":"  # Add colon

                    # Symptom cell with narrower width
                    pdf.cell(55, 8, label, border=1, align="L")

                    # Value cell
                    pdf.cell(25, 8, val, border=1, align="C")
                else:
                    # Maintain 2-column layout
                    pdf.cell(55, 8, "", border=1)
                    pdf.cell(25, 8, "", border=1)
            pdf.ln()

    # Disclaimer
    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 8, "Disclaimer: This is a preliminary diagnostic report based on machine learning predictions. Always consult a medical professional for proper diagnosis.")

    # Save PDF
    filename = f"{name.replace(' ', '_')}_SmartHealth_Report.pdf"
    pdf.output(filename)
    return filename
    
# Define diagnosis labels (ensure this matches the model's output)
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

# Load feature columns
feature_columns = [
    'irregular_heartbeat', 'sore_throat', 'dark_urine', 'slow_healing_wounds',
    'unexplained_weight_loss', 'muscle_cramps', 'fatigue', 'nausea', 'fever',
    'chest_pain', 'jaundice', 'shortness_of_breath', 'skin_changes',
    'wheezing', 'chest_tightness', 'body_pain', 'cough', 'loss_of_taste',
    'abdominal_pain', 'trouble_sleeping', 'frequent_urination', 'headache',
    'swelling_in_legs', 'increased_thirst', 'blurred_vision', 'dizziness'
]

input_df = input_df[feature_columns]

patient_name = st.text_input("Enter patient's full name:")

# Prediction
if st.button("Predict Diagnosis"):
    if patient_name.strip() == "":
        st.warning("Please enter the patient's name.")
    else:
        try:
            prediction = model.predict(input_df)[0]
            diagnosis = diagnosis_map.get(prediction, "Unknown")
            st.success(f"🩺 Predicted Diagnosis for {patient_name}: **{diagnosis}**")

            # Generate and download PDF
            report_file = generate_pdf(patient_name, input_df, diagnosis)
            with open(report_file, "rb") as f:
                st.download_button(
                    label="📄 Download Diagnosis Report",
                    data=f,
                    file_name=report_file,
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Add disclaimer
st.caption("Note: This is a prediction based on the symptoms provided. Always consult a medical professional for a definitive diagnosis.")

# Patient engagement: Health tips section
st.markdown("---")
st.markdown("### 💡 General Health Tips")
st.info("""
- Stay hydrated and eat a balanced diet.
- Get regular checkups even if you're feeling fine.
- Avoid self-medication. Seek expert advice.
- Keep track of chronic symptoms.
- Get enough sleep and exercise regularly.
""")

# Patient engagement: Personalized greeting
st.markdown("---")
st.subheader("💬 Tell us about your experience")
feedback_name = st.text_input("What's your name?", "")
if feedback_name:
    st.success(f"Thank you for using SmartHealth, {feedback_name}! We hope this App helps you stay informed about your health.")

# Patient engagement: Feedback form
st.markdown("### 📝 We’d love your feedback!")
feedback = st.text_area("Do you have suggestions or comments about SmartHealth?")
if st.button("Submit Feedback"):
    if feedback:
        st.success("✅ Thank you for your feedback! We look forward to seeing you again soon for your health monitoring.")
    else:
        st.warning("Please enter some feedback before submitting.")

# Set OpenAI API key
try:
    openai.api_key = st.secrets["openai"]["auth_token"]
except Exception as e:
    st.error(f"OpenAI authentication failed: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your SmartHealth Assistant. How can I help you today?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me about symptoms, conditions, or health tips..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

# Generate assistant response
client = OpenAI()

try:
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful SmartHealth assistant specializing in providing general health tips and information based on symptoms. Always recommend seeing a doctor for serious concerns."},
        *st.session_state.messages
    ],
    stream=True,  # Streaming allows you to update the message in real-time
)
except Exception as e:
    st.error(f"Failed to get assistant response: {e}")

for chunk in response:
    if chunk.choices[0].delta.get("content"):
        full_response += chunk.choices[0].delta["content"]
        message_placeholder.markdown(full_response + "▌")  # ▌ for typing effect

message_placeholder.markdown(full_response)  # Remove the typing cursor when done

# Save assistant message
st.session_state.messages.append({"role": "assistant", "content": full_response})

# Initialize chat history with system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in providing information about health conditions, symptoms, and general wellness tips."},
        {"role": "assistant", "content": "Hello! I'm your SmartHealth Assistant. How can I help you today?"}
    ]
