import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# Sidebar note
st.sidebar.markdown("## ℹ️ About SmartHealth")
st.sidebar.info("This app uses a machine learning model to predict possible diagnoses based on selected symptoms. It is not a substitute for professional medical advice.")

st.title("SmartHealth - Diagnostic Predictor")

st.write("Enter patient data to predict diagnosis:")

# Get user input
irregular_heartbeat = st.selectbox("Irregular heartbeat", ["No", "Yes"])
sore_throat = st.selectbox("Sore throat", ["No", "Yes"])
dark_urine = st.selectbox("Dark urine", ["No", "Yes"])
slow_healing_wounds = st.selectbox("Slow healing wounds", ["No", "Yes"])
unexplained_weight_loss = st.selectbox("Unexplained weight loss", ["No", "Yes"])
muscle_cramps = st.selectbox("Muscle cramps", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
nausea = st.selectbox("Nausea", ["No", "Yes"])
fever = st.selectbox("Fever", ["No", "Yes"])
chest_pain = st.selectbox("Chest pain", ["No", "Yes"])
jaundice = st.selectbox("Jaundice", ["No", "Yes"])
shortness_of_breath = st.selectbox("Shortness of breath", ["No", "Yes"])
skin_changes = st.selectbox("Skin changes", ["No", "Yes"])
wheezing = st.selectbox("Wheezing", ["No", "Yes"])
chest_tightness = st.selectbox("Chest tightness", ["No", "Yes"])
body_pain = st.selectbox("Body pain", ["No", "Yes"])
cough = st.selectbox("Cough", ["No", "Yes"])
loss_of_taste = st.selectbox("Loss of taste", ["No", "Yes"])
abdominal_pain = st.selectbox("Abdominal pain", ["No", "Yes"])
trouble_sleeping = st.selectbox("Trouble sleeping", ["No", "Yes"])
frequent_urination = st.selectbox("Frequent urination", ["No", "Yes"])
headache = st.selectbox("Headache", ["No", "Yes"])
swelling_in_legs = st.selectbox("Swelling in legs", ["No", "Yes"])
increased_thirst = st.selectbox("Increased thirst", ["No", "Yes"])
blurred_vision = st.selectbox("Blurred vision", ["No", "Yes"])
dizziness = st.selectbox("Dizziness", ["No", "Yes"])

# Convert categorical input to numeric
def convert_input(val):
    return 1 if val == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([[
    convert_input(irregular_heartbeat),
    convert_input(sore_throat),
    convert_input(dark_urine),
    convert_input(slow_healing_wounds),
    convert_input(unexplained_weight_loss),
    convert_input(muscle_cramps),
    convert_input(fatigue),
    convert_input(nausea),
    convert_input(fever),
    convert_input(chest_pain),
    convert_input(jaundice),
    convert_input(shortness_of_breath),
    convert_input(skin_changes),
    convert_input(wheezing),
    convert_input(chest_tightness),
    convert_input(body_pain),
    convert_input(cough),
    convert_input(loss_of_taste),
    convert_input(abdominal_pain),
    convert_input(trouble_sleeping),
    convert_input(frequent_urination),
    convert_input(headache),
    convert_input(swelling_in_legs),
    convert_input(increased_thirst),
    convert_input(blurred_vision),
    convert_input(dizziness)
]], columns=[
    'irregular_heartbeat', 'sore_throat', 'dark_urine', 'slow_healing_wounds',
    'unexplained_weight_loss', 'muscle_cramps', 'fatigue', 'nausea', 'fever',
    'chest_pain', 'jaundice', 'shortness_of_breath', 'skin_changes',
    'wheezing', 'chest_tightness', 'body_pain', 'cough', 'loss_of_taste',
    'abdominal_pain', 'trouble_sleeping', 'frequent_urination', 'headache',
    'swelling_in_legs', 'increased_thirst', 'blurred_vision', 'dizziness'
])

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

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        diagnosis = diagnosis_map.get(prediction, "Unknown")
        st.success(f"🩺 Predicted Diagnosis: **{diagnosis}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Add disclaimer
st.caption("Note: This is a machine learning prediction based on the entered symptoms. Always consult a medical professional for a definitive diagnosis.")
