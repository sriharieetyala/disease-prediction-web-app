import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from openai import OpenAI
from fpdf import FPDF
import base64

# Load the trained model
model = load_model(r"F:\disease_prediction\disease_prediction_ann_model .h5")

# Load the label classes
label_classes = np.load(r"F:\disease_prediction\label_classes (2).npy", allow_pickle=True)

# Load the scaler
scaler = joblib.load(r"F:\disease_prediction\scaler (1).pkl")

# Load the symptom names
with open(r"F:\disease_prediction\symptom_order.txt", "r") as file:
    symptom_names = file.read().splitlines()

# Initialize OpenAI client
client = OpenAI(api_key="api key")  # Replace with your actual API key

# Function to generate chatbot response
def generate_chatbot_response(predicted_disease):
    input_message = f"The disease is {predicted_disease}. Please provide prevention methods and possible medications."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_message}]
    )
    return completion.choices[0].message.content

# Function to generate PDF report
def generate_pdf(disease, response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Disease Prevention and Medication Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Disease: {disease}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Prevention and Medications:\n{response}")
    return pdf.output(dest='S').encode('latin1')

# Function to create a download link
def create_download_link(pdf_data, filename):
    b64 = base64.b64encode(pdf_data).decode('latin1')
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="color: #007BFF; font-weight: bold;">📄 Download PDF</a>'

# Streamlit UI configuration
st.set_page_config(page_title="Disease Prediction & Health Bot", layout="wide")

# Custom header
st.markdown("""
    <style>
    h1 {
        text-align: center;
        font-size: 2.8em;
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    hr {
        border: none;
        height: 4px;
        background: linear-gradient(90deg, #6a11cb, #2575fc, #00ffe7);
        margin-bottom: 2em;
    }
    </style>
    <h1>Disease Prediction and Assistance 🧬</h1>
    <hr>
""", unsafe_allow_html=True)

# Sidebar for symptoms
st.sidebar.header("🩺 Select Your Symptoms")
selected_symptoms = st.sidebar.multiselect("Choose symptoms from the list below:", symptom_names)

# Session state
if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = None

# Prediction logic
if st.sidebar.button("🔍 Predict Disease"):
    if selected_symptoms:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_names]
        input_vector = np.array(input_vector).reshape(1, -1)
        input_vector_scaled = scaler.transform(input_vector)

        prediction = model.predict(input_vector_scaled)
        st.session_state.predicted_disease = label_classes[np.argmax(prediction)]
        st.session_state.chatbot_response = None  # Reset chatbot
    else:
        st.warning("⚠️ Please select at least one symptom.")

# Display prediction
if st.session_state.predicted_disease:
    st.success(f"🧾 **Predicted Disease:** {st.session_state.predicted_disease}")

    if st.button("💊 Get Assistance"):
        if st.session_state.chatbot_response is None:
            st.session_state.chatbot_response = generate_chatbot_response(st.session_state.predicted_disease)

        st.subheader("🩹 Prevention and Medications")
        st.write(st.session_state.chatbot_response)

        # PDF Download
        pdf_data = generate_pdf(st.session_state.predicted_disease, st.session_state.chatbot_response)
        st.markdown(create_download_link(pdf_data, "disease_report.pdf"), unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Made with ❤️ using Streamlit and OpenAI API <br>
        <strong>Developed by Srihari</strong>
    </div>
""", unsafe_allow_html=True)
