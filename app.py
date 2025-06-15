# ecosnap/app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model & class names
model = tf.keras.models.load_model("model/food_model.h5")
with open("data/class_names.txt") as f:
    class_names = f.read().splitlines()

# Load and clean carbon data
carbon_df = pd.read_csv("data/carbon_data.csv")
carbon_df["Food"] = carbon_df["Food"].astype(str).str.strip().str.lower()

def predict(img):
    img = img.resize((224, 224))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    idx = np.argmax(pred)
    return class_names[idx].strip().lower()

def get_carbon_info(label):
    df = pd.read_csv("data/carbon_data.csv")
    df["Food"] = df["Food"].str.strip().str.lower()  # Clean and normalize CSV entries
    label_clean = label.lower().replace(" ", "_").strip()  # Normalize model output
    row = df[df["Food"] == label_clean]
    
    if not row.empty:
        co2 = row["CO2_per_kg"].values[0]
        tip = row["Tip"].values[0]
        return co2, tip
    else:
        return None, None

# Streamlit UI
st.set_page_config(page_title="EcoSnap 🌱")
st.title("EcoSnap – AI Food Classifier with Carbon Estimator")
st.markdown("Upload a food image to estimate its environmental impact 🌍")

img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
analyze = st.button("🔍 Analyze")

if img_file and analyze:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        label = predict(image)
        co2, tip = get_carbon_info(label)

    st.subheader(f"🍽️ Food: **{label.replace('_', ' ').title()}**")
    if co2:
        st.success(f"🌍 CO₂ Footprint: {co2} kg CO₂ / kg")
        st.info(f"💡 Tip: {tip}")
    else:
        st.warning("⚠️ Carbon data not found.")
