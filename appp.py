import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

st.set_page_config(page_title="FarmAI", layout="centered")
st.title("🐄 FarmAI: AI for Cow Forecasting, Health & Identity")

# ========== PRICE FORECASTING ==========

def dummy_forecast(days=30):
    np.random.seed(42)
    base = 180000  # Assume base cow price in PKR
    trend = np.linspace(0, 5000, days)  # Upward trend
    noise = np.random.normal(0, 2000, days)
    prices = base + trend + noise
    return pd.Series(prices)

# ========== HEALTH CLASSIFIER ==========

def classify_health(image):
    resized = image.resize((128, 128))
    np_img = np.array(resized) / 255.0
    brightness = np.mean(np_img)

    if brightness > 0.6:
        return "🟢 Healthy"
    elif brightness > 0.4:
        return "🟡 Under Observation"
    else:
        return "🔴 Sick"

# ========== NOSEPRINT MATCHING ==========

def match_noseprint(uploaded_image, registry_dir="nose_registry"):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    uploaded_gray = np.array(uploaded_image.convert("L"))
    kp1, des1 = orb.detectAndCompute(uploaded_gray, None)

    best_match = None
    max_matches = 0

    for file in os.listdir(registry_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            reg_img = Image.open(os.path.join(registry_dir, file)).convert("L")
            reg_np = np.array(reg_img)
            kp2, des2 = orb.detectAndCompute(reg_np, None)
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                if len(matches) > max_matches:
                    max_matches = len(matches)
                    best_match = file

    return best_match if max_matches > 10 else None

# ========== UI ==========

tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "🩺 Health Check", "🆔 Noseprint Identity"])

# --------- PRICE ---------
with tab1:
    st.header("📈 Cow Price Forecast (Demo)")
    days = st.slider("Select days", 7, 60, 30)
    forecast = dummy_forecast(days)
    st.line_chart(forecast)

# --------- HEALTH ---------
with tab2:
    st.header("🩺 Cow Health Classifier (Demo)")
    image_file = st.file_uploader("Upload a cow image", type=["jpg", "png", "jpeg"], key="health")
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Cow Image", use_column_width=True)
        health_status = classify_health(image)
        st.success(f"Health Status: {health_status}")

# --------- NOSE ID ---------
with tab3:
    st.header("🆔 Cow Noseprint Identity Match")
    st.write("Uploads compared to local registry in `/nose_registry` folder.")
    image_file = st.file_uploader("Upload nose image", type=["jpg", "png", "jpeg"], key="nose")
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Nose Image", use_column_width=True)

        # Ensure directory exists
        os.makedirs("nose_registry", exist_ok=True)

        match = match_noseprint(image)
        if match:
            st.success(f"✅ Match found with: {match}")
        else:
            st.error("❌ No matching cow found.")
