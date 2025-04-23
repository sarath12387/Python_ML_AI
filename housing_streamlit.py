import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")

# DEBUG: Confirming app has loaded
st.write("📦 Loading model...")

try:
    with open(r"C:\Users\pottu\OneDrive\Desktop\ML & AI\Machine_learning\housing_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

st.markdown("### Enter house details")

sqft_living = st.slider("Living Area (sqft)", 500, 10000, 1500)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1.0, 10.0, 2.0, step=0.25)
floors = st.slider("Floors", 1.0, 4.0, 1.0, step=0.5)

if st.button("Predict Price"):
    features = np.array([[sqft_living, bedrooms, bathrooms, floors]])
    price = model.predict(features)[0]
    st.success(f"💰 Estimated Price: ${price:,.2f}")
    st.balloons()
