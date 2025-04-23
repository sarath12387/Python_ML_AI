import streamlit as st
import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyBtA019VbkXBDu9qtFb6CpNN1b6Hatgyq8")

# Initialize model
model = genai.GenerativeModel("gemini-2.0-flash")

# Streamlit app
st.set_page_config(page_title="Gemini AI Q&A", layout="centered")
st.title("🤖 Ask Gemini AI")

st.markdown("Type any question and let Gemini AI answer it using Google's generative model.")

# Input field
user_input = st.text_area("💬 Enter your question here:", placeholder="e.g. What is model context protocol?", height=100)

if st.button("🔍 Get Answer"):
    if user_input.strip():
        try:
            with st.spinner("Thinking..."):
                response = model.generate_content(user_input)
                st.success("✅ Response:")
                st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter a valid question.")

