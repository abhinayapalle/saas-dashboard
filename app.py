import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- Streamlit App Header ---
st.set_page_config(page_title="📊 SaaS Dashboard", layout="wide")
st.title("📊 SaaS Dashboard for Data Analysis")

# --- File Upload Section ---
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv", "xlsx"])

if uploaded_file:
    # Read CSV/XLSX File
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext == "xlsx":
        df = pd.read_excel(uploaded_file)

    st.write("✅ Data Preview:")
    st.dataframe(df)

    # --- AI-powered Insights ---
    st.subheader("🧠 AI-Powered Insights")
    
    @st.cache_resource
    def load_model():
        """Load AI model for sentiment analysis."""
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

    ai_model = load_model()

    # AI Sentiment Analysis on Text Column
    text_columns = [col for col in df.columns if df[col].dtype == "O"]
    if text_columns:
        selected_text_col = st.selectbox("📝 Select Text Column for Analysis", text_columns)
        df
