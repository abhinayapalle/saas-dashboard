import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.set_page_config(page_title="📊 AI-Powered SaaS Dashboard", layout="wide")
st.title("📊 AI-Powered SaaS Business Insights")

# ========================== 📂 File Upload Handling ==========================
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.warning("⚠️ Please upload a CSV file to continue.")
    st.stop()

# ========================== 📌 Data Preview ==========================
st.subheader("📌 Data Preview")
st.write(df.head())

# ========================== 📊 Data Visualization (Without Date Column) ==========================
st.subheader("📊 Data Visualizations")

# List of numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if numerical_cols:
    # Bar Chart
    bar_col_x = st.selectbox("Select X-axis for Bar Chart", numerical_cols, key="bar_x")
    bar_col_y = st.selectbox("Select Y-axis for Bar Chart", numerical_cols, key="bar_y")
    
    fig_bar = px.bar(df, x=bar_col_x, y=bar_col_y, title=f"{bar_col_y} vs {bar_col_x}")
    st.plotly_chart(fig_bar)

    # Scatter Plot
    scatter_col_x = st.selectbox("Select X-axis for Scatter Plot", numerical_cols, key="scatter_x")
    scatter_col_y = st.selectbox("Select Y-axis for Scatter Plot", numerical_cols, key="scatter_y")

    fig_scatter = px.scatter(df, x=scatter_col_x, y=scatter_col_y, title=f"{scatter_col_y} vs {scatter_col_x}")
    st.plotly_chart(fig_scatter)

    # Pie Chart
    pie_col = st.selectbox("Select Column for Pie Chart", numerical_cols, key="pie_col")
    fig_pie = px.pie(df, names=pie_col, title=f"Distribution of {pie_col}")
    st.plotly_chart(fig_pie)

else:
    st.warning("⚠️ No numerical columns found in the dataset for visualization.")

# ========================== 🎯 Industry-Specific Dashboard Options ==========================
st.sidebar.header("🎯 Industry Customization")

industry = st.sidebar.selectbox("Select Industry", ["E-commerce", "Healthcare", "Finance & Stock Market"])

if industry == "E-commerce":
    st.sidebar.write("🛒 Showing e-commerce insights (customer churn, order trends)")
elif industry == "Healthcare":
    st.sidebar.write("🏥 Showing healthcare insights (patient monitoring, drug stock tracking)")
elif industry == "Finance & Stock Market":
    st.sidebar.write("💰 Showing finance insights (portfolio tracking, fraud detection)")

st.sidebar.write("✨ Customize insights based on industry needs!")

st.sidebar.markdown("---")
st.sidebar.info("📌 Built with Python, Streamlit, Plotly & AI!")

