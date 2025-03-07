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

# ========================== 🧠 Sentiment Analysis ==========================
st.subheader("🧠 AI Sentiment Analysis")

# Select a text column for sentiment analysis
text_cols = df.select_dtypes(include=['object']).columns.tolist()
if text_cols:
    sentiment_col = st.selectbox("Select Text Column for Sentiment Analysis", text_cols)
    
    def analyze_sentiment(text):
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['Sentiment'] = df[sentiment_col].astype(str).apply(analyze_sentiment)
    
    # Show sentiment distribution
    st.write("### Sentiment Analysis Results")
    st.write(df[[sentiment_col, "Sentiment"]].head())

    fig_sentiment = px.histogram(df, x="Sentiment", title="Sentiment Distribution", color="Sentiment")
    st.plotly_chart(fig_sentiment)
else:
    st.warning("⚠️ No text column found for sentiment analysis.")

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

# ========================== 🔮 AI Forecasting (Prophet) ==========================
st.subheader("🔮 AI-Based Forecasting (Time Series Prediction)")

# Check for a date column
date_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist()

if date_cols:
    date_col = st.selectbox("Select Date Column for Forecasting", date_cols)
    target_col = st.selectbox("Select Target Column to Predict", numerical_cols)

    # Convert date column to datetime format
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Prepare data for Prophet
    forecast_df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})

    # Train the model
    model = Prophet()
    model.fit(forecast_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot predictions
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted"))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], mode="markers", name="Actual"))
    fig_forecast.update_layout(title="Future Predictions", xaxis_title="Date", yaxis_title=target_col)

    st.plotly_chart(fig_forecast)
else:
    st.warning("⚠️ No date column found. Please ensure your dataset contains a valid date column for AI-based forecasting.")

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

