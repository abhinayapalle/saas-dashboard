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

# Ensure 'date' column exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
else:
    st.warning("⚠️ The dataset does not contain a 'date' column. Some features may not work properly.")

# ========================== 📈 AI-Powered Forecasting ==========================
st.subheader("📈 AI-Powered Sales Forecasting")

if "date" in df.columns and "sales" in df.columns:
    forecast_period = st.slider("Select Forecasting Period (Days)", min_value=7, max_value=365, value=30)
    
    # Prepare data for Prophet
    prophet_df = df[['date', 'sales']].dropna().rename(columns={"date": "ds", "sales": "y"})
    
    # Train Prophet Model
    model = Prophet()
    model.fit(prophet_df)
    
    # Make Future Predictions
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    
    # Plot Forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Sales'))
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', name='Actual Sales'))
    st.plotly_chart(fig)
else:
    st.warning("⚠️ 'date' and 'sales' columns are required for forecasting.")

# ========================== 🧠 AI Sentiment Analysis ==========================
st.subheader("🧠 AI Sentiment Analysis")

# Let user select a text column
text_col = st.selectbox("Select Text Column for Sentiment Analysis", df.columns)

if text_col:
    # Ensure column is string type
    df[text_col] = df[text_col].astype(str)

    # Compute Sentiment Scores
    df['vader_sentiment'] = df[text_col].apply(lambda text: sia.polarity_scores(text)['compound'])
    df['textblob_sentiment'] = df[text_col].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Assign Sentiment Category
    df['sentiment_category'] = df['vader_sentiment'].apply(lambda score: 
        "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral")
    )

    # Display Sentiment Data
    st.write(df[[text_col, "vader_sentiment", "textblob_sentiment", "sentiment_category"]])

    # Sentiment Pie Chart
    sentiment_counts = df['sentiment_category'].value_counts()
    fig_pie = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment Distribution")
    st.plotly_chart(fig_pie)

# ========================== 🚨 Real-Time Alerts ==========================
st.subheader("🚨 Real-Time Alerts & Notifications")

alert_threshold = st.slider("Set Revenue Drop Alert Threshold (%)", min_value=5, max_value=50, value=10)

if "sales" in df.columns:
    df['daily_change'] = df['sales'].pct_change() * 100
    alert_df = df[df['daily_change'] < -alert_threshold]

    if not alert_df.empty:
        st.error(f"⚠️ {len(alert_df)} instances detected where sales dropped by more than {alert_threshold}%!")
        st.write(alert_df[['date', 'sales', 'daily_change']])
    else:
        st.success("✅ No significant revenue drops detected!")

# ========================== 📊 Data Visualization ==========================
st.subheader("📊 Data Visualizations")

# Line Chart for Sales Trends
if "date" in df.columns and "sales" in df.columns:
    fig_line = px.line(df, x="date", y="sales", title="Sales Over Time")
    st.plotly_chart(fig_line)

# Bar Chart with Error Handling
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if numerical_cols:
    bar_col = st.selectbox("Select Column for Bar Chart", numerical_cols)

    # Ensure data is clean before plotting
    if bar_col in df.columns:
        df.dropna(subset=[bar_col], inplace=True)  # Remove NaN values

        if "date" in df.columns:
            fig_bar = px.bar(df, x="date", y=bar_col, title=f"{bar_col} Trends Over Time")
            st.plotly_chart(fig_bar)
        else:
            st.warning("⚠️ The dataset does not contain a 'date' column for visualization.")
    else:
        st.warning("⚠️ The selected column is not available in the dataset.")
else:
    st.warning("⚠️ No numeric columns found in the dataset.")

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
st.sidebar.info("📌 Built with Python, Streamlit, Prophet, NLP, and AI!")

