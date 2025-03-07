import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import plotly.graph_objects as go
from textblob import TextBlob

# 📌 Load Data
st.title("📊 AI-Powered SaaS Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    # 📌 Drop columns with too many NaN values
    df.dropna(axis=1, thresh=len(df) * 0.5, inplace=True)

    # 📌 Select Numeric Columns for Visualization
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found for visualization!")
    else:
        # 📊 Data Visualization (Without Date Column)
        st.subheader("📊 Data Visualization")
        selected_col = st.selectbox("Select Column to Visualize", numeric_cols)
        fig_bar = px.bar(df, x=df.index, y=selected_col, title=f"{selected_col} Distribution")
        st.plotly_chart(fig_bar)

    # 📌 AI Forecasting with Prophet (Handles Missing Values)
    st.subheader("📈 AI Forecasting")
    date_col = st.selectbox("Select Date Column", df.columns)
    target_col = st.selectbox("Select Column to Forecast", numeric_cols)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col])

    if len(df) < 2:
        st.error("❌ Not enough valid rows for AI forecasting. Please check your data.")
    else:
        forecast_df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted"))
        fig_forecast.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], mode="markers", name="Actual"))
        fig_forecast.update_layout(title="Future Predictions", xaxis_title="Date", yaxis_title=target_col)

        st.plotly_chart(fig_forecast)

    # 📌 Sentiment Analysis
    st.subheader("🧠 AI Sentiment Analysis")
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    if text_columns:
        sentiment_col = st.selectbox("Select Text Column for Sentiment Analysis", text_columns)
        df["Sentiment"] = df[sentiment_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df["Sentiment_Label"] = df["Sentiment"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")

        fig_sentiment = px.histogram(df, x="Sentiment_Label", title="Sentiment Distribution")
        st.plotly_chart(fig_sentiment)
    else:
        st.warning("⚠️ No text column found for sentiment analysis!")

else:
    st.warning("⚠️ Please upload a CSV file!")
