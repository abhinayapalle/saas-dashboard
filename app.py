import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from transformers import pipeline

# --- TITLE ---
st.title("📊 SaaS Dashboard for Data Analysis")

# --- FILE UPLOAD ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- DATA PREVIEW ---
    st.subheader("📜 Data Preview")
    st.write(df.head())

    # --- ANOMALY DETECTION ---
    st.subheader("🚨 Anomaly Detection")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_columns:
        selected_column = st.selectbox("Select Numeric Column for Anomaly Detection", numeric_columns)

        # Calculate Z-score
        df["z_score"] = (df[selected_column] - df[selected_column].mean()) / df[selected_column].std()
        df["Anomaly"] = df["z_score"].apply(lambda x: "Yes" if abs(x) > 2 else "No")

        st.write(df[[selected_column, "z_score", "Anomaly"]])
        fig = px.scatter(df, x=df.index, y=selected_column, color=df["Anomaly"])
        st.plotly_chart(fig)

    else:
        st.warning("⚠️ No numeric columns found for anomaly detection.")

    # --- PREDICTIVE ANALYTICS (FORECASTING) ---
    st.subheader("🔮 Predictive Analytics (Forecasting)")
    date_columns = [col for col in df.columns if "date" in col.lower()]

    if date_columns:
        date_column = st.selectbox("Select Date Column", date_columns)
        value_column = st.selectbox("Select Value Column", [col for col in df.columns if col != date_column])

        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column])
        df = df.rename(columns={date_column: "ds", value_column: "y"})

        if df.empty:
            st.error("❌ No valid data after cleaning. Please check your file.")
        else:
            # Prophet Model
            model = Prophet()
            model.fit(df)

            period = st.slider("📅 Forecast Period (Days)", 7, 365, 30)
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)

            st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
            st.line_chart(forecast.set_index("ds")["yhat"])

    else:
        st.warning("⚠️ No date column found for forecasting.")

    # --- AI-POWERED INSIGHTS ---
    st.subheader("🤖 AI-Powered Insights")
    
    # Load AI model
    ai_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Generate insights
    sample_text = "The dataset contains trends over time, showing growth in key metrics."
    ai_insights = ai_model(sample_text)

    st.write("📝 AI Insights:", ai_insights)

else:
    st.warning("⚠️ Please upload a CSV file to start analysis.")

