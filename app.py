import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from transformers import pipeline
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="SaaS Dashboard", layout="wide")

# --- AI MODEL FOR TEXT ANALYSIS ---
ai_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def generate_ai_summary(text):
    result = ai_model(text)
    return result[0]["label"]

# --- FILE UPLOAD ---
st.title("📊 SaaS Dashboard for Data Analysis")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    
    # Read file
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 🔍 Data Preview")
    st.dataframe(df)

    # --- DATA CLEANING ---
    st.subheader("⚙️ Data Cleaning")
    df = df.dropna()  # Remove missing values
    df = df.drop_duplicates()  # Remove duplicate rows
    st.write("✅ Data cleaned successfully!")

    # --- DATA VISUALIZATION ---
    st.subheader("📊 Data Visualization")
    
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("Select X-axis", numeric_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig)
    else:
        st.warning("❗ At least two numeric columns are required for visualization.")

    # --- ANOMALY DETECTION ---
    st.subheader("🚨 Anomaly Detection")
    if len(numeric_columns) > 0:
        selected_col = st.selectbox("Select Column for Anomaly Detection", numeric_columns)
        
        mean = df[selected_col].mean()
        std_dev = df[selected_col].std()

        df["Anomaly"] = (df[selected_col] < mean - 3 * std_dev) | (df[selected_col] > mean + 3 * std_dev)
        anomalies = df[df["Anomaly"]]

        st.write(f"🔍 Detected {len(anomalies)} anomalies in `{selected_col}`")
        st.dataframe(anomalies)
    else:
        st.warning("❗ No numeric columns available for anomaly detection.")

    # --- PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Predictive Analytics (Forecasting)")
    
    date_columns = [col for col in df.columns if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
    if date_columns:
        date_column = st.selectbox("Select Date Column:", date_columns)
        value_column = st.selectbox("Select Value Column:", [col for col in numeric_columns if col != date_column])

        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column])  # Remove NaN values

        if df.empty:
            st.error("❌ No valid data available after cleaning.")
        else:
            # Rename for Prophet
            df = df.rename(columns={date_column: "ds", value_column: "y"})
            df = df[df["ds"].notna()]

            # Train Prophet model
            model = Prophet()
            model.fit(df)

            # Forecast future data
            period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)

            st.write("🔮 Forecasted Data:")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
            st.line_chart(forecast.set_index("ds")["yhat"])
    else:
        st.warning("❗ No date columns found for forecasting.")

    # --- AI-POWERED INSIGHTS ---
    st.subheader("🤖 AI-Powered Insights")
    user_input = st.text_area("Enter text for AI analysis:")
    if user_input:
        insights = generate_ai_summary(user_input)
        st.write("🔍 AI Insight:", insights)

else:
    st.warning("⚠️ Please upload a CSV or Excel file to start analysis.")
