import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from textblob import TextBlob
import numpy as np
import openai  # For AI-powered insights

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="SaaS Dashboard", layout="wide", page_icon="📊")

# --- HEADER ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>📊 AI-Powered SaaS Dashboard</h1>
    <h4 style='text-align: center; color: gray;'>Get Real-time Insights, Forecasting & Anomaly Detection</h4>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX File", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]

    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.success("✅ File uploaded successfully!")
    
    # --- DATA PREVIEW ---
    st.subheader("📄 Data Preview")
    st.write(df.head())

    # --- AI-POWERED DATA INSIGHTS ---
    st.subheader("🧠 AI-Powered Insights")

    def generate_ai_summary(data):
        prompt = f"Analyze the following dataset and provide key insights:\n{data.head().to_string()}\nSummarize in 3 bullet points."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]

    if st.button("🔍 Generate AI Insights"):
        with st.spinner("Analyzing data..."):
            insights = generate_ai_summary(df)
            st.write(insights)

    # --- ANOMALY DETECTION ---
    st.subheader("🚨 Anomaly Detection")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        column_to_check = st.selectbox("Select Column for Anomaly Detection:", numeric_columns)

        df["z_score"] = (df[column_to_check] - df[column_to_check].mean()) / df[column_to_check].std()
        df["Anomaly"] = df["z_score"].apply(lambda x: "Anomaly" if abs(x) > 3 else "Normal")

        st.write("🔎 Detected Anomalies:")
        st.write(df[df["Anomaly"] == "Anomaly"])

        fig = px.scatter(df, x=df.index, y=column_to_check, color="Anomaly", title="Anomaly Detection")
        st.plotly_chart(fig)

        df.drop(columns=["z_score", "Anomaly"], inplace=True)
    else:
        st.warning("⚠️ No numeric columns found for anomaly detection.")

    # --- PREDICTIVE ANALYTICS (FORECASTING) ---
    st.subheader("🔮 Predictive Analytics (AI Forecasting)")

    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
    
    if date_columns:
        date_column = st.selectbox("Select Date Column:", date_columns)
        value_column = st.selectbox("Select Value Column:", [col for col in df.columns if col != date_column])

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column])

        if df.empty:
            st.error("❌ No valid data available after cleaning. Please check your file.")
        else:
            df = df.rename(columns={date_column: "ds", value_column: "y"})
            df = df[df["ds"].notna()]
            df = df[df["ds"].apply(lambda x: isinstance(x, pd.Timestamp))]

            if df.empty:
                st.error("❌ No valid date values found in the selected column.")
            else:
                model = Prophet()
                model.fit(df)

                period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
                future = model.make_future_dataframe(periods=period)
                future = future.dropna()

                forecast = model.predict(future)
                st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                st.line_chart(forecast.set_index("ds")["yhat"])
    else:
        st.warning("⚠️ No date column found for forecasting.")

else:
    st.warning("⚠️ Please upload a CSV/XLSX file first.")
