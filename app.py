import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from transformers import pipeline  # AI-powered insights

# --- Streamlit UI Config ---
st.set_page_config(page_title="SaaS Dashboard", layout="wide")

# --- Title ---
st.title("📊 SaaS Dashboard for Data Analysis")

# --- File Upload ---
st.sidebar.header("⬆️ Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.success("✅ File uploaded successfully!")
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # --- Data Filtering ---
    st.sidebar.subheader("🔍 Data Filtering")
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    if not numeric_columns.empty:
        selected_numeric_column = st.sidebar.selectbox("Select Numeric Column", numeric_columns)
    if not categorical_columns.empty:
        selected_category = st.sidebar.selectbox("Select Category", categorical_columns)
    
    # --- Anomaly Detection ---
    st.subheader("🚨 Anomaly Detection")

    if not numeric_columns.empty:
        outlier_model = IsolationForest(contamination=0.05)
        df['Anomaly'] = outlier_model.fit_predict(df[[selected_numeric_column]])

        fig = px.scatter(df, x=df.index, y=selected_numeric_column, color=df["Anomaly"].map({1: "Normal", -1: "Anomaly"}))
        st.plotly_chart(fig)

    # --- Predictive Analytics (Forecasting) ---
    st.subheader("📈 Predictive Analytics (Forecasting)")

    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
    
    if date_columns:
        date_column = st.selectbox("Select Date Column:", date_columns)
        value_column = st.selectbox("Select Value Column:", [col for col in df.columns if col != date_column])

        # Convert to DateTime format
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column])

        if df.empty:
            st.error("❌ No valid data after cleaning. Please check your file.")
        else:
            df = df.rename(columns={date_column: "ds", value_column: "y"})

            if df["ds"].isnull().any():
                st.error("❌ Found NaN values in the Date column. Please clean your data.")
            else:
                # Train Prophet Model
                model = Prophet()
                model.fit(df)

                period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
                future = model.make_future_dataframe(periods=period)

                forecast = model.predict(future)

                st.write("🔮 Forecasted Data:")
                st.line_chart(forecast.set_index("ds")["yhat"])
    else:
        st.warning("⚠️ Please include a date column for forecasting.")

    # --- AI Insights using Transformers ---
    st.subheader("🤖 AI-Powered Insights")
    ai_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    user_input = st.text_area("Enter text for AI analysis:")
    if user_input:
        prediction = ai_model(user_input)
        st.write("🔍 Sentiment Analysis:", prediction[0]["label"])

else:
    st.warning("⚠️ Please upload a CSV file first.")
