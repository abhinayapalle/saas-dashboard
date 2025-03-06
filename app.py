import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from textblob import TextBlob
import numpy as np

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="SaaS Dashboard", layout="wide")

# --- HEADER ---
st.title("📊 SaaS Dashboard for Data Analysis")

# --- FILE UPLOAD ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("✅ File uploaded successfully!")

    # --- DATA CLEANING ---
    st.subheader("📂 Data Preview")
    st.write(df.head())

    # --- ANOMALY DETECTION ---
    st.subheader("🚨 Anomaly Detection")
    
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    
    if numeric_columns:
        column_to_check = st.selectbox("Select Column for Anomaly Detection:", numeric_columns)

        # Calculate z-score
        df["z_score"] = (df[column_to_check] - df[column_to_check].mean()) / df[column_to_check].std()
        df["Anomaly"] = df["z_score"].apply(lambda x: "Anomaly" if abs(x) > 3 else "Normal")

        # Display anomalies
        st.write("🔎 Detected Anomalies:")
        st.write(df[df["Anomaly"] == "Anomaly"])

        # Visualization
        fig = px.scatter(df, x=df.index, y=column_to_check, color="Anomaly", title="Anomaly Detection")
        st.plotly_chart(fig)

        # Drop temporary columns
        df.drop(columns=["z_score", "Anomaly"], inplace=True)

    else:
        st.warning("⚠️ No numeric columns found for anomaly detection.")

    # --- PREDICTIVE ANALYTICS (FORECASTING) ---
    st.subheader("🔮 Predictive Analytics (Forecasting)")

    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
    
    if date_columns:
        date_column = st.selectbox("Select Date Column:", date_columns)
        value_column = st.selectbox("Select Value Column:", [col for col in df.columns if col != date_column])

        # Convert to DateTime format
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")  
        
        # Remove NaN values (invalid/missing dates)
        df = df.dropna(subset=[date_column, value_column])

        # Ensure dataset is not empty
        if df.empty:
            st.error("❌ No valid data available after cleaning. Please check your file.")
        else:
            # Rename for Prophet
            df = df.rename(columns={date_column: "ds", value_column: "y"})

            # Ensure 'ds' column has valid dates
            df = df[df["ds"].notna()]  
            df = df[df["ds"].apply(lambda x: isinstance(x, pd.Timestamp))]  

            if df.empty:
                st.error("❌ No valid date values found in the selected column.")
            else:
                # Prophet Model Training
                model = Prophet()
                model.fit(df)

                # Forecasting
                period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
                future = model.make_future_dataframe(periods=period)

                # Ensure future data does not contain NaN dates
                future = future.dropna()

                # Make Predictions
                forecast = model.predict(future)
                st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                st.line_chart(forecast.set_index("ds")["yhat"])
    else:
        st.warning("⚠️ No date column found for forecasting.")

else:
    st.warning("⚠️ Please upload a CSV/XLSX file first.")
