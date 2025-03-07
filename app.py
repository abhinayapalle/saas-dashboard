import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from transformers import pipeline

# --- Set Streamlit Page Config ---
st.set_page_config(page_title="📊 SaaS Dashboard", layout="wide")
st.title("📊 SaaS Dashboard for Data Analysis & AI-Powered Insights")

# --- Sidebar for File Upload ---
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
    
    st.write("✅ **Data Preview:**")
    st.dataframe(df)

    # --- AI Sentiment Analysis ---
    st.subheader("🧠 AI Sentiment Analysis")
    try:
        sentiment_model = pipeline("sentiment-analysis")
        text_column = st.selectbox("Select Text Column for Sentiment Analysis", df.columns)
        df["Sentiment"] = df[text_column].apply(lambda x: sentiment_model(str(x))[0]["label"])
        st.dataframe(df[[text_column, "Sentiment"]])
    except Exception as e:
        st.error(f"⚠️ Sentiment Analysis Error: {e}")

    # --- AI-Powered Summary ---
    st.subheader("🧠 AI-Powered Data Insights")
    try:
        summary_model = pipeline("summarization")
        summary_text = " ".join(df[text_column].astype(str).tolist())[:1000]  # Limit text for processing
        summary = summary_model(summary_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        st.write("📌 AI Summary:", summary)
    except Exception as e:
        st.error(f"⚠️ AI Summary Error: {e}")

    # --- Data Visualization ---
    st.subheader("📊 Data Visualization")
    x_axis = st.selectbox("Select X-Axis", df.columns)
    y_axis = st.selectbox("Select Y-Axis", df.columns)
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])

    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} Over Time")
    else:
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")

    st.plotly_chart(fig)

    # --- Time-Series Forecasting ---
    st.subheader("📈 Predictive Analytics (Forecasting)")
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
                try:
                    model = Prophet()
                    model.fit(df)

                    period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
                    future = model.make_future_dataframe(periods=period)
                    future = future.dropna()

                    forecast = model.predict(future)
                    st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                    st.line_chart(forecast.set_index("ds")["yhat"])
                except Exception as e:
                    st.error(f"⚠️ Forecasting Error: {e}")
    else:
        st.warning("❌ No date columns found.")

else:
    st.warning("⚠️ Please upload a CSV or Excel file to proceed.")
