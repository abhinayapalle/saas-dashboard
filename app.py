import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- Streamlit App Header ---
st.set_page_config(page_title="📊 SaaS Dashboard", layout="wide")
st.title("📊 SaaS Dashboard for Data Analysis")

# --- File Upload Section ---
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv", "xlsx"])

if uploaded_file:
    # Read CSV/XLSX File
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext == "xlsx":
        df = pd.read_excel(uploaded_file)

    st.write("✅ Data Preview:")
    st.dataframe(df)

    # --- AI-powered Insights ---
    st.subheader("🧠 AI-Powered Insights")

    @st.cache_resource
    def load_model():
        """Load AI model for sentiment analysis with error handling."""
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
        except ImportError as e:
            st.error("⚠️ Missing dependencies. Please install required packages using:")
            st.code("pip install transformers torch torchvision torchaudio", language="bash")
            return None
        except Exception as e:
            st.error(f"⚠️ Error loading AI model: {e}")
            return None

    ai_model = load_model()
    
    if ai_model:
        text_columns = [col for col in df.columns if df[col].dtype == "O"]
        if text_columns:
            selected_text_col = st.selectbox("📝 Select Text Column for Analysis", text_columns)
            df["Sentiment"] = df[selected_text_col].astype(str).apply(lambda x: ai_model(x)[0]['label'])
            st.write("✅ Sentiment Analysis Done")
            st.dataframe(df[[selected_text_col, "Sentiment"]])

    # --- Time-Series Forecasting with Prophet ---
    st.subheader("📈 Predictive Analytics (Forecasting)")

    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
    if date_columns:
        date_column = st.selectbox("📅 Select Date Column", date_columns)
        value_column = st.selectbox("📊 Select Value Column", [col for col in df.columns if col != date_column])

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column])

        if not df.empty:
            df = df.rename(columns={date_column: "ds", value_column: "y"})
            df = df[df["ds"].notna()]

            if not df.empty:
                model = Prophet()
                model.fit(df)

                period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
                future = model.make_future_dataframe(periods=period)
                future = future.dropna()

                forecast = model.predict(future)
                st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                st.line_chart(forecast.set_index("ds")["yhat"])
            else:
                st.error("❌ No valid date values found.")
        else:
            st.error("❌ No valid data available after cleaning.")

    else:
        st.warning("⚠️ No date column found. Please upload a dataset with a proper date column.")

else:
    st.warning("⚠️ Please upload a CSV/XLSX file first.")
