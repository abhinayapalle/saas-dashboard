import streamlit as st
import pandas as pd
from textblob import TextBlob
from prophet import Prophet

# Set page title
st.title("📊 SaaS Dashboard for Data Analysis")

# Upload File
st.sidebar.header("🔍 Data Filtering")
uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX File", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file based on type
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # Display Data Preview
    st.write("📂 Uploaded Data:", df.head())

    # Category Filtering
    if "Category" in df.columns:
        selected_category = st.sidebar.selectbox("Select Category", df["Category"].dropna().unique())
        filtered_data = df[df["Category"] == selected_category]
        st.write("🔍 Filtered Data:", filtered_data)
    else:
        st.warning("⚠️ No 'Category' column found in the uploaded file.")

    # --- SENTIMENT ANALYSIS ---
    st.subheader("💬 Sentiment Analysis")
    text_columns = [col for col in df.columns if df[col].dtype == 'O']  # Select text columns
    
    if text_columns:
        text_column = st.selectbox("Select Text Column:", text_columns)
        df["Sentiment Score"] = df[text_column].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        st.write("📊 Sentiment Scores:", df[[text_column, "Sentiment Score"]])
    else:
        st.warning("⚠️ No text column found for sentiment analysis.")

    # --- TIME-SERIES FORECASTING ---
    st.subheader("📈 Predictive Analytics (Forecasting)")
    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]

    if date_columns:
        date_column = st.selectbox("Select Date Column:", date_columns)
        value_column = st.selectbox("Select Value Column:", [col for col in df.columns if col != date_column])

        df[date_column] = pd.to_datetime(df[date_column])  # Convert to DateTime
        df = df[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})

        model = Prophet()
        model.fit(df)

        period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

        st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
        st.line_chart(forecast.set_index("ds")["yhat"])
    else:
        st.warning("⚠️ No date column found for forecasting.")

else:
    st.warning("⚠️ Please upload a CSV or XLSX file to proceed.")
import streamlit as st
import pandas as pd
from prophet import Prophet

# --- TIME-SERIES FORECASTING ---
st.subheader("📈 Predictive Analytics (Forecasting)")

if uploaded_file:
    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]

    if date_columns:
        date_column = st.selectbox("Select Date Column:", date_columns)
        value_column = st.selectbox("Select Value Column:", [col for col in df.columns if col != date_column])

        # ✅ Convert to DateTime format
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")  

        # ✅ Remove NaN values (invalid/missing dates)
        df = df.dropna(subset=[date_column, value_column])

        # ✅ Check if dataset is empty after cleaning
        if df.empty:
            st.error("❌ No valid data available after cleaning. Please check your file.")
        else:
            # ✅ Rename for Prophet
            df = df.rename(columns={date_column: "ds", value_column: "y"})

            # ✅ Ensure 'ds' is in correct format
            df = df[df["ds"].notna()]  # Remove invalid dates
            df = df[df["ds"].apply(lambda x: isinstance(x, pd.Timestamp))]  # Ensure proper date format

            if df.empty:
                st.error("❌ No valid date values found in the selected column.")
            else:
                # ✅ Prophet Model Training
                model = Prophet()
                model.fit(df)

                period = st.slider("📅 Select Forecast Period (Days)", 7, 365, 30)
                future = model.make_future_dataframe(periods=period)

                # ✅ Ensure future data does not contain NaN dates
                future = future.dropna()

                # ✅ Make Predictions
                forecast = model.predict(future)
                st.write("🔮 Forecasted Data:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
                st.line_chart(forecast.set_index("ds")["yhat"])
    else:
        st.warning("⚠️ No date column found for forecasting.")
