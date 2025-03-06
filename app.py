import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import zscore
from prophet import Prophet
import smtplib
from textblob import TextBlob
from email.mime.text import MIMEText

# Function to send email alert
def send_email_alert(subject, message, recipient_email="your_email@gmail.com"):
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"  # Use an App Password (not your real password)

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("📧 Email alert sent successfully!")
    except Exception as e:
        print("⚠️ Error sending email:", e)

# 🎨 Streamlit UI Setup
st.title("📊 SaaS Dashboard for Data Analysis")
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure Date Column Exists
    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])  # Remove rows with missing dates
    df = df.sort_values(by=date_col)   # Sort by date

    st.success("✅ File uploaded successfully!")
    st.write(df.head())

    # Select Numeric Column for Analysis
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    col = st.selectbox("Select Numeric Column for Analysis", numeric_cols)

    # 📊 **1️⃣ Data Visualization**
    st.subheader(f"📊 {col} Trend Over Time")
    fig = px.line(df, x=date_col, y=col, title=f"{col} Over Time")
    st.plotly_chart(fig)

    # 🚨 **2️⃣ Anomaly Detection**
    st.subheader("🚨 Anomaly Detection")
    
    df["zscore"] = zscore(df[col])  # Calculate Z-score
    anomalies = df[np.abs(df["zscore"]) > 2.5]  # Find anomalies

    if not anomalies.empty:
        st.warning(f"⚠️ {len(anomalies)} anomalies detected in {col}!")
        st.write(anomalies[[col, "zscore"]])
        
        # Highlight anomalies in the chart
        anomaly_fig = px.scatter(df, x=date_col, y=col, title=f"Anomalies in {col}")
        anomaly_fig.add_scatter(x=anomalies[date_col], y=anomalies[col], mode="markers", 
                                marker=dict(color="red", size=10), name="Anomalies")
        st.plotly_chart(anomaly_fig)

        # Send email alert
        subject = f"🚨 Alert: {len(anomalies)} Anomalies Detected in {col}"
        message = f"Anomalies found in {col}: {anomalies.to_string()}"
        send_email_alert(subject, message, "recipient_email@gmail.com")
    else:
        st.success("✅ No anomalies detected!")

    # 🔮 **3️⃣ Predictive Analytics (Forecasting)**
    st.subheader("🔮 Predictive Analytics (Forecasting)")

    forecast_days = st.slider("Select Forecast Period (Days)", min_value=7, max_value=365, value=30)
    
    # Prophet requires "ds" (date) and "y" (value) columns
    forecast_df = df[[date_col, col]].rename(columns={date_col: "ds", col: "y"})

    # Train Prophet model
    model = Prophet()
    model.fit(forecast_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Show forecast chart
    forecast_fig = px.line(forecast, x="ds", y="yhat", title="📈 Forecasted Trends")
    st.plotly_chart(forecast_fig)

else:
    st.info("⬆️ Upload a CSV file to start analysis.")
    # Sidebar Filters
selected_category = st.sidebar.selectbox("Select Category", df['Category'].unique())
filtered_data = df[['Category'] == selected_category]

st.write("Filtered Data", filtered_data)
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

df['Sentiment'] = df['Review_Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
st.write("Sentiment Analysis", df[['Review_Text', 'Sentiment']])
