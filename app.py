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
        # 📊 Multiple Data Visualizations
        st.subheader("📊 Data Visualization")
        selected_col = st.selectbox("Select Column to Visualize", numeric_cols)

        # Select visualization type
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"]
        )

        # Generate selected visualization
        if viz_type == "Bar Chart":
            fig = px.bar(df, x=df.index, y=selected_col, title=f"{selected_col} Distribution")
        elif viz_type == "Line Chart":
            fig = px.line(df, x=df.index, y=selected_col, title=f"{selected_col} Trend Over Time")
        elif viz_type == "Scatter Plot":
            fig = px.scatter(df, x=df.index, y=selected_col, title=f"{selected_col} Scatter Plot")
        elif viz_type == "Pie Chart":
            fig = px.pie(df, names=selected_col, title=f"{selected_col} Pie Chart")
        elif viz_type == "Histogram":
            fig = px.histogram(df, x=selected_col, title=f"{selected_col} Histogram")
        elif viz_type == "Box Plot":
            fig = px.box(df, y=selected_col, title=f"{selected_col} Box Plot")

        st.plotly_chart(fig)

    # 📌 AI Forecasting with Prophet
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
        df["Sentiment Score"] = df[sentiment_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df["Sentiment Label"] = df["Sentiment Score"].apply(lambda x: "😊 Positive" if x > 0 else "😠 Negative" if x < 0 else "😐 Neutral")

        # Display sentiment summary
        sentiment_counts = df["Sentiment Label"].value_counts()
        st.write("### Sentiment Summary")
        st.write(sentiment_counts)

        # Display sample positive & negative comments (only if available)
        st.write("### Example Comments:")

        positive_examples = df[df["Sentiment Score"] > 0][sentiment_col]
        negative_examples = df[df["Sentiment Score"] < 0][sentiment_col]

        if not positive_examples.empty:
            st.write("✅ **Positive Example:**", positive_examples.sample(1, random_state=42).values[0])
        else:
            st.write("✅ **Positive Example:** No positive comments found.")

        if not negative_examples.empty:
            st.write("❌ **Negative Example:**", negative_examples.sample(1, random_state=42).values[0])
        else:
            st.write("❌ **Negative Example:** No negative comments found.")

        # Sentiment Distribution Plot
        fig_sentiment = px.histogram(df, x="Sentiment Label", title="Sentiment Distribution", color="Sentiment Label")
        st.plotly_chart(fig_sentiment)
    else:
        st.warning("⚠️ No text column found for sentiment analysis!")

else:
    st.warning("⚠️ Please upload a CSV file!")
