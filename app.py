from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import openai
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import requests
import os

# Initialize Flask app
app = Flask(__name__)
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your API key

# Load sample sales dataset
df = pd.read_csv("sales_data.csv")  # Your sales data file

def detect_anomalies(data):
    """Detect anomalies using Isolation Forest."""
    model = IsolationForest(contamination=0.05)
    data['anomaly'] = model.fit_predict(data[['sales']])
    anomalies = data[data['anomaly'] == -1]
    return anomalies

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Return dashboard data (AI insights, charts)."""
    anomalies = detect_anomalies(df)
    fig = px.line(df, x='date', y='sales', title='Sales Trend')
    return jsonify({
        "chart": fig.to_json(),
        "anomalies": anomalies.to_dict(orient='records')
    })

@app.route('/nlq', methods=['POST'])
def natural_language_query():
    """Handle user queries and generate charts."""
    user_query = request.json.get('query')
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Analyze the dataset and generate insights for: {user_query}",
        max_tokens=100
    )
    return jsonify({"response": response['choices'][0]['text']})

@app.route('/forecast', methods=['GET'])
def forecast():
    """Predict future sales using Prophet."""
    sales_data = df[['date', 'sales']]
    sales_data.columns = ['ds', 'y']  # Prophet requires 'ds' (date) and 'y' (value)
    model = Prophet()
    model.fit(sales_data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].to_json(orient='records')

@app.route('/alerts', methods=['POST'])
def send_alert():
    """Send an alert when anomalies are detected."""
    alert_type = request.json.get('alert_type')
    message = "Sales dropped significantly! Check your dashboard."
    
    if alert_type == "email":
        # Replace with actual email API
        requests.post("https://api.emailservice.com/send", json={"to": "admin@example.com", "message": message})
    elif alert_type == "slack":
        requests.post("https://slack.com/api/chat.postMessage", json={"channel": "#alerts", "text": message})
    
    return jsonify({"status": "Alert Sent!"})

if __name__ == '__main__':
    app.run(debug=True)
