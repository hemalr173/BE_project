from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load Trained LSTM Model & Scaler
lstm_model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Dummy Database
customers = [
    {"id": 1, "name": "Alice", "email": "alice@email.com", "transactions": []},
    {"id": 2, "name": "Bob", "email": "bob@email.com", "transactions": []}
]

# Features used for training
FEATURES = ["amt", "category", "job", "merch_lat", "merch_long", "lat", "long",
            "transaction_hour", "transaction_day"]

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        data = request.json
        X_input = np.array([data[feature] for feature in FEATURES]).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # Reshape for LSTM
        X_lstm = X_scaled.reshape(1, X_scaled.shape[1], 1)
        lstm_pred = lstm_model.predict(X_lstm)[0][0]

        # Decision Threshold (Adjustable)
        is_fraud = 1 if lstm_pred > 0.5 else 0

        return jsonify({"fraud_prediction": int(is_fraud)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/customers', methods=['GET'])
def get_customers():
    return jsonify(customers)

@app.route('/create_payment', methods=['POST'])
def create_payment():
    try:
        data = request.json
        payment_link = f"http://paymentgateway.com/{data['amount']}/{data['country']}"
        return jsonify({"payment_link": payment_link})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
