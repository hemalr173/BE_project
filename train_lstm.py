import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("credit_card_transactions.csv")

if "trans_date_trans_time" in df.columns:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day"] = df["trans_date_trans_time"].dt.day
    df["transaction_month"] = df["trans_date_trans_time"].dt.month
    df["transaction_weekday"] = df["trans_date_trans_time"].dt.weekday
else:
    print("Column 'trans_date_trans_time' missing.")

if "dob" in df.columns:
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = (pd.to_datetime("today") - df["dob"]).dt.days // 365
else:
    print("Column 'dob' missing.")


# Features & target
FEATURES = ["amt", "category", "job", "merch_lat", "merch_long", "lat", "long",
            "transaction_hour", "transaction_day", "transaction_month", "age"]
X = df[FEATURES]
y = df["is_fraud"]

# One-Hot Encoding categorical variables
X = pd.get_dummies(X, columns=["category", "job"], drop_first=True)
X = X.select_dtypes(include=[float, int])

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM (samples, time_steps, features)
X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save Model & Scaler
model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("LSTM Model Trained & Saved!")
