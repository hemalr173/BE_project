import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Ensure the file exists
file_path = "credit_card_transactions.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found.")

# Load dataset
df = pd.read_csv(file_path)

# Check if required columns exist
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

# Define features & target
FEATURES = ["amt", "category", "job", "merch_lat", "merch_long", "lat", "long",
            "transaction_hour", "transaction_day", "transaction_month", "age"]

# Ensure features exist
df = df.dropna(subset=FEATURES)  # Remove rows with missing values in key features

X = df[FEATURES]
y = df["is_fraud"]

# Convert categorical variables
X = pd.get_dummies(X, columns=["category", "job"], drop_first=True, dtype=int)

# Ensure all columns are numeric
X = X.select_dtypes(include=[float, int])

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully!")
