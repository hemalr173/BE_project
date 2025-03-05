import requests

# Test Fraud Detection
data = {
    "amt": 100,  # Very high amount
    "category":5,  # Rare category
    "job": 20,  # Uncommon job code
    "merch_lat": 45, "merch_long": -120.0,  # Distant merchant location
    "lat": 25.0, "long": -100.0,  # Customer's location far from merchant
    "transaction_hour": 2,  # Unusual late-night transaction
    "transaction_day": 12
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)
print("Fraud Prediction Response:", response.json())


# Test Customer Retrieval
response = requests.get("http://127.0.0.1:5000/customers")
print("Customers Data:", response.json())

# Test Payment Link Creation
response = requests.post("http://127.0.0.1:5000/create_payment", json={"amount": 500, "country": "US"})
print("Payment Link Response:", response.json())
