import requests

# Define the API endpoint URL
url = "http://localhost:8000/predict"

# Define the input features
features = {
    "feature1": 1,
    "feature2": 2,
    "feature3": 3
}

# Send a POST request to the API
response = requests.post(url, json={"features": features})

# Check the response
if response.status_code == 200:
    data = response.json()
    prediction = data["prediction"]
    feature_names = data["feature_names"]
    print("Prediction:", prediction)
    print("Feature Names:", feature_names)
else:
    print("Error:", response.status_code)
