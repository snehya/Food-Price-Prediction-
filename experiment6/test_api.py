import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "Tomato_Lag1": 47.61,
    "Tomato_Lag2": 46.47,
    "Tomato_PastRollingMean3": 47.43,
    "Month": 7,
    "Quarter": 3,
    "Week": 29,
    "Month_Sin": -0.5,
    "Month_Cos": -0.8660254038
}

response = requests.post(url, json=payload, timeout=10)

print("Status code:", response.status_code)
print("Response:")
print(response.json())
response.raise_for_status()
