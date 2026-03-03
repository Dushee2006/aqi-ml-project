import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# 🔴 REPLACE WITH YOUR REAL API KEY
API_KEY = "2283a1787b937dd569ea23e00bfa72a9"

LAT = 6.9271
LON = 79.8612


def fetch_historical_data(days=120):
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=days)).timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={API_KEY}"
    )

    print("Request URL:", url)  # Debug

    response = requests.get(url)

    print("Status Code:", response.status_code)  # Debug

    data = response.json()

    print("API Response Preview:", str(data)[:500])  # Debug (first 500 chars)

    if "list" not in data:
        print("Error: 'list' not found in API response.")
        return pd.DataFrame()

    rows = []

    for item in data["list"]:
        row = {
            "timestamp": datetime.fromtimestamp(item["dt"]),
            "aqi": item["main"]["aqi"],
            "pm2_5": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "co": item["components"]["co"],
            "no2": item["components"]["no2"],
            "o3": item["components"]["o3"],
            "so2": item["components"]["so2"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":
    df = fetch_historical_data(days=120)

    if df.empty:
        print("No data returned.")
    else:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/aqi_data.csv", index=False)

        print("Historical data saved.")
        print("Total rows:", len(df))