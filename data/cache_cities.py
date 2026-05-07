import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "offline_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CITIES = [
    {"name": "Lucknow",   "lat": 26.85, "lon": 80.95},
    {"name": "Delhi",     "lat": 28.61, "lon": 77.21},
    {"name": "Mumbai",    "lat": 19.08, "lon": 72.88},
    {"name": "Cairo",     "lat": 30.04, "lon": 31.24},
    {"name": "Bangkok",   "lat": 13.75, "lon": 100.52},
    {"name": "Rajasthan", "lat": 26.91, "lon": 75.79},
    {"name": "Phoenix",   "lat": 33.45, "lon": -112.07},
    {"name": "Lagos",     "lat": 6.52,  "lon": 3.38},
    {"name": "Karachi",   "lat": 24.86, "lon": 67.01},
    {"name": "Doha",      "lat": 25.29, "lon": 51.53},
]

FEATURE_COLS = ['temp','humidity','wind_speed','solar_rad',
                'pressure','precip','hour_sin','hour_cos',
                'month_sin','month_cos']

end   = datetime.utcnow().date()
start = end - timedelta(days=30)

for city in CITIES:
    print(f"Caching {city['name']}...")
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  city["lat"],
            "longitude": city["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,"
                      "shortwave_radiation,surface_pressure,precipitation",
            "start_date": str(start),
            "end_date":   str(end),
            "timezone":   "UTC"
        }
        r    = requests.get(url, params=params, timeout=30)
        data = r.json()["hourly"]

        records = []
        for i, ts in enumerate(data["time"]):
            dt = datetime.fromisoformat(ts)
            h  = dt.hour
            m  = dt.month
            records.append({
                "temp":       data["temperature_2m"][i],
                "humidity":   data["relative_humidity_2m"][i],
                "wind_speed": data["windspeed_10m"][i],
                "solar_rad":  data["shortwave_radiation"][i],
                "pressure":   data["surface_pressure"][i],
                "precip":     data["precipitation"][i],
                "hour_sin":   np.sin(2 * np.pi * h / 24),
                "hour_cos":   np.cos(2 * np.pi * h / 24),
                "month_sin":  np.sin(2 * np.pi * m / 12),
                "month_cos":  np.cos(2 * np.pi * m / 12),
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(CACHE_DIR, f"{city['name'].lower()}.csv"), index=False)
        print(f"  ✓ Saved {len(df)} rows")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n✓ Cache complete. Files saved to data/offline_cache/")