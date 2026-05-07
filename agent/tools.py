import numpy as np
import pandas as pd
import pickle, json, os, requests
from datetime import datetime, timedelta

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CACHE_DIR  = os.path.join(BASE_DIR, "data", "offline_cache")
ALERT_LOG  = os.path.join(BASE_DIR, "data", "alert_log.txt")

with open(os.path.join(MODEL_DIR, "scaler_X.pkl"), "rb") as f:
    scaler_X = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler_y.pkl"), "rb") as f:
    scaler_y = pickle.load(f)

FEATURE_COLS = ['temp','humidity','wind_speed','solar_rad',
                'pressure','precip','hour_sin','hour_cos',
                'month_sin','month_cos']

def get_weather_history(lat: float, lon: float, city_name: str = "") -> dict:
    end   = datetime.utcnow().date()
    start = end - timedelta(days=30)
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,"
                      "shortwave_radiation,surface_pressure,precipitation",
            "start_date": str(start), "end_date": str(end),
            "timezone": "UTC"
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["hourly"]

        records = []
        for i, ts in enumerate(data["time"]):
            dt = datetime.fromisoformat(ts)
            h  = dt.hour
            m  = dt.month
            records.append({
                "timestamp":  ts,
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
        return {"source": "live", "records": records[-720:]}

    except Exception as e:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")] if os.path.exists(CACHE_DIR) else []
        if cache_files:
            # Try to match city name first
            city_key = city_name.lower() if city_name else ""
            matched  = [f for f in cache_files if city_key in f]
            chosen   = matched[0] if matched else cache_files[0]
            path     = os.path.join(CACHE_DIR, chosen)
            df   = pd.read_csv(path).tail(720)
            return {"source": "cache", "records": df[FEATURE_COLS].to_dict("records")}
        return {"source": "error", "records": [], "error": str(e)}


def run_lstm_forecast(history: dict) -> dict:
    import torch
    from agent.model_def import AegisLSTM

    records = history["records"]
    if len(records) < 720:
        return {"error": f"Need 720 records, got {len(records)}"}

    df  = pd.DataFrame(records)[FEATURE_COLS].astype(np.float32)
    X   = scaler_X.transform(df.values)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    model_path = os.path.join(MODEL_DIR, "lstm_heatwave.pt")
    model = AegisLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred = model(X_t).squeeze(0).numpy()

    pred_inv = scaler_y.inverse_transform(pred)

    hourly = []
    for i in range(24):
        t   = float(np.clip(pred_inv[i, 0], -50, 60))
        rh  = float(np.clip(pred_inv[i, 1], 1, 100))
        wbt = calculate_wet_bulb(t, rh)
        hourly.append({"hour": i, "temp": round(t, 2),
                        "humidity": round(rh, 2), "wet_bulb": round(wbt, 2)})

    peak    = max(hourly, key=lambda x: x["wet_bulb"])
    rl      = get_risk_level(peak["wet_bulb"])

    return {
        "source":            history["source"],
        "forecast_hours":    24,
        "peak_risk_hour":    peak["hour"],
        "peak_temp_c":       peak["temp"],
        "peak_humidity_pct": peak["humidity"],
        "peak_wet_bulb_c":   peak["wet_bulb"],
        "risk_level":        rl["level"],
        "risk_label":        rl["label"],
        "hourly":            hourly
    }


def calculate_wet_bulb(temp_c: float, humidity_pct: float) -> float:
    T  = np.clip(temp_c, 10.0, 50.0)
    RH = np.clip(humidity_pct, 1.0, 99.0)
    return round(float(
        T * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    ), 2)


ADVISORIES = {
    0: {"label": "Safe",    "color": "green",  "advice": "Conditions are safe. Stay hydrated."},
    1: {"label": "Caution", "color": "yellow", "advice": "Reduce outdoor exertion. Drink water frequently. Wear light clothing."},
    2: {"label": "Danger",  "color": "orange", "advice": "Elderly and children at serious risk. Seek shade. Avoid outdoor work between 11am–4pm."},
    3: {"label": "Severe",  "color": "red",    "advice": "Life-threatening for vulnerable people. Activate cooling centres. Wet cloth on neck/wrists. Call health services if dizzy."},
    4: {"label": "Lethal",  "color": "black",  "advice": "UNSURVIVABLE conditions. Full evacuation required. Do NOT go outside. Call emergency services immediately."},
}

def get_advisory(risk_level: int, user_profile: str = "general") -> dict:
    base = ADVISORIES.get(risk_level, ADVISORIES[0]).copy()
    if user_profile == "farmer":
        base["advice"] += " Stop all field work immediately. Move livestock to shade."
    elif user_profile == "elderly":
        base["advice"] += " Check on neighbours. Call family. Do not leave home."
    elif user_profile == "child":
        base["advice"] += " Keep children indoors. Close curtains. Use wet towels."
    return base


def send_alert(message: str, recipient: str = "community") -> dict:
    ts    = datetime.utcnow().isoformat()
    entry = f"[{ts}] TO:{recipient} | {message}\n"
    os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
    with open(ALERT_LOG, "a") as f:
        f.write(entry)
    print(f"🚨 ALERT SENT → {recipient}: {message}")
    return {"status": "sent", "timestamp": ts, "recipient": recipient}


def get_risk_level(wbt: float) -> dict:
    if wbt < 26:   return {"level": 0, "label": "Safe"}
    elif wbt < 28: return {"level": 1, "label": "Caution"}
    elif wbt < 31: return {"level": 2, "label": "Danger"}
    elif wbt < 35: return {"level": 3, "label": "Severe"}
    else:          return {"level": 4, "label": "Lethal"}


CITY_LOOKUP = {
    "lucknow":      {"lat": 26.85, "lon": 80.95,   "name": "Lucknow"},
    "ahmedabad":    {"lat": 23.03, "lon": 72.58,   "name": "Ahmedabad"},
    "nagpur":       {"lat": 21.15, "lon": 79.09,   "name": "Nagpur"},
    "patna":        {"lat": 25.59, "lon": 85.14,   "name": "Patna"},
    "bhubaneswar":  {"lat": 20.30, "lon": 85.82,   "name": "Bhubaneswar"},
    "jaipur":       {"lat": 26.91, "lon": 75.79,   "name": "Jaipur"},
    "jodhpur":      {"lat": 26.29, "lon": 73.02,   "name": "Jodhpur"},
    "varanasi":     {"lat": 25.32, "lon": 82.97,   "name": "Varanasi"},
    "kolkata":      {"lat": 22.57, "lon": 88.36,   "name": "Kolkata"},
    "chennai":      {"lat": 13.08, "lon": 80.27,   "name": "Chennai"},
    "bhopal":       {"lat": 23.26, "lon": 77.41,   "name": "Bhopal"},
    "allahabad":    {"lat": 25.44, "lon": 81.84,   "name": "Allahabad"},
    "gwalior":      {"lat": 26.22, "lon": 78.18,   "name": "Gwalior"},
    "bikaner":      {"lat": 28.02, "lon": 73.31,   "name": "Bikaner"},
    "pune":         {"lat": 18.52, "lon": 73.86,   "name": "Pune"},
    "delhi":        {"lat": 28.61, "lon": 77.21,   "name": "New Delhi"},
    "mumbai":       {"lat": 19.08, "lon": 72.88,   "name": "Mumbai"},
    "hyderabad":    {"lat": 17.38, "lon": 78.49,   "name": "Hyderabad"},
    "karachi":      {"lat": 24.86, "lon": 67.01,   "name": "Karachi"},
    "baghdad":      {"lat": 33.34, "lon": 44.40,   "name": "Baghdad"},
    "tehran":       {"lat": 35.69, "lon": 51.39,   "name": "Tehran"},
    "cairo":        {"lat": 30.04, "lon": 31.24,   "name": "Cairo"},
    "riyadh":       {"lat": 24.69, "lon": 46.72,   "name": "Riyadh"},
    "kuwait city":  {"lat": 29.37, "lon": 47.98,   "name": "Kuwait City"},
    "doha":         {"lat": 25.29, "lon": 51.53,   "name": "Doha"},
    "muscat":       {"lat": 23.61, "lon": 58.59,   "name": "Muscat"},
    "abu dhabi":    {"lat": 24.47, "lon": 54.37,   "name": "Abu Dhabi"},
    "phoenix":      {"lat": 33.45, "lon": -112.07, "name": "Phoenix"},
    "las vegas":    {"lat": 36.17, "lon": -115.14, "name": "Las Vegas"},
    "bangkok":      {"lat": 13.75, "lon": 100.52,  "name": "Bangkok"},
    "lagos":        {"lat": 6.52,  "lon": 3.38,    "name": "Lagos"},
    "jakarta":      {"lat": -6.21, "lon": 106.85,  "name": "Jakarta"},
    "dhaka":        {"lat": 23.81, "lon": 90.41,   "name": "Dhaka"},
    "kuala lumpur": {"lat": 3.14,  "lon": 101.69,  "name": "Kuala Lumpur"},
    "accra":        {"lat": 5.56,  "lon": -0.20,   "name": "Accra"},
    "colombo":      {"lat": 6.93,  "lon": 79.85,   "name": "Colombo"},
    "moscow":       {"lat": 55.75, "lon": 37.62,   "name": "Moscow"},
    "chicago":      {"lat": 41.85, "lon": -87.65,  "name": "Chicago"},
    "beijing":      {"lat": 39.91, "lon": 116.39,  "name": "Beijing"},
    "tokyo":        {"lat": 35.69, "lon": 139.69,  "name": "Tokyo"},
    "sydney":       {"lat": -33.87,"lon": 151.21,  "name": "Sydney"},
    "madrid":       {"lat": 40.42, "lon": -3.70,   "name": "Madrid"},
    "paris":        {"lat": 48.85, "lon": 2.35,    "name": "Paris"},
    "nairobi":      {"lat": -1.29, "lon": 36.82,   "name": "Nairobi"},
    "addis ababa":  {"lat": 9.03,  "lon": 38.74,   "name": "Addis Ababa"},
    "bogota":       {"lat": 4.71,  "lon": -74.07,  "name": "Bogota"},
    "lima":         {"lat": -12.05,"lon": -77.04,  "name": "Lima"},
    "lhasa":        {"lat": 29.65, "lon": 91.17,   "name": "Lhasa"},
    "kathmandu":    {"lat": 27.71, "lon": 85.31,   "name": "Kathmandu"},
}


def geocode_city(city_name: str, online: bool = True) -> dict:
    key = city_name.strip().lower()
    if online:
        try:
            url     = "https://nominatim.openstreetmap.org/search"
            params  = {"q": city_name, "format": "json", "limit": 1}
            headers = {"User-Agent": "aegis-gemma/1.0"}
            r       = requests.get(url, params=params, headers=headers, timeout=10)
            data    = r.json()
            if data:
                return {
                    "name": data[0]["display_name"].split(",")[0],
                    "lat":  float(data[0]["lat"]),
                    "lon":  float(data[0]["lon"])
                }
        except:
            pass
    if key in CITY_LOOKUP:
        return CITY_LOOKUP[key]
    return {
        "error": f"City '{city_name}' not found offline. "
                 f"Available: {', '.join(sorted(CITY_LOOKUP.keys())[:10])}..."
    }