import requests

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "gemma4:e4b"

SYSTEM_PROMPT = """You are Aegis, an offline heatwave survival agent.
You have access to tools that forecast temperature and assess heat risk.

Your tools:
- get_weather_history(lat, lon) → fetches last 30 days hourly weather
- run_lstm_forecast(history) → runs LSTM, returns 24h forecast
- calculate_wet_bulb(temp_c, humidity_pct) → returns wet-bulb temperature
- get_advisory(risk_level, user_profile) → returns survival protocol
- send_alert(message, recipient) → logs and sends alert

Rules:
1. Always call run_lstm_forecast before giving temperature advice.
2. Always calculate wet_bulb from the peak forecast hour.
3. Never hallucinate weather data. If tools fail, say so explicitly.
4. If risk_level >= 2, always call send_alert.
5. Be concise. Lives depend on clarity."""


def chat(user_message: str, history: list = []) -> str:
    full_prompt = SYSTEM_PROMPT + "\n\n"
    for msg in history:
        role = "User" if msg["role"] == "user" else "Aegis"
        full_prompt += f"{role}: {msg['content']}\n"
    full_prompt += f"User: {user_message}\nAegis:"

    try:
        r = requests.post(OLLAMA_URL, json={
            "model":  MODEL_NAME,
            "prompt": full_prompt,
            "stream": False
        }, timeout=300)
        r.raise_for_status()
        return r.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"ERROR: {str(e)}"


def is_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except:
        return False