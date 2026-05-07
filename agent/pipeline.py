from agent.tools import (
    get_weather_history, run_lstm_forecast,
    calculate_wet_bulb, get_advisory, send_alert
)
from agent.gemma_client import chat
from agent.prompt_templates import format_forecast_prompt

def run_pipeline(lat: float, lon: float, location_name: str,
                 user_profile: str = "general",
                 chat_history: list = []) -> dict:
    results = {}

    print("⏳ Fetching weather history...")
    history = get_weather_history(lat, lon, city_name=location_name)
    results["data_source"] = history["source"]

    if history["source"] == "error":
        return {"error": history.get("error", "Unknown fetch error")}

    print("⏳ Running LSTM forecast...")
    forecast = run_lstm_forecast(history)
    results["forecast"] = forecast

    if "error" in forecast:
        return {"error": forecast["error"]}

    advisory = get_advisory(forecast["risk_level"], user_profile)
    results["advisory"] = advisory

    if forecast["risk_level"] >= 2:
        alert_msg = (
            f"⚠️ {forecast['risk_label']} heatwave alert for {location_name}. "
            f"Peak WBT: {forecast['peak_wet_bulb_c']}°C at hour {forecast['peak_risk_hour']}. "
            f"{advisory['advice']}"
        )
        send_alert(alert_msg, "community")
        results["alert_sent"] = True
    else:
        results["alert_sent"] = False

    print("⏳ Asking Gemma to synthesize response...")
    prompt   = format_forecast_prompt(location_name, forecast)
    response = chat(prompt, history=chat_history)
    results["gemma_response"] = response

    return results