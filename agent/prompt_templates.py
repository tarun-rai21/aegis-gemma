import json

def format_forecast_prompt(location_name: str, forecast: dict) -> str:
    return f"""Location: {location_name}
Peak forecast — Hour {forecast['peak_risk_hour']}:00: {forecast['peak_temp_c']}°C | RH {forecast['peak_humidity_pct']}% | WBT {forecast['peak_wet_bulb_c']}°C
Risk: {forecast['risk_level']} — {forecast['risk_label']}
Source: {forecast['source']}

Provide brief survival advice for this heat risk level. Be concise."""


def format_tool_result(tool_name: str, result: dict) -> str:
    return f"[Tool: {tool_name}] Result: {json.dumps(result, indent=2)}"