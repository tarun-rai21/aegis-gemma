from agent.pipeline import run_pipeline

result = run_pipeline(26.85, 80.95, 'Lucknow', 'general')

if "error" in result:
    print(f"ERROR: {result['error']}")
else:
    print(f"Risk level  : {result['forecast']['risk_level']}")
    print(f"Risk label  : {result['forecast']['risk_label']}")
    print(f"Peak WBT    : {result['forecast']['peak_wet_bulb_c']}°C")
    print(f"Alert sent  : {result['alert_sent']}")
    print(f"Data source : {result['data_source']}")
    print(f"\nGemma says:\n{result['gemma_response'][:500]}")