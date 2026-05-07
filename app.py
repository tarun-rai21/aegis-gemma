import streamlit as st
import requests
from agent.tools import geocode_city
from agent.pipeline import run_pipeline

st.set_page_config(
    page_title="Aegis-Gemma | Heatwave Guardian",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def is_online() -> bool:
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

ONLINE = is_online()

if not ONLINE:
    st.error("✈️ OFFLINE MODE ACTIVE — Running fully on local models. No internet required.")
else:
    st.success("🌐 Online Mode — Live weather data enabled.")

st.title("🔥 Aegis-Gemma: Global Heatwave Guardian")
st.caption("Offline-capable AI agent · LSTM Forecaster + Gemma 4 Reasoning · Global Resilience Track")
st.divider()

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/fire.png", width=60)
    st.title("Aegis-Gemma")
    st.markdown("**Offline Heatwave Guardian**")
    st.divider()
    st.markdown("#### System Status")
    st.markdown(f"🌐 Network: {'Online' if ONLINE else '**OFFLINE**'}")
    st.markdown(f"🤖 Gemma 4: `gemma4:e4b` via Ollama")
    st.markdown(f"🧠 LSTM: `lstm_heatwave.pt`")
    st.divider()
    st.markdown("#### User Profile")
    user_profile = st.selectbox(
        "Who needs advice?",
        ["general", "farmer", "elderly", "child"],
        index=0
    )
    st.divider()
    st.caption("Gemma 4 Good Hackathon · May 2026")

if "result" not in st.session_state:
    st.session_state.result = None
if "location" not in st.session_state:
    st.session_state.location = None

col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 📍 Location")
    st.info("Enter city name — coordinates fetched automatically.")
    city_input = st.text_input("City name", placeholder="e.g. Lucknow, Cairo, Bangkok")
    run_button = st.button("🔍 Get Heatwave Forecast", use_container_width=True)

    if run_button and city_input.strip():
        with st.spinner("📡 Geocoding location..."):
            geo = geocode_city(city_input.strip(), online=ONLINE)

        if "error" in geo:
            st.error(geo["error"])
        else:
            st.success(f"📍 Found: {geo['name']} ({geo['lat']:.2f}, {geo['lon']:.2f})")
            st.session_state.location = geo

            with st.spinner("⏳ Fetching weather + running LSTM + asking Gemma..."):
                result = run_pipeline(
                    lat=geo["lat"],
                    lon=geo["lon"],
                    location_name=geo["name"],
                    user_profile=user_profile
                )
            st.session_state.result = result

    elif run_button and not city_input.strip():
        st.warning("Please enter a city name.")

with col_right:
    st.markdown("### 📊 Forecast")
    if st.session_state.result is None:
        st.caption("Forecast will appear here after you click the button.")
    elif "error" in st.session_state.result:
        st.error(f"Pipeline error: {st.session_state.result['error']}")
    else:
        result = st.session_state.result
        forecast = result["forecast"]
        advisory = result["advisory"]

        # Risk badge
        risk_colors = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "⚫"}
        risk_emoji  = risk_colors.get(forecast["risk_level"], "🟢")
        st.markdown(f"## {risk_emoji} Risk Level {forecast['risk_level']} — {forecast['risk_label']}")

        # Key metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak Temp", f"{forecast['peak_temp_c']}°C")
        m2.metric("Peak Humidity", f"{forecast['peak_humidity_pct']}%")
        m3.metric("Peak WBT", f"{forecast['peak_wet_bulb_c']}°C")

        st.divider()

        # Forecast chart
        import plotly.graph_objects as go
        hours  = [h["hour"] for h in forecast["hourly"]]
        temps  = [h["temp"] for h in forecast["hourly"]]
        wbts   = [h["wet_bulb"] for h in forecast["hourly"]]
        humids = [h["humidity"] for h in forecast["hourly"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=temps,  mode="lines+markers", name="Temperature (°C)", line=dict(color="orange", width=2)))
        fig.add_trace(go.Scatter(x=hours, y=wbts,   mode="lines+markers", name="Wet-Bulb Temp (°C)", line=dict(color="red", width=2)))
        fig.add_trace(go.Scatter(x=hours, y=humids, mode="lines+markers", name="Humidity (%)", line=dict(color="skyblue", width=2)))
        fig.update_layout(
            title="24-Hour Forecast",
            xaxis_title="Hour",
            yaxis_title="Value",
            legend=dict(orientation="h"),
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Advisory panel
        st.markdown("### 🛡️ Survival Advisory")
        st.warning(advisory["advice"])

        # Alert log
        if result["alert_sent"]:
            st.error("🚨 Alert fired — community notified.")

        # Gemma response
        st.divider()
        st.markdown("### 🤖 Gemma 4 Analysis")
        st.markdown(result["gemma_response"])

        # Data source
        st.caption(f"Data source: `{result['data_source']}` | Model: `gemma4:e4b`")