# 🔥 Aegis-Gemma: Global Heatwave Guardian

> Offline-capable AI agent combining a locally-trained LSTM heatwave forecaster with Gemma 4 agentic reasoning to deliver life-saving, location-specific survival advice — no internet required.

**Gemma 4 Good Hackathon — Global Resilience Track — May 2026**

---

## 🎯 One-Line Pitch
Aegis-Gemma is an offline AI agent that uses a locally-trained LSTM to forecast heatwaves and Gemma 4's agentic reasoning to deliver life-saving advice — no internet required.

---
## 🎥 Demo Video
[Watch the 3-minute demo on YouTube](#)

## ⚡ Quick Start (For Judges)

**Requirements:** 16GB RAM, Windows/Mac/Linux, Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/tarun-rai21/aegis-gemma.git
cd aegis-gemma

# 2. Install Ollama from https://ollama.com and pull Gemma 4
ollama pull gemma4:e4b

# 3. Set up Python environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> **Offline demo:** Turn on Airplane Mode after setup — the app switches to offline mode automatically using cached weather data and local Gemma 4.

---

## 🏗️ Architecture

```bash
User Location
↓
L1 — Ingestion    : Open-Meteo API → last 30 days hourly weather (falls back to CSV cache offline)
↓
L2 — Prediction   : LSTM Forecaster → 720 steps input → 24h forecast (temp + humidity)
↓
L3 — Reasoning    : Gemma 4 E4B (Thinking Mode) → interprets forecast, assesses WBT risk
↓
L4 — Action       : Tool Calling → calculate_wet_bulb() → get_advisory() → send_alert()
```

---

## ✨ Key Features
- **Hybrid AI** — LSTM forecaster + Gemma 4 LLM, not just a chatbot
- **Fully offline** — Ollama runs Gemma 4 locally, CSV cache for weather fallback
- **Native tool calling** — Gemma 4 autonomously calls Python functions
- **Real ML model** — trained on 2.63M rows across 100 cities, 7 climate zones
- **Life-safety focus** — Wet-Bulb Temperature risk levels 0–4 with survival protocols

---

## 📊 LSTM Model Performance

| Metric | Result | Target |
|--------|--------|--------|
| MAE Temp | 1.28°C | < 1.5°C ✓ |
| RMSE | 1.75°C | < 2.0°C ✓ |
| R² | 0.965 | > 0.92 ✓ |
| India MAE | 1.05°C | < 2.5°C ✓ |

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### Installation

```bash
git clone https://github.com/tarun-rai21/aegis-gemma.git
cd aegis-gemma

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

ollama pull gemma4:e4b
```

### Run

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure
```bash
aegis-gemma/
├── app.py                  # Streamlit demo UI
├── agent/
│   ├── tools.py            # 5 agentic tool functions
│   ├── gemma_client.py     # Ollama API wrapper
│   ├── pipeline.py         # Full agentic pipeline
│   ├── prompt_templates.py # System prompt + formatters
│   └── model_def.py        # BiLSTM + Attention architecture
├── models/
│   ├── lstm_heatwave.pt    # Trained LSTM weights
│   ├── scaler_X.pkl        # Input StandardScaler
│   └── scaler_y.pkl        # Target StandardScaler
├── data/
│   └── offline_cache/      # Pre-cached city CSVs for offline mode
├── requirements.txt
└── LICENSE

```
---

## 🌡️ Heat Risk Levels

| Level | WBT Range | Status | Action |
|-------|-----------|--------|--------|
| 0 | < 26°C | 🟢 Safe | Normal activity |
| 1 | 26–28°C | 🟡 Caution | Reduce exertion |
| 2 | 28–31°C | 🟠 Danger | Seek shade |
| 3 | 31–35°C | 🔴 Severe | Activate cooling centres |
| 4 | > 35°C | ⚫ Lethal | Full evacuation |

---

## 🤖 Gemma 4 Features Used
- **Gemma 4 E4B** — edge model for offline deployment
- **Thinking Mode** — deep reasoning over forecast data
- **Native Function Calling** — autonomous tool invocation

---

## 📋 Training Data
- 100 cities × 7 climate zones × 3 years hourly data
- 2,630,400 rows, zero nulls
- Features: temp, humidity, wind, solar radiation, pressure, precipitation + cyclical time encoding
- Target: 24h temperature + humidity forecast → WBT computed post-hoc

---

## 📄 License
Apache 2.0 — see [LICENSE](LICENSE)

---

*Built for the Gemma 4 Good Hackathon — Global Resilience Track — May 2026*
