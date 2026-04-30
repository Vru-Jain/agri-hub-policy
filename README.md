# Sovereign Agri-Policy Hub

Agricultural Intelligence Dashboard for Maharashtra & Delhi | 2026

A high-performance Streamlit dashboard providing real-time agricultural policy analysis, ML-powered yield predictions, and intervention recommendations for government officials.

---

##  Features

| Feature | Description |
|---------|-------------|
|  **LSTM Yield Predictions** | PyTorch model trained on 19,000+ crop records from Kaggle |
|  **Satellite Imagery** | Sentinel Hub integration for NDVI and vegetation analysis |
|  **Live Market Prices** | Real-time mandi prices from Data.gov.in (Agmarknet) |
|  **Geospatial Heatmaps** | Interactive Folium maps with district-level risk visualization |
|  **MSP Rate Display** | Current Minimum Support Prices for all major crops |
|  **Executive Dashboard** | Intervention status cards with revenue impact analysis |

---

##  Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-repo/agri-hub-policy.git
cd agri-hub-policy

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (see API Setup section below)
```

### 3. Train the ML Model (Optional)

```bash
# Download Kaggle data and train the LSTM model
python models/train.py
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

---

##  API Setup

| Service | Purpose | How to Get |
|---------|---------|------------|
| **Kaggle** | ML training data | [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create Token |
| **Sentinel Hub** | Satellite imagery | [sentinel-hub.com](https://www.sentinel-hub.com/) → Free tier signup |
| **Data.gov.in** | Mandi prices | [data.gov.in/user/register](https://data.gov.in/user/register) |

> **Note**: The dashboard works in demo mode without API keys, using simulated data.

---

##  Project Structure

```
agri-hub-policy/
├── app.py                   # Main Streamlit entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template (copy to .env)
│
├── config/                  # Configuration
│   ├── msp_rates.py         # MSP rate constants
│   └── theme.py             # CSS themes (Maharashtra/Delhi)
│
├── services/                # Live Data APIs
│   ├── config.py            # API key management
│   ├── kaggle_service.py    # Kaggle dataset integration
│   ├── sentinel.py          # Sentinel Hub satellite data
│   ├── agmarknet.py         # Data.gov.in mandi prices
│   ├── imd_weather.py       # IMD weather data
│   └── data_service.py      # Unified data orchestration
│
├── models/                  # ML Models
│   ├── crop_yield_model.py  # PyTorch LSTM architecture
│   ├── train.py             # Training script
│   └── checkpoints/         # Saved model weights (gitignored)
│
├── utils/                   # Utilities
│   ├── data_loader.py       # Cached data loading
│   ├── economics.py         # Economic calculations
│   └── predictions.py       # Yield prediction interface
│
├── components/              # UI Components
│   ├── sidebar.py           # Sidebar controls
│   ├── metrics.py           # KPI cards
│   ├── alerts.py            # Priority alerts
│   ├── charts.py            # Plotly visualizations
│   └── map.py               # Folium map rendering
│
└── data/                    # Data cache
    └── kaggle_cache/        # Downloaded datasets (gitignored)
```

---

##  ML Model

The crop yield prediction model uses a **PyTorch LSTM** architecture with:

- **Embedding layers** for categorical features (State, Crop, Season)
- **Continuous features**: NDVI, Soil Moisture, Rainfall, Area, Year
- **Training data**: 19,000+ historical crop yield records from Kaggle

### Retrain the Model

```bash
python models/train.py
```

### Use Predictions in Code

```python
from utils.predictions import get_yield_prediction

result = get_yield_prediction(
    ndvi=0.65,
    soil_moisture=0.5,
    season='Rabi',
    state='Maharashtra',
    crop='Rice'
)
# Returns: {'yield_quintals_per_ha': 28.5, 'model_type': 'LSTM', ...}
```

---

##  Security Notes

- **Never commit `.env`** - Contains API secrets (already in `.gitignore`)
- **Model checkpoints are gitignored** - Regenerate with `train.py`
- **Kaggle cache is gitignored** - Data downloads on-demand

---

##  Data Sources

| Source | Data Type |
|--------|-----------|
| [Kaggle](https://www.kaggle.com) | Historical crop yields, agricultural datasets |
| [Sentinel Hub](https://www.sentinel-hub.com) | Satellite imagery, NDVI |
| [Data.gov.in](https://data.gov.in) | Mandi prices (Agmarknet) |
| [IMD](https://mausam.imd.gov.in) | Weather and rainfall data |

---

##  Performance

The dashboard uses Streamlit caching for optimal performance:

- `@st.cache_data` - CSV and API data caching
- `@st.cache_resource` - Folium map objects and ML models

---

##  License

© 2026 Government of India | All Rights Reserved. Built for the Digital India Initiative
