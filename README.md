# Sovereign Agri-Policy Hub

Agricultural Intelligence Dashboard for Maharashtra & Delhi | 2026

A high-performance Streamlit dashboard providing real-time agricultural policy analysis, yield predictions, and intervention recommendations for government officials.

## Features

- **State-specific themes** - Dynamic UI with Maharashtra (Sahyadri Sunset) and Delhi (Modern Metro) color schemes
- **LSTM Yield Predictions** - Seasonal weighting for Rabi and Kharif crops
- **Geospatial Heatmaps** - Interactive Folium maps with district-level risk visualization
- **MSP Rate Display** - Current Minimum Support Prices for all major crops
- **Executive Dashboard** - Intervention status cards with revenue impact analysis
- **Supply Chain Forecasting** - Mandi arrival predictions with urgency levels

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_data.py

# Run the dashboard
streamlit run app.py
```

## Project Structure

```
agri-hub-policy/
├── app.py                  # Main entry point (~85 lines)
├── requirements.txt        # Dependencies
├── generate_data.py        # Sample data generator
├── simulated_data.csv      # Generated data file
│
├── config/                 # Configuration modules
│   ├── msp_rates.py        # MSP rate constants
│   └── theme.py            # CSS themes and styling
│
├── utils/                  # Utility functions
│   ├── data_loader.py      # Cached data loading
│   ├── economics.py        # Economic calculations
│   └── predictions.py      # LSTM yield simulation
│
└── components/             # UI components
    ├── sidebar.py          # Sidebar controls
    ├── metrics.py          # KPI cards
    ├── alerts.py           # Alert boxes
    ├── charts.py           # Plotly visualizations
    └── map.py              # Folium map rendering
```

## Configuration

### MSP Rates
Edit `config/msp_rates.py` to update Minimum Support Prices:
- `MSP_RATES_RABI_2026_27` - Rabi season rates
- `MSP_RATES_KHARIF_2025_26` - Kharif season rates

### Themes
Modify `config/theme.py` to customize state-specific color schemes.

## Data Sources

- **IMD** - India Meteorological Department (rainfall data)
- **ICAR** - Indian Council of Agricultural Research
- **Ministry of Agriculture & Farmers Welfare**

## Performance

The dashboard uses Streamlit caching for optimal performance:
- `@st.cache_data` - CSV data loading
- `@st.cache_resource` - Folium map objects

## License

© 2026 Government of India | All Rights Reserved
