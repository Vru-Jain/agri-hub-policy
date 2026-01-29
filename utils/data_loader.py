"""
Data Loading Utilities for Sovereign Agri-Policy Hub.

Provides cached data loading functions with automatic fallback between
live API data and demo mode using simulated CSV data.
"""

import streamlit as st
import pandas as pd

# Import data service for live/demo mode
try:
    from services.data_service import get_district_data, get_data_mode
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data() -> pd.DataFrame:
    """
    Load agricultural data from live APIs or CSV fallback.
    
    Uses Streamlit's cache_data decorator to cache the data across reruns.
    TTL of 5 minutes ensures data refreshes periodically in live mode.
    
    The function automatically determines whether to use:
    - Live mode: Fetches from IMD, Agmarknet, Sentinel Hub APIs
    - Demo mode: Uses simulated_data.csv (when API keys not configured)
    
    Returns:
        pd.DataFrame: DataFrame containing district-level agricultural data
        
    Note:
        - Set API keys in .env file for live mode
        - Run `python generate_data.py` to create demo data file
    """
    if SERVICES_AVAILABLE:
        try:
            df = get_district_data(use_cache=True)
            return df
        except Exception as e:
            st.warning(f"Live data fetch failed, using demo mode: {e}")
    
    # Fallback to CSV
    try:
        df = pd.read_csv('simulated_data.csv')
        df['data_source'] = 'DEMO'
    except FileNotFoundError:
        st.error("Data file not found. Please run `python generate_data.py` first.")
        st.stop()
    return df


def get_current_data_mode() -> str:
    """
    Get the current data mode (LIVE or DEMO).
    
    Returns:
        String 'LIVE' or 'DEMO' indicating current data source
    """
    if SERVICES_AVAILABLE:
        try:
            return get_data_mode()
        except Exception:
            pass
    return 'DEMO'
