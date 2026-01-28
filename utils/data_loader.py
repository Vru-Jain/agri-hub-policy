"""
Data Loading Utilities for Sovereign Agri-Policy Hub.

Provides cached data loading functions to improve performance.
"""

import streamlit as st
import pandas as pd


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load simulated agricultural data from CSV file.
    
    Uses Streamlit's cache_data decorator to cache the data across reruns,
    significantly improving load performance.
    
    Returns:
        pd.DataFrame: DataFrame containing district-level agricultural data
        
    Raises:
        FileNotFoundError: If simulated_data.csv is not found
        
    Note:
        Run `python generate_data.py` to create the data file if missing.
    """
    try:
        df = pd.read_csv('simulated_data.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please run `python generate_data.py` first.")
        st.stop()
    return df
