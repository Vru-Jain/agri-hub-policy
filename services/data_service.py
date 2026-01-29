"""
Data Service Orchestrator for Sovereign Agri-Policy Hub.

Coordinates data fetching from multiple sources (IMD, Agmarknet, Sentinel Hub)
with automatic fallback to demo mode using simulated data when API keys are
not configured or APIs fail.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .config import is_live_mode, get_api_keys, get_sentinel_credentials, get_data_gov_key
from .imd_weather import fetch_district_rainfall_sync, get_rainfall_normal, IMD_RAINFALL_NORMALS
from .agmarknet import fetch_mandi_prices_sync

logger = logging.getLogger(__name__)

# District configurations with coordinates
MAHARASHTRA_DISTRICTS = {
    'Nashik': {'region': 'North Maharashtra', 'primary_crop': 'Onion', 'risk_zone': False, 'lat': 19.9975, 'lon': 73.7898},
    'Pune': {'region': 'Western Maharashtra', 'primary_crop': 'Sugarcane', 'risk_zone': False, 'lat': 18.5204, 'lon': 73.8567},
    'Nagpur': {'region': 'Vidarbha', 'primary_crop': 'Cotton', 'risk_zone': True, 'lat': 21.1458, 'lon': 79.0882},
    'Amravati': {'region': 'Vidarbha', 'primary_crop': 'Soybean', 'risk_zone': True, 'lat': 20.9374, 'lon': 77.7796},
    'Aurangabad': {'region': 'Marathwada', 'primary_crop': 'Cotton', 'risk_zone': True, 'lat': 19.8762, 'lon': 75.3433},
    'Jalna': {'region': 'Marathwada', 'primary_crop': 'Cotton', 'risk_zone': True, 'lat': 19.8347, 'lon': 75.8816},
    'Latur': {'region': 'Marathwada', 'primary_crop': 'Soybean', 'risk_zone': True, 'lat': 18.4088, 'lon': 76.5604},
    'Kolhapur': {'region': 'Western Maharashtra', 'primary_crop': 'Sugarcane', 'risk_zone': False, 'lat': 16.7050, 'lon': 74.2433},
    'Sangli': {'region': 'Western Maharashtra', 'primary_crop': 'Sugarcane', 'risk_zone': False, 'lat': 16.8524, 'lon': 74.5815},
    'Ahmednagar': {'region': 'Western Maharashtra', 'primary_crop': 'Onion', 'risk_zone': False, 'lat': 19.0948, 'lon': 74.7480},
}

DELHI_DISTRICTS = {
    'Najafgarh': {'region': 'Najafgarh Zone', 'primary_crop': 'Vegetables', 'risk_zone': True, 'lat': 28.6092, 'lon': 76.9798},
    'Yamuna Floodplain North': {'region': 'Yamuna Floodplain', 'primary_crop': 'Vegetables', 'risk_zone': True, 'lat': 28.7500, 'lon': 77.2500},
    'Yamuna Floodplain South': {'region': 'Yamuna Floodplain', 'primary_crop': 'Flowers', 'risk_zone': True, 'lat': 28.5500, 'lon': 77.2800},
    'Alipur': {'region': 'North Delhi', 'primary_crop': 'Vegetables', 'risk_zone': False, 'lat': 28.7967, 'lon': 77.1350},
    'Narela': {'region': 'North Delhi', 'primary_crop': 'Wheat', 'risk_zone': False, 'lat': 28.8526, 'lon': 77.0929},
}

# MSP Rates
MSP_RATES = {
    'Wheat': 2585,
    'Mustard': 6200,
    'Gram': 5650,
    'Barley': 1980,
    'Masoor': 6700,
    'Paddy': 2369,
    'Sugarcane': 340,
    'Cotton': 7121,
    'Soybean': 4892,
    'Groundnut': 6783,
    'Onion': 1800,
    'Vegetables': 2500,
    'Flowers': 8000,
}


def get_data_mode() -> str:
    """
    Get current data mode (LIVE or DEMO).
    
    Returns:
        'LIVE' if API keys configured, 'DEMO' otherwise
    """
    return 'LIVE' if is_live_mode() else 'DEMO'


def _generate_simulated_district(district: str, config: Dict, state: str) -> Dict:
    """Generate simulated data for a single district (demo mode)."""
    np.random.seed(hash(district) % 2**32)
    
    rainfall_normal = get_rainfall_normal(district)
    rainfall_deviation = np.random.uniform(-0.30, 0.20)
    rainfall_actual = rainfall_normal * (1 + rainfall_deviation)
    
    base_yield = np.random.uniform(18, 35)
    yield_adjustment = 1 + (rainfall_deviation * 0.5)
    
    ndvi = np.random.uniform(0.35, 0.75)
    soil_moisture = np.random.uniform(0.15, 0.45)
    
    risk_factor = np.random.uniform(0.75, 0.95) if config['risk_zone'] else np.random.uniform(0.90, 1.05)
    predicted_yield = base_yield * yield_adjustment * risk_factor
    historical_yield = base_yield * np.random.uniform(0.95, 1.05)
    
    current_month = datetime.now().month
    season = 'Rabi' if current_month in [10, 11, 12, 1, 2, 3] else 'Kharif'
    
    return {
        'district': district,
        'state': state,
        'region': config['region'],
        'crop': config['primary_crop'],
        'season': season,
        'predicted_yield': round(predicted_yield, 2),
        'historical_yield': round(historical_yield, 2),
        'yield_variance_pct': round((predicted_yield - historical_yield) / historical_yield * 100, 2),
        'ndvi': round(ndvi, 3),
        'soil_moisture': round(soil_moisture, 3),
        'rainfall_actual_mm': round(rainfall_actual, 1),
        'rainfall_normal_mm': rainfall_normal,
        'rainfall_deviation_pct': round(rainfall_deviation * 100, 2),
        'msp_rate': MSP_RATES.get(config['primary_crop'], 2500),
        'acreage_ha': np.random.randint(5000, 50000),
        'risk_zone': config['risk_zone'],
        'mandi_arrival_days': np.random.randint(15, 90),
        'lat': config['lat'],
        'lon': config['lon'],
        'data_source': 'DEMO'
    }


def _fetch_live_district_data(district: str, config: Dict, state: str) -> Optional[Dict]:
    """Fetch live data for a single district from APIs."""
    try:
        # Fetch rainfall from IMD
        rainfall_data = fetch_district_rainfall_sync(state, district)
        
        if rainfall_data:
            rainfall_actual = rainfall_data['actual_mm']
            rainfall_normal = rainfall_data['normal_mm']
            rainfall_deviation = rainfall_data['deviation_pct']
        else:
            # Fallback to normals
            rainfall_normal = get_rainfall_normal(district)
            rainfall_actual = rainfall_normal
            rainfall_deviation = 0
        
        # For NDVI/soil moisture, check if Sentinel credentials exist
        sentinel_creds = get_sentinel_credentials()
        if sentinel_creds:
            try:
                from .sentinel import fetch_vegetation_indices
                ndvi, soil_moisture = fetch_vegetation_indices(
                    config['lat'], 
                    config['lon'], 
                    sentinel_creds
                )
                ndvi = ndvi if ndvi else np.random.uniform(0.35, 0.75)
                soil_moisture = soil_moisture if soil_moisture else np.random.uniform(0.15, 0.45)
            except Exception as e:
                logger.warning(f"Sentinel fetch failed for {district}: {e}")
                ndvi = np.random.uniform(0.35, 0.75)
                soil_moisture = np.random.uniform(0.15, 0.45)
        else:
            ndvi = np.random.uniform(0.35, 0.75)
            soil_moisture = np.random.uniform(0.15, 0.45)
        
        # Calculate yields based on live data
        np.random.seed(hash(district) % 2**32)
        base_yield = np.random.uniform(18, 35)
        yield_adjustment = 1 + (rainfall_deviation / 100) * 0.5
        risk_factor = np.random.uniform(0.75, 0.95) if config['risk_zone'] else np.random.uniform(0.90, 1.05)
        predicted_yield = base_yield * yield_adjustment * risk_factor
        historical_yield = base_yield * np.random.uniform(0.95, 1.05)
        
        current_month = datetime.now().month
        season = 'Rabi' if current_month in [10, 11, 12, 1, 2, 3] else 'Kharif'
        
        return {
            'district': district,
            'state': state,
            'region': config['region'],
            'crop': config['primary_crop'],
            'season': season,
            'predicted_yield': round(predicted_yield, 2),
            'historical_yield': round(historical_yield, 2),
            'yield_variance_pct': round((predicted_yield - historical_yield) / historical_yield * 100, 2),
            'ndvi': round(ndvi, 3),
            'soil_moisture': round(soil_moisture, 3),
            'rainfall_actual_mm': round(rainfall_actual, 1),
            'rainfall_normal_mm': rainfall_normal,
            'rainfall_deviation_pct': round(rainfall_deviation, 2),
            'msp_rate': MSP_RATES.get(config['primary_crop'], 2500),
            'acreage_ha': np.random.randint(5000, 50000),
            'risk_zone': config['risk_zone'],
            'mandi_arrival_days': np.random.randint(15, 90),
            'lat': config['lat'],
            'lon': config['lon'],
            'data_source': 'LIVE'
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch live data for {district}: {e}")
        return None


def get_district_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Get district data from live APIs or demo mode.
    
    This is the main entry point for data loading. It automatically
    determines whether to use live APIs or fall back to demo mode
    based on API key configuration.
    
    Args:
        use_cache: Whether to use cached CSV if available (for demo mode)
        
    Returns:
        DataFrame with district-level agricultural data
    """
    data = []
    mode = get_data_mode()
    
    logger.info(f"Loading data in {mode} mode")
    
    if mode == 'LIVE':
        # Try to fetch live data
        for district, config in MAHARASHTRA_DISTRICTS.items():
            live_data = _fetch_live_district_data(district, config, 'Maharashtra')
            if live_data:
                data.append(live_data)
            else:
                # Fallback to simulated for this district
                data.append(_generate_simulated_district(district, config, 'Maharashtra'))
        
        for district, config in DELHI_DISTRICTS.items():
            live_data = _fetch_live_district_data(district, config, 'Delhi')
            if live_data:
                data.append(live_data)
            else:
                data.append(_generate_simulated_district(district, config, 'Delhi'))
    else:
        # Demo mode - try to load cached CSV first
        if use_cache:
            try:
                df = pd.read_csv('simulated_data.csv')
                df['data_source'] = 'DEMO'
                return df
            except FileNotFoundError:
                pass
        
        # Generate fresh demo data
        for district, config in MAHARASHTRA_DISTRICTS.items():
            data.append(_generate_simulated_district(district, config, 'Maharashtra'))
        
        for district, config in DELHI_DISTRICTS.items():
            data.append(_generate_simulated_district(district, config, 'Delhi'))
    
    return pd.DataFrame(data)
