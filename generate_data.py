"""
Simulated Data Generator for Sovereign Agri-Policy Hub
Generates district-level agricultural data for Maharashtra and Delhi
"""

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# District configurations
MAHARASHTRA_DISTRICTS = {
    'Nashik': {'region': 'North Maharashtra', 'primary_crop': 'Onion', 'risk_zone': False},
    'Pune': {'region': 'Western Maharashtra', 'primary_crop': 'Sugarcane', 'risk_zone': False},
    'Nagpur': {'region': 'Vidarbha', 'primary_crop': 'Cotton', 'risk_zone': True},
    'Amravati': {'region': 'Vidarbha', 'primary_crop': 'Soybean', 'risk_zone': True},
    'Aurangabad': {'region': 'Marathwada', 'primary_crop': 'Cotton', 'risk_zone': True},
    'Jalna': {'region': 'Marathwada', 'primary_crop': 'Cotton', 'risk_zone': True},
    'Latur': {'region': 'Marathwada', 'primary_crop': 'Soybean', 'risk_zone': True},
    'Kolhapur': {'region': 'Western Maharashtra', 'primary_crop': 'Sugarcane', 'risk_zone': False},
    'Sangli': {'region': 'Western Maharashtra', 'primary_crop': 'Sugarcane', 'risk_zone': False},
    'Ahmednagar': {'region': 'Western Maharashtra', 'primary_crop': 'Onion', 'risk_zone': False},
}

DELHI_DISTRICTS = {
    'Najafgarh': {'region': 'Najafgarh Zone', 'primary_crop': 'Vegetables', 'risk_zone': True},
    'Yamuna Floodplain North': {'region': 'Yamuna Floodplain', 'primary_crop': 'Vegetables', 'risk_zone': True},
    'Yamuna Floodplain South': {'region': 'Yamuna Floodplain', 'primary_crop': 'Flowers', 'risk_zone': True},
    'Alipur': {'region': 'North Delhi', 'primary_crop': 'Vegetables', 'risk_zone': False},
    'Narela': {'region': 'North Delhi', 'primary_crop': 'Wheat', 'risk_zone': False},
}

# MSP Rates (₹ per quintal)
MSP_RATES = {
    # 2026-27 Rabi
    'Wheat': 2585,
    'Mustard': 6200,
    'Gram': 5650,
    'Barley': 1980,
    'Masoor': 6700,
    # 2025-26 Kharif
    'Paddy': 2369,
    'Sugarcane': 340,  # FRP
    'Cotton': 7121,
    'Soybean': 4892,
    'Groundnut': 6783,
    # Non-MSP (market rates)
    'Onion': 1800,
    'Vegetables': 2500,
    'Flowers': 8000,
}

# IMD 50-year rainfall normals (mm)
IMD_RAINFALL_NORMALS = {
    'Nashik': 650,
    'Pune': 720,
    'Nagpur': 980,
    'Amravati': 850,
    'Aurangabad': 680,
    'Jalna': 620,
    'Latur': 750,
    'Kolhapur': 1100,
    'Sangli': 580,
    'Ahmednagar': 520,
    'Najafgarh': 650,
    'Yamuna Floodplain North': 680,
    'Yamuna Floodplain South': 680,
    'Alipur': 620,
    'Narela': 600,
}


def generate_district_data(district, config, state):
    """Generate simulated data for a single district"""
    
    rainfall_normal = IMD_RAINFALL_NORMALS.get(district, 700)
    # Simulate rainfall deviation (-30% to +20%)
    rainfall_deviation = np.random.uniform(-0.30, 0.20)
    rainfall_actual = rainfall_normal * (1 + rainfall_deviation)
    
    # Base yield varies by crop and region
    base_yield = np.random.uniform(18, 35)  # quintals per hectare
    
    # Adjust yield based on rainfall deviation
    yield_adjustment = 1 + (rainfall_deviation * 0.5)  # 50% correlation with rainfall
    
    # Add NDVI and soil moisture
    ndvi = np.random.uniform(0.35, 0.75)  # Vegetation index
    soil_moisture = np.random.uniform(0.15, 0.45)  # Soil moisture fraction
    
    # Calculate predicted yield
    if config['risk_zone']:
        risk_factor = np.random.uniform(0.75, 0.95)  # Risk zones have lower yields
    else:
        risk_factor = np.random.uniform(0.90, 1.05)
    
    predicted_yield = base_yield * yield_adjustment * risk_factor
    historical_yield = base_yield * np.random.uniform(0.95, 1.05)
    
    # Acreage in hectares
    acreage = np.random.randint(5000, 50000)
    
    # Get MSP rate
    crop = config['primary_crop']
    msp_rate = MSP_RATES.get(crop, 2500)
    
    # Mandi arrival timing (days from now)
    mandi_arrival_days = np.random.randint(15, 90)
    
    # Season determination
    current_month = datetime.now().month
    season = 'Rabi' if current_month in [10, 11, 12, 1, 2, 3] else 'Kharif'
    
    return {
        'district': district,
        'state': state,
        'region': config['region'],
        'crop': crop,
        'season': season,
        'predicted_yield': round(predicted_yield, 2),
        'historical_yield': round(historical_yield, 2),
        'yield_variance_pct': round((predicted_yield - historical_yield) / historical_yield * 100, 2),
        'ndvi': round(ndvi, 3),
        'soil_moisture': round(soil_moisture, 3),
        'rainfall_actual_mm': round(rainfall_actual, 1),
        'rainfall_normal_mm': rainfall_normal,
        'rainfall_deviation_pct': round(rainfall_deviation * 100, 2),
        'msp_rate': msp_rate,
        'acreage_ha': acreage,
        'risk_zone': config['risk_zone'],
        'mandi_arrival_days': mandi_arrival_days,
        'lat': np.random.uniform(18.5, 21.5) if state == 'Maharashtra' else np.random.uniform(28.4, 28.9),
        'lon': np.random.uniform(73.0, 80.0) if state == 'Maharashtra' else np.random.uniform(76.8, 77.4),
    }


def generate_all_data():
    """Generate complete dataset for all districts"""
    data = []
    
    # Maharashtra districts
    for district, config in MAHARASHTRA_DISTRICTS.items():
        data.append(generate_district_data(district, config, 'Maharashtra'))
    
    # Delhi districts
    for district, config in DELHI_DISTRICTS.items():
        data.append(generate_district_data(district, config, 'Delhi'))
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    df = generate_all_data()
    df.to_csv('simulated_data.csv', index=False)
    print(f"✅ Generated simulated_data.csv with {len(df)} district records")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
