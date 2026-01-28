"""
Prediction Models for Sovereign Agri-Policy Hub.

Provides LSTM-based yield prediction simulation with seasonal weighting.
"""

import numpy as np


def simulate_lstm_prediction(ndvi: float, soil_moisture: float, season: str = 'Rabi') -> float:
    """
    Simulate LSTM predictive engine with seasonal weighting.
    
    The model uses different feature weights based on crop season:
    - Rabi season: NDVI weighted higher (70%) as vegetation health is 
      critical for winter crops
    - Kharif season: Soil Moisture weighted higher (60%) as crops are 
      monsoon-dependent
    
    Args:
        ndvi: Normalized Difference Vegetation Index (0-1 scale)
        soil_moisture: Soil moisture fraction (0-1 scale)
        season: Crop season, either 'Rabi' or 'Kharif' (default: 'Rabi')
    
    Returns:
        Predicted yield in quintals per hectare (range: 10-40 q/ha)
    """
    if season == 'Rabi':
        ndvi_weight = 0.7
        sm_weight = 0.3
    else:  # Kharif
        ndvi_weight = 0.4
        sm_weight = 0.6
    
    # Normalize and combine features
    ndvi_score = ndvi * 100  # Scale 0-100
    sm_score = soil_moisture * 100  # Scale 0-100
    
    combined_score = (ndvi_weight * ndvi_score) + (sm_weight * sm_score)
    
    # Map to yield prediction (quintals/hectare)
    base_yield = 15 + (combined_score / 100) * 25  # Range: 15-40 q/ha
    
    # Add some randomness for realism
    noise = np.random.uniform(-2, 2)
    predicted_yield = base_yield + noise
    
    return max(10, round(predicted_yield, 2))
