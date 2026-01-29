"""
MSP (Minimum Support Price) Rates Configuration.

Current MSP rates for various crops as per Government of India.
These rates are updated annually and should be verified against
official government sources.
"""

from typing import Dict

# MSP Rates in INR per quintal (100 kg)
# Updated for 2024-25 marketing season
MSP_RATES: Dict[str, int] = {
    # Rabi Crops
    'Wheat': 2585,
    'Mustard': 6200,
    'Gram': 5650,
    'Barley': 1980,
    'Masoor': 6700,
    
    # Kharif Crops
    'Paddy': 2369,
    'Cotton': 7121,
    'Soybean': 4892,
    'Groundnut': 6783,
    
    # Commercial Crops
    'Sugarcane': 340,  # Per quintal FRP
    
    # Horticultural (estimated market rates, not official MSP)
    'Onion': 1800,
    'Vegetables': 2500,
    'Flowers': 8000,
}


def get_msp_rate(crop: str) -> int:
    """
    Get MSP rate for a crop.
    
    Args:
        crop: Crop name
        
    Returns:
        MSP rate in INR per quintal, or 2500 if not found
    """
    return MSP_RATES.get(crop, 2500)


def get_all_msp_rates() -> Dict[str, int]:
    """Get all MSP rates."""
    return MSP_RATES.copy()
