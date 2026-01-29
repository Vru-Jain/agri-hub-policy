"""
District Configurations for Agri-Policy Hub.

Contains geographical and agricultural configuration data for supported districts.
This includes coordinates, primary crops, and risk zone classifications.
"""

from typing import Dict

# Maharashtra District Configurations
MAHARASHTRA_DISTRICTS: Dict[str, Dict] = {
    'Nashik': {
        'region': 'North Maharashtra',
        'primary_crop': 'Onion',
        'risk_zone': False,
        'lat': 19.9975,
        'lon': 73.7898
    },
    'Pune': {
        'region': 'Western Maharashtra',
        'primary_crop': 'Sugarcane',
        'risk_zone': False,
        'lat': 18.5204,
        'lon': 73.8567
    },
    'Nagpur': {
        'region': 'Vidarbha',
        'primary_crop': 'Cotton',
        'risk_zone': True,
        'lat': 21.1458,
        'lon': 79.0882
    },
    'Amravati': {
        'region': 'Vidarbha',
        'primary_crop': 'Soybean',
        'risk_zone': True,
        'lat': 20.9374,
        'lon': 77.7796
    },
    'Aurangabad': {
        'region': 'Marathwada',
        'primary_crop': 'Cotton',
        'risk_zone': True,
        'lat': 19.8762,
        'lon': 75.3433
    },
    'Jalna': {
        'region': 'Marathwada',
        'primary_crop': 'Cotton',
        'risk_zone': True,
        'lat': 19.8347,
        'lon': 75.8816
    },
    'Latur': {
        'region': 'Marathwada',
        'primary_crop': 'Soybean',
        'risk_zone': True,
        'lat': 18.4088,
        'lon': 76.5604
    },
    'Kolhapur': {
        'region': 'Western Maharashtra',
        'primary_crop': 'Sugarcane',
        'risk_zone': False,
        'lat': 16.7050,
        'lon': 74.2433
    },
    'Sangli': {
        'region': 'Western Maharashtra',
        'primary_crop': 'Sugarcane',
        'risk_zone': False,
        'lat': 16.8524,
        'lon': 74.5815
    },
    'Ahmednagar': {
        'region': 'Western Maharashtra',
        'primary_crop': 'Onion',
        'risk_zone': False,
        'lat': 19.0948,
        'lon': 74.7480
    },
}


# Delhi District Configurations
DELHI_DISTRICTS: Dict[str, Dict] = {
    'Najafgarh': {
        'region': 'Najafgarh Zone',
        'primary_crop': 'Vegetables',
        'risk_zone': True,
        'lat': 28.6092,
        'lon': 76.9798
    },
    'Yamuna Floodplain North': {
        'region': 'Yamuna Floodplain',
        'primary_crop': 'Vegetables',
        'risk_zone': True,
        'lat': 28.7500,
        'lon': 77.2500
    },
    'Yamuna Floodplain South': {
        'region': 'Yamuna Floodplain',
        'primary_crop': 'Flowers',
        'risk_zone': True,
        'lat': 28.5500,
        'lon': 77.2800
    },
    'Alipur': {
        'region': 'North Delhi',
        'primary_crop': 'Vegetables',
        'risk_zone': False,
        'lat': 28.7967,
        'lon': 77.1350
    },
    'Narela': {
        'region': 'North Delhi',
        'primary_crop': 'Wheat',
        'risk_zone': False,
        'lat': 28.8526,
        'lon': 77.0929
    },
}


# All districts combined
ALL_DISTRICTS = {
    'Maharashtra': MAHARASHTRA_DISTRICTS,
    'Delhi': DELHI_DISTRICTS,
}


def get_all_districts() -> Dict[str, Dict[str, Dict]]:
    """Get all district configurations grouped by state."""
    return ALL_DISTRICTS


def get_district_config(state: str, district: str) -> Dict:
    """
    Get configuration for a specific district.
    
    Args:
        state: State name
        district: District name
        
    Returns:
        District configuration dict or empty dict if not found
    """
    state_districts = ALL_DISTRICTS.get(state, {})
    return state_districts.get(district, {})
