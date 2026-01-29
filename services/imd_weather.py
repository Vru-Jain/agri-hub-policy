"""
IMD (India Meteorological Department) Weather API Integration.

Fetches district-wise rainfall data from the IMD Mausam API.
API Endpoint: https://mausam.imd.gov.in/api/districtwise_rainfall_api.php

Note: IMD API requires IP whitelisting. Contact IMD for access.
"""

import httpx
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# IMD API Configuration
IMD_API_BASE = "https://mausam.imd.gov.in/api"
IMD_RAINFALL_ENDPOINT = f"{IMD_API_BASE}/districtwise_rainfall_api.php"

# District codes for Maharashtra and Delhi (sample)
# Full list available from IMD documentation
DISTRICT_CODES = {
    'Maharashtra': {
        'Nashik': 'MH01',
        'Pune': 'MH02',
        'Nagpur': 'MH03',
        'Amravati': 'MH04',
        'Aurangabad': 'MH05',
        'Jalna': 'MH06',
        'Latur': 'MH07',
        'Kolhapur': 'MH08',
        'Sangli': 'MH09',
        'Ahmednagar': 'MH10',
    },
    'Delhi': {
        'Najafgarh': 'DL01',
        'Yamuna Floodplain North': 'DL02',
        'Yamuna Floodplain South': 'DL03',
        'Alipur': 'DL04',
        'Narela': 'DL05',
    }
}

# IMD 50-year rainfall normals (mm) - for deviation calculation
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


async def fetch_district_rainfall(state: str, district: str) -> Optional[Dict]:
    """
    Fetch current rainfall data for a specific district from IMD API.
    
    Args:
        state: State name (Maharashtra or Delhi)
        district: District name
        
    Returns:
        Dictionary with rainfall data or None if API fails:
        {
            'actual_mm': float,
            'normal_mm': float,
            'deviation_pct': float,
            'period': str
        }
    """
    try:
        district_code = DISTRICT_CODES.get(state, {}).get(district)
        if not district_code:
            logger.warning(f"No district code found for {district}, {state}")
            return None
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                IMD_RAINFALL_ENDPOINT,
                params={
                    'district': district_code,
                    'format': 'json'
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse IMD response format
            actual_rainfall = float(data.get('actual_rainfall', 0))
            normal_rainfall = IMD_RAINFALL_NORMALS.get(district, 700)
            
            deviation = ((actual_rainfall - normal_rainfall) / normal_rainfall * 100) if normal_rainfall > 0 else 0
            
            return {
                'actual_mm': round(actual_rainfall, 1),
                'normal_mm': normal_rainfall,
                'deviation_pct': round(deviation, 2),
                'period': data.get('period', f"Monsoon {datetime.now().year}")
            }
            
    except httpx.HTTPError as e:
        logger.error(f"IMD API error for {district}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching rainfall for {district}: {e}")
        return None


def fetch_district_rainfall_sync(state: str, district: str) -> Optional[Dict]:
    """
    Synchronous wrapper for fetch_district_rainfall.
    
    Args:
        state: State name
        district: District name
        
    Returns:
        Rainfall data dict or None
    """
    try:
        import asyncio
        return asyncio.run(fetch_district_rainfall(state, district))
    except Exception as e:
        logger.error(f"Sync rainfall fetch failed: {e}")
        return None


def get_rainfall_normal(district: str) -> float:
    """
    Get IMD 50-year normal rainfall for a district.
    
    Args:
        district: District name
        
    Returns:
        Normal rainfall in mm (default 700 if not found)
    """
    return IMD_RAINFALL_NORMALS.get(district, 700)
