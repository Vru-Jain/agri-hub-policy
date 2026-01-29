"""
Agmarknet (Agricultural Marketing Network) API Integration.

Fetches real-time mandi (market) prices from the Data.gov.in Open Government Data platform.
API Base: https://api.data.gov.in/resource/{resource_id}

Data Source: Agmarknet Portal (agmarknet.gov.in) via OGD Platform
"""

import httpx
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Data.gov.in API Configuration
DATA_GOV_API_BASE = "https://api.data.gov.in/resource"

# Resource IDs for Agmarknet commodity prices
# These IDs may need periodic updates from data.gov.in
RESOURCE_IDS = {
    'daily_prices': '9ef84268-d588-465a-a308-a864a43d0070',  # Daily commodity prices
    'variety_prices': '35985678-0d79-46b4-9ed6-6f13308a1d24',  # Variety-wise prices
}

# State codes used by Agmarknet
STATE_CODES = {
    'Maharashtra': 'MH',
    'Delhi': 'DL',
}

# Commodity name mappings (local names to Agmarknet names)
COMMODITY_MAPPINGS = {
    'Wheat': 'Wheat',
    'Cotton': 'Cotton',
    'Sugarcane': 'Sugarcane',
    'Soybean': 'Soyabean',
    'Onion': 'Onion',
    'Vegetables': 'Tomato',  # Use tomato as proxy for vegetables
    'Flowers': 'Marigold',
    'Paddy': 'Paddy(Dhan)(Common)',
}


async def fetch_mandi_prices(
    state: str, 
    commodity: str, 
    api_key: str,
    limit: int = 10
) -> Optional[List[Dict]]:
    """
    Fetch current mandi prices for a commodity in a state.
    
    Args:
        state: State name (Maharashtra or Delhi)
        commodity: Commodity name
        api_key: Data.gov.in API key
        limit: Maximum number of records to fetch
        
    Returns:
        List of price records or None if API fails:
        [
            {
                'market': str,
                'district': str,
                'min_price': float,
                'max_price': float,
                'modal_price': float,
                'date': str
            }
        ]
    """
    try:
        state_code = STATE_CODES.get(state)
        agmarknet_commodity = COMMODITY_MAPPINGS.get(commodity, commodity)
        
        if not state_code:
            logger.warning(f"No state code found for {state}")
            return None
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{DATA_GOV_API_BASE}/{RESOURCE_IDS['daily_prices']}",
                params={
                    'api-key': api_key,
                    'format': 'json',
                    'filters[state]': state,
                    'filters[commodity]': agmarknet_commodity,
                    'limit': limit,
                    'sort[arrival_date]': 'desc'
                }
            )
            response.raise_for_status()
            data = response.json()
            
            records = data.get('records', [])
            
            return [
                {
                    'market': record.get('market', 'Unknown'),
                    'district': record.get('district', 'Unknown'),
                    'min_price': float(record.get('min_price', 0)),
                    'max_price': float(record.get('max_price', 0)),
                    'modal_price': float(record.get('modal_price', 0)),
                    'date': record.get('arrival_date', datetime.now().strftime('%Y-%m-%d')),
                    'commodity': record.get('commodity', commodity),
                }
                for record in records
            ]
            
    except httpx.HTTPError as e:
        logger.error(f"Agmarknet API error for {commodity} in {state}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching mandi prices: {e}")
        return None


def fetch_mandi_prices_sync(
    state: str, 
    commodity: str, 
    api_key: str
) -> Optional[List[Dict]]:
    """
    Synchronous wrapper for fetch_mandi_prices.
    
    Args:
        state: State name
        commodity: Commodity name
        api_key: Data.gov.in API key
        
    Returns:
        List of price records or None
    """
    try:
        import asyncio
        return asyncio.run(fetch_mandi_prices(state, commodity, api_key))
    except Exception as e:
        logger.error(f"Sync mandi fetch failed: {e}")
        return None


def calculate_price_trend(prices: List[Dict]) -> Dict:
    """
    Calculate price trend from historical records.
    
    Args:
        prices: List of price records
        
    Returns:
        Dictionary with trend analysis:
        {
            'current_modal': float,
            'avg_modal': float,
            'trend': str ('up', 'down', 'stable'),
            'change_pct': float
        }
    """
    if not prices:
        return {
            'current_modal': 0,
            'avg_modal': 0,
            'trend': 'unknown',
            'change_pct': 0
        }
    
    modal_prices = [p['modal_price'] for p in prices if p['modal_price'] > 0]
    
    if not modal_prices:
        return {
            'current_modal': 0,
            'avg_modal': 0,
            'trend': 'unknown',
            'change_pct': 0
        }
    
    current = modal_prices[0]
    avg = sum(modal_prices) / len(modal_prices)
    change_pct = ((current - avg) / avg * 100) if avg > 0 else 0
    
    if change_pct > 5:
        trend = 'up'
    elif change_pct < -5:
        trend = 'down'
    else:
        trend = 'stable'
    
    return {
        'current_modal': current,
        'avg_modal': round(avg, 2),
        'trend': trend,
        'change_pct': round(change_pct, 2)
    }
