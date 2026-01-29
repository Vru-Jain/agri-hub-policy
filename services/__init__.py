"""
Services Package for Agri-Policy Hub.

This package provides data services and API integrations:

Subpackages:
    apis/   - External API integrations (IMD, Agmarknet, Sentinel, Kaggle)
    data/   - Data configurations (districts, MSP rates)

Main Modules:
    config.py        - Environment and API key configuration
    data_service.py  - Main data orchestrator (live + demo modes)
    
Quick Usage:
    # Get district data (live or demo)
    from services import get_district_data, get_data_mode
    df = get_district_data()
    
    # Access APIs directly
    from services.apis import fetch_mandi_prices_sync, get_kaggle_service
"""

# Re-export main components for convenience
from .config import (
    is_live_mode,
    get_api_keys,
    get_sentinel_credentials,
    get_data_gov_key
)

from .data_service import (
    get_district_data,
    get_data_mode
)

from .kaggle_service import get_kaggle_service, KaggleService

__all__ = [
    # Config
    'is_live_mode',
    'get_api_keys',
    'get_sentinel_credentials',
    'get_data_gov_key',
    # Data Service
    'get_district_data',
    'get_data_mode',
    # Kaggle
    'get_kaggle_service',
    'KaggleService',
]
