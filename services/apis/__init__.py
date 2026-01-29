"""
External APIs Package.

This package provides unified access to external data sources:
- IMD Weather API (rainfall data)
- Agmarknet API (mandi prices via Data.gov.in)
- Sentinel Hub API (satellite imagery)
- Kaggle API (historical datasets)

Each module handles its own authentication, rate limiting, and error handling.
"""

# Re-export from parent services package for convenience
# The actual implementations remain in services/ to avoid breaking changes
from services.imd_weather import (
    fetch_district_rainfall_sync,
    get_rainfall_normal,
    IMD_RAINFALL_NORMALS
)

from services.agmarknet import (
    fetch_mandi_prices_sync,
    calculate_price_trend
)

from services.sentinel import fetch_vegetation_indices

from services.kaggle_service import get_kaggle_service, KaggleService

__all__ = [
    # IMD Weather
    'fetch_district_rainfall_sync',
    'get_rainfall_normal',
    'IMD_RAINFALL_NORMALS',
    # Agmarknet
    'fetch_mandi_prices_sync',
    'calculate_price_trend',
    # Sentinel Hub
    'fetch_vegetation_indices',
    # Kaggle
    'get_kaggle_service',
    'KaggleService',
]
