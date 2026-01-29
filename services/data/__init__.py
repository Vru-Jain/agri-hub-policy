"""
Data Package for Agri-Policy Hub.

This package contains data configuration and constants:
- districts.py: District configurations and coordinates
- msp_rates.py: Minimum Support Price rates

Usage:
    from services.data import MAHARASHTRA_DISTRICTS, DELHI_DISTRICTS, MSP_RATES
"""

from .districts import (
    MAHARASHTRA_DISTRICTS,
    DELHI_DISTRICTS,
    ALL_DISTRICTS,
    get_all_districts,
    get_district_config
)

from .msp_rates import (
    MSP_RATES,
    get_msp_rate,
    get_all_msp_rates
)

__all__ = [
    # Districts
    'MAHARASHTRA_DISTRICTS',
    'DELHI_DISTRICTS',
    'ALL_DISTRICTS',
    'get_all_districts',
    'get_district_config',
    # MSP Rates
    'MSP_RATES',
    'get_msp_rate',
    'get_all_msp_rates',
]
