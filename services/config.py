"""
Configuration management for API keys and environment settings.

Handles loading of API credentials from environment variables or .env file,
and determines whether the application should run in live or demo mode.
"""

import os
from pathlib import Path
from typing import Dict, Optional

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass


def get_api_keys() -> Dict[str, Optional[str]]:
    """
    Retrieve all API keys from environment variables.
    
    Returns:
        Dictionary containing API keys (values may be None if not set)
    """
    return {
        'sentinel_hub_client_id': os.getenv('SENTINEL_HUB_CLIENT_ID'),
        'sentinel_hub_client_secret': os.getenv('SENTINEL_HUB_CLIENT_SECRET'),
        'data_gov_api_key': os.getenv('DATA_GOV_API_KEY'),
        'kaggle_username': os.getenv('KAGGLE_USERNAME'),
        'kaggle_key': os.getenv('KAGGLE_KEY'),
    }


def is_live_mode() -> bool:
    """
    Check if all required API keys are configured for live mode.
    
    Returns:
        True if all API keys are present, False otherwise (demo mode)
    """
    keys = get_api_keys()
    # Require at least the Data.gov.in key for live mode
    # Sentinel Hub is optional (expensive, may not have)
    return keys['data_gov_api_key'] is not None


def get_sentinel_credentials() -> Optional[Dict[str, str]]:
    """
    Get Sentinel Hub OAuth2 credentials.
    
    Returns:
        Dict with client_id and client_secret, or None if not configured
    """
    keys = get_api_keys()
    if keys['sentinel_hub_client_id'] and keys['sentinel_hub_client_secret']:
        return {
            'client_id': keys['sentinel_hub_client_id'],
            'client_secret': keys['sentinel_hub_client_secret'],
        }
    return None


def get_data_gov_key() -> Optional[str]:
    """
    Get Data.gov.in API key.
    
    Returns:
        API key string or None if not configured
    """
    return get_api_keys()['data_gov_api_key']


def get_kaggle_credentials() -> Optional[Dict[str, str]]:
    """
    Get Kaggle API credentials.
    
    Returns:
        Dict with username and key, or None if not configured
    """
    keys = get_api_keys()
    if keys['kaggle_username'] and keys['kaggle_key']:
        return {
            'username': keys['kaggle_username'],
            'key': keys['kaggle_key'],
        }
    return None
