"""
Sentinel Hub Satellite Imagery Integration.

Fetches NDVI (Normalized Difference Vegetation Index) and soil moisture data
from Sentinel-2 satellite imagery using the Sentinel Hub API.

API: https://services.sentinel-hub.com
Package: sentinelhub-py

Note: Requires Sentinel Hub account with OAuth2 credentials.
Free tier available at https://www.sentinel-hub.com/
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Check if sentinelhub package is available
SENTINEL_AVAILABLE = False
try:
    from sentinelhub import (
        SHConfig,
        SentinelHubStatistical,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        CRS,
        BBox,
        bbox_to_dimensions,
        Geometry,
    )
    SENTINEL_AVAILABLE = True
except ImportError:
    logger.warning("sentinelhub package not installed. Satellite features disabled.")


# NDVI Evalscript for Sentinel-2
NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B04", "B08"],
            units: "DN"
        }],
        output: {
            bands: 1,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    return [ndvi];
}
"""

# Soil Moisture Index Evalscript (using NDWI as proxy)
SOIL_MOISTURE_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B03", "B08"],
            units: "DN"
        }],
        output: {
            bands: 1,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    // NDWI (Normalized Difference Water Index) as soil moisture proxy
    let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
    // Normalize to 0-1 range for soil moisture representation
    let moisture = (ndwi + 1) / 2;
    return [moisture];
}
"""


def configure_sentinel_hub(client_id: str, client_secret: str) -> Optional[object]:
    """
    Configure Sentinel Hub with OAuth2 credentials.
    
    Args:
        client_id: Sentinel Hub OAuth2 client ID
        client_secret: Sentinel Hub OAuth2 client secret
        
    Returns:
        SHConfig object or None if package not available
    """
    if not SENTINEL_AVAILABLE:
        logger.error("sentinelhub package not available")
        return None
    
    try:
        config = SHConfig()
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
        return config
    except Exception as e:
        logger.error(f"Failed to configure Sentinel Hub: {e}")
        return None


def fetch_ndvi(
    lat: float, 
    lon: float, 
    config: object,
    buffer_km: float = 5.0,
    days_back: int = 30
) -> Optional[Dict]:
    """
    Fetch NDVI value for a location from Sentinel-2 imagery.
    
    Args:
        lat: Latitude of the center point
        lon: Longitude of the center point
        config: SHConfig object with credentials
        buffer_km: Buffer radius in kilometers around the point
        days_back: Number of days to look back for cloud-free imagery
        
    Returns:
        Dictionary with NDVI data or None:
        {
            'ndvi': float (0-1 scale),
            'date': str,
            'cloud_coverage': float
        }
    """
    if not SENTINEL_AVAILABLE or config is None:
        return None
    
    try:
        # Calculate bounding box (approx 1 degree = 111 km)
        buffer_deg = buffer_km / 111.0
        bbox = BBox(
            [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg],
            crs=CRS.WGS84
        )
        
        # Time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Statistical API request for mean NDVI
        request = SentinelHubStatistical(
            aggregation=SentinelHubStatistical.aggregation(
                evalscript=NDVI_EVALSCRIPT,
                time_interval=time_interval,
                aggregation_interval='P1D',
                size=[100, 100]
            ),
            input_data=[
                SentinelHubStatistical.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    maxcc=0.3  # Max 30% cloud coverage
                )
            ],
            bbox=bbox,
            config=config
        )
        
        response = request.get_data()
        
        if response and len(response) > 0:
            # Get the most recent valid observation
            for interval in reversed(response[0].get('data', [])):
                stats = interval.get('outputs', {}).get('data', {}).get('bands', {}).get('B0', {}).get('stats', {})
                if stats:
                    return {
                        'ndvi': round(stats.get('mean', 0.5), 3),
                        'date': interval.get('interval', {}).get('to', datetime.now().strftime('%Y-%m-%d')),
                        'cloud_coverage': interval.get('outputs', {}).get('dataMask', {}).get('noData', 0)
                    }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch NDVI for ({lat}, {lon}): {e}")
        return None


def fetch_soil_moisture(
    lat: float, 
    lon: float, 
    config: object,
    buffer_km: float = 5.0,
    days_back: int = 30
) -> Optional[Dict]:
    """
    Fetch soil moisture proxy (NDWI-based) for a location.
    
    Args:
        lat: Latitude
        lon: Longitude
        config: SHConfig object
        buffer_km: Buffer radius in km
        days_back: Days to look back
        
    Returns:
        Dictionary with soil moisture data or None:
        {
            'soil_moisture': float (0-1 scale),
            'date': str
        }
    """
    if not SENTINEL_AVAILABLE or config is None:
        return None
    
    try:
        buffer_deg = buffer_km / 111.0
        bbox = BBox(
            [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg],
            crs=CRS.WGS84
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        request = SentinelHubStatistical(
            aggregation=SentinelHubStatistical.aggregation(
                evalscript=SOIL_MOISTURE_EVALSCRIPT,
                time_interval=time_interval,
                aggregation_interval='P1D',
                size=[100, 100]
            ),
            input_data=[
                SentinelHubStatistical.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    maxcc=0.3
                )
            ],
            bbox=bbox,
            config=config
        )
        
        response = request.get_data()
        
        if response and len(response) > 0:
            for interval in reversed(response[0].get('data', [])):
                stats = interval.get('outputs', {}).get('data', {}).get('bands', {}).get('B0', {}).get('stats', {})
                if stats:
                    return {
                        'soil_moisture': round(stats.get('mean', 0.3), 3),
                        'date': interval.get('interval', {}).get('to', datetime.now().strftime('%Y-%m-%d'))
                    }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch soil moisture for ({lat}, {lon}): {e}")
        return None


def fetch_vegetation_indices(
    lat: float, 
    lon: float, 
    credentials: Dict[str, str]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Convenience function to fetch both NDVI and soil moisture.
    
    Args:
        lat: Latitude
        lon: Longitude
        credentials: Dict with 'client_id' and 'client_secret'
        
    Returns:
        Tuple of (ndvi, soil_moisture) - values may be None
    """
    config = configure_sentinel_hub(
        credentials.get('client_id', ''),
        credentials.get('client_secret', '')
    )
    
    if config is None:
        return None, None
    
    ndvi_data = fetch_ndvi(lat, lon, config)
    moisture_data = fetch_soil_moisture(lat, lon, config)
    
    ndvi = ndvi_data.get('ndvi') if ndvi_data else None
    moisture = moisture_data.get('soil_moisture') if moisture_data else None
    
    return ndvi, moisture
