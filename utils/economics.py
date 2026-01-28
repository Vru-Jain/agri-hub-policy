"""
Economic Calculation Functions for Sovereign Agri-Policy Hub.

Provides functions for:
- Revenue calculations using MSP rates
- Monsoon deviation analysis
- Yield adjustments based on rainfall
- Intervention status determination
- Mandi arrival forecasting
"""

from datetime import datetime
from typing import Dict, Tuple
import pandas as pd


def calculate_economic_impact(predicted_yield: float, msp_rate: float, acreage_ha: float) -> float:
    """
    Calculate total regional revenue based on predicted yield and MSP.
    
    Args:
        predicted_yield: Yield in quintals per hectare
        msp_rate: MSP in ₹ per quintal
        acreage_ha: Total acreage in hectares
    
    Returns:
        Total revenue in ₹ Crores
    """
    total_production_quintals = predicted_yield * acreage_ha
    total_revenue = total_production_quintals * msp_rate
    revenue_crores = total_revenue / 1e7  # Convert to Crores
    return revenue_crores


def calculate_monsoon_deviation(actual_rainfall: float, normal_rainfall: float) -> float:
    """
    Calculate rainfall deviation from IMD 50-year normal.
    
    Args:
        actual_rainfall: Actual rainfall in mm
        normal_rainfall: IMD 50-year normal rainfall in mm
    
    Returns:
        Deviation percentage (-ve for deficit, +ve for excess)
    """
    if normal_rainfall == 0:
        return 0
    deviation = ((actual_rainfall - normal_rainfall) / normal_rainfall) * 100
    return round(deviation, 2)


def adjust_yield_for_monsoon(base_yield: float, monsoon_deviation_pct: float) -> float:
    """
    Adjust yield prediction based on monsoon deviation.
    
    Uses a correlation factor of 0.5 (50% impact of rainfall on yield).
    The adjustment is capped between 0.6 and 1.3 to prevent extreme values.
    
    Args:
        base_yield: Base yield in quintals per hectare
        monsoon_deviation_pct: Monsoon deviation percentage
    
    Returns:
        Adjusted yield in quintals per hectare
    """
    adjustment_factor = 1 + (monsoon_deviation_pct / 100) * 0.5
    # Cap adjustment between 0.6 and 1.3
    adjustment_factor = max(0.6, min(1.3, adjustment_factor))
    return base_yield * adjustment_factor


def get_intervention_status(yield_variance_pct: float) -> Tuple[str, str]:
    """
    Determine intervention status based on yield variance.
    
    Args:
        yield_variance_pct: Yield variance percentage from historical average
    
    Returns:
        Tuple of (status_code, status_label):
        - 'red', 'CRITICAL': yield < -15%
        - 'amber', 'WARNING': yield < -5%
        - 'green', 'NORMAL': otherwise
    """
    if yield_variance_pct < -15:
        return 'red', 'CRITICAL'
    elif yield_variance_pct < -5:
        return 'amber', 'WARNING'
    else:
        return 'green', 'NORMAL'


def generate_intervention_alert(district: str, yield_variance_pct: float, crop: str) -> str:
    """
    Generate intervention alert text based on yield variance.
    
    Args:
        district: District name
        yield_variance_pct: Yield variance percentage
        crop: Crop name
    
    Returns:
        Formatted alert message string
    """
    if yield_variance_pct < -15:
        return (f"CRITICAL: Predicted yield in {district} is {abs(yield_variance_pct):.1f}% "
                f"below average. Recommend activating PM Fasal Bima Yojana claims pipeline "
                f"and coordinating with district collector for immediate assessment.")
    elif yield_variance_pct < -5:
        return (f"WARNING: {district} showing {abs(yield_variance_pct):.1f}% yield deficit "
                f"for {crop}. Consider pre-positioning relief supplies and alerting "
                f"agricultural extension officers.")
    else:
        return (f"STABLE: {district} yield projections within normal range. "
                f"Continue standard monitoring protocols.")


def calculate_mandi_arrivals(mandi_days: int, current_stock_pct: int = 50) -> Dict:
    """
    Predict mandi arrival logistics and provide recommendations.
    
    Args:
        mandi_days: Days until expected mandi arrival
        current_stock_pct: Current stock percentage (default 50)
    
    Returns:
        Dictionary containing:
        - arrival_days: Days until arrival
        - urgency: 'HIGH', 'MEDIUM', or 'LOW'
        - recommendation: Action recommendation string
        - estimated_date: Formatted arrival date
    """
    if mandi_days < 20:
        urgency = "HIGH"
        recommendation = "Immediate storage capacity expansion required. Alert FCI godowns."
    elif mandi_days < 45:
        urgency = "MEDIUM"
        recommendation = "Begin procurement logistics planning. Notify APMC officials."
    else:
        urgency = "LOW"
        recommendation = "Standard preparation timeline. Monitor market conditions."
    
    return {
        'arrival_days': mandi_days,
        'urgency': urgency,
        'recommendation': recommendation,
        'estimated_date': (datetime.now() + pd.Timedelta(days=mandi_days)).strftime('%d %b %Y')
    }
