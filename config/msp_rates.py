"""
Minimum Support Price (MSP) Rates for Agricultural Commodities.

MSP rates are announced by the Government of India and represent the minimum price
at which the government purchases crops from farmers.

Data Sources:
- Ministry of Agriculture & Farmers Welfare
- Commission for Agricultural Costs & Prices (CACP)
"""

# Rabi Season 2026-27 MSP Rates (₹ per quintal)
MSP_RATES_RABI_2026_27 = {
    'Wheat': 2585,
    'Mustard': 6200,
    'Gram (Chana)': 5650,
    'Barley': 1980,
    'Masoor (Lentil)': 6700,
    'Safflower': 5850,
}

# Kharif Season 2025-26 MSP Rates (₹ per quintal)
MSP_RATES_KHARIF_2025_26 = {
    'Paddy (Common)': 2369,
    'Paddy (Grade A)': 2409,
    'Jowar (Hybrid)': 3371,
    'Jowar (Maldandi)': 3421,
    'Bajra': 2625,
    'Maize': 2225,
    'Ragi': 4290,
    'Cotton (Medium)': 7121,
    'Cotton (Long)': 7521,
    'Soybean (Yellow)': 4892,
    'Groundnut': 6783,
    'Sugarcane (FRP)': 340,  # Fair & Remunerative Price
}
