"""
Map Component for Sovereign Agri-Policy Hub.

Renders the Folium geospatial heatmap with district markers.
"""

import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from utils.economics import get_intervention_status


@st.cache_resource
def create_folium_map(_df: pd.DataFrame, state: str) -> folium.Map:
    """
    Create Folium map with district heatmap.
    
    Uses cache_resource decorator to cache the map object across reruns,
    significantly improving performance when switching between states.
    
    Args:
        _df: Full DataFrame (underscore prefix for unhashable type with st.cache_resource)
        state: Either "Maharashtra" or "Delhi"
    
    Returns:
        Configured Folium Map object
    """
    if state == "Maharashtra":
        center = [19.7515, 75.7139]
        zoom = 6
        highlight_regions = ['Vidarbha', 'Marathwada']
    else:  # Delhi
        center = [28.6139, 77.2090]
        zoom = 10
        highlight_regions = ['Najafgarh Zone', 'Yamuna Floodplain']
    
    # Create base map with dark theme
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='CartoDB dark_matter'
    )
    
    # Filter data for selected state
    state_df = _df[_df['state'] == state]
    
    # Add markers for each district
    for _, row in state_df.iterrows():
        # Determine color based on yield variance
        status, _ = get_intervention_status(row['yield_variance_pct'])
        if status == 'red':
            color = 'red'
            icon = 'exclamation-triangle'
        elif status == 'amber':
            color = 'orange'
            icon = 'exclamation-circle'
        else:
            color = 'green'
            icon = 'check-circle'
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="color: #333; margin: 0 0 10px 0;">{row['district']}</h4>
            <p><b>Region:</b> {row['region']}</p>
            <p><b>Crop:</b> {row['crop']}</p>
            <p><b>Predicted Yield:</b> {row['predicted_yield']} q/ha</p>
            <p><b>Yield Variance:</b> {row['yield_variance_pct']}%</p>
            <p><b>Rainfall Deviation:</b> {row['rainfall_deviation_pct']}%</p>
            <p><b>NDVI:</b> {row['ndvi']}</p>
            <p><b>Soil Moisture:</b> {row['soil_moisture']}</p>
        </div>
        """
        
        # Add highlight for risk zones
        is_risk_zone = row['region'] in highlight_regions
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
            tooltip=f"{row['district']} - {row['crop']}"
        ).add_to(m)
        
        # Add circle for risk zones
        if is_risk_zone or row['risk_zone']:
            folium.Circle(
                location=[row['lat'], row['lon']],
                radius=15000 if state == "Maharashtra" else 3000,
                color='red' if status == 'red' else 'orange',
                fill=True,
                fillOpacity=0.2,
                tooltip=f"Risk Zone: {row['region']}"
            ).add_to(m)
    
    # Add heatmap layer based on yield variance (inverted - lower yield = higher heat)
    # Scaled by 10x to ensure visibility even for 5-10% variance
    heat_data = [[row['lat'], row['lon'], max(0, -row['yield_variance_pct']) * 10] 
                 for _, row in state_df.iterrows()]
    
    if heat_data:
        HeatMap(heat_data, radius=25, blur=15, max_zoom=1).add_to(m)
    
    return m


def render_map_section(df: pd.DataFrame, selected_state: str) -> None:
    """
    Render the geospatial map section.
    
    Args:
        df: Full DataFrame with all data
        selected_state: Currently selected state
    """
    st.markdown(f'<h2 class="section-header">{selected_state} District Heatmap</h2>', 
                unsafe_allow_html=True)
    
    region_desc = ("Vidarbha & Marathwada drought-risk zones" 
                   if selected_state == "Maharashtra" 
                   else "Najafgarh & Yamuna Floodplain agriculture zones")
    
    st.markdown(f"""
    <div class="glass-card">
        <h3>Regional Focus: {region_desc}</h3>
        <p>Interactive map showing district-level yield predictions and risk zones. Red markers indicate critical intervention needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    folium_map = create_folium_map(df, selected_state)
    st_folium(folium_map, width=None, height=500, returned_objects=[])
