"""
Sidebar Component for Sovereign Agri-Policy Hub.

Renders the sidebar with state selector, executive view toggle,
monsoon deviation slider, and season selector.
"""

import streamlit as st
from datetime import datetime
from typing import Tuple


def render_sidebar() -> Tuple[str, bool, int, str]:
    """
    Render the sidebar UI and return user selections.
    
    Returns:
        Tuple containing:
        - selected_state: Either "Maharashtra" or "Delhi"
        - executive_view: Boolean for executive dashboard toggle
        - monsoon_adjust: Integer for rainfall deviation adjustment (-40 to +30)
        - selected_season: Either "Rabi" or "Kharif"
    """
    with st.sidebar:
        st.markdown("### Control Panel")
        st.markdown("---")
        
        # State selector
        selected_state = st.selectbox(
            "Select State",
            ["Maharashtra", "Delhi"],
            index=0
        )
        
        st.markdown("---")
        
        # Executive View Toggle
        executive_view = st.toggle(
            "Enable Executive View",
            value=False,
            help="Display detailed intervention recommendations with status indicators"
        )
        
        st.markdown("---")
        
        # Monsoon Deviation Slider
        st.markdown("### Monsoon Deviation")
        monsoon_adjust = st.slider(
            "Adjust Rainfall Deviation (%)",
            min_value=-40,
            max_value=30,
            value=0,
            step=5,
            help="Simulate rainfall deviation from IMD 50-year normal"
        )
        
        st.markdown("---")
        
        # Season selector
        current_month = datetime.now().month
        default_season = 'Rabi' if current_month in [10, 11, 12, 1, 2, 3] else 'Kharif'
        selected_season = st.radio(
            "Crop Season",
            ["Rabi", "Kharif"],
            index=0 if default_season == 'Rabi' else 1
        )
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.75rem;">
            Last Updated: {datetime.now().strftime('%d %b %Y, %H:%M IST')}
        </div>
        """, unsafe_allow_html=True)
    
    return selected_state, executive_view, monsoon_adjust, selected_season
