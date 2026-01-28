"""
Sovereign Agri-Policy Hub: Maharashtra & Delhi 2026
A high-performance Streamlit dashboard for agricultural policy analysis.

This is the main entry point that orchestrates all modular components.
See the config/, utils/, and components/ directories for implementation details.
"""

import streamlit as st
from config.theme import get_theme_css
from utils.data_loader import load_data
from utils.economics import adjust_yield_for_monsoon
from components.sidebar import render_sidebar
from components.metrics import render_msp_cards, render_key_metrics
from components.alerts import render_priority_alerts, render_executive_dashboard
from components.charts import (
    render_yield_chart, 
    render_monsoon_analysis, 
    render_mandi_table, 
    render_lstm_insights
)
from components.map import render_map_section


# Page configuration
st.set_page_config(
    page_title="Sovereign Agri-Policy Hub | Maharashtra & Delhi 2026",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point."""
    # Load data
    df = load_data()
    
    # Render sidebar and get user selections
    selected_state, executive_view, monsoon_adjust, selected_season = render_sidebar()
    
    # Inject theme CSS
    st.markdown(get_theme_css(selected_state), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <h1 class="main-header">Sovereign Agri-Policy Hub</h1>
        <p class="sub-header">Maharashtra & Delhi Agricultural Intelligence Dashboard | 2026</p>
    """, unsafe_allow_html=True)
    
    # Filter data for selected state
    state_df = df[df['state'] == selected_state].copy()
    
    # Apply monsoon adjustment if needed
    if monsoon_adjust != 0:
        state_df['rainfall_deviation_pct'] = state_df['rainfall_deviation_pct'] + monsoon_adjust
        state_df['predicted_yield'] = state_df.apply(
            lambda row: adjust_yield_for_monsoon(row['historical_yield'], row['rainfall_deviation_pct']),
            axis=1
        )
        state_df['yield_variance_pct'] = (
            (state_df['predicted_yield'] - state_df['historical_yield']) / 
            state_df['historical_yield'] * 100
        ).round(2)
    
    # Render dashboard components
    render_msp_cards()
    render_key_metrics(state_df, selected_state)
    render_priority_alerts(state_df)
    
    if executive_view:
        render_executive_dashboard(state_df)
    
    render_map_section(df, selected_state)
    render_yield_chart(state_df)
    render_monsoon_analysis(state_df, monsoon_adjust)
    render_mandi_table(state_df)
    render_lstm_insights(state_df, selected_season)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.8rem; padding: 1rem;">
        <p>Sovereign Agri-Policy Hub v1.0 | Built for Digital India Initiative</p>
        <p>Data Sources: IMD, ICAR, Ministry of Agriculture & Farmers Welfare</p>
        <p>© 2026 Government of India | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
