"""
Metrics Components for Sovereign Agri-Policy Hub.

Renders MSP cards and key performance metrics.
"""

import streamlit as st
import pandas as pd
from config.msp_rates import MSP_RATES_RABI_2026_27, MSP_RATES_KHARIF_2025_26
from utils.economics import calculate_economic_impact


def render_msp_cards() -> None:
    """
    Render MSP rate cards for Rabi and Kharif seasons.
    
    Displays the Minimum Support Prices in styled glass cards
    for both seasons.
    """
    st.markdown('<h2 class="section-header">Minimum Support Prices</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>Rabi Season 2026-27</h3>
        """, unsafe_allow_html=True)
        
        msp_html = ""
        for crop, rate in MSP_RATES_RABI_2026_27.items():
            msp_html += f'<span class="msp-badge">{crop}: ₹{rate:,}/q</span> '
        st.markdown(msp_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>Kharif Season 2025-26</h3>
        """, unsafe_allow_html=True)
        
        msp_html = ""
        for crop, rate in list(MSP_RATES_KHARIF_2025_26.items())[:6]:  # Show first 6
            msp_html += f'<span class="msp-badge">{crop}: ₹{rate:,}/q</span> '
        st.markdown(msp_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_key_metrics(state_df: pd.DataFrame, selected_state: str) -> None:
    """
    Render key metrics dashboard for the selected state.
    
    Args:
        state_df: Filtered DataFrame for the selected state
        selected_state: Name of the selected state
    """
    st.markdown(f'<h2 class="section-header">{selected_state} Key Metrics</h2>', unsafe_allow_html=True)
    
    # Calculate aggregate metrics
    total_acreage = state_df['acreage_ha'].sum()
    avg_variance = state_df['yield_variance_pct'].mean()
    
    # Calculate total economic impact
    total_revenue = sum([
        calculate_economic_impact(row['predicted_yield'], row['msp_rate'], row['acreage_ha'])
        for _, row in state_df.iterrows()
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(state_df)}</div>
            <div class="metric-label">Districts Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_acreage/1000:.0f}K ha</div>
            <div class="metric-label">Total Acreage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        variance_color = "#e74c3c" if avg_variance < -5 else "#2ecc71"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {variance_color}">{avg_variance:+.1f}%</div>
            <div class="metric-label">Avg Yield Variance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{total_revenue:.0f} Cr</div>
            <div class="metric-label">Projected Revenue</div>
        </div>
        """, unsafe_allow_html=True)
