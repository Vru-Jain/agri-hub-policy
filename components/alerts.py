"""
Alerts Components for Sovereign Agri-Policy Hub.

Renders priority alerts and executive intervention dashboard.
"""

import streamlit as st
import pandas as pd
from utils.economics import (
    get_intervention_status,
    generate_intervention_alert,
    calculate_economic_impact,
)


def render_priority_alerts(state_df: pd.DataFrame) -> None:
    """
    Render priority alerts for districts with critical yield variance.
    
    Args:
        state_df: Filtered DataFrame for the selected state
    """
    critical_districts = state_df[state_df['yield_variance_pct'] < -10]
    
    if len(critical_districts) > 0:
        st.markdown('<h2 class="section-header">Priority Alerts</h2>', unsafe_allow_html=True)
        
        for _, row in critical_districts.iterrows():
            alert_text = generate_intervention_alert(
                row['district'], 
                row['yield_variance_pct'], 
                row['crop']
            )
            st.markdown(f"""
            <div class="alert-box">
                <div class="alert-title">Alert: {row['district']} District</div>
                <div class="alert-text">{alert_text}</div>
            </div>
            """, unsafe_allow_html=True)


def render_executive_dashboard(state_df: pd.DataFrame) -> None:
    """
    Render executive intervention dashboard with status cards.
    
    Args:
        state_df: Filtered DataFrame for the selected state
    """
    st.markdown('<h2 class="section-header">Executive Intervention Dashboard</h2>', 
                unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(state_df.iterrows()):
        status, label = get_intervention_status(row['yield_variance_pct'])
        revenue_impact = calculate_economic_impact(
            row['predicted_yield'], 
            row['msp_rate'], 
            row['acreage_ha']
        )
        
        with cols[i % 3]:
            st.markdown(f"""
            <div class="intervention-card status-{status}">
                <div class="intervention-title">
                    <span class="status-dot dot-{status}"></span>
                    <span style="color: white;">{row['district']}</span>
                </div>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0; font-size: 0.85rem;">
                    <b>Crop:</b> {row['crop']}<br>
                    <b>Status:</b> {label}<br>
                    <b>Yield Variance:</b> {row['yield_variance_pct']:+.1f}%<br>
                    <b>Revenue Impact:</b> ₹{revenue_impact:.1f} Cr
                </p>
            </div>
            """, unsafe_allow_html=True)
