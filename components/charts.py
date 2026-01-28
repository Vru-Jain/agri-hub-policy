"""
Chart Components for Sovereign Agri-Policy Hub.

Renders Plotly charts for yield predictions, monsoon analysis,
mandi arrivals table, and LSTM insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.economics import calculate_mandi_arrivals
from utils.predictions import simulate_lstm_prediction


def render_yield_chart(state_df: pd.DataFrame) -> None:
    """
    Render yield predictions bar chart with historical comparison.
    
    Args:
        state_df: Filtered DataFrame for the selected state
    """
    st.markdown('<h2 class="section-header">Yield Predictions by District</h2>', 
                unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Add predicted yield bars with color coding
    colors = ['#e74c3c' if v < -15 else '#f1c40f' if v < -5 else '#2ecc71' 
              for v in state_df['yield_variance_pct']]
    
    fig.add_trace(go.Bar(
        x=state_df['district'],
        y=state_df['predicted_yield'],
        name='Predicted Yield',
        marker_color=colors,
        text=[f"{y:.1f} q/ha" for y in state_df['predicted_yield']],
        textposition='outside'
    ))
    
    # Add historical yield line
    fig.add_trace(go.Scatter(
        x=state_df['district'],
        y=state_df['historical_yield'],
        name='Historical Avg',
        mode='lines+markers',
        line=dict(color='#9b59b6', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(title='District', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='Yield (quintals/hectare)', gridcolor='rgba(255,255,255,0.1)'),
        height=400,
        margin=dict(l=50, r=50, t=50, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_monsoon_analysis(state_df: pd.DataFrame, monsoon_adjust: int) -> None:
    """
    Render monsoon deviation analysis charts.
    
    Args:
        state_df: Filtered DataFrame for the selected state
        monsoon_adjust: User-selected monsoon adjustment value
    """
    st.markdown('<h2 class="section-header">Monsoon Deviation Analysis</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rainfall comparison chart
        fig_rain = go.Figure()
        
        fig_rain.add_trace(go.Bar(
            x=state_df['district'],
            y=state_df['rainfall_normal_mm'],
            name='IMD Normal (50yr)',
            marker_color='#3498db',
            opacity=0.6
        ))
        
        fig_rain.add_trace(go.Bar(
            x=state_df['district'],
            y=state_df['rainfall_actual_mm'],
            name='Actual Rainfall',
            marker_color='#2ecc71' if monsoon_adjust >= 0 else '#e74c3c'
        ))
        
        fig_rain.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            barmode='group',
            font=dict(family='Inter', color='white'),
            legend=dict(orientation='h', y=1.1),
            xaxis=dict(title='', tickangle=45),
            yaxis=dict(title='Rainfall (mm)'),
            height=350
        )
        
        st.plotly_chart(fig_rain, use_container_width=True)
    
    with col2:
        # NDVI vs Soil Moisture scatter
        fig_ml = px.scatter(
            state_df,
            x='ndvi',
            y='soil_moisture',
            size='acreage_ha',
            color='yield_variance_pct',
            hover_name='district',
            color_continuous_scale=['#e74c3c', '#f1c40f', '#2ecc71'],
            range_color=[-20, 10],
            labels={
                'ndvi': 'NDVI (Vegetation Index)',
                'soil_moisture': 'Soil Moisture',
                'yield_variance_pct': 'Yield Var %'
            }
        )
        
        fig_ml.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='white'),
            height=350
        )
        
        st.plotly_chart(fig_ml, use_container_width=True)


def render_mandi_table(state_df: pd.DataFrame) -> None:
    """
    Render mandi arrivals forecast table.
    
    Args:
        state_df: Filtered DataFrame for the selected state
    """
    st.markdown('<h2 class="section-header">Supply Chain: Mandi Arrivals Forecast</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3>Procurement & Logistics Planning</h3>
        <p>Estimated arrival timelines for major crops to help officials plan storage and transportation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    mandi_data = []
    for _, row in state_df.iterrows():
        arrival_info = calculate_mandi_arrivals(row['mandi_arrival_days'])
        mandi_data.append({
            'District': row['district'],
            'Crop': row['crop'],
            'Est. Arrival': arrival_info['estimated_date'],
            'Days': arrival_info['arrival_days'],
            'Urgency': arrival_info['urgency'],
            'Action': arrival_info['recommendation']
        })
    
    mandi_df = pd.DataFrame(mandi_data)
    
    # Create urgency-colored dataframe display
    st.dataframe(
        mandi_df.style.apply(
            lambda x: ['background-color: rgba(231, 76, 60, 0.3)' if v == 'HIGH' 
                      else 'background-color: rgba(241, 196, 15, 0.3)' if v == 'MEDIUM'
                      else 'background-color: rgba(46, 204, 113, 0.3)' 
                      for v in x] if x.name == 'Urgency' else [''] * len(x), 
            axis=0
        ),
        use_container_width=True,
        height=300
    )


def render_lstm_insights(state_df: pd.DataFrame, selected_season: str) -> None:
    """
    Render LSTM predictive engine insights.
    
    Args:
        state_df: Filtered DataFrame for the selected state
        selected_season: Currently selected crop season
    """
    st.markdown('<h2 class="section-header">LSTM Predictive Engine</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h3>Seasonal Weighting Configuration</h3>
            <p>The LSTM model dynamically adjusts feature weights based on crop season:</p>
            <ul style="color: var(--text-secondary);">
                <li><b>Rabi Season:</b> NDVI weight = 70%, Soil Moisture = 30% (vegetation health critical for winter crops)</li>
                <li><b>Kharif Season:</b> NDVI weight = 40%, Soil Moisture = 60% (monsoon-dependent crops)</li>
            </ul>
            <p><b>Current Season:</b> <span class="msp-badge">{selected_season}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Show sample LSTM prediction
        sample_ndvi = state_df['ndvi'].mean()
        sample_sm = state_df['soil_moisture'].mean()
        lstm_pred = simulate_lstm_prediction(sample_ndvi, sample_sm, selected_season)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{lstm_pred} q/ha</div>
            <div class="metric-label">LSTM Avg Prediction</div>
            <p style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 0.5rem;">
                Based on mean NDVI: {sample_ndvi:.3f}<br>
                Mean Soil Moisture: {sample_sm:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
