"""
Theme Configuration for Sovereign Agri-Policy Hub.

Provides state-specific CSS themes with glassmorphism styling.
- Maharashtra: Sahyadri Sunset (Deep Oranges and Purples)
- Delhi: Modern Metro (Steel Blue and Cool Gray)
"""


def get_theme_css(state: str) -> str:
    """
    Generate CSS based on selected state theme.
    
    Args:
        state: Either "Maharashtra" or "Delhi"
        
    Returns:
        Complete CSS string with theme variables and styles
    """
    if state == "Maharashtra":
        # Sahyadri Sunset: Deep Oranges and Purples
        primary_gradient = "linear-gradient(135deg, #ff6b35 0%, #9b59b6 50%, #6c3483 100%)"
        accent_color = "#ff6b35"
        secondary_color = "#9b59b6"
        glow_color = "rgba(255, 107, 53, 0.4)"
        card_bg = "rgba(155, 89, 182, 0.15)"
        border_color = "rgba(255, 107, 53, 0.3)"
    else:
        # Modern Metro: Steel Blue and Cool Gray
        primary_gradient = "linear-gradient(135deg, #2c3e50 0%, #3498db 50%, #1abc9c 100%)"
        accent_color = "#3498db"
        secondary_color = "#1abc9c"
        glow_color = "rgba(52, 152, 219, 0.4)"
        card_bg = "rgba(52, 152, 219, 0.15)"
        border_color = "rgba(26, 188, 156, 0.3)"
    
    return f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        /* Root Variables */
        :root {{
            --primary-gradient: {primary_gradient};
            --accent-color: {accent_color};
            --secondary-color: {secondary_color};
            --glow-color: {glow_color};
            --card-bg: {card_bg};
            --border-color: {border_color};
            --dark-bg: #0a0a0f;
            --card-dark: rgba(20, 20, 30, 0.85);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.7);
        }}
        
        /* Global Styles */
        .stApp {{
            background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
            font-family: 'Inter', sans-serif;
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(20, 20, 35, 0.95) 0%, rgba(10, 10, 20, 0.98) 100%);
            border-right: 1px solid var(--border-color);
        }}
        
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stToggle label {{
            color: var(--text-primary) !important;
            font-weight: 500;
        }}
        
        /* Main Header */
        .main-header {{
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 1rem;
        }}
        
        .sub-header {{
            color: var(--text-secondary);
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }}
        
        /* Glassmorphism Cards */
        .glass-card {{
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }}
        
        .glass-card:hover {{
            box-shadow: 0 8px 32px var(--glow-color);
            border-color: var(--accent-color);
        }}
        
        .glass-card h3 {{
            color: var(--accent-color);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        
        .glass-card p {{
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: linear-gradient(135deg, var(--card-bg), rgba(0, 0, 0, 0.3));
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px var(--glow-color);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-color);
            font-family: 'Space Grotesk', sans-serif;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.5rem;
        }}
        
        /* Status Indicators */
        .status-red {{
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.2), rgba(192, 57, 43, 0.3));
            border: 2px solid #e74c3c;
            box-shadow: 0 0 20px rgba(231, 76, 60, 0.4);
        }}
        
        .status-amber {{
            background: linear-gradient(135deg, rgba(241, 196, 15, 0.2), rgba(243, 156, 18, 0.3));
            border: 2px solid #f1c40f;
            box-shadow: 0 0 20px rgba(241, 196, 15, 0.4);
        }}
        
        .status-green {{
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.2), rgba(39, 174, 96, 0.3));
            border: 2px solid #2ecc71;
            box-shadow: 0 0 20px rgba(46, 204, 113, 0.4);
        }}
        
        /* Alert Box */
        .alert-box {{
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.2));
            backdrop-filter: blur(10px);
            border-left: 4px solid #e74c3c;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
            animation: pulse-glow 2s infinite;
        }}
        
        @keyframes pulse-glow {{
            0%, 100% {{ box-shadow: 0 0 10px rgba(231, 76, 60, 0.3); }}
            50% {{ box-shadow: 0 0 25px rgba(231, 76, 60, 0.6); }}
        }}
        
        .alert-box .alert-title {{
            color: #e74c3c;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }}
        
        .alert-box .alert-text {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        /* Intervention Cards */
        .intervention-card {{
            backdrop-filter: blur(20px);
            border-radius: 12px;
            padding: 1.25rem;
            margin: 0.75rem 0;
        }}
        
        .intervention-title {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}
        
        .status-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        
        .dot-red {{ background: #e74c3c; box-shadow: 0 0 10px #e74c3c; }}
        .dot-amber {{ background: #f1c40f; box-shadow: 0 0 10px #f1c40f; }}
        .dot-green {{ background: #2ecc71; box-shadow: 0 0 10px #2ecc71; }}
        
        /* Data Table Styling */
        .dataframe {{
            background: var(--card-dark) !important;
            border-radius: 8px;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.3);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--accent-color);
            border-radius: 4px;
        }}
        
        /* MSP Badge */
        .msp-badge {{
            display: inline-block;
            background: linear-gradient(135deg, var(--accent-color), var(--secondary-color));
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.25rem;
        }}
        
        /* Section Headers */
        .section-header {{
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
    </style>
    """
