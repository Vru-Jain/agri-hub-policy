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
        # Sahyadri Sunset: Deep Oranges and Purples - High Contrast
        primary_gradient = "linear-gradient(135deg, #e65100 0%, #8e44ad 100%)" # Darker orange/purple
        accent_color = "#d35400" # Darker orange for text
        secondary_color = "#8e44ad"
        glow_color = "rgba(230, 81, 0, 0.15)" # Subtle glow
        card_bg = "rgba(255, 255, 255, 0.98)" # Almost solid
        border_color = "rgba(230, 81, 0, 0.3)" # More visible border
    else:
        # Modern Metro: Steel Blue and Cool Gray - High Contrast
        primary_gradient = "linear-gradient(135deg, #2c3e50 0%, #2980b9 100%)" # Darker blue gradient
        accent_color = "#2980b9" # Darker blue for text
        secondary_color = "#16a085" # Darker teal
        glow_color = "rgba(41, 128, 185, 0.15)" # Subtle glow
        card_bg = "rgba(255, 255, 255, 0.98)" # Almost solid
        border_color = "rgba(44, 62, 80, 0.3)" # More visible border
    
    return f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
        
        /* Root Variables */
        :root {{
            --primary-gradient: {primary_gradient};
            --accent-color: {accent_color};
            --secondary-color: {secondary_color};
            --glow-color: {glow_color};
            --card-bg: {card_bg};
            --border-color: {border_color};
            --dark-bg: #f8f9fa;
            --card-dark: #ffffff; /* Solid white */
            --text-primary: #000000; /* Pure black for max contrast */
            --text-secondary: #333333; /* Dark gray for secondary text */
        }}
        
        /* Global Styles */
        .stApp {{
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: #ffffff;
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
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
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
            background: linear-gradient(135deg, var(--card-bg), rgba(255, 255, 255, 0.5));
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px var(--glow-color);
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
