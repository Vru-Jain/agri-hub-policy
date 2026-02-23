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
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        
        /* Root Variables - Indian Flag Premium Theme */
        :root {{
            --saffron-primary: #FF9933;
            --saffron-glow: rgba(255, 153, 51, 0.3);
            --white-pure: #FFFFFF;
            --green-primary: #138808;
            --green-glow: rgba(19, 136, 8, 0.3);
            --navy-dark: #000080;  /* Ashoka Chakra base */
            --navy-light: #0d1b2a; 
            
            --bg-color: #f4f7f6; /* Off-white for depth */
            --card-bg: rgba(255, 255, 255, 0.95); /* Solidified glassmorphism */
            --border-color: rgba(0, 0, 128, 0.1); 
            
            --text-heading: var(--navy-dark);
            --text-primary: #1e293b;
            --text-secondary: #475569;
        }}
        
        /* Global Styles */
        .stApp {{
            background: var(--bg-color);
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: var(--white-pure);
            border-right: 1px solid var(--border-color);
            box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        }}
        
        section[data-testid="stSidebar"] * {{
            font-family: 'Inter', sans-serif !important;
        }}
        
        /* Main Header */
        .main-header {{
            background: linear-gradient(135deg, var(--saffron-primary) 0%, var(--navy-dark) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
            letter-spacing: -0.5px;
        }}
        
        .sub-header {{
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            text-align: center;
            font-size: 1.15rem;
            margin-bottom: 2.5rem;
        }}
        
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Premium Cards */
        .glass-card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.75rem;
            margin: 1.25rem 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.6s ease-out forwards;
        }}
        
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
            border-color: rgba(255, 153, 51, 0.4); /* Saffron tint on hover */
        }}
        
        .glass-card h3 {{
            color: var(--text-heading);
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
            margin-bottom: 1.2rem;
            border-bottom: 2px solid rgba(0,0,128,0.05);
            padding-bottom: 0.5rem;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: var(--white-pure);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            animation: fadeInUp 0.5s ease-out forwards;
            position: relative;
            overflow: hidden;
        }}
        
        /* Subtle top border color for metrics */
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 4px;
            background: var(--saffron-primary);
            opacity: 0.8;
        }}
        
        .metric-card:hover {{
            transform: translateY(-6px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.06);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--navy-dark);
            font-family: 'Plus Jakarta Sans', sans-serif;
            line-height: 1.2;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        
        /* Target Streamlit Buttons to make them pop! */
        .stButton button, .stDownloadButton button {{
            background: linear-gradient(135deg, var(--saffron-primary) 0%, #e67e22 100%) !important;
            color: var(--white-pure) !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
            letter-spacing: 0.5px !important;
            padding: 0.6rem 1.5rem !important;
            box-shadow: 0 4px 14px var(--saffron-glow) !important;
            transition: all 0.3s ease !important;
            width: 100% !important; /* ensure full width if inside columns */
        }}
        
        .stButton button:hover, .stDownloadButton button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255, 153, 51, 0.4) !important;
            filter: brightness(1.05) !important;
        }}
        
        .stButton button:active {{
            transform: translateY(1px) !important;
        }}

        /* Secondary Button Style if we need it (using green) */
        div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton button {{
             background: linear-gradient(135deg, var(--green-primary) 0%, #0e6606 100%) !important;
             box-shadow: 0 4px 14px var(--green-glow) !important;
        }}
        
        
        /* Status Indicators (Using Indian Green/Saffron where possible) */
        .status-red {{
            background: rgba(231, 76, 60, 0.1);
            border: 1px solid rgba(231, 76, 60, 0.3);
            border-left: 4px solid #e74c3c;
        }}
        
        .status-amber {{
            /* Saffron instead of yellow */
            background: rgba(255, 153, 51, 0.1);
            border: 1px solid rgba(255, 153, 51, 0.3);
            border-left: 4px solid var(--saffron-primary);
        }}
        
        .status-green {{
            /* Indian Green */
            background: rgba(19, 136, 8, 0.1);
            border: 1px solid rgba(19, 136, 8, 0.3);
            border-left: 4px solid var(--green-primary);
        }}
        
        /* Alert Box */
        .alert-box {{
            background: var(--white-pure);
            border: 1px solid #e2e8f0;
            border-left: 5px solid #e74c3c;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.08);
        }}
        
        .alert-box .alert-title {{
            color: #e74c3c;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        /* Intervention Cards */
        .intervention-card {{
            background: var(--white-pure);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
            transition: transform 0.2s ease;
        }}
        .intervention-card:hover {{
            transform: translateX(4px);
        }}
        
        .intervention-title {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            color: var(--text-heading);
            margin-bottom: 0.75rem;
        }}
        
        .status-dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
        }}
        
        .dot-red {{ background: #e74c3c; box-shadow: 0 0 10px rgba(231,76,60,0.4); }}
        .dot-amber {{ background: var(--saffron-primary); box-shadow: 0 0 10px var(--saffron-glow); }}
        .dot-green {{ background: var(--green-primary); box-shadow: 0 0 10px var(--green-glow); }}
        
        /* Data Table Styling */
        .dataframe {{
            background: var(--white-pure) !important;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        
        .dataframe th {{
            background: rgba(0, 0, 128, 0.03) !important;
            color: var(--navy-dark) !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 600 !important;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{
            background: #cbd5e1;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
        
        /* MSP Badge */
        .msp-badge {{
            display: inline-block;
            background: rgba(19, 136, 8, 0.1);
            color: var(--green-primary);
            border: 1px solid rgba(19, 136, 8, 0.2);
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin: 0.25rem;
        }}
        
        /* Section Headers */
        .section-header {{
            color: var(--navy-dark);
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            margin: 2.5rem 0 1.5rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid rgba(0,0,128,0.08);
            letter-spacing: -0.5px;
        }}
    </style>
    """
