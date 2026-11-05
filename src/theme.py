"""One place for colours and CSS. To change the look, change it here only.

Palette is taken from the proposal deck (aqua, green, gold). If Africa CDC gives you
official brand hex codes, replace the values below and everything follows.
"""
DARK = "#0A3B34"      # deep teal: headings, sidebar
TEAL = "#0E7C73"
AQUA = "#17A398"      # main accent: forecast line and bands
AQUA_LIGHT = "#5FD6CA"
PALE = "#F2FAF8"      # page background
MIST = "#DDEFEB"
GREEN = "#2F8F4E"
GOLD = "#E0A526"      # warnings, "synthetic" strip
INK = "#36504B"       # body text

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;700&family=Nunito+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Nunito Sans', sans-serif; color: {INK}; }}
h1, h2, h3 {{ font-family: 'DM Sans', sans-serif !important; color: {DARK}; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {DARK} 0%, #0D5249 100%); }}
[data-testid="stSidebar"] * {{ color: #F2FAF8 !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: #FFFFFF; border-radius: 10px; }}
[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {DARK} !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: {DARK} !important; }}
/* newer Streamlit versions build the dropdown differently (react-aria), so style both */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {{ background: #FFFFFF !important; border-radius: 10px !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {{ color: {DARK} !important; -webkit-text-fill-color: {DARK} !important; opacity: 1 !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] button, [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {{ color: {DARK} !important; fill: {DARK} !important; }}
.hero {{ background: linear-gradient(135deg, {DARK} 0%, #0D5249 70%, {TEAL} 100%); color: #F2FAF8;
  border-radius: 18px; padding: 28px 34px; position: relative; overflow: hidden; margin-bottom: 14px; }}
.hero::after {{ content: ""; position: absolute; right: -60px; top: -80px; width: 260px; height: 260px;
  border-radius: 50%; background: {AQUA}; opacity: .25; }}
.hero .eyebrow {{ letter-spacing: 3px; text-transform: uppercase; font-weight: 700; font-size: .78rem; color: {AQUA_LIGHT}; }}
.hero h1 {{ color: #F2FAF8 !important; margin: 4px 0 6px 0; font-size: 2rem; }}
.hero p {{ margin: 0; color: #D3ECE7; max-width: 760px; }}
.stripe {{ height: 6px; border-radius: 3px; background: linear-gradient(to right, {AQUA}, {GREEN}, {GOLD}); margin: 0 0 14px 0; }}
.synthetic {{ background: {GOLD}; color: {DARK}; font-weight: 700; border-radius: 10px; padding: 8px 16px; margin-bottom: 14px; font-size: .92rem; }}
.card {{ background: #FFFFFF; border: 1px solid {MIST}; border-radius: 16px; padding: 18px 22px;
  box-shadow: 0 2px 10px rgba(10,59,52,.06); margin-bottom: 14px; }}
.kpi .label {{ font-size: .82rem; text-transform: uppercase; letter-spacing: 1.5px; color: {TEAL}; font-weight: 700; }}
.kpi .value {{ font-family: 'DM Sans', sans-serif; font-size: 2.3rem; font-weight: 700; color: {DARK}; line-height: 1.15; }}
.kpi .sub {{ color: {INK}; font-size: .9rem; }}
.kpi.main {{ border-top: 5px solid {AQUA}; }}
.section {{ font-family: 'DM Sans', sans-serif; font-weight: 700; color: {DARK}; font-size: 1.15rem; margin: 6px 0 8px 0; }}
.chip {{ display: block; background: #FFF6DD; border-left: 5px solid {GOLD}; border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px; color: {DARK}; font-size: .95rem; }}
.driver {{ border-left: 5px solid {AQUA}; padding-left: 14px; font-size: 1.02rem; color: {DARK}; }}
</style>
"""
