"""
EduSense — Student Performance Predictor
Enhanced Premium UI — v2.0
Run: streamlit run student_performance_app.py
Place Final_Marks_Data.csv in the same directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings, os

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EduSense",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Enhanced v2.0
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── Design tokens ── */
:root {
    --bg-base:       #05080f;
    --bg-surface:    #090d1a;
    --bg-card:       #0d1120;
    --bg-card-hover: #111827;
    --border:        #1a2540;
    --border-bright: #2a3a60;
    --text-pri:      #f0f4ff;
    --text-sec:      #94a3c8;
    --text-muted:    #566480;
    --accent:        #5b7eff;
    --accent-2:      #7c3aed;
    --accent-glow:   rgba(91,126,255,0.18);
    --green:         #10d48e;
    --amber:         #fbbf24;
    --red:           #f43f5e;
    --purple:        #a78bfa;
    --teal:          #2dd4bf;
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     24px;
    --radius-xl:     32px;
    --shadow-card:   0 4px 24px rgba(0,0,0,0.5), 0 1px 4px rgba(0,0,0,0.3);
    --shadow-hover:  0 20px 48px rgba(0,0,0,0.65), 0 4px 16px rgba(0,0,0,0.4);
    --transition:    all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg-base) !important;
    color: var(--text-pri) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    scroll-behavior: smooth;
    font-size: 15px;
    line-height: 1.6;
}
.stApp { background: var(--bg-base) !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Subtle noise texture overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Container ── */
.block-container {
    max-width: 1320px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 48px 80px 48px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 4px; }

/* ═══ BUTTONS ═══ */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
    letter-spacing: -0.1px !important;
}

/* Primary gradient buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3a54e0 0%, #6366f1 50%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.95rem !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4),
                0 1px 3px rgba(0,0,0,0.3),
                inset 0 1px 0 rgba(255,255,255,0.1) !important;
    width: 100%;
    position: relative;
    overflow: hidden;
}
.stButton > button[kind="primary"]::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    transition: left 0.5s ease;
}
.stButton > button[kind="primary"]:hover::before { left: 100%; }
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.005) !important;
    box-shadow: 0 12px 36px rgba(99,102,241,0.55),
                0 4px 12px rgba(0,0,0,0.4),
                inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) scale(0.995) !important;
}

/* Secondary nav pills */
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-sec) !important;
    border-radius: 999px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #5b7eff !important;
    color: var(--text-pri) !important;
    background: rgba(91,126,255,0.08) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(91,126,255,0.18) !important;
}

/* Active nav pill overrides */
.stButton > button[kind="primary"].nav-active {
    border-radius: 999px !important;
    padding: 10px 22px !important;
    font-size: 0.84rem !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.4) !important;
}

/* ═══ SLIDERS ═══ */
.stSlider {
    padding: 0 !important;
}
.stSlider [data-baseweb="slider"] {
    padding: 4px 0 !important;
}
div[data-testid="stSlider"] > label {
    color: var(--text-sec) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    margin-bottom: 6px !important;
    display: block;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stSlider"] .stMarkdown {
    display: none;
}
[data-baseweb="slider"] [data-testid="stTickBarMin"],
[data-baseweb="slider"] [data-testid="stTickBarMax"] {
    font-family: 'DM Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.68rem !important;
}
/* Slider track */
[data-baseweb="slider"] > div:first-child {
    background: var(--border) !important;
    border-radius: 999px !important;
    height: 5px !important;
}
/* Slider fill */
[data-baseweb="slider"] [role="progressbar"] {
    background: linear-gradient(90deg, #5b7eff, #a78bfa) !important;
    border-radius: 999px !important;
}
/* Slider thumb */
[data-baseweb="slider"] [role="slider"] {
    background: #fff !important;
    border: 3px solid #5b7eff !important;
    box-shadow: 0 0 0 4px rgba(91,126,255,0.22), 0 2px 8px rgba(0,0,0,0.5) !important;
    width: 20px !important;
    height: 20px !important;
    top: -8px !important;
    transition: box-shadow 0.2s !important;
}
[data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 0 0 6px rgba(91,126,255,0.28), 0 4px 12px rgba(0,0,0,0.6) !important;
}

/* ═══ NUMBER INPUTS ═══ */
div[data-testid="stNumberInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 2px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"]:focus-within {
    border-color: #5b7eff !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stNumberInput input {
    background: transparent !important;
    border: none !important;
    color: var(--text-pri) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    padding: 12px 8px !important;
}
.stNumberInput input:focus { box-shadow: none !important; outline: none !important; }
.stNumberInput label {
    color: var(--text-sec) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    margin-bottom: 8px !important;
    display: block;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] button {
    background: rgba(91,126,255,0.07) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-sec) !important;
    min-width: 36px !important;
    height: 36px !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
    margin: 4px !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: #5b7eff !important;
    border-color: #5b7eff !important;
    color: #fff !important;
    transform: scale(1.05) !important;
}

/* ═══ SELECTBOX (Study Hours) ═══ */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-pri) !important;
    padding: 10px 16px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    font-family: 'DM Mono', monospace !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    min-height: 52px !important;
}
.stSelectbox > div > div:focus-within,
.stSelectbox > div > div:hover {
    border-color: #5b7eff !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
/* Make selected value text clearly visible */
.stSelectbox [data-baseweb="select"] span {
    color: var(--text-pri) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}
.stSelectbox [data-baseweb="select"] svg {
    fill: var(--text-sec) !important;
}
/* Dropdown list */
[data-baseweb="popover"] [role="listbox"] {
    background: #0f1628 !important;
    border: 1.5px solid var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 20px 48px rgba(0,0,0,0.7) !important;
}
[data-baseweb="popover"] [role="option"] {
    background: transparent !important;
    color: var(--text-sec) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
    transition: background 0.15s !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(91,126,255,0.12) !important;
    color: var(--text-pri) !important;
}
.stSelectbox label {
    color: var(--text-sec) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ═══ TABS (Nerd Zone) ═══ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 6px;
    gap: 4px;
    width: fit-content;
    margin: 0 0 36px 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted) !important;
    border-radius: var(--radius-md);
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 22px;
    white-space: nowrap;
    transition: var(--transition) !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-sec) !important;
    background: rgba(91,126,255,0.07) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(91,126,255,0.15), rgba(124,58,237,0.12)) !important;
    color: #818cf8 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3), inset 0 1px 0 rgba(91,126,255,0.2) !important;
}
/* Tab indicator line fix */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ═══ NAV PILLS — uniform sizing fix ═══ */
/* Force all nav buttons same height/padding */
[data-testid="column"]:has(button[kind="primary"]) button,
[data-testid="column"]:has(button[kind="secondary"]) button {
    height: 44px !important;
    min-height: 44px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* ═══ MISC ═══ */
hr { border-color: var(--border) !important; margin: 36px 0 !important; }

/* Fade-in animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 30px var(--glow-color, rgba(91,126,255,0.3)); }
    50%       { box-shadow: 0 0 50px var(--glow-color, rgba(91,126,255,0.5)); }
}
@keyframes shimmer {
    0%   { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.fade-in-up   { animation: fadeInUp 0.55s cubic-bezier(0.22, 1, 0.36, 1) both; }
.fade-in      { animation: fadeIn 0.4s ease both; }
.slide-right  { animation: slideInRight 0.45s cubic-bezier(0.22, 1, 0.36, 1) both; }

.anim-d1 { animation-delay: 0.05s; }
.anim-d2 { animation-delay: 0.12s; }
.anim-d3 { animation-delay: 0.20s; }
.anim-d4 { animation-delay: 0.28s; }
.anim-d5 { animation-delay: 0.36s; }

@media (max-width: 1024px) {
    .block-container { padding: 0 28px 56px 28px !important; }
}
@media (max-width: 640px) {
    .block-container { padding: 0 16px 40px 16px !important; }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════════
DATA_PATH = "Final_Marks_Data.csv"

@st.cache_data
def load_data():
    for path in [DATA_PATH, os.path.join(os.path.dirname(__file__), DATA_PATH)]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

@st.cache_resource
def train_models(df):
    df = df.copy()
    df["Internal_Avg"] = (
        df["Internal Test 1 (out of 40)"] + df["Internal Test 2 (out of 40)"]
    ) / 2
    FEATURES = [
        "Attendance (%)", "Internal Test 1 (out of 40)",
        "Internal Test 2 (out of 40)", "Assignment Score (out of 10)",
        "Daily Study Hours", "Internal_Avg",
    ]
    TARGET = "Final Exam Marks (out of 100)"
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)
    lr = LinearRegression().fit(Xtr_s, ytr)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_split=5, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    yp_lr, yp_rf = lr.predict(Xte_s), rf.predict(Xte)

    def met(yt, yp):
        return dict(R2=float(r2_score(yt, yp)),
                    RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
                    MAE=float(mean_absolute_error(yt, yp)))

    return dict(lr=lr, rf=rf, scaler=scaler, features=FEATURES,
                lr_met=met(yte, yp_lr), rf_met=met(yte, yp_rf),
                fi=dict(zip(FEATURES, rf.feature_importances_)),
                df=df, Xte=Xte, yte=yte, yp_lr=yp_lr, yp_rf=yp_rf)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
GRADES = {
    "A+": (90,   100,  "#10d48e", "#02180e", "Outstanding! Top of the class. 🏆"),
    "A":  (80,   89.99, "#22c55e", "#031a09", "Excellent performance! 🌟"),
    "B":  (65,   79.99, "#5b7eff", "#0d1535", "Good job — well above average. 👍"),
    "C":  (50,   64.99, "#fbbf24", "#241a00", "Passing. More effort will push you higher. 💪"),
    "F":  (0,    49.99, "#f43f5e", "#200610", "Needs improvement. You can turn this around! 🔥"),
}

def get_grade(s):
    if s >= 90: return "A+", GRADES["A+"][2], GRADES["A+"][3], GRADES["A+"][4]
    if s >= 80: return "A",  GRADES["A"][2],  GRADES["A"][3],  GRADES["A"][4]
    if s >= 65: return "B",  GRADES["B"][2],  GRADES["B"][3],  GRADES["B"][4]
    if s >= 50: return "C",  GRADES["C"][2],  GRADES["C"][3],  GRADES["C"][4]
    return "F", GRADES["F"][2], GRADES["F"][3], GRADES["F"][4]

def predict_rf(M, att, t1, t2, asgn, hrs):
    avg = (t1 + t2) / 2
    inp = pd.DataFrame([[att, t1, t2, asgn, hrs, avg]], columns=M["features"])
    return float(np.clip(M["rf"].predict(inp)[0], 0, 100))

def predict_lr(M, att, t1, t2, asgn, hrs):
    avg = (t1 + t2) / 2
    inp = np.array([[att, t1, t2, asgn, hrs, avg]])
    return float(np.clip(M["lr"].predict(M["scaler"].transform(inp))[0], 0, 100))

def smart_tips(att, t1, t2, asgn, hrs, score):
    """
    Returns list of dicts: {icon, title, body, priority, color, category}
    priority: "critical" | "important" | "positive"
    """
    tips = []
    int_avg = (t1 + t2) / 2

    # ── Score-band context message ─────────────────────────────────────────
    if score >= 90:
        tips.append(dict(icon="🏆", title="Outstanding performance!",
            body="You are in the top grade band. Focus on maintaining consistency and challenging yourself with harder practice sets to stay sharp.",
            priority="positive", color="#10d48e", category="Overall"))
    elif score >= 80:
        tips.append(dict(icon="🌟", title="Excellent — one step from the top",
            body="You are just points away from an A+. A small push in your weaker areas — even 2 more marks in internal tests — can close the gap.",
            priority="positive", color="#22c55e", category="Overall"))
    elif score >= 65:
        tips.append(dict(icon="📈", title="Good standing — upgrade is within reach",
            body="You are solidly in Grade B territory. Targeted improvement in 1–2 areas could move you into the A range within the next assessment cycle.",
            priority="important", color="#5b7eff", category="Overall"))
    elif score >= 50:
        tips.append(dict(icon="⚡", title="Passing — but there is room to grow",
            body="You are above the passing threshold, but a few weak spots are holding back your score. Address them now before they compound.",
            priority="important", color="#fbbf24", category="Overall"))
    else:
        tips.append(dict(icon="🔥", title="Critical — urgent improvement needed",
            body="Your predicted score is below passing. This is recoverable — students who act early on attendance and test prep can move up a full grade band.",
            priority="critical", color="#f43f5e", category="Overall"))

    # ── Attendance ─────────────────────────────────────────────────────────
    if att < 60:
        tips.append(dict(icon="📅", title="Attendance is critically low",
            body=f"At {att}%, you are missing over a third of classes. Attendance is the single strongest non-test predictor of final score. Aim for 75%+ immediately — each additional class attended compounds your understanding.",
            priority="critical", color="#f43f5e", category="Attendance"))
    elif att < 75:
        tips.append(dict(icon="📅", title="Attendance needs improvement",
            body=f"You are at {att}% — just below the 75% threshold that strongly correlates with passing grades. Missing even 2–3 fewer classes per month can make a measurable difference to your final score.",
            priority="important", color="#fbbf24", category="Attendance"))
    elif att < 90:
        tips.append(dict(icon="📅", title="Good attendance — keep it up",
            body=f"At {att}%, your attendance is solid. Students who push above 90% attendance tend to score 5–8 points higher on average due to better concept retention.",
            priority="positive", color="#10d48e", category="Attendance"))

    # ── Internal Tests ─────────────────────────────────────────────────────
    if int_avg < 20:
        tips.append(dict(icon="📝", title="Internal test scores need urgent attention",
            body=f"Your combined average of {int_avg:.1f}/40 is below 50%. Focus on past papers and concept revision — even getting to 25/40 avg would significantly lift your predicted final score.",
            priority="critical", color="#f43f5e", category="Tests"))
    elif int_avg < 28:
        tips.append(dict(icon="📝", title="Strengthen your test performance",
            body=f"An average of {int_avg:.1f}/40 means you are leaving marks on the table. Review topics where you lost points, practice timed tests, and target 30+ average for a meaningful grade jump.",
            priority="important", color="#fbbf24", category="Tests"))
    else:
        tips.append(dict(icon="📝", title="Strong test scores — maintain the momentum",
            body=f"Your {int_avg:.1f}/40 average is excellent. Keep revising consistently; do not let up before the final exam as test performance is heavily weighted in the model.",
            priority="positive", color="#10d48e", category="Tests"))

    # ── Study Hours ────────────────────────────────────────────────────────
    if hrs < 2:
        tips.append(dict(icon="⏰", title="Study time is too low",
            body=f"At {hrs}h/day you are in the bottom tier for study time. Students with 3+ hours per day outperform 1–2h students by roughly one full grade band. Even adding 45 minutes daily makes a difference.",
            priority="critical", color="#f43f5e", category="Study Habits"))
    elif hrs < 3:
        tips.append(dict(icon="⏰", title="Consider increasing study time",
            body=f"You are studying {hrs}h/day. Bumping this to 3–4 hours with focused, distraction-free sessions — rather than passive re-reading — typically yields a 6–10 point score improvement.",
            priority="important", color="#fbbf24", category="Study Habits"))
    else:
        tips.append(dict(icon="⏰", title="Study time is strong",
            body=f"At {hrs}h/day you are in a good range. Quality matters as much as quantity — use active recall, spaced repetition, and past paper practice to get the most from your study sessions.",
            priority="positive", color="#10d48e", category="Study Habits"))

    # ── Assignments ────────────────────────────────────────────────────────
    if asgn < 5:
        tips.append(dict(icon="📋", title="Assignment score is dragging you down",
            body=f"A score of {asgn}/10 on assignments is significantly below average. Assignments signal consistent effort to instructors and directly contribute to internal marks. Prioritise completion and quality.",
            priority="critical", color="#f43f5e", category="Assignments"))
    elif asgn < 8:
        tips.append(dict(icon="📋", title="Assignment scores can be improved",
            body=f"At {asgn}/10 there is room to improve. Spending an extra 20–30 minutes reviewing assignment feedback and acting on it typically pushes this to 8–9/10 quickly.",
            priority="important", color="#fbbf24", category="Assignments"))
    else:
        tips.append(dict(icon="📋", title="Excellent assignment performance",
            body=f"Scoring {asgn}/10 on assignments shows consistent effort and engagement. This positive signal extends to how you approach exam preparation — keep it up.",
            priority="positive", color="#10d48e", category="Assignments"))

    return tips

def set_dark_mpl():
    plt.rcParams.update({
        "axes.facecolor":    "#090d1a",
        "figure.facecolor":  "#05080f",
        "axes.edgecolor":    "#1a2540",
        "axes.labelcolor":   "#7b8db8",
        "xtick.color":       "#3d4e70",
        "ytick.color":       "#3d4e70",
        "grid.color":        "#141e36",
        "text.color":        "#eef2ff",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
    })

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
df_raw = load_data()
if df_raw is None:
    st.error("⚠️  **`Final_Marks_Data.csv` not found.** Place it in the same folder and rerun.")
    st.stop()

with st.spinner("Warming up the models..."):
    M = train_models(df_raw)

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
defaults = dict(page="home", predicted=False, score_rf=None, score_lr=None,
                att=80, test1=28, test2=28, asgn=7, hrs=3)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP NAV BAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fade-in" style="
    background: linear-gradient(180deg, rgba(9,13,26,0.98) 0%, rgba(5,8,15,0.95) 100%);
    border-bottom: 1px solid #1a2540;
    padding: 16px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
    margin-bottom: 0;">
    <div style="display:flex; align-items:center; gap:14px;">
        <div style="width:36px; height:36px; background:linear-gradient(135deg,#5b7eff,#7c3aed);
                    border-radius:10px; display:flex; align-items:center; justify-content:center;
                    font-size:1.1rem; box-shadow:0 4px 14px rgba(91,126,255,0.4);">🎓</div>
        <span style="font-family:'Inter',sans-serif; font-weight:800; font-size:1.25rem;
                     color:#eef2ff; letter-spacing:-0.5px;">
            Edu<span style="background:linear-gradient(135deg,#5b7eff,#a78bfa);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text;">Sense</span>
        </span>
    </div>
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#5b7eff;
                    background:rgba(91,126,255,0.08); border:1px solid rgba(91,126,255,0.2);
                    border-radius:6px; padding:4px 12px; letter-spacing:0.8px;">
            ML-Powered · 2,000 Students
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Nav pills — uniform sizing with consistent styling ────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
cur = st.session_state.page

# Inject uniform nav pill styles
st.markdown(f"""
<style>
/* Force uniform nav pill sizing */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) button,
div[data-testid="stHorizontalBlock"] > div:nth-child(3) button,
div[data-testid="stHorizontalBlock"] > div:nth-child(4) button {{
    border-radius: 999px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    height: 44px !important;
    width: 100% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
/* Active nav pill */
div[data-testid="stHorizontalBlock"] > div:nth-child({'2' if cur == 'home' else '3' if cur == 'result' else '4'}) button {{
    background: linear-gradient(135deg, rgba(91,126,255,0.18), rgba(124,58,237,0.14)) !important;
    border: 1.5px solid rgba(91,126,255,0.45) !important;
    color: #818cf8 !important;
    box-shadow: 0 2px 14px rgba(91,126,255,0.2), inset 0 1px 0 rgba(91,126,255,0.1) !important;
}}
</style>
""", unsafe_allow_html=True)

_, nc1, nc2, nc3, _ = st.columns([3, 1.2, 1.2, 1.2, 3])
with nc1:
    if st.button("🏠  Home", use_container_width=True,
                 type="primary" if cur == "home" else "secondary"):
        st.session_state.page = "home"; st.rerun()
with nc2:
    if st.button("📊  My Result", use_container_width=True,
                 disabled=not st.session_state.predicted,
                 type="primary" if cur == "result" else "secondary"):
        st.session_state.page = "result"; st.rerun()
with nc3:
    if st.button("🔬  Nerd Zone", use_container_width=True,
                 type="primary" if cur == "nerd" else "secondary"):
        st.session_state.page = "nerd"; st.rerun()

st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  ██  HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="fade-in-up" style="text-align:center; padding:4px 0 56px 0; max-width:720px; margin:0 auto;">
        <div style="display:inline-flex; align-items:center; gap:8px;
                    background:rgba(91,126,255,0.07); border:1px solid rgba(91,126,255,0.2);
                    border-radius:999px; padding:6px 20px; margin-bottom:28px;">
            <span style="width:6px; height:6px; border-radius:50%; background:#5b7eff;
                         display:inline-block; box-shadow:0 0 10px #5b7eff;
                         animation: glow-pulse 2s ease-in-out infinite;"></span>
            <span style="font-family:'DM Mono',monospace; font-size:0.68rem;
                         color:#5b7eff; letter-spacing:2px;">PREDICT · REFLECT · IMPROVE</span>
        </div>
        <h1 style="font-family:'Inter',sans-serif; font-size:3.4rem; font-weight:800;
                   color:#eef2ff; line-height:1.1; margin:0 0 20px 0; letter-spacing:-2px;">
            What will you score<br>
            <span style="background:linear-gradient(135deg,#5b7eff 0%,#818cf8 45%,#a78bfa 100%);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                         background-clip:text;">
                in your final exam?
            </span>
        </h1>
        <p style="font-size:1.05rem; color:#7b8db8; line-height:1.85; margin:0 auto;
                  max-width:520px; font-weight:400;">
            Enter your academic stats below. Our ML model — trained on
            <strong style="color:#eef2ff; font-weight:600;">2,000 real students</strong> —
            predicts your score and grade <em>instantly</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-panel layout ──────────────────────────────────────────────────────
    form_col, gap_col, preview_col = st.columns([5, 0.3, 3])

    # ── LEFT: Input form ──────────────────────────────────────────────────────
    with form_col:
        st.markdown("""
        <div class="fade-in-up anim-d1" style="display:flex; align-items:center; gap:12px; margin-bottom:32px;">
            <div style="width:3px; height:22px; background:linear-gradient(180deg,#5b7eff,#7c3aed);
                        border-radius:2px;"></div>
            <span style="font-family:'Inter',sans-serif; font-weight:700;
                         font-size:1.05rem; color:#eef2ff; letter-spacing:-0.3px;">
                Your Academic Stats
            </span>
            <span style="font-family:'DM Mono',monospace; font-size:0.65rem;
                         color:#3d4e70; letter-spacing:1px;">— 5 inputs</span>
        </div>
        """, unsafe_allow_html=True)

        # ── CSS to style st.container(border=True) as premium cards ──────────
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(160deg, rgba(14,20,40,0.97) 0%, rgba(8,12,24,0.99) 100%) !important;
            border: 1.5px solid #1e2d4a !important;
            border-radius: 16px !important;
            padding: 2px 6px 10px 6px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.45) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] {
            padding: 0 2px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Attendance ────────────────────────────────────────────────────────
        with st.container(border=True):
            att_init = st.session_state.att
            att_col_init = "#10d48e" if att_init >= 75 else "#fbbf24"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:12px 6px 10px 6px;border-bottom:1px solid #1a2540;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:38px;height:38px;border-radius:10px;
                                background:rgba(91,126,255,0.12);border:1px solid rgba(91,126,255,0.25);
                                display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📅</div>
                    <div>
                        <div style="font-family:'Inter',sans-serif;font-weight:700;font-size:0.97rem;
                                    color:#e4ecff;letter-spacing:-0.2px;">Attendance</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.73rem;
                                    color:#4e6080;margin-top:2px;">Percentage of classes attended</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Inter',sans-serif;font-size:1.8rem;font-weight:800;
                                color:{att_col_init};line-height:1;letter-spacing:-1px;">{att_init}<span style="font-size:1rem;font-weight:600;color:#4e6080;">%</span></div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;color:{att_col_init};
                                margin-top:3px;">{"Good ✓" if att_init >= 75 else "Low ⚠"}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            att = st.slider("Attendance", 0, 100, att_init, 1, label_visibility="collapsed")
            att_color = "#10d48e" if att >= 75 else "#fbbf24"
            st.markdown(f"""
            <div style="padding:0 6px 14px 6px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#4e6080;">Progress</span>
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;font-weight:600;
                                 color:{att_color};">{"Good attendance ✓" if att >= 75 else "Aim for 75%+ ⚠️"}</span>
                </div>
                <div style="background:#16213a;border-radius:999px;height:6px;">
                    <div style="width:{att}%;background:linear-gradient(90deg,#5b7eff,{att_color});
                                height:6px;border-radius:999px;transition:width 0.3s;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Internal Tests ────────────────────────────────────────────────────
        with st.container(border=True):
            int_avg_init = (st.session_state.test1 + st.session_state.test2) / 2
            avg_col_init = "#10d48e" if int_avg_init >= 25 else "#fbbf24" if int_avg_init >= 18 else "#f43f5e"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:12px 6px 10px 6px;border-bottom:1px solid #1a2540;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:38px;height:38px;border-radius:10px;
                                background:rgba(167,139,250,0.12);border:1px solid rgba(167,139,250,0.25);
                                display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📝</div>
                    <div>
                        <div style="font-family:'Inter',sans-serif;font-weight:700;font-size:0.97rem;
                                    color:#e4ecff;letter-spacing:-0.2px;">Internal Tests</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.73rem;
                                    color:#4e6080;margin-top:2px;">Each scored out of 40 marks</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;color:#4e6080;margin-bottom:2px;">AVERAGE</div>
                    <div style="font-family:'Inter',sans-serif;font-size:1.8rem;font-weight:800;
                                color:{avg_col_init};line-height:1;letter-spacing:-1px;">{int_avg_init:.0f}<span style="font-size:1rem;font-weight:600;color:#4e6080;">/40</span></div>
                </div>
            </div>""", unsafe_allow_html=True)
            t_col1, t_col2 = st.columns(2, gap="medium")
            with t_col1:
                st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.75rem;font-weight:600;color:#5e7090;margin:0 0 4px 6px;">Test 1 <span style="color:#2d3f5e;font-weight:400;">/40</span></p>', unsafe_allow_html=True)
                test1 = st.slider("Test 1", 0, 40, st.session_state.test1, 1, label_visibility="collapsed")
                t1_c = "#10d48e" if test1 >= 28 else "#fbbf24" if test1 >= 20 else "#f43f5e"
                st.markdown(f'<p style="font-family:Inter,sans-serif;font-size:0.82rem;font-weight:700;color:{t1_c};margin:2px 0 0 6px;">{test1} / 40</p>', unsafe_allow_html=True)
            with t_col2:
                st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.75rem;font-weight:600;color:#5e7090;margin:0 0 4px 6px;">Test 2 <span style="color:#2d3f5e;font-weight:400;">/40</span></p>', unsafe_allow_html=True)
                test2 = st.slider("Test 2", 0, 40, st.session_state.test2, 1, label_visibility="collapsed")
                t2_c = "#10d48e" if test2 >= 28 else "#fbbf24" if test2 >= 20 else "#f43f5e"
                st.markdown(f'<p style="font-family:Inter,sans-serif;font-size:0.82rem;font-weight:700;color:{t2_c};margin:2px 0 0 6px;">{test2} / 40</p>', unsafe_allow_html=True)
            int_avg_val = (test1 + test2) / 2
            int_avg_pct = int(int_avg_val / 40 * 100)
            avg_col = "#10d48e" if int_avg_val >= 25 else "#fbbf24" if int_avg_val >= 18 else "#f43f5e"
            st.markdown(f"""
            <div style="margin:8px 6px 0 6px;background:rgba(0,0,0,0.25);border-radius:10px;padding:10px 14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.73rem;
                                 font-weight:600;color:#4e6080;">Combined Average</span>
                    <span style="font-family:'Inter',sans-serif;font-size:0.88rem;
                                 font-weight:700;color:{avg_col};">{int_avg_val:.1f} / 40</span>
                </div>
                <div style="background:#16213a;border-radius:999px;height:5px;">
                    <div style="width:{int_avg_pct}%;background:linear-gradient(90deg,#a78bfa,{avg_col});
                                height:5px;border-radius:999px;transition:width 0.3s;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Assignment ────────────────────────────────────────────────────────
        with st.container(border=True):
            asgn_init = st.session_state.asgn
            asgn_col_init = "#10d48e" if asgn_init >= 7 else "#fbbf24" if asgn_init >= 5 else "#f43f5e"
            asgn_lbl_init = "Excellent ✓" if asgn_init >= 7 else "Average" if asgn_init >= 5 else "Low ⚠"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:12px 6px 10px 6px;border-bottom:1px solid #1a2540;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:38px;height:38px;border-radius:10px;
                                background:rgba(16,212,142,0.1);border:1px solid rgba(16,212,142,0.22);
                                display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📋</div>
                    <div>
                        <div style="font-family:'Inter',sans-serif;font-weight:700;font-size:0.97rem;
                                    color:#e4ecff;letter-spacing:-0.2px;">Assignment Score</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.73rem;
                                    color:#4e6080;margin-top:2px;">Total marks out of 10</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Inter',sans-serif;font-size:1.8rem;font-weight:800;
                                color:{asgn_col_init};line-height:1;letter-spacing:-1px;">{asgn_init}<span style="font-size:1rem;font-weight:600;color:#4e6080;">/10</span></div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;color:{asgn_col_init};
                                margin-top:3px;">{asgn_lbl_init}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            asgn = st.slider("Assignment", 0, 10, asgn_init, 1, label_visibility="collapsed")
            asgn_color = "#10d48e" if asgn >= 7 else "#fbbf24" if asgn >= 5 else "#f43f5e"
            asgn_label = "Excellent ✓" if asgn >= 7 else "Average" if asgn >= 5 else "Low ⚠️"
            st.markdown(f"""
            <div style="padding:0 6px 14px 6px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#4e6080;">Progress</span>
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;font-weight:600;
                                 color:{asgn_color};">{asgn_label}</span>
                </div>
                <div style="background:#16213a;border-radius:999px;height:6px;">
                    <div style="width:{asgn*10}%;background:linear-gradient(90deg,#10d48e,{asgn_color});
                                height:6px;border-radius:999px;transition:width 0.3s;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Study Hours ───────────────────────────────────────────────────────
        current_hrs = st.session_state.hrs if st.session_state.hrs in [1,2,3,4,5] else 3
        with st.container(border=True):
            hrs_col_init = "#10d48e" if current_hrs >= 3 else "#fbbf24"
            hrs_tip_init = "Great dedication ✓" if current_hrs >= 4 else "On track" if current_hrs >= 3 else "Try more"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:12px 6px 10px 6px;border-bottom:1px solid #1a2540;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:38px;height:38px;border-radius:10px;
                                background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.22);
                                display:flex;align-items:center;justify-content:center;font-size:1.1rem;">⏰</div>
                    <div>
                        <div style="font-family:'Inter',sans-serif;font-weight:700;font-size:0.97rem;
                                    color:#e4ecff;letter-spacing:-0.2px;">Daily Study Hours</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.73rem;
                                    color:#4e6080;margin-top:2px;">Hours of focused study per day</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Inter',sans-serif;font-size:1.8rem;font-weight:800;
                                color:{hrs_col_init};line-height:1;letter-spacing:-1px;">{current_hrs}<span style="font-size:1rem;font-weight:600;color:#4e6080;">h</span></div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;color:{hrs_col_init};
                                margin-top:3px;">{hrs_tip_init}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            hrs = st.slider("Study Hours", 1, 5, current_hrs, 1, label_visibility="collapsed")
            hrs_color = "#10d48e" if hrs >= 3 else "#fbbf24"
            hrs_bar = int(hrs / 5 * 100)
            hrs_tip = "Great dedication ✓" if hrs >= 4 else "On track" if hrs >= 3 else "Try to study more"
            st.markdown(f"""
            <div style="padding:0 6px 14px 6px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#4e6080;">Progress</span>
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;font-weight:600;
                                 color:{hrs_color};">{hrs_tip}</span>
                </div>
                <div style="background:#16213a;border-radius:999px;height:6px;">
                    <div style="width:{hrs_bar}%;background:linear-gradient(90deg,#fbbf24,{hrs_color});
                                height:6px;border-radius:999px;transition:width 0.3s;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        # Predict button
        go = st.button("🔮  Predict My Score", use_container_width=True, type="primary")
        st.markdown("""
        <div style="text-align:center; margin-top:14px; color:#3d4e70;
                    font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.5px;">
            Random Forest · Linear Regression · scikit-learn
        </div>
        """, unsafe_allow_html=True)

        # ── How It Works section ──────────────────────────────────────────────
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="border-top:1px solid #1a2540;padding-top:32px;margin-top:4px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:24px;">
                <span style="font-size:1.1rem;">⚙️</span>
                <span style="font-family:'Inter',sans-serif;font-weight:700;font-size:0.95rem;color:#eef2ff;">
                    How It Works
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:rgba(91,126,255,0.06);border:1px solid rgba(91,126,255,0.15);
                            border-radius:14px;padding:18px 16px;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#5b7eff;
                                letter-spacing:1px;margin-bottom:8px;">STEP 01</div>
                    <div style="font-weight:600;font-size:0.85rem;color:#eef2ff;margin-bottom:6px;">Data Collection</div>
                    <div style="font-size:0.78rem;color:#7b8db8;line-height:1.6;">
                        2,000 student records with 6 academic features collected and cleaned.
                    </div>
                </div>
                <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.15);
                            border-radius:14px;padding:18px 16px;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#a78bfa;
                                letter-spacing:1px;margin-bottom:8px;">STEP 02</div>
                    <div style="font-weight:600;font-size:0.85rem;color:#eef2ff;margin-bottom:6px;">Model Training</div>
                    <div style="font-size:0.78rem;color:#7b8db8;line-height:1.6;">
                        Random Forest (100 trees) and Linear Regression trained on 80% split.
                    </div>
                </div>
                <div style="background:rgba(16,212,142,0.06);border:1px solid rgba(16,212,142,0.15);
                            border-radius:14px;padding:18px 16px;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#10d48e;
                                letter-spacing:1px;margin-bottom:8px;">STEP 03</div>
                    <div style="font-weight:600;font-size:0.85rem;color:#eef2ff;margin-bottom:6px;">Prediction</div>
                    <div style="font-size:0.78rem;color:#7b8db8;line-height:1.6;">
                        RF captures non-linear patterns; LR provides interpretable baseline.
                    </div>
                </div>
                <div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);
                            border-radius:14px;padding:18px 16px;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#fbbf24;
                                letter-spacing:1px;margin-bottom:8px;">STEP 04</div>
                    <div style="font-weight:600;font-size:0.85rem;color:#eef2ff;margin-bottom:6px;">Insights</div>
                    <div style="font-size:0.78rem;color:#7b8db8;line-height:1.6;">
                        Personalised tips generated based on your weakest performance areas.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── GAP ───────────────────────────────────────────────────────────────────
    with gap_col:
        st.markdown("")

    # ── RIGHT: Live Preview Panel ─────────────────────────────────────────────
    with preview_col:
        try:
            live_rf  = predict_rf(M, att, test1, test2, asgn, hrs)
            live_lr  = predict_lr(M, att, test1, test2, asgn, hrs)
            live_avg = (live_rf + live_lr) / 2
            prev_grade, prev_col, prev_bg, prev_msg = get_grade(live_avg)
        except Exception:
            live_avg, prev_grade, prev_col, prev_bg, prev_msg = 50.0, "C", "#fbbf24", "#241a00", ""

        int_avg  = round((test1 + test2) / 2, 1)
        pct      = min(live_avg, 100)
        bar_width = f"{pct:.1f}%"

        # rgba maps for sanitizer safety
        grade_rgba_map = {
            "#10d48e": ("rgba(16,212,142,0.12)",  "rgba(16,212,142,0.35)"),
            "#22c55e": ("rgba(34,197,94,0.12)",   "rgba(34,197,94,0.35)"),
            "#5b7eff": ("rgba(91,126,255,0.12)",  "rgba(91,126,255,0.35)"),
            "#fbbf24": ("rgba(251,191,36,0.12)",  "rgba(251,191,36,0.35)"),
            "#f43f5e": ("rgba(244,63,94,0.12)",   "rgba(244,63,94,0.35)"),
        }
        pc_bg_rgba, pc_border_rgba = grade_rgba_map.get(
            prev_col, ("rgba(91,126,255,0.12)", "rgba(91,126,255,0.35)")
        )

        att_ok  = att     >= 75
        avg_ok  = int_avg >= 25
        hrs_ok  = hrs     >= 3
        asgn_ok = asgn    >= 7

        def pill_border(ok):  return "rgba(16,212,142,0.22)" if ok else "rgba(251,191,36,0.22)"
        def pill_ibg(ok):     return "rgba(16,212,142,0.1)"  if ok else "rgba(251,191,36,0.1)"
        def pill_ibrd(ok):    return "rgba(16,212,142,0.5)"  if ok else "rgba(251,191,36,0.5)"
        def ok_color(ok):     return "#10d48e" if ok else "#fbbf24"
        def ok_icon(ok):      return "✓" if ok else "!"

        def make_pill(lbl, val, ok):
            return (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:11px 16px;background:rgba(0,0,0,0.25);'
                f'border-radius:12px;border:1px solid {pill_border(ok)};'
                f'margin-bottom:8px;transition:all 0.2s;">'
                f'<span style="font-size:0.82rem;color:#7b8db8;font-weight:500;">{lbl}</span>'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-family:\'DM Mono\',monospace;font-weight:500;'
                f'font-size:0.88rem;color:#eef2ff;">{val}</span>'
                f'<span style="width:22px;height:22px;border-radius:50%;'
                f'background:{pill_ibg(ok)};border:1.5px solid {pill_ibrd(ok)};'
                f'color:{ok_color(ok)};font-size:0.6rem;'
                f'display:inline-flex;align-items:center;justify-content:center;font-weight:800;">'
                f'{ok_icon(ok)}</span></div></div>'
            )

        pills_html = (
            make_pill("Attendance",   f"{att}%",        att_ok)
            + make_pill("Internal Avg", f"{int_avg}/40", avg_ok)
            + make_pill("Study Time",   f"{hrs} hr/day", hrs_ok)
            + make_pill("Assignment",   f"{asgn}/10",    asgn_ok)
        )

        # Score percentage for arc display
        score_pct = int(live_avg)
        confidence = "High" if abs(live_rf - live_lr) <= 3 else "Moderate" if abs(live_rf - live_lr) <= 7 else "Low"
        conf_color = "#10d48e" if confidence == "High" else "#fbbf24" if confidence == "Moderate" else "#f43f5e"

        preview_html = (
            f'<div style="background:linear-gradient(145deg,rgba(13,17,32,0.95),rgba(9,13,26,0.98));'
            f'border:1.5px solid #1a2540;border-radius:24px;padding:28px 24px 24px;'
            f'box-shadow:0 8px 40px rgba(0,0,0,0.5),0 2px 8px rgba(0,0,0,0.3);'
            f'position:sticky;top:90px;">'

            # Header
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
            f'color:#5b7eff;letter-spacing:2px;">LIVE PREVIEW</div>'
            f'<div style="width:7px;height:7px;border-radius:50%;background:#5b7eff;'
            f'box-shadow:0 0 8px #5b7eff;animation:glow-pulse 2s infinite;"></div>'
            f'</div>'

            # Score circle
            f'<div style="text-align:center;margin-bottom:24px;">'
            f'<div style="position:relative;display:inline-flex;align-items:center;'
            f'justify-content:center;width:140px;height:140px;border-radius:50%;'
            f'background:conic-gradient({prev_col} {int(pct/100*360)}deg, #1a2540 0deg);'
            f'margin-bottom:18px;">'
            f'<div style="position:absolute;width:108px;height:108px;border-radius:50%;'
            f'background:#090d1a;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;gap:2px;">'
            f'<span style="font-family:\'Syne\',sans-serif;font-size:2.6rem;'
            f'font-weight:800;color:#eef2ff;line-height:1;">{live_avg:.0f}</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.55rem;'
            f'color:#3d4e70;letter-spacing:1px;">/100</span>'
            f'</div></div>'

            # Grade badge
            f'<div style="display:inline-flex;align-items:center;gap:8px;'
            f'background:{pc_bg_rgba};border:1.5px solid {pc_border_rgba};'
            f'border-radius:999px;padding:7px 22px;margin-bottom:8px;">'
            f'<span style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1rem;'
            f'color:{prev_col};">Grade {prev_grade}</span>'
            f'</div>'

            f'<div style="display:flex;justify-content:center;gap:16px;margin-top:10px;">'
            f'<div style="text-align:center;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#3d4e70;letter-spacing:0.5px;">RF</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:#eef2ff;font-weight:500;">{live_rf:.1f}</div>'
            f'</div>'
            f'<div style="width:1px;background:#1a2540;"></div>'
            f'<div style="text-align:center;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#3d4e70;letter-spacing:0.5px;">LR</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:#eef2ff;font-weight:500;">{live_lr:.1f}</div>'
            f'</div>'
            f'<div style="width:1px;background:#1a2540;"></div>'
            f'<div style="text-align:center;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#3d4e70;letter-spacing:0.5px;">CONF</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:{conf_color};font-weight:500;">{confidence}</div>'
            f'</div>'
            f'</div>'
            f'</div>'

            # Divider
            f'<div style="border-top:1px solid #1a2540;margin:16px 0;"></div>'

            # Input status pills
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;color:#3d4e70;'
            f'letter-spacing:1px;margin-bottom:10px;">INPUT STATUS</div>'
            f'<div>{pills_html}</div>'

            # Prompt
            f'<div style="margin-top:14px;text-align:center;font-family:\'DM Mono\',monospace;'
            f'font-size:0.62rem;color:#3d4e70;letter-spacing:0.5px;">'
            f'← Updates live as you change inputs</div>'
            f'</div>'
        )

        st.markdown(preview_html, unsafe_allow_html=True)

    # ── Handle predict ────────────────────────────────────────────────────────
    if go:
        sc_rf = predict_rf(M, att, test1, test2, asgn, hrs)
        sc_lr = predict_lr(M, att, test1, test2, asgn, hrs)
        for k, v in [("score_rf", sc_rf), ("score_lr", sc_lr), ("att", att),
                     ("test1", test1), ("test2", test2), ("asgn", asgn),
                     ("hrs", hrs), ("predicted", True), ("page", "result")]:
            st.session_state[k] = v
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  ██  RESULT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "result" and st.session_state.predicted:

    sc_rf = st.session_state.score_rf
    sc_lr = st.session_state.score_lr
    att   = st.session_state.att
    t1    = st.session_state.test1
    t2    = st.session_state.test2
    asgn  = st.session_state.asgn
    hrs   = st.session_state.hrs

    grade, gcol, gbg, gmsg = get_grade(sc_rf)
    lr_grade, lr_col, lr_bg, _ = get_grade(sc_lr)
    tips = smart_tips(att, t1, t2, asgn, hrs, sc_rf)
    ring_deg = int(sc_rf / 100 * 360)

    # ── Score Hero ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    @keyframes score-glow {{
        0%, 100% {{ box-shadow: 0 0 50px {gcol}33, 0 0 100px {gcol}18; }}
        50%       {{ box-shadow: 0 0 70px {gcol}55, 0 0 140px {gcol}28; }}
    }}
    @keyframes count-up {{
        from {{ opacity: 0; transform: scale(0.8); }}
        to   {{ opacity: 1; transform: scale(1); }}
    }}
    .score-ring-v2 {{ animation: score-glow 3s ease-in-out infinite; }}
    .score-number  {{ animation: count-up 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s both; }}
    </style>
    <div class="fade-in" style="text-align:center; padding:16px 0 56px 0;">
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                    color:#5b7eff; letter-spacing:2.5px; margin-bottom:32px;">
            YOUR RESULT
        </div>
        <div class="score-ring-v2" style="
            display:inline-flex; align-items:center; justify-content:center;
            width:220px; height:220px; border-radius:50%;
            background:conic-gradient({gcol} {ring_deg}deg, #1a2540 0deg);
            margin:0 auto 30px auto; position:relative;">
            <div style="position:absolute; width:170px; height:170px; border-radius:50%;
                        background:#05080f;
                        display:flex; flex-direction:column; align-items:center;
                        justify-content:center; gap:4px;">
                <span class="score-number" style="font-family:'Inter',sans-serif;
                             font-size:3.4rem; font-weight:800; color:#eef2ff;
                             line-height:1;">{sc_rf:.1f}</span>
                <span style="font-family:'DM Mono',monospace; font-size:0.65rem;
                             color:#3d4e70; letter-spacing:1.5px;">OUT OF 100</span>
            </div>
        </div>
        <div style="display:inline-flex; align-items:center; gap:12px;
                    background:{gbg}; border:1.5px solid {gcol}55;
                    border-radius:999px; padding:10px 36px; margin-bottom:16px;">
            <span style="font-family:'Inter',sans-serif; font-weight:800;
                         font-size:1.4rem; color:{gcol};">Grade {grade}</span>
        </div>
        <p style="color:#7b8db8; font-size:1rem; margin:10px auto 6px auto;
                  max-width:480px; font-weight:400;">{gmsg}</p>
        <p style="color:#3d4e70; font-size:0.75rem; margin:0; font-family:'DM Mono',monospace;">
            Primary · Random Forest &nbsp;·&nbsp;
            Linear Regression: <span style="color:{lr_col}; font-weight:500;">{sc_lr:.1f}</span>
            (Grade <span style="color:{lr_col};">{lr_grade}</span>)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model cards + Input recap ─────────────────────────────────────────────
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.62rem;
                    color:#5b7eff; letter-spacing:2px; margin-bottom:18px;">
            BOTH MODELS PREDICT
        </div>
        """, unsafe_allow_html=True)

        mc1, mc2 = st.columns(2, gap="medium")
        for col, name, sc, gc, gl, badge, is_primary in [
            (mc1, "Random Forest",     sc_rf, gcol,   grade,    "⭐ Primary", True),
            (mc2, "Linear Regression", sc_lr, lr_col, lr_grade, "Alternative", False),
        ]:
            gc_safe = gc.replace('#','')
            col.markdown(f"""
            <style>
            .mcard-v2-{gc_safe} {{
                background: linear-gradient(145deg, #0d1120, #090d1a);
                border: 1px solid rgba(91,126,255,0.12);
                border-top: 3px solid {gc};
                border-radius: 20px;
                padding: 26px 18px;
                text-align: center;
                box-shadow: 0 4px 24px rgba(0,0,0,0.4);
                transition: transform 0.25s cubic-bezier(0.4,0,0.2,1),
                            box-shadow 0.25s, border-color 0.25s;
                cursor: default;
            }}
            .mcard-v2-{gc_safe}:hover {{
                transform: translateY(-6px);
                box-shadow: 0 20px 48px rgba(0,0,0,0.6), 0 0 28px {gc}22;
                border-color: {gc}44;
            }}
            </style>
            <div class="mcard-v2-{gc_safe}">
                <div style="font-size:0.6rem;color:{gc};font-family:'DM Mono',monospace;
                             letter-spacing:1.5px;margin-bottom:10px;">{badge}</div>
                <div style="font-size:0.78rem;color:#7b8db8;margin-bottom:14px;font-weight:400;">{name}</div>
                <div style="font-family:'Inter',sans-serif;font-size:2.6rem;
                             font-weight:800;color:#eef2ff;line-height:1;margin-bottom:14px;">{sc:.1f}</div>
                <div style="display:inline-block;background:{gc}18;
                            border:1.5px solid {gc}44;border-radius:999px;padding:5px 18px;">
                    <span style="color:{gc};font-weight:700;font-size:0.85rem;">Grade {gl}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Grade scale
        st.markdown("""
        <div style="margin-top:28px;font-family:'DM Mono',monospace;font-size:0.62rem;
                    color:#5b7eff;letter-spacing:2px;margin-bottom:14px;">GRADE SCALE</div>
        """, unsafe_allow_html=True)
        grade_info = [("A+","90–100","#10d48e"),("A","80–89","#22c55e"),
                      ("B","65–79","#5b7eff"),("C","50–64","#fbbf24"),("F","0–49","#f43f5e")]
        gs = st.columns(5, gap="small")
        for col, (g, rng, gc) in zip(gs, grade_info):
            is_active = g == grade
            border_style = f"border:2px solid {gc};box-shadow:0 0 14px {gc}33;" if is_active else "border:1px solid #1a2540;"
            col.markdown(f"""
            <div style="background:#0d1120;{border_style}border-radius:12px;
                        padding:14px 6px;text-align:center;transition:var(--transition);">
                <div style="font-family:'Inter',sans-serif;font-weight:800;color:{gc};font-size:1.05rem;">{g}</div>
                <div style="font-size:0.55rem;color:#3d4e70;margin-top:4px;
                             font-family:'DM Mono',monospace;">{rng}</div>
            </div>
            """, unsafe_allow_html=True)

        # Agreement bar
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        diff = abs(sc_rf - sc_lr)
        ac   = "#10d48e" if diff <= 3 else "#fbbf24" if diff <= 7 else "#f43f5e"
        al   = "High" if diff <= 3 else "Moderate" if diff <= 7 else "Low"
        st.markdown(f"""
        <div style="background:#0d1120;border:1px solid #1a2540;border-radius:18px;padding:20px 22px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                        color:#5b7eff;letter-spacing:2px;margin-bottom:14px;">MODEL AGREEMENT</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;align-items:center;">
                <span style="font-size:0.8rem;color:#7b8db8;">RF: {sc_rf:.1f}</span>
                <span style="font-size:0.72rem;color:{ac};font-family:'DM Mono',monospace;font-weight:600;">
                    {al} agreement · Δ{diff:.1f}
                </span>
                <span style="font-size:0.8rem;color:#7b8db8;">LR: {sc_lr:.1f}</span>
            </div>
            <div style="background:#1a2540;border-radius:999px;height:7px;overflow:hidden;">
                <div style="width:{sc_rf:.1f}%;
                            background:linear-gradient(90deg,{gcol}cc,{gcol});
                            height:7px;border-radius:999px;transition:width 0.6s ease;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:6px;">
                <span style="font-size:0.6rem;color:#3d4e70;font-family:'DM Mono',monospace;">0</span>
                <span style="font-size:0.6rem;color:#3d4e70;font-family:'DM Mono',monospace;">100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                    color:#5b7eff;letter-spacing:2px;margin-bottom:18px;">YOUR INPUTS</div>
        """, unsafe_allow_html=True)

        input_rows = [
            [("📅","Attendance", f"{att}%",      att >= 75),
             ("📝","Test 1",     f"{t1}/40",     t1 >= 25),
             ("📝","Test 2",     f"{t2}/40",     t2 >= 25)],
            [("📋","Assignment", f"{asgn}/10",   asgn >= 7),
             ("⏰","Study",      f"{hrs}h/day",  hrs >= 3),
             ("📊","Int. Avg",  f"{(t1+t2)/2:.1f}/40", (t1+t2)/2 >= 25)],
        ]
        for row in input_rows:
            cols = st.columns(3, gap="medium")
            for col, (icon, lbl, val, ok) in zip(cols, row):
                sc_color = "#10d48e" if ok else "#fbbf24"
                col.markdown(f"""
                <div style="background:#0d1120;border:1px solid #1a2540;
                            border-radius:16px;padding:20px 10px;text-align:center;
                            margin-bottom:12px;transition:all 0.25s cubic-bezier(0.4,0,0.2,1);
                            cursor:default;"
                     onmouseover="this.style.transform='translateY(-5px)';
                                  this.style.boxShadow='0 14px 32px rgba(0,0,0,0.5)';
                                  this.style.borderColor='{sc_color}44';"
                     onmouseout="this.style.transform='';
                                 this.style.boxShadow='';
                                 this.style.borderColor='#1a2540';">
                    <div style="font-size:1.3rem;margin-bottom:8px;">{icon}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:1rem;
                                 font-weight:500;color:{sc_color};margin-bottom:6px;">{val}</div>
                    <div style="font-size:0.58rem;color:#3d4e70;text-transform:uppercase;
                                 letter-spacing:0.8px;font-family:'DM Mono',monospace;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)



    # ── Personalized Tips Section ────────────────────────────────────────────
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    priority_meta = {
        "critical":  {"label": "Needs Attention", "bg": "rgba(244,63,94,0.08)",
                      "border": "rgba(244,63,94,0.35)", "badge_bg": "rgba(244,63,94,0.15)",
                      "badge_color": "#f43f5e"},
        "important": {"label": "Worth Improving",  "bg": "rgba(251,191,36,0.06)",
                      "border": "rgba(251,191,36,0.3)",  "badge_bg": "rgba(251,191,36,0.12)",
                      "badge_color": "#fbbf24"},
        "positive":  {"label": "Keep It Up",       "bg": "rgba(16,212,142,0.06)",
                      "border": "rgba(16,212,142,0.28)", "badge_bg": "rgba(16,212,142,0.12)",
                      "badge_color": "#10d48e"},
    }

    # Section header
    overall_tip = tips[0]
    oh = priority_meta[overall_tip["priority"]]
    st.markdown(f"""
    <style>
    @keyframes tip-slide-in {{
        from {{ opacity:0; transform:translateY(18px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    .tip-card {{ animation: tip-slide-in 0.45s ease both; }}
    </style>
    <div style="border-top:1px solid #1a2540;padding-top:36px;margin-bottom:28px;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    flex-wrap:wrap;gap:16px;margin-bottom:8px;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                            color:#5b7eff;letter-spacing:2px;margin-bottom:8px;">
                    PERSONALISED TIPS
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:1.35rem;
                            font-weight:800;color:#e4ecff;letter-spacing:-0.4px;">
                    Your action plan based on this result
                </div>
            </div>
            <div style="background:{oh["bg"]};border:1px solid {oh["border"]};
                        border-radius:14px;padding:14px 20px;min-width:220px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                            color:{overall_tip["color"]};letter-spacing:1.5px;margin-bottom:6px;">
                    OVERALL VERDICT
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:1.4rem;">{overall_tip["icon"]}</span>
                    <div style="font-family:'Inter',sans-serif;font-weight:700;
                                font-size:0.9rem;color:{overall_tip["color"]};">
                        {overall_tip["title"]}
                    </div>
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
                            color:#6a7fa8;margin-top:8px;line-height:1.6;">
                    {overall_tip["body"]}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tip cards in 2-column grid (skip index 0 — already shown as verdict above)
    remaining_tips = tips[1:]
    for row_start in range(0, len(remaining_tips), 2):
        row_tips = remaining_tips[row_start:row_start+2]
        tip_cols = st.columns(len(row_tips), gap="large")
        for col, tip in zip(tip_cols, row_tips):
            m = priority_meta[tip["priority"]]
            col.markdown(f"""
            <div class="tip-card" style="
                background:{m["bg"]};
                border:1.5px solid {m["border"]};
                border-radius:18px;
                padding:22px 22px 20px 22px;
                margin-bottom:18px;
                transition:transform 0.22s ease, box-shadow 0.22s ease;
                cursor:default;"
                onmouseover="this.style.transform='translateY(-4px)';
                             this.style.boxShadow='0 16px 40px rgba(0,0,0,0.5)'"
                onmouseout="this.style.transform='';this.style.boxShadow=''">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            margin-bottom:14px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div style="width:36px;height:36px;border-radius:10px;
                                    background:{m["badge_bg"]};
                                    display:flex;align-items:center;justify-content:center;
                                    font-size:1.1rem;flex-shrink:0;">{tip["icon"]}</div>
                        <div>
                            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                        color:{tip["color"]};letter-spacing:1px;margin-bottom:3px;">
                                {tip["category"].upper()}
                            </div>
                            <div style="font-family:'Inter',sans-serif;font-weight:700;
                                        color:#e4ecff;font-size:0.92rem;line-height:1.25;">
                                {tip["title"]}
                            </div>
                        </div>
                    </div>
                    <div style="background:{m["badge_bg"]};border:1px solid {m["border"]};
                                border-radius:999px;padding:3px 10px;white-space:nowrap;flex-shrink:0;">
                        <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;
                                     font-weight:700;color:{m["badge_color"]};">
                            {m["label"]}
                        </span>
                    </div>
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.8rem;
                            color:#7080a0;line-height:1.75;border-top:1px solid {m["border"]};
                            padding-top:12px;">
                    {tip["body"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Back button ───────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    _, back_col, _ = st.columns([3, 2, 3])
    with back_col:
        if st.button("← Try Different Values", use_container_width=True, type="primary"):
            st.session_state.page = "home"; st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="border-top:1px solid #1a2540;margin-top:48px;padding-top:28px;
                display:flex;justify-content:space-between;align-items:center;
                flex-wrap:wrap;gap:8px;">
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4e70;">
            sklearn · RandomForestRegressor · LinearRegression
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4e70;">
            100 trees · max_depth=10 · 80/20 split · seed=42
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4e70;">
            WillIPass? v2.0 ✦ ML Dashboard
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ██  NERD ZONE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "nerd":

    set_dark_mpl()

    st.markdown("""
    <div class="fade-in-up" style="padding:4px 0 44px 0;">
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
                    border-radius:999px;padding:6px 18px;margin-bottom:18px;">
            <span style="font-size:0.9rem;">⚗️</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.68rem;
                         color:#a78bfa;letter-spacing:1.5px;">FOR THE CURIOUS MINDS</span>
        </div>
        <h2 style="font-family:'Inter',sans-serif;font-weight:800;font-size:2.4rem;
                   color:#eef2ff;margin:0 0 12px 0;letter-spacing:-1px;">Nerd Zone</h2>
        <p style="color:#7b8db8;font-size:0.95rem;max-width:560px;line-height:1.75;margin:0;">
            Full ML analysis — model performance, feature importance, score distribution,
            and feature correlations across <strong style="color:#eef2ff;">2,000 students</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Model Metrics",
        "🌲  Feature Importance",
        "📈  Score Distribution",
        "🔗  Correlations",
    ])

    # ── TAB 1: Model Metrics ──────────────────────────────────────────────────
    with tab1:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                    color:#5b7eff;letter-spacing:2px;margin-bottom:24px;">
            TEST SET PERFORMANCE · 20% HOLDOUT · 400 STUDENTS
        </div>
        """, unsafe_allow_html=True)

        for model_name, mets, accent in [
            ("Random Forest",     M["rf_met"], "#5b7eff"),
            ("Linear Regression", M["lr_met"], "#a78bfa"),
        ]:
            st.markdown(f"""
            <div style="font-family:'Inter',sans-serif;font-size:0.88rem;font-weight:700;
                        color:{accent};margin:22px 0 14px 0;">
                {model_name}
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3, gap="large")
            for col, key, lbl, desc, good in [
                (mc1, "R2",   "R² Score",  "Variance explained",  "high"),
                (mc2, "RMSE", "RMSE",      "Root mean sq. error", "low"),
                (mc3, "MAE",  "MAE",       "Mean absolute error", "low"),
            ]:
                val = mets[key]
                ok  = (good == "high" and val > 0.78) or (good == "low" and val < 5.5)
                vc  = "#10d48e" if ok else "#fbbf24"
                safe_key = f"{model_name.replace(' ','')}{key}"
                col.markdown(f"""
                <style>
                .mcard-v2-{safe_key} {{
                    background:#0d1120; border:1px solid #1a2540;
                    border-top:3px solid {vc}; border-radius:20px;
                    padding:28px 18px; text-align:center;
                    box-shadow:0 4px 24px rgba(0,0,0,0.4);
                    margin-bottom:14px;
                    transition:transform 0.25s cubic-bezier(0.4,0,0.2,1), box-shadow 0.25s;
                    cursor:default;
                }}
                .mcard-v2-{safe_key}:hover {{
                    transform:translateY(-6px);
                    box-shadow:0 20px 40px rgba(0,0,0,0.55), 0 0 24px {vc}20;
                }}
                </style>
                <div class="mcard-v2-{safe_key}">
                    <div style="font-family:'Inter',sans-serif;font-size:2rem;
                                 font-weight:800;color:{vc};margin-bottom:8px;line-height:1;">{val:.3f}</div>
                    <div style="font-size:0.85rem;font-weight:600;color:#eef2ff;margin-bottom:4px;">{lbl}</div>
                    <div style="font-size:0.7rem;color:#3d4e70;font-family:'DM Mono',monospace;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.patch.set_facecolor("#05080f")
        for ax, yp, name, clr in [
            (axes[0], M["yp_rf"], "Random Forest",     "#5b7eff"),
            (axes[1], M["yp_lr"], "Linear Regression", "#a78bfa"),
        ]:
            ax.set_facecolor("#090d1a")
            ax.scatter(M["yte"], yp, alpha=0.35, s=16, color=clr, edgecolors="none")
            mn, mx = float(M["yte"].min()), float(M["yte"].max())
            ax.plot([mn, mx], [mn, mx], "--", color="#f43f5e", lw=1.5, alpha=0.7, label="Perfect fit")
            r2v = r2_score(M["yte"], yp)
            ax.set_title(f"{name}  ·  R² = {r2v:.3f}", fontsize=11,
                         fontweight="bold", color="#eef2ff", pad=10)
            ax.set_xlabel("Actual Marks", fontsize=9.5)
            ax.set_ylabel("Predicted Marks", fontsize=9.5)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.07)
            ax.legend(fontsize=8.5, framealpha=0.1)
        fig.tight_layout(pad=2.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div style="background:#0d1120;border:1px solid #1a2540;
                    border-left:3px solid #5b7eff;
                    border-radius:0 14px 14px 0;padding:14px 20px;margin-top:16px;
                    font-size:0.82rem;color:#7b8db8;line-height:1.7;">
            💡 Points along the red dashed line = perfect predictions. Random Forest captures
            non-linear patterns better, while Linear Regression offers a fast interpretable baseline.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Feature Importance ─────────────────────────────────────────────
    with tab2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        fi_vals  = list(M["fi"].values())
        fn_short = ["Attendance","Int Test 1","Int Test 2","Assignment","Study Hrs","Internal Avg"]
        idx    = np.argsort(fi_vals)
        COLORS = ["#5b7eff","#10d48e","#fbbf24","#f43f5e","#a78bfa","#2dd4bf"]

        ch_col, ins_col = st.columns([3, 2], gap="large")
        with ch_col:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            fig.patch.set_facecolor("#05080f")
            ax.set_facecolor("#090d1a")
            bars = ax.barh([fn_short[i] for i in idx],
                           [fi_vals[i] for i in idx],
                           color=[COLORS[i % len(COLORS)] for i in idx],
                           edgecolor="#05080f", height=0.6)
            for b, v in zip(bars, [fi_vals[i] for i in idx]):
                ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                        f"{v*100:.1f}%", va="center", color="#7b8db8",
                        fontsize=9.5, fontfamily="monospace")
            ax.set_xlabel("Relative Importance", fontsize=10)
            ax.set_title("Random Forest — Feature Importance",
                         fontweight="bold", color="#eef2ff", fontsize=12, pad=12)
            ax.set_xlim(0, max(fi_vals) * 1.22)
            ax.tick_params(labelsize=9.5)
            ax.grid(True, axis="x", alpha=0.07)
            fig.tight_layout(pad=2.0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with ins_col:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            for color, feat, pct, desc in [
                ("#2dd4bf","Internal Avg", f"{fi_vals[5]*100:.1f}%",
                 "Strongest single feature — compresses both test scores into one high-signal metric."),
                ("#5b7eff","Attendance",   f"{fi_vals[0]*100:.1f}%",
                 "Second most predictive. Showing up consistently is half the battle."),
                ("#fbbf24","Study Hours",  f"{fi_vals[4]*100:.1f}%",
                 "Weak in isolation — quality of study matters far more than raw hours logged."),
            ]:
                safe_feat = feat.replace(' ','')
                st.markdown(f"""
                <style>
                .ins-v2-{safe_feat} {{
                    background:#0d1120;border:1px solid #1a2540;
                    border-left:3px solid {color};border-radius:0 16px 16px 0;
                    padding:18px 20px;margin-bottom:14px;
                    transition:transform 0.22s cubic-bezier(0.4,0,0.2,1),box-shadow 0.22s;
                    cursor:default;
                }}
                .ins-v2-{safe_feat}:hover {{
                    transform:translateX(5px);
                    box-shadow:0 8px 24px rgba(0,0,0,0.45);
                }}
                </style>
                <div class="ins-v2-{safe_feat}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span style="font-family:'Inter',sans-serif;font-weight:700;color:#eef2ff;font-size:0.88rem;">{feat}</span>
                        <span style="font-family:'DM Mono',monospace;font-weight:500;font-size:0.82rem;color:{color};">{pct}</span>
                    </div>
                    <div style="font-size:0.78rem;color:#7b8db8;line-height:1.7;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Score Distribution ─────────────────────────────────────────────
    with tab3:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        final = M["df"]["Final Exam Marks (out of 100)"]

        dist_col, stat_col = st.columns([3, 2], gap="large")
        with dist_col:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            fig.patch.set_facecolor("#05080f")
            ax.set_facecolor("#090d1a")
            ax.hist(final, bins=30, color="#5b7eff", edgecolor="#05080f", alpha=0.85, zorder=3)
            for g, (lo, hi, col, _, _) in GRADES.items():
                ax.axvspan(lo, min(hi,100), alpha=0.05, color=col, zorder=1)
                ax.text((lo+min(hi,100))/2, ax.get_ylim()[1]*0.9, g,
                        ha="center", fontsize=8.5, color=col,
                        fontfamily="monospace", fontweight="bold")
            ax.axvline(float(final.mean()),   color="#f43f5e", lw=1.5,
                       linestyle="--", label=f"Mean: {final.mean():.1f}")
            ax.axvline(float(final.median()), color="#10d48e", lw=1.5,
                       linestyle="--", label=f"Median: {final.median():.1f}")
            ax.legend(fontsize=9.5, framealpha=0.12)
            ax.set_xlabel("Final Exam Marks", fontsize=10)
            ax.set_ylabel("No. of Students", fontsize=10)
            ax.set_title("Score Distribution  (n = 2,000)",
                         fontweight="bold", color="#eef2ff", fontsize=12, pad=12)
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", alpha=0.07)
            fig.tight_layout(pad=2.0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with stat_col:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            bins_b = [0, 50, 65, 80, 90, 101]
            lbl_b  = ["F","C","B","A","A+"]
            g_cnt  = pd.cut(final, bins=bins_b, labels=lbl_b, right=False).value_counts()
            gmap   = {"A+":92,"A":82,"B":70,"C":55,"F":30}

            st.markdown("""
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                        color:#5b7eff;letter-spacing:2px;margin-bottom:14px;">
                GRADE BREAKDOWN
            </div>
            """, unsafe_allow_html=True)

            for g in ["A+","A","B","C","F"]:
                cnt = int(g_cnt.get(g, 0))
                _, gc_color, _, _ = get_grade(gmap[g])
                pct_v = cnt / len(final) * 100
                lo, hi = GRADES[g][:2]
                st.markdown(f"""
                <div style="background:#0d1120;border:1px solid #1a2540;
                            border-radius:14px;padding:14px 18px;margin-bottom:10px;
                            display:flex;align-items:center;gap:14px;
                            transition:transform 0.22s,box-shadow 0.22s;cursor:default;"
                     onmouseover="this.style.transform='translateX(4px)';
                                  this.style.boxShadow='0 6px 20px rgba(0,0,0,0.4)';"
                     onmouseout="this.style.transform='';this.style.boxShadow='';">
                    <div style="width:36px;height:36px;border-radius:10px;
                                background:{gc_color}18;border:1.5px solid {gc_color}44;
                                display:inline-flex;align-items:center;justify-content:center;
                                font-family:'Inter',sans-serif;font-weight:800;font-size:0.95rem;
                                color:{gc_color};flex-shrink:0;">{g}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                            <span style="font-size:0.75rem;color:#7b8db8;">{lo}–{min(hi,100)} marks</span>
                            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;
                                         color:#eef2ff;font-weight:500;">{cnt}</span>
                        </div>
                        <div style="background:#1a2540;border-radius:999px;height:4px;">
                            <div style="width:{pct_v:.1f}%;background:{gc_color};
                                        height:4px;border-radius:999px;"></div>
                        </div>
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                 font-weight:500;color:{gc_color};min-width:40px;
                                 text-align:right;">{pct_v:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#0d1120;border:1px solid #1a2540;border-radius:14px;
                        padding:18px 20px;margin-top:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="font-size:0.8rem;color:#7b8db8;">Mean score</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.85rem;
                                 font-weight:500;color:#eef2ff;">{final.mean():.1f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="font-size:0.8rem;color:#7b8db8;">Median score</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.85rem;
                                 font-weight:500;color:#eef2ff;">{final.median():.1f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.8rem;color:#7b8db8;">Std deviation</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.85rem;
                                 font-weight:500;color:#eef2ff;">{final.std():.1f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: Correlations ───────────────────────────────────────────────────
    with tab4:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        num_cols = [
            "Attendance (%)", "Internal Test 1 (out of 40)",
            "Internal Test 2 (out of 40)", "Assignment Score (out of 10)",
            "Daily Study Hours", "Final Exam Marks (out of 100)",
        ]
        short = ["Attendance","Int Test 1","Int Test 2","Assignment","Study Hrs","Final Marks"]
        corr  = M["df"][num_cols].corr()
        corr.columns = short; corr.index = short

        ch2_col, ins2_col = st.columns([3, 2], gap="large")
        with ch2_col:
            fig, ax = plt.subplots(figsize=(7.5, 6))
            fig.patch.set_facecolor("#05080f")
            ax.set_facecolor("#090d1a")
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                        center=0, ax=ax, square=True, linewidths=0.4,
                        linecolor="#05080f", cbar_kws={"shrink":0.6},
                        annot_kws={"size":9.5,"color":"#eef2ff","weight":"600"})
            ax.set_title("Feature Correlation Matrix",
                         fontweight="bold", color="#eef2ff", fontsize=12, pad=12)
            ax.tick_params(labelsize=9.5, colors="#7b8db8")
            fig.tight_layout(pad=2.0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with ins2_col:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            for color, pair, strength, desc in [
                ("#5b7eff","Internal Tests ↔ Final","~0.7–0.8",
                 "Highest correlation in the dataset. Internal test performance is the strongest direct predictor of final marks."),
                ("#10d48e","Attendance ↔ Final","~0.4–0.5",
                 "Moderate positive correlation. Students who attend more tend to score significantly higher."),
                ("#fbbf24","Study Hours ↔ Final","~0.1–0.2",
                 "Weak correlation. Time alone doesn't predict performance — how you study matters more."),
            ]:
                safe_pair = pair.replace(' ','').replace('↔','').replace('~','')
                st.markdown(f"""
                <style>
                .corr-v2-{safe_pair} {{
                    background:#0d1120;border:1px solid #1a2540;
                    border-left:3px solid {color};border-radius:0 16px 16px 0;
                    padding:18px 20px;margin-bottom:16px;
                    transition:transform 0.22s cubic-bezier(0.4,0,0.2,1),box-shadow 0.22s;
                    cursor:default;
                }}
                .corr-v2-{safe_pair}:hover {{
                    transform:translateX(4px);
                    box-shadow:0 8px 24px rgba(0,0,0,0.45);
                }}
                </style>
                <div class="corr-v2-{safe_pair}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span style="font-family:'Inter',sans-serif;font-weight:700;color:#eef2ff;font-size:0.88rem;">{pair}</span>
                        <span style="font-family:'DM Mono',monospace;font-weight:500;font-size:0.8rem;color:{color};">
                            {strength}
                        </span>
                    </div>
                    <div style="font-size:0.78rem;color:#7b8db8;line-height:1.7;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <hr>
    <div style="display:flex;justify-content:space-between;align-items:center;
                flex-wrap:wrap;gap:8px;padding-bottom:8px;">
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4e70;">
            sklearn · RandomForestRegressor · LinearRegression
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4e70;">
            100 trees · max_depth=10 · 80/20 split · seed=42
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#5b7eff;">
            EduSense v2.0 ✦ Built with Streamlit
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "result" and not st.session_state.predicted:
    st.markdown("""
    <div style="text-align:center; padding:90px 20px 40px 20px;">
        <div style="width:72px;height:72px;border-radius:20px;
                    background:linear-gradient(135deg,rgba(91,126,255,0.12),rgba(124,58,237,0.08));
                    border:1.5px solid rgba(91,126,255,0.2);
                    display:inline-flex;align-items:center;justify-content:center;
                    font-size:2rem;margin-bottom:20px;">🎓</div>
        <h3 style="font-family:'Inter',sans-serif;color:#eef2ff;font-weight:800;
                   margin-bottom:10px;letter-spacing:-0.5px;">No prediction yet</h3>
        <p style="color:#7b8db8;font-size:0.92rem;max-width:320px;margin:0 auto;">
            Head to Home, fill in your academic stats, and hit Predict.
        </p>
    </div>
    """, unsafe_allow_html=True)
    _, fb_col, _ = st.columns([3, 2, 3])
    with fb_col:
        if st.button("← Go to Home", use_container_width=True, type="primary"):
            st.session_state.page = "home"; st.rerun()