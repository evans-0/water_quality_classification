import streamlit as st
import pickle
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaSafe — Water Quality Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1e2e 100%);
    color: #e2eaf4;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem; max-width: 1100px; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid rgba(99,179,237,0.15);
    margin-bottom: 2rem;
}
.hero-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.hero h1 {
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #e8f4fd;
    margin: 0 0 0.4rem;
}
.hero p {
    color: #7aadcc;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

/* Section labels */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a9bc4;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(74,155,196,0.2);
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #5a9ab8;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #c8e6f7;
}
.metric-card .unit {
    font-size: 0.75rem;
    color: #4a7a94;
    margin-left: 4px;
}

/* Result banner */
.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.35);
    border-left: 4px solid #10b981;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-unsafe {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.08));
    border: 1px solid rgba(239,68,68,0.35);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.result-sub {
    font-size: 0.9rem;
    opacity: 0.75;
    font-weight: 300;
}

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 8px;
    margin-top: 1rem;
    overflow: hidden;
}
.conf-bar-fill-safe {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.6s ease;
}
.conf-bar-fill-unsafe {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 0.6s ease;
}
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #5a9ab8;
    margin-top: 0.4rem;
}

/* WHO thresholds info box */
.info-box {
    background: rgba(99,179,237,0.06);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #7aadcc;
    margin-top: 1rem;
    line-height: 1.7;
}
.info-box strong { color: #a8d4ec; }

/* Slider tweaks */
.stSlider > div > div > div > div {
    background: #1e4a6e !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1a6fa8, #1e90cc);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    margin-top: 0.5rem;
}
.stButton > button:hover { opacity: 0.88; }

/* Divider */
.hdivider {
    border: none;
    border-top: 1px solid rgba(99,179,237,0.1);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = pickle.load(open("rf_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl",   "rb"))
    return model, scaler

model, scaler = load_artifacts()

# ── Feature config ─────────────────────────────────────────────────────────────
feature_config = {
    "ph":              {"min": 0.0,    "max": 14.0,    "default": 7.0,     "step": 0.1,   "unit": "pH",      "who": "6.5 – 8.5"},
    "Hardness":        {"min": 47.0,   "max": 323.0,   "default": 196.0,   "step": 1.0,   "unit": "mg/L",    "who": "< 300"},
    "Solids":          {"min": 320.0,  "max": 61227.0, "default": 20927.0, "step": 100.0, "unit": "ppm",     "who": "< 500"},
    "Chloramines":     {"min": 0.35,   "max": 13.13,   "default": 7.12,    "step": 0.1,   "unit": "ppm",     "who": "< 4"},
    "Sulfate":         {"min": 129.0,  "max": 481.0,   "default": 333.0,   "step": 1.0,   "unit": "mg/L",    "who": "< 250"},
    "Conductivity":    {"min": 181.0,  "max": 753.0,   "default": 426.0,   "step": 1.0,   "unit": "μS/cm",   "who": "< 400"},
    "Organic_carbon":  {"min": 2.2,    "max": 28.3,    "default": 14.28,   "step": 0.1,   "unit": "ppm",     "who": "< 2"},
    "Trihalomethanes": {"min": 0.738,  "max": 124.0,   "default": 66.4,    "step": 0.1,   "unit": "μg/L",    "who": "< 80"},
    "Turbidity":       {"min": 1.45,   "max": 6.74,    "default": 3.97,    "step": 0.01,  "unit": "NTU",     "who": "< 5"},
}

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">💧</div>
    <h1>AquaSafe</h1>
    <p>Water Potability Prediction · Random Forest Classifier · WHO Guidelines</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: two columns ────────────────────────────────────────────────────────
col_inputs, col_result = st.columns([1.1, 0.9], gap="large")

with col_inputs:
    st.markdown('<div class="section-label">Physicochemical Parameters</div>', unsafe_allow_html=True)

    inputs = []
    cols_a, cols_b = st.columns(2)

    feature_items = list(feature_config.items())
    left_features  = feature_items[:5]
    right_features = feature_items[5:]

    with cols_a:
        for feature, cfg in left_features:
            val = st.slider(
                f"{feature}  ({cfg['unit']})",
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                step=cfg["step"],
                key=feature
            )
            inputs.append((feature, val, cfg))

    with cols_b:
        for feature, cfg in right_features:
            val = st.slider(
                f"{feature}  ({cfg['unit']})",
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                step=cfg["step"],
                key=feature
            )
            inputs.append((feature, val, cfg))

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
    predict_clicked = st.button("Analyse Water Sample", use_container_width=True)

with col_result:
    st.markdown('<div class="section-label">Current Values</div>', unsafe_allow_html=True)

    # Show live metric cards (3 key ones)
    key_metrics = ["ph", "Turbidity", "Chloramines"]
    metric_html = '<div class="metric-row">'
    for feat, val, cfg in inputs:
        if feat in key_metrics:
            metric_html += f"""
            <div class="metric-card">
                <div class="label">{feat}</div>
                <div class="value">{val:.2f}<span class="unit">{cfg['unit']}</span></div>
            </div>"""
    metric_html += "</div>"
    st.markdown(metric_html, unsafe_allow_html=True)

    # WHO reference info
    st.markdown('<div class="section-label" style="margin-top:1rem">WHO Safe Thresholds</div>', unsafe_allow_html=True)
    who_html = '<div class="info-box">'
    for feat, val, cfg in inputs:
        flag = ""
        who_val = cfg["who"]
        who_html += f"<strong>{feat}</strong>: {val:.2f} {cfg['unit']} &nbsp;·&nbsp; WHO: {who_val}<br>"
    who_html += "</div>"
    st.markdown(who_html, unsafe_allow_html=True)

    # ── Prediction result ──────────────────────────────────────────────────────
    if predict_clicked:
        feature_values = [val for _, val, _ in inputs]
        X_input = scaler.transform([feature_values])
        pred    = model.predict(X_input)[0]
        prob    = model.predict_proba(X_input)[0][1]

        if pred == 1:
            conf_pct = int(prob * 100)
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-title">✅ Potable — Safe to Drink</div>
                <div class="result-sub">Model confidence: {prob:.1%} · Classification threshold: 0.50</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill-safe" style="width:{conf_pct}%"></div>
                </div>
                <div class="conf-label"><span>0%</span><span>{conf_pct}% confidence</span><span>100%</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            conf_pct = int((1 - prob) * 100)
            st.markdown(f"""
            <div class="result-unsafe">
                <div class="result-title">❌ Not Potable — Unsafe</div>
                <div class="result-sub">Model confidence: {(1-prob):.1%} · Recommend lab verification</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill-unsafe" style="width:{conf_pct}%"></div>
                </div>
                <div class="conf-label"><span>0%</span><span>{conf_pct}% confidence</span><span>100%</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Breakdown of features outside WHO range
        flagged = []
        for feat, val, cfg in inputs:
            who = cfg["who"]
            # Simple flag logic for known thresholds
            thresholds = {
                "ph": (6.5, 8.5), "Turbidity": (None, 5.0),
                "Chloramines": (None, 4.0), "Sulfate": (None, 250.0),
                "Conductivity": (None, 400.0), "Organic_carbon": (None, 2.0),
                "Trihalomethanes": (None, 80.0), "Solids": (None, 500.0),
            }
            if feat in thresholds:
                lo, hi = thresholds[feat]
                if (lo and val < lo) or (hi and val > hi):
                    flagged.append(f"⚠️ **{feat}** = {val:.2f} {cfg['unit']} (WHO: {who})")

        if flagged:
            st.markdown('<div class="section-label" style="margin-top:1.5rem">Parameters Outside WHO Range</div>', unsafe_allow_html=True)
            for f in flagged:
                st.markdown(f)

    else:
        st.markdown("""
        <div style="
            margin-top: 2rem;
            padding: 2rem;
            border: 1px dashed rgba(99,179,237,0.2);
            border-radius: 12px;
            text-align: center;
            color: #3d7a99;
            font-size: 0.9rem;
            line-height: 1.8;
        ">
            Adjust the parameters on the left<br>and click <strong style="color:#5a9ab8">Analyse Water Sample</strong><br>to get the prediction.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="hdivider" style="margin-top:3rem">
<div style="text-align:center; color:#2d5f7a; font-size:0.78rem; padding-bottom:1rem">
    AquaSafe · Random Forest · Trained on Water Potability Dataset (Kaggle) ·
    For environmental monitoring use only — not a substitute for certified lab testing
</div>
""", unsafe_allow_html=True)