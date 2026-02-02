# app.py — CC/CSP Recommender (GEX-style tiles, refined layout + alignment fix)
# Includes:
# - Ticker input + Spot tile aligned on the same baseline (caption spacer fix)
# - Strategy + Risk under ticker, above expiration
# - Recommended Match tile (green highlight)
# - Metrics split into 2 rows of 3 (Premium/Total/Return then Delta/Keep/Annualized)
# - Narrow app width (doesn't use entire screen)
# - Compare Strikes (Top 5)

import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

def top_nav(active: str = "cc"):
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] { display: none; }
          div[data-testid="stAppViewContainer"] { margin-left: 0; }
          .navbtn div[data-testid="stButton"] > button {
            height: 28px !important;
            padding: 0 10px !important;
            font-size: 0.78rem !important;
            font-weight: 650 !important;
            border-radius: 999px !important;
            min-width: unset !important;
            width: auto !important;
          }
          .navbtn.active div[data-testid="stButton"] > button {
            border: 1.5px solid rgba(80,170,255,0.95) !important;
            background: rgba(80,170,255,0.18) !important;
            box-shadow: inset 0 -2px 0 rgba(80,170,255,0.95) !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, _ = st.columns([0.15, 0.45, 0.25, 1])

    with c1:
        st.markdown(f'<div class="navbtn {"active" if active=="gex" else ""}">', unsafe_allow_html=True)
        if st.button("GEX", key="nav_gex"):
            st.switch_page("app.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="navbtn {"active" if active=="lev" else ""}">', unsafe_allow_html=True)
        if st.button("Leverage Equivalence", key="nav_lev"):
            st.switch_page("pages/1_Leverage_Equivalence.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="navbtn {"active" if active=="cc" else ""}">', unsafe_allow_html=True)
        if st.button("CC / CSP", key="nav_cc"):
            st.switch_page("pages/2_CC_CSP.py")
        st.markdown("</div>", unsafe_allow_html=True)

st.set_page_config(page_title="CC / CSP Recommender", layout="wide")
top_nav(active="cc")
st.title("Covered Call / CSP Identifier")

# ... then keep the rest of your CC/CSP code unchanged ...


# =========================
# Page config + CSS
# =========================
st.set_page_config(page_title="CC / CSP Recommender", layout="wide")

st.markdown(
    """
    <style>
      /* Narrow the whole app content so it doesn't span full screen */
      .block-container{
        max-width: 980px;
        padding-top: 1.0rem;
      }

      /* Compact ticker input */
      div[data-testid="stTextInput"] input {
        width: 145px !important;
        min-width: 145px !important;
        max-width: 145px !important;
        text-align: left;
        font-weight: 700;
      }

      /* Slightly tighter dataframe header */
      div[data-testid="stDataFrame"] thead tr th { font-weight: 800 !important; }

      /* Make radios a bit tighter */
      div[role="radiogroup"] label { margin-right: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Covered Call / CSP Identifier")

if not YF_OK:
    st.error("Missing dependency: yfinance. Install with: pip install yfinance")
    st.stop()


# =========================
# GEX-style tile system
# =========================
def snap_cell(
    label: str,
    value: str,
    label_px: int = 9,
    value_px: int = 14,
    wrap_value: bool = False,
    border_rgba: str = "rgba(255,255,255,0.06)",
    bg_rgba: str = "rgba(255,255,255,0.03)",
    value_color: str = "#e6e6e6",
    value_weight: int = 700,
):
    white_space = "normal" if wrap_value else "nowrap"
    return f"""
    <div style="
      padding:6px 6px 5px 10px;
      border-radius:10px;
      background:{bg_rgba};
      border:1px solid {border_rgba};
      min-width:0;
    ">
      <div style="
        font-size:{label_px}px;
        opacity:0.78;
        line-height:1.1;
        margin-bottom:3px;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        color:#cfcfcf;
      ">{label}</div>

      <div style="
        font-size:{value_px}px;
        font-weight:{value_weight};
        line-height:1.15;
        white-space:{white_space};
        overflow:hidden;
        text-overflow:ellipsis;
        word-break:break-word;
        color:{value_color};
      ">{value}</div>
    </div>
    """


def tile_grid(items, cols: int = 3):
    html = f"""
    <div style="
      display:grid;
      grid-template-columns: repeat({cols}, minmax(0, 1fr));
      gap:10px 14px;
      margin-top:6px;
      margin-bottom:8px;
    ">
    """
    for it in items:
        html += snap_cell(
            it["label"],
            it["value"],
            label_px=it.get("label_px", 9),
            value_px=it.get("value_px", 14),
            wrap_value=it.get("wrap", False),
            border_rgba=it.get("border", "rgba(255,255,255,0.06)"),
            bg_rgba=it.get("bg", "rgba(255,255,255,0.03)"),
            value_color=it.get("value_color", "#e6e6e6"),
            value_weight=it.get("value_weight", 700),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# =========================
# Value-only tile for spot
# =========================
def snap_value_only(
    value: str,
    value_px: int = 14,
    value_weight: int = 800,
    border_rgba: str = "rgba(255,255,255,0.06)",
    bg_rgba: str = "rgba(255,255,255,0.03)",
    value_color: str = "#e6e6e6",
):
    return f"""
    <div style="
      padding:10px 10px 13px 12px;
      border-radius:10px;
      background:{bg_rgba};
      border:1px solid {border_rgba};
      min-width:0;
    ">
      <div style="
        font-size:{value_px}px;
        font-weight:{value_weight};
        line-height:1.15;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        color:{value_color};
      ">{value}</div>
    </div>
    """
    
# =========================
# Helpers
# =========================
def money2(x):
    return "—" if not np.isfinite(x) else f"${x:,.2f}"


def money0(x):
    return "—" if not np.isfinite(x) else f"${x:,.0f}"


def pct1(x):
    return "—" if not np.isfinite(x) else f"{x * 100:.1f}%"


def pct0(x):
    return "—" if not np.isfinite(x) else f"{x * 100:.0f}%"


def strike_str(k):
    if not np.isfinite(k):
        return "—"
    s = f"{k:,.2f}".rstrip("0").rstrip(".")
    return f"${s}"


def fmt_exp_pretty(exp_yyyy_mm_dd: str) -> str:
    try:
        d = dt.datetime.strptime(exp_yyyy_mm_dd, "%Y-%m-%d").date()
        return d.strftime("%A, %b %d").replace(" 0", " ")
    except Exception:
        return exp_yyyy_mm_dd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S, K, T, r, sigma, option_type):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    except Exception:
        return np.nan
    return _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0


def annualized_yield(premium_dollars, denom_dollars, dte):
    if denom_dollars <= 0 or dte <= 0:
        return np.nan
    return (premium_dollars / denom_dollars) * (365.0 / dte)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def liquidity_penalty(bid, ask):
    bid, ask = safe_float(bid), safe_float(ask)
    if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0:
        return 0.75
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.70
    spread = (ask - bid) / mid
    return float(np.clip(1.0 - 2.5 * spread, 0.20, 1.00))


def oi_bonus(oi):
    oi = safe_float(oi)
    if not np.isfinite(oi) or oi <= 0:
        return 0.95
    return float(np.clip(0.90 + 0.06 * math.log10(oi + 1.0), 0.90, 1.20))


def final_score(ann_yld, bid, ask, oi):
    if not np.isfinite(ann_yld):
        return np.nan
    return ann_yld * liquidity_penalty(bid, ask) * oi_bonus(oi)


# =========================
# Risk buckets (ABS delta)
# =========================
@dataclass
# delta ranges interpreted as ABS(delta)
class RiskBucket:
    dmin: float
    dmax: float


RISK_BUCKETS = {
    "Aggressive": RiskBucket(0.30, 0.50),
    "Neutral": RiskBucket(0.15, 0.30),
    "Risk averse": RiskBucket(0.05, 0.15),
}


# =========================
# Data fetch (cache-safe)
# =========================
@st.cache_data(ttl=120)
def fetch_spot_and_exps(ticker: str):
    t = yf.Ticker(ticker)

    spot = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            spot = fi.get("last_price") or fi.get("lastPrice")
            spot = float(spot) if spot is not None else None
    except Exception:
        pass

    if spot is None:
        try:
            hist = t.history(period="5d")
            if hist is not None and not hist.empty:
                spot = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    exps = []
    try:
        exps = list(t.options)
    except Exception:
        exps = []

    return spot, exps


@st.cache_data(ttl=120)
def fetch_chain(ticker: str, exp: str):
    oc = yf.Ticker(ticker).option_chain(exp)
    return oc.calls.copy(), oc.puts.copy()


# =========================
# Controls (alignment fixed)
# =========================
# Row 1: Ticker input + Spot tile aligned by adding a caption spacer in Spot column
row1 = st.columns([0.22, 0.28, 0.50], vertical_alignment="top")

LABEL_STYLE = "font-size:12px;opacity:0.78;margin:0 0 6px 0;color:#cfcfcf;"

with row1[0]:
    st.markdown(f"<div style='{LABEL_STYLE}'>Ticker</div>", unsafe_allow_html=True)
    ticker = st.text_input(
        label="ticker",
        value="",
        placeholder="Input a ticker",
        max_chars=5,
        label_visibility="collapsed",
    ).strip().upper()

with row1[1]:
    st.markdown(f"<div style='{LABEL_STYLE}'>Spot</div>", unsafe_allow_html=True)
    spot_holder = st.empty()

with row1[2]:
    st.empty()

# Row 2/3: Strategy + Risk UNDER ticker, ABOVE expiration
mode = st.radio("Strategy", ["Covered Call", "Cash-Secured Put"], horizontal=True)
risk = st.radio("Risk tolerance", ["Aggressive", "Neutral", "Risk averse"], horizontal=True, index=1)
bucket = RISK_BUCKETS[risk]

if not ticker:
    spot_holder.markdown(snap_cell("Spot", "—", value_px=14, value_weight=800), unsafe_allow_html=True)
    st.info("Enter a ticker to load expirations and compare strikes.")
    st.stop()

with st.spinner("Loading spot + expirations..."):
    spot, expirations = fetch_spot_and_exps(ticker)

if spot is None or not np.isfinite(spot) or not expirations:
    spot_holder.markdown(snap_cell("Spot", "—", value_px=14, value_weight=800), unsafe_allow_html=True)
    st.error("Invalid ticker or no options available.")
    st.stop()

spot_holder.markdown(
    snap_value_only(money2(float(spot)), value_px=14, value_weight=800),
    unsafe_allow_html=True,
)

# Expiration below Strategy/Risk
exp = st.selectbox("Expiration date", expirations, index=0)

today = dt.date.today()
dte = max((dt.datetime.strptime(exp, "%Y-%m-%d").date() - today).days, 0)
T = dte / 365.0
r = 0.05

if dte <= 0:
    st.warning("Select a future expiration.")
    st.stop()

# =========================
# Fetch option chain
# =========================
with st.spinner("Fetching option chain..."):
    calls, puts = fetch_chain(ticker, exp)

opt_type = "call" if mode == "Covered Call" else "put"
df = calls.copy() if opt_type == "call" else puts.copy()

for c in ["strike", "bid", "ask", "lastPrice", "openInterest", "impliedVolatility"]:
    if c not in df.columns:
        df[c] = np.nan

df["K"] = pd.to_numeric(df["strike"], errors="coerce")
df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce")
df["oi"] = pd.to_numeric(df["openInterest"], errors="coerce")
df["iv"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")

mid = (df["bid"] + df["ask"]) / 2.0
df["price_used"] = np.where(np.isfinite(mid) & (mid > 0), mid, df["lastPrice"])
df["premium_$"] = df["price_used"] * 100.0

df["delta"] = [
    bs_delta(float(spot), k, T, r, iv, opt_type) if np.isfinite(k) and np.isfinite(iv) else np.nan
    for k, iv in zip(df["K"], df["iv"])
]
df["abs_delta"] = df["delta"].abs()

# OTM filter + denom + keep proxy
if opt_type == "call":
    df = df[df["K"] >= float(spot)].copy()
    df["denom_$"] = float(spot) * 100.0
    df["keep_%"] = 1.0 - df["delta"]
else:
    df = df[df["K"] <= float(spot)].copy()
    df["denom_$"] = df["K"] * 100.0
    df["keep_%"] = 1.0 - df["abs_delta"]

df["return_%"] = df["premium_$"] / df["denom_$"]
df["ann_%"] = df.apply(lambda row: annualized_yield(row["premium_$"], row["denom_$"], dte), axis=1)
df["score"] = df.apply(lambda row: final_score(row["ann_%"], row["bid"], row["ask"], row["oi"]), axis=1)

df = df[
    np.isfinite(df["abs_delta"]) &
    (df["abs_delta"] >= bucket.dmin) &
    (df["abs_delta"] <= bucket.dmax) &
    np.isfinite(df["price_used"]) & (df["price_used"] > 0) &
    np.isfinite(df["ann_%"]) & (df["ann_%"] > 0)
].copy()

if df.empty:
    st.warning("No strikes match your delta bucket + OTM filter. Try another expiration or risk tolerance.")
    st.stop()

df = df.sort_values("score", ascending=False).reset_index(drop=True)
best = df.iloc[0]
top = df.head(5).copy()

# =========================
# Recommended Match tile
# =========================
strike = float(best["K"])
strike_label = strike_str(strike)
pretty_exp = fmt_exp_pretty(exp)

recommended_value = (
    f"{strike_label} Strike<br/>"
    f"<span style='opacity:0.85;'>Expiration {pretty_exp} ({dte} DTE)</span>"
)

tile_grid(
    items=[
        {
            "label": "✅ Recommended Match",
            "value": recommended_value,
            "label_px": 10,
            "value_px": 18,
            "wrap": True,
            "border": "rgba(25, 211, 162, 0.55)",
            "bg": "rgba(0, 70, 60, 0.14)",
            "value_weight": 800,
        }
    ],
    cols=1,
)

# =========================
# Metrics in 2 rows of 3
# =========================
premium_per_share = float(best["price_used"])
total_premium = float(best["premium_$"])  # 1 contract (100 shares)
ret = float(best["return_%"])
ann = float(best["ann_%"])
delta = float(best["delta"])
keep_p = float(best["keep_%"])

keep_color = "#19d3a2" if keep_p >= 0.75 else ("#ff5c5c" if keep_p < 0.60 else "#ffcc66")

tile_grid(
    items=[
        {"label": "Premium", "value": f"<span style='color:#19d3a2;font-weight:800;'>{money2(premium_per_share)}</span>", "value_px": 16, "wrap": True},
        {"label": "Total Premium", "value": f"<span style='color:#19d3a2;font-weight:800;'>{money0(total_premium)}</span>", "value_px": 16, "wrap": True},
        {"label": "Return", "value": f"<span style='color:#19d3a2;font-weight:800;'>{pct1(ret)}</span>", "value_px": 16, "wrap": True},
    ],
    cols=3,
)

tile_grid(
    items=[
        {"label": "Delta", "value": f"{delta:.2f}", "value_px": 16},
        {"label": "Keep Probability", "value": f"<span style='color:{keep_color};font-weight:800;'>~{pct0(keep_p)}</span>", "value_px": 16, "wrap": True},
        {"label": "Annualized Return", "value": f"<span style='color:#19d3a2;font-weight:800;'>{pct1(ann)}</span>", "value_px": 16, "wrap": True},
    ],
    cols=3,
)

st.divider()

# =========================
# Compare Strikes (Top 5)
# =========================
st.subheader("Compare Strikes")

def strike_label_row(k: float, best_k: float) -> str:
    if not np.isfinite(k):
        return "—"
    s = f"${k:,.2f}".rstrip("0").rstrip(".")
    return f"{s} ✓" if abs(k - best_k) < 1e-12 else s

compare = pd.DataFrame({
    "Strike": [strike_label_row(k, float(best["K"])) for k in top["K"].values],
    "Total Premium": [money0(x) for x in top["premium_$"].values],
    "Return": [pct1(x) for x in top["return_%"].values],
    "Ann.": [pct1(x) for x in top["ann_%"].values],
    "Delta": [("" if not np.isfinite(x) else f"{x:.2f}") for x in top["delta"].values],
    "IV": [pct0(x) for x in top["iv"].values],
    "Keep %": [pct0(x) for x in top["keep_%"].values],
})

st.dataframe(compare, use_container_width=True, hide_index=True)

st.caption(
    "Keep Probability/Keep % is a quick approximation using delta (calls: 1−Δ, puts: 1−|Δ|). "
    "Deltas are estimated via Black–Scholes using Yahoo IV."
)









