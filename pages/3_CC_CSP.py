# app.py — CC/CSP Recommender (GEX-style tiles, refined layout + alignment fix)
# Includes:
# - Ticker input + Spot tile aligned on the same baseline (caption spacer fix)
# - Strategy + Risk under ticker, above expiration
# - Recommended Match tile (green highlight)
# - Metrics split into 2 rows of 3 (Premium/Total/Return then Delta/Keep/Annualized)
# - Narrow app width (doesn't use entire screen)

import math
from dataclasses import dataclass
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# Page config + CSS
# =========================
st.set_page_config(page_title="CC/CSP Identifier", page_icon="🧩", layout="centered")

CSS = """
<style>
/* Narrow content wrapper */
.block-container { max-width: 920px; padding-top: 1.25rem; }

/* Header */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Tile grid */
.tile-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 880px){
  .tile-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px){
  .tile-grid { grid-template-columns: 1fr; }
}
.tile {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 1px 10px rgba(0,0,0,0.12);
}
.tile .label {
  font-size: 0.83rem;
  opacity: 0.75;
  margin-bottom: 6px;
}
.tile .value {
  font-size: 1.25rem;
  font-weight: 650;
  line-height: 1.2;
}
.tile .value_small {
  font-size: 1.05rem;
  font-weight: 650;
  line-height: 1.25;
}
.tile.reco {
  border-color: rgba(54, 235, 136, 0.55);
  box-shadow: 0 0 0 2px rgba(54, 235, 136, 0.14) inset, 0 1px 12px rgba(0,0,0,0.16);
}

/* Compact radio buttons */
div[role="radiogroup"] label { padding: 0.12rem 0.45rem; }

/* Helper for aligning the Spot tile with ticker input caption */
.caption-spacer { height: 1.4rem; } /* matches st.text_input label/caption space */
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
@dataclass
class RiskBucket:
    dmin: float
    dmax: float

RISK_BUCKETS = {
    "Aggressive": RiskBucket(0.25, 0.40),
    "Neutral": RiskBucket(0.15, 0.25),
    "Risk averse": RiskBucket(0.08, 0.15),
}

def fmt_money(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.2f}"

def fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.{decimals}f}%"

def fmt_num(x: float, decimals: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{decimals}f}"

def fmt_int(x) -> str:
    try:
        x = int(x)
        return f"{x:,}"
    except Exception:
        return "—"

def strike_str(k: float) -> str:
    if not np.isfinite(k):
        return "—"
    if abs(k - round(k)) < 1e-8:
        return f"{int(round(k))}"
    return f"{k:.2f}"

def years_to_expiry(exp_date: date) -> float:
    today = date.today()
    days = (exp_date - today).days
    return max(days, 0) / 365.0

def dte(exp_date: date) -> int:
    today = date.today()
    return max((exp_date - today).days, 0)

def fmt_exp_pretty(exp: str) -> str:
    # yfinance returns yyyy-mm-dd
    try:
        dt = datetime.strptime(exp, "%Y-%m-%d")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return exp

def tile_grid(items):
    html = '<div class="tile-grid">'
    for it in items:
        extra_cls = "reco" if it.get("reco") else ""
        value_class = "value_small" if it.get("small") else "value"
        html += f"""
          <div class="tile {extra_cls}">
            <div class="label">{it['label']}</div>
            <div class="{value_class}">{it['value']}</div>
          </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    """
    Black-Scholes delta (no dividends).
    opt_type: "call" or "put"
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    except Exception:
        return np.nan

    if opt_type == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0

def annualized_yield(premium_dollars: float, denom_dollars: float, dte_days: int) -> float:
    if not np.isfinite(premium_dollars) or not np.isfinite(denom_dollars) or denom_dollars <= 0 or dte_days <= 0:
        return np.nan
    period = premium_dollars / denom_dollars
    return (1.0 + period) ** (365.0 / dte_days) - 1.0

def final_score(ann_yield: float, bid: float, ask: float, oi: float) -> float:
    """
    A simple ranking score:
    - Higher annualized yield is better
    - Tighter spreads are better
    - Higher open interest is better (log-scaled)
    """
    if not np.isfinite(ann_yield):
        return -np.inf

    spread = np.nan
    if np.isfinite(bid) and np.isfinite(ask) and ask > 0:
        spread = (ask - bid) / ask

    spread_pen = 0.0
    if np.isfinite(spread):
        spread_pen = min(max(spread, 0.0), 1.0)  # 0..1

    oi_boost = 0.0
    if np.isfinite(oi) and oi > 0:
        oi_boost = math.log10(oi + 10.0) / 5.0  # ~0..1 range

    return (ann_yield * 100.0) + (oi_boost * 2.0) - (spread_pen * 3.0)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_spot(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker)
        # fast_info usually best for spot
        fi = getattr(t, "fast_info", None)
        if fi and "lastPrice" in fi and fi["lastPrice"] is not None:
            return float(fi["lastPrice"])
        # fallback to recent close
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_expirations(ticker: str):
    try:
        t = yf.Ticker(ticker)
        return list(getattr(t, "options", []) or [])
    except Exception:
        return []

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_chain(ticker: str, exp: str):
    t = yf.Ticker(ticker)
    chain = t.option_chain(exp)
    return chain.calls.copy(), chain.puts.copy()


# =========================
# UI — Header
# =========================
st.title("🧩 Covered Call / CSP Identifier")
st.caption("Find a strike that matches your risk bucket and ranks by annualized yield, liquidity, and spreads.")


# =========================
# UI — Input Row (Ticker
