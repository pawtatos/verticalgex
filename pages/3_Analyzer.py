import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from plotly.subplots import make_subplots

# Optional: enables hover readout + a simple price ruler line
try:
    from streamlit_plotly_events import plotly_events  # pip install streamlit-plotly-events
except Exception:
    plotly_events = None

# =========================
# Indicator Settings
# =========================
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_LEN = 14
STOCH_RSI_LEN = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3
STOCH_K_SMOOTH = STOCH_RSI_SMOOTH_K
STOCH_D_SMOOTH = STOCH_RSI_SMOOTH_D

YFINANCE_PERIOD = "2y"

# =========================
# Chart Colors
# =========================
BULL_LINE = "#00C853"
BEAR_LINE = "#FF3D00"
BULL_FILL = "rgba(0, 200, 83, 0.35)"
BEAR_FILL = "rgba(255, 61, 0, 0.35)"


# Default number of bars to show initially
DEFAULT_BARS = 220
st.set_page_config(
    page_title="Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- Mobile helpers ---
def _detect_mobile() -> bool:
    """Best-effort mobile heuristic.
    - Uses Streamlit screen width when available.
    - Falls back to `?mobile=true` query param (supports both new and old Streamlit APIs).
    """
    # 1) Screen width (only exists in some builds / custom frontends)
    w = st.session_state.get("_st_screen_width", None)
    if isinstance(w, (int, float)):
        return w <= 768

    # 2) Query params (Streamlit 1.30+: st.query_params, older: st.experimental_get_query_params)
    try:
        qp = st.query_params
        mobile_flag = qp.get("mobile", "")
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            mobile_flag = (qp.get("mobile", [""])[0] if isinstance(qp.get("mobile", ""), list) else qp.get("mobile", ""))
        except Exception:
            mobile_flag = ""

    if str(mobile_flag).lower() in ("1", "true", "yes"):
        return True
    return False

MOBILE = _detect_mobile()
st.session_state["MOBILE"] = MOBILE

st.markdown(
    """
<style>
@media (max-width: 768px) {
  .block-container { padding-top: 0.55rem !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
  h1 { font-size: 1.25rem !important; line-height: 1.15 !important; }
  h2, h3 { font-size: 1.02rem !important; }
  div[data-testid="stPlotlyChart"] { margin-top: -4px; }
  .js-plotly-plot .modebar { display: none !important; }
  .js-plotly-plot .plotly .modebar { display: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

if not MOBILE:
    st.markdown("""
    <style>
    /* move plotly toolbar down (desktop only) */
    .js-plotly-plot .modebar {
        top: 30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # On mobile we keep the modebar hidden to avoid overlaps
    pass


st.markdown("""
<style>
  [data-testid="stSidebar"] { display: none; }
  [data-testid="stSidebarNav"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

def top_nav(active: str = "analyzer"):
    # Responsive top nav (2x2 on mobile, 1x4 on desktop)
    st.markdown(
        """
        <style>
          [data-testid="stSidebarNav"] { display: none; }
          .navwrap { padding: 6px 0 10px 0; }
          .navwrap div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
          .navbtn button {
            border-radius: 12px !important;
            font-weight: 850 !important;
            height: 42px !important;
            padding: 0 10px !important;
            white-space: nowrap !important;
          }
          .navbtn.active button {
            border: 2px solid rgba(80,170,255,0.95) !important;
            background: rgba(80,170,255,0.20) !important;
            box-shadow: inset 0 -4px 0 rgba(80,170,255,0.95) !important;
          }
          @media (max-width: 768px) {
            .navbtn button {
              height: 34px !important;
              font-size: 11px !important;
              padding: 0 6px !important;
              white-space: normal !important;
              line-height: 1.05 !important;
            }
          }
        </style>
        """
        , unsafe_allow_html=True
    )

    def _nav_button(label: str, page: str, key: str):
        st.markdown('<div class="navbtn %s">' % ("active" if active==key else ""), unsafe_allow_html=True)
        if st.button(label, use_container_width=True):
            st.switch_page(page)
        st.markdown("</div>", unsafe_allow_html=True)

    if MOBILE:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            _nav_button("GEX", "app.py", "gex")
        with r1c2:
            _nav_button("Leverage", "pages/1_Leverage_Equivalence.py", "lev")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            _nav_button("Put DCA", "pages/2_Synthetic_Put_DCA.py", "dca")
        with r2c2:
            _nav_button("Analyzer", "pages/3_Analyzer.py", "analyzer")
    else:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            _nav_button("GEX", "app.py", "gex")
        with c2:
            _nav_button("Leveraged Converter", "pages/1_Leverage_Equivalence.py", "lev")
        with c3:
            _nav_button("Synthetic Put DCA", "pages/2_Synthetic_Put_DCA.py", "dca")
        with c4:
            _nav_button("Analyzer", "pages/3_Analyzer.py", "analyzer")

top_nav(active="analyzer")

@st.cache_data(ttl=30)
def get_quote(symbol: str):
    symbol = str(symbol).strip().upper()
    last = None
    prev = None
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        if fi:
            last = fi.get("last_price") or fi.get("lastPrice")
            prev = fi.get("previous_close") or fi.get("previousClose")
    except Exception:
        pass

    if last is None or prev is None:
        try:
            h = yf.Ticker(symbol).history(period="5d", interval="1d")
            if h is not None and not h.empty and "Close" in h.columns:
                closes = h["Close"].dropna().astype(float)
                if len(closes) >= 1 and last is None:
                    last = float(closes.iloc[-1])
                if len(closes) >= 2 and prev is None:
                    prev = float(closes.iloc[-2])
        except Exception:
            pass

    try:
        last = float(last) if last is not None else float("nan")
    except Exception:
        last = float("nan")
    try:
        prev = float(prev) if prev is not None else float("nan")
    except Exception:
        prev = float("nan")

    chg = last - prev if (np.isfinite(last) and np.isfinite(prev)) else float("nan")
    pct = (chg / prev) if (np.isfinite(chg) and np.isfinite(prev) and prev != 0) else float("nan")
    return last, chg, pct



# appv56.py
# Daily Technical Dashboard (TV-style) — Plotly + reliable, explainable analysis
#
# v33 (from your current v32 lineage) changes:
# - Version consistency: title/page config now match v33 (your file had mismatched v11/v23 labels).
# - True Stoch RSI (not Stoch of price) added and used in analysis + oscillator panel.
# - Analysis made more reliable & explainable:
#     * EMA stack + price vs EMA levels + EMA slope (5-bar) to avoid “it broke EMA but still not ready” confusion.
#     * A transparent points breakdown shown in the UI.
# - Support/Resistance alignment:
#     * Uses candle-derived S/R for “Next/Major” levels (snapshot + analysis).
#     * Keeps TV-style pivot step lines on chart for visual context.
# - Chart UX:
#     * dragmode = pan (so you drag to move chart, not zoom boxes).
#     * Right-edge S/R labels use paper coords + white text and are collision-safe.
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
# Run:
#   streamlit run appv56.py




# =========================
# Config
# =========================
VERSION = 57

# =========================
def snap_cell(label: str, value: str, label_px: int = 10, value_px: int = 16, wrap_value: bool = False) -> str:
    white_space = "normal" if wrap_value else "nowrap"
    return f"""
    <div style="
      padding:10px 12px;
      border-radius:14px;
      background: rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.12);
      box-shadow: 0 6px 18px rgba(0,0,0,0.10);
      min-width:0;
    ">
      <div style="
        font-size:{label_px}px;
        opacity:0.78;
        line-height:1.1;
        margin-bottom:6px;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        color: rgba(255,255,255,0.80);
        font-weight:700;
        letter-spacing:0.2px;
      ">{label}</div>

      <div style="
        font-size:{value_px}px;
        font-weight:900;
        line-height:1.18;
        white-space:{white_space};
        overflow:hidden;
        text-overflow:ellipsis;
        word-break:break-word;
        color: rgba(255,255,255,0.98);
      ">{value}</div>
    </div>
    """

def tile_grid(items: List[dict], cols: int = 4):
    cols = max(1, int(cols))
    html = f"""
    <div style="
      display:grid;
      grid-template-columns: repeat({cols}, minmax(0, 1fr));
      gap:12px 14px;
      margin-top:8px;
      margin-bottom:12px;
    ">
    """
    for it in items:
        html += snap_cell(
            it.get("label",""),
            it.get("value","—"),
            label_px=it.get("label_px", 10),
            value_px=it.get("value_px", 16),
            wrap_value=it.get("wrap", False),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Indicator helpers
# =========================
def tv_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()

def tv_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    chg = close.diff()
    up = rma(chg.clip(lower=0), length)
    dn = rma((-chg).clip(lower=0), length)
    rs = up / dn
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(dn == 0, 100, np.where(up == 0, 0, rsi))
    return pd.Series(rsi, index=close.index, name="RSI")

def stoch_rsi(
    close: pd.Series,
    rsi_len: int = 14,
    stoch_len: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    True Stoch RSI:
      RSI = rsi(close, rsi_len)
      StochRSI = (RSI - min(RSI, stoch_len)) / (max(RSI, stoch_len) - min(RSI, stoch_len)) * 100
      then smooth to %K and %D
    """
    rsi = tv_rsi(close, rsi_len).astype(float)
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    rng = (rsi_max - rsi_min)
    raw = np.where(rng != 0, 100.0 * (rsi - rsi_min) / rng, np.nan)
    raw = pd.Series(raw, index=close.index, name="StochRSI_raw")
    k = raw.rolling(k_smooth).mean().rename("%K")
    d = k.rolling(d_smooth).mean().rename("%D")
    return k, d

def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    n = len(high)
    out = pd.Series(np.nan, index=high.index)
    h = high.values
    for i in range(left, n - right):
        window = h[i - left : i + right + 1]
        if np.isfinite(h[i]) and h[i] == np.max(window):
            out.iat[i] = h[i]
    return out

def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    out = pd.Series(np.nan, index=low.index)
    l = low.values
    for i in range(left, n - right):
        window = l[i - left : i + right + 1]
        if np.isfinite(l[i]) and l[i] == np.min(window):
            out.iat[i] = l[i]
    return out

def tv_sr_step_lines(df: pd.DataFrame, left: int, right: int) -> Tuple[pd.Series, pd.Series]:
    """TradingView-ish step S/R based on confirmed pivots."""
    ph = pivot_high(df["High"], left, right)
    pl = pivot_low(df["Low"], left, right)

    ph_plot = ph.shift(1).shift(-(right + 1))
    pl_plot = pl.shift(1).shift(-(right + 1))

    res_line = ph_plot.ffill()
    sup_line = pl_plot.ffill()

    if not res_line.notna().any():
        res_line = pd.Series(np.nan, index=df.index, name="Resistance")
        res_line.iloc[-1] = float(df["High"].tail(120).max())
        res_line = res_line.ffill()
    if not sup_line.notna().any():
        sup_line = pd.Series(np.nan, index=df.index, name="Support")
        sup_line.iloc[-1] = float(df["Low"].tail(120).min())
        sup_line = sup_line.ffill()

    res_line.name = "Resistance"
    sup_line.name = "Support"
    return res_line, sup_line

def ensure_sr_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "SR_R" in df.columns and "SR_S" in df.columns:
        return df
    try:
        sr_r, sr_s = tv_sr_step_lines(df, PIVOT_LEFT, PIVOT_RIGHT)
        df["SR_R"] = sr_r
        df["SR_S"] = sr_s
    except Exception:
        df["SR_R"] = pd.Series(index=df.index, data=float(df["High"].tail(120).max()) if len(df) else float("nan")).ffill()
        df["SR_S"] = pd.Series(index=df.index, data=float(df["Low"].tail(120).min()) if len(df) else float("nan")).ffill()
    return df


# =========================
# Candle-derived Support/Resistance engine
# =========================
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hi = df["High"].astype(float)
    lo = df["Low"].astype(float)
    cl = df["Close"].astype(float)
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def _cluster_levels(points: np.ndarray, tol: float) -> List[Dict]:
    clusters: List[Dict] = []
    for p in points:
        placed = False
        for c in clusters:
            if abs(p - c["center"]) <= tol:
                c["n"] += 1
                c["center"] = c["center"] + (p - c["center"]) / c["n"]
                placed = True
                break
        if not placed:
            clusters.append({"center": float(p), "n": 1})
    return clusters

def compute_sr_from_candles(
    df: pd.DataFrame,
    close_now: float,
    lookback: int = 220,
    max_levels: int = 8,
) -> Dict:
    if df is None or df.empty:
        return {"levels": [], "next_support": None, "next_resistance": None, "major_support": None, "major_resistance": None}

    w = df.tail(lookback).copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in w.columns:
            raise KeyError(f"Missing column {c} for S/R engine")

    w["Close"] = w["Close"].astype(float)
    w["High"] = w["High"].astype(float)
    w["Low"] = w["Low"].astype(float)
    w["Volume"] = pd.to_numeric(w["Volume"], errors="coerce").fillna(0.0)

    a = atr(w, 14)
    atr_now = float(a.dropna().iloc[-1]) if a is not None and a.dropna().size else float(w["High"].sub(w["Low"]).rolling(14, min_periods=1).mean().iloc[-1])
    tol = max(0.25 * atr_now, 0.002 * close_now)

    k = 5
    hi_roll = w["High"].rolling(k, center=True).max()
    lo_roll = w["Low"].rolling(k, center=True).min()
    swing_hi = w.loc[w["High"] >= hi_roll - 1e-9, ["High","Volume"]].copy()
    swing_lo = w.loc[w["Low"] <= lo_roll + 1e-9, ["Low","Volume"]].copy()

    points = np.concatenate([swing_hi["High"].values, swing_lo["Low"].values])
    points = points[np.isfinite(points)]
    points.sort()

    clusters = _cluster_levels(points, tol=tol)
    if not clusters:
        return {"levels": [], "next_support": None, "next_resistance": None, "major_support": None, "major_resistance": None}

    levels = []
    idx_arr = np.arange(len(w))
    vol = w["Volume"].values
    highs = w["High"].values
    lows = w["Low"].values

    o = w["Open"].astype(float).values
    cl = w["Close"].astype(float).values
    upper_wick = highs - np.maximum(o, cl)
    lower_wick = np.minimum(o, cl) - lows

    vol_mean = float(np.nanmean(vol)) if np.isfinite(np.nanmean(vol)) else 1.0

    for c in clusters:
        lvl = c["center"]
        near_hi = np.abs(highs - lvl) <= tol
        near_lo = np.abs(lows - lvl) <= tol
        touched = near_hi | near_lo
        touches = int(np.sum(touched))
        vol_touch = float(np.sum(vol[touched]))
        touched_idx = idx_arr[touched]
        last_touch = int(touched_idx.max()) if touched_idx.size else -1
        rec = 0.0 if last_touch < 0 else (last_touch / max(len(w)-1, 1))
        rej = float(np.sum(upper_wick[near_hi])) + float(np.sum(lower_wick[near_lo]))

        score = (touches * 1.0) + (vol_touch / (vol_mean + 1e-9)) * 0.35 + (rec * 3.0) + (rej / (tol + 1e-9)) * 0.15

        levels.append({"level": float(lvl), "touches": touches, "score": float(score)})

    levels.sort(key=lambda x: x["score"], reverse=True)
    picked = []
    for lv in levels:
        if any(abs(lv["level"] - p["level"]) <= tol for p in picked):
            continue
        picked.append(lv)
        if len(picked) >= max_levels:
            break
    picked.sort(key=lambda x: x["level"])

    supports = [x for x in picked if x["level"] < close_now]
    resistances = [x for x in picked if x["level"] > close_now]

    next_support = max(supports, key=lambda x: x["level"])["level"] if supports else None
    next_resistance = min(resistances, key=lambda x: x["level"])["level"] if resistances else None
    major_support = max(supports, key=lambda x: x["score"])["level"] if supports else None
    major_resistance = max(resistances, key=lambda x: x["score"])["level"] if resistances else None

    if next_support is None:
        next_support = float(w["Low"].tail(60).min()) if len(w) else None
    if major_support is None:
        major_support = next_support
    if next_resistance is None:
        next_resistance = float(w["High"].tail(60).max()) if len(w) else None
    if major_resistance is None:
        major_resistance = next_resistance

    return {
        "levels": picked,
        "tol": float(tol),
        "next_support": next_support,
        "next_resistance": next_resistance,
        "major_support": major_support,
        "major_resistance": major_resistance,
    }


# =========================
# Volume confluence helpers
# =========================
def volume_confluence(df: pd.DataFrame, level_sup: Optional[float], level_res: Optional[float]) -> Dict:
    out = {
        "vol_ratio": None,
        "vol_state": "—",
        "green_dom": None,
        "green_state": "—",
        "breakout_state": "—",
        "points": 0,
    }
    if df is None or df.empty or "Volume" not in df.columns:
        return out

    w = df.tail(60).copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in w.columns:
            return out

    vol = pd.to_numeric(w["Volume"], errors="coerce").fillna(0.0)
    v20 = vol.rolling(20, min_periods=10).mean()
    v_now = float(vol.iloc[-1])
    v20_now = float(v20.iloc[-1]) if pd.notna(v20.iloc[-1]) and float(v20.iloc[-1]) != 0 else None

    if v20_now:
        ratio = v_now / v20_now
        out["vol_ratio"] = ratio
        if ratio >= 1.30:
            out["vol_state"] = "Above avg (>=1.3×)"
            out["points"] += 1
        elif ratio <= 0.90:
            out["vol_state"] = "Below avg (<=0.9×)"
            out["points"] -= 1
        else:
            out["vol_state"] = "Near avg"

    last10 = w.tail(10).copy()
    green = last10["Close"].astype(float) >= last10["Open"].astype(float)
    gv = float(pd.to_numeric(last10.loc[green, "Volume"], errors="coerce").fillna(0.0).sum())
    rv = float(pd.to_numeric(last10.loc[~green, "Volume"], errors="coerce").fillna(0.0).sum())
    denom = gv + rv
    if denom > 0:
        green_share = gv / denom
        out["green_dom"] = green_share
        if green_share >= 0.58:
            out["green_state"] = "Green vol dominant"
            out["points"] += 1
        elif green_share <= 0.42:
            out["green_state"] = "Red vol dominant"
            out["points"] -= 1
        else:
            out["green_state"] = "Mixed"

    close_now = float(w["Close"].astype(float).iloc[-1])
    thresh = 0.0035 * close_now

    def confirmed() -> bool:
        return (out["vol_ratio"] is not None) and (out["vol_ratio"] >= 1.30)

    def weak() -> bool:
        return (out["vol_ratio"] is not None) and (out["vol_ratio"] <= 0.90)

    if level_res is not None and close_now >= level_res + thresh:
        out["breakout_state"] = "Breakout confirmed" if confirmed() else ("Breakout weak" if weak() else "Breakout unconfirmed")
        out["points"] += 1 if confirmed() else (-1 if weak() else 0)

    if level_sup is not None and close_now <= level_sup - thresh:
        out["breakout_state"] = "Breakdown confirmed" if confirmed() else ("Breakdown weak" if weak() else "Breakdown unconfirmed")
        out["points"] += 1 if confirmed() else (-1 if weak() else 0)

    return out


# =========================
# Analysis
# =========================
@dataclass
class Verdict:
    label: str
    headline: str
    components: List[tuple]
    explanations: List[str]
    score: int
    confidence_pct: int
    confidence_label: str

def _pct_dist(a: float, b: float) -> float:
    if b == 0 or (isinstance(b, float) and math.isnan(b)):
        return float("nan")
    return (a / b - 1.0) * 100.0

def _slope(series: pd.Series, n: int = 5) -> Optional[float]:
    """Simple slope over last n bars: (last - n_ago) / n."""
    if series is None or series.dropna().size < (n + 1):
        return None
    s = series.dropna()
    return float((s.iloc[-1] - s.iloc[-(n+1)]) / n)

def score_to_confidence(score: int) -> tuple[int, str]:
    """Map model score to a simple confidence % + label.
    This is NOT a probability of profit; it is a signal-strength gauge.
    """
    # Typical score range is about -7 to +7.
    s = max(-7, min(7, int(score)))
    pct = int(round((s + 7) / 14 * 100))
    if pct >= 70:
        lbl = "High"
    elif pct >= 50:
        lbl = "Moderate"
    else:
        lbl = "Low"
    return pct, lbl


def build_verdict(
    close: float,
    ema20: float,
    ema50: float,
    ema200: float,
    ema20_slope: Optional[float],
    ema50_slope: Optional[float],
    rsi: float,
    k: float,
    d: float,
    next_support: Optional[float],
    next_resistance: Optional[float],
    volc: Optional[Dict],
    prev_close: Optional[float],
    prev_ema20: Optional[float],
    prev_ema50: Optional[float],
) -> Verdict:
    """
    Points system (signal alignment score):
      + points = signals agree bullish
      0 points = neutral / no clear edge
      - points = signals agree bearish
    Higher total score => higher confidence (signals more aligned).
    """
    explanations: List[str] = []
    components: List[tuple] = []
    score = 0

    def add(name: str, pts: int, explanation: str):
        nonlocal score
        score += pts
        components.append((name, pts))
        explanations.append(explanation)

    # 1) Trend structure (EMA stack)
    stack_bull = (ema20 > ema50) and (ema50 > ema200)
    stack_bear = (ema20 < ema50) and (ema50 < ema200)
    if stack_bull:
        add("Trend (20/50/200)", +2, "Trend looks healthy overall: short-, medium-, and long-term trend lines are stacked upward.")
    elif stack_bear:
        add("Trend (20/50/200)", -2, "Trend looks weak overall: short-, medium-, and long-term trend lines are stacked downward.")
    else:
        add("Trend (20/50/200)", 0, "Trend is mixed: the main trend lines disagree, which often leads to choppy/sideways price action.")

    # 2) Price vs short-term trend lines (EMA20/EMA50)
    above_20 = close > ema20
    above_50 = close > ema50
    reclaimed_20_50 = False
    if prev_close is not None and prev_ema20 is not None and prev_ema50 is not None:
        was_below = (prev_close <= prev_ema20) or (prev_close <= prev_ema50)
        now_above = above_20 and above_50
        reclaimed_20_50 = bool(was_below and now_above)

    if above_20 and above_50:
        if reclaimed_20_50:
            add("Price vs EMA20/50", +1, "Price just moved back above the short-term trend lines (20/50-day), which is a positive shift.")
        else:
            add("Price vs EMA20/50", +1, "Price is holding above the short-term trend lines (20/50-day), which is generally a good sign.")
    elif above_20 and not above_50:
        add("Price vs EMA20/50", 0, "Price bounced above the 20-day line but is still below the 50-day line, so it’s not fully confirmed yet.")
    else:
        add("Price vs EMA20/50", -1, "Price is still below the short-term trend lines (20/50-day), which usually means weaker momentum.")

    # 3) Bigger picture (EMA200)
    if close > ema200:
        add("Price vs EMA200", +1, "Bigger picture is supportive: price is above the 200-day line (a common long-term trend reference).")
    else:
        add("Price vs EMA200", -1, "Bigger picture is still weak: price is below the 200-day line, so the long-term trend isn’t supportive yet.")

    # 4) Trend direction filter (EMA slopes)
    slope_pts = 0
    if ema20_slope is not None and ema50_slope is not None:
        if ema20_slope > 0 and ema50_slope > 0:
            slope_pts = +1
            exp = "The short-term trend lines are sloping upward, which suggests momentum is improving."
        elif ema20_slope < 0 and ema50_slope < 0:
            slope_pts = -1
            exp = "The short-term trend lines are sloping downward, which can make rallies less reliable."
        else:
            slope_pts = 0
            exp = "The trend lines’ slopes disagree, which can mean the market is indecisive."
    else:
        exp = "Not enough data to judge the slope of the trend lines."
    add("EMA slopes", slope_pts, exp)

    # 5) Momentum (RSI)
    if rsi >= 55:
        add("RSI momentum", +1, f"Momentum favors buyers: RSI is {rsi:.1f}.")
    elif rsi <= 45:
        add("RSI momentum", -1, f"Momentum favors sellers: RSI is {rsi:.1f}.")
    else:
        add("RSI momentum", 0, f"Momentum is balanced: RSI is {rsi:.1f}.")

    # 6) Short-term turning signal (Stoch RSI)
    stoch_pts = 0
    if np.isfinite(k) and np.isfinite(d):
        if (k < 20 and d < 20) and (k > d):
            stoch_pts = +1
            exp = f"Short-term bounce signal: Stoch RSI is turning up from a very low level (%K {k:.1f} > %D {d:.1f})."
        elif (k > 80 and d > 80) and (k < d):
            stoch_pts = -1
            exp = f"Short-term pullback risk: Stoch RSI is turning down from a very high level (%K {k:.1f} < %D {d:.1f})."
        else:
            exp = f"No strong Stoch RSI signal right now (%K {k:.1f}, %D {d:.1f})."
    else:
        exp = "Not enough data to calculate Stoch RSI."
    add("Stoch RSI", stoch_pts, exp)

    # 7) Volume (conviction)
    vol_pts = 0
    if isinstance(volc, dict):
        vr = volc.get("vol_ratio")
        if vr is not None:
            if vr >= 1.30:
                vol_pts += 1
                explanations.append(f"Trading activity is higher than normal (about {vr:.2f}× recent average), which can make moves more believable.")
            elif vr <= 0.90:
                vol_pts -= 1
                explanations.append(f"Trading activity is lighter than normal (about {vr:.2f}× recent average), which can make moves less reliable.")
            else:
                explanations.append(f"Trading activity is around normal (about {vr:.2f}× recent average).")
        else:
            explanations.append("Volume ratio isn’t available, so we can’t judge how strong today’s participation is.")

        gs = volc.get("green_dom")
        if gs is not None:
            if gs >= 0.58:
                vol_pts += 1
                explanations.append(f"More volume has been happening on up days lately (~{gs:.0%} of the last-10-day volume), which supports buyers.")
            elif gs <= 0.42:
                vol_pts -= 1
                explanations.append(f"More volume has been happening on down days lately (~{(1-gs):.0%} of the last-10-day volume), which supports sellers.")
            else:
                explanations.append("Up-day and down-day volume have been fairly balanced recently.")

        bs = volc.get("breakout_state")
        if bs and bs not in ("—",):
            explanations.append(f"Level check: {bs}.")

    add("Volume", vol_pts, "Volume is used as a ‘conviction’ check: higher participation usually makes signals more trustworthy.")

    # 8) Support / Resistance context (not scored)
    if next_support is not None:
        explanations.append(f"Nearby support: around {next_support:.2f} (a level where buyers previously stepped in).")
    if next_resistance is not None:
        explanations.append(f"Nearby resistance: around {next_resistance:.2f} (a level where sellers previously stepped in).")

    # Label thresholds (kept intentionally simple)
    if score >= 4:
        label = "Ready"
        headline = "Conditions are **favorable** (trend + momentum + confirmation mostly aligned)."
    elif score >= 2:
        label = "Getting ready"
        headline = "Signals are improving, but it’s **not fully confirmed** yet."
    else:
        label = "Not ready"
        headline = "Signals are **not aligned** (higher chop / lower edge)."

    confidence_pct, confidence_label = score_to_confidence(score)
    return Verdict(
        label=label,
        headline=headline,
        components=components,
        explanations=explanations,
        score=int(score),
        confidence_pct=int(confidence_pct),
        confidence_label=confidence_label,
    )



# =========================
# Data
# =========================
@st.cache_data(show_spinner=False)
def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=YFINANCE_PERIOD,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close", "Volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns {miss}. Got: {list(df.columns)}")

    df = df.dropna(subset=need)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


# =========================
# Chart
# =========================
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha_close = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4.0

    ha_open = pd.Series(index=ha.index, dtype="float64")
    ha_open.iloc[0] = (ha["Open"].iloc[0] + ha["Close"].iloc[0]) / 2.0
    for i in range(1, len(ha)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = pd.concat([ha["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([ha["Low"], ha_open, ha_close], axis=1).min(axis=1)

    out = ha.copy()
    out["Open"] = ha_open
    out["High"] = ha_high
    out["Low"] = ha_low
    out["Close"] = ha_close
    return out

def make_chart(df: pd.DataFrame, ruler_y: float | None = None) -> go.Figure:
    raw_df = df.copy()
    ha_df = to_heikin_ashi(raw_df)
    vol_colors = np.where(raw_df["Close"] >= raw_df["Open"], BULL_FILL, BEAR_FILL)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.13, 0.27],
    )

    # Candles: Regular + HA toggle
    fig.add_trace(
        go.Candlestick(
            x=raw_df.index,
            open=raw_df["Open"], high=raw_df["High"], low=raw_df["Low"], close=raw_df["Close"],
            name="Regular",
            visible=True,
            increasing=dict(fillcolor=BULL_FILL, line=dict(color=BULL_LINE, width=1)),
            decreasing=dict(fillcolor=BEAR_FILL, line=dict(color=BEAR_LINE, width=1)),
            whiskerwidth=0.35,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Candlestick(
            x=ha_df.index,
            open=ha_df["Open"], high=ha_df["High"], low=ha_df["Low"], close=ha_df["Close"],
            name="Heikin Ashi",
            visible=False,
            increasing=dict(fillcolor=BULL_FILL, line=dict(color=BULL_LINE, width=1)),
            decreasing=dict(fillcolor=BEAR_FILL, line=dict(color=BEAR_LINE, width=1)),
            whiskerwidth=0.35,
        ),
        row=1, col=1
    )

    # Invisible helper trace so hover always returns a single "price" (Close) per bar
    fig.add_trace(
        go.Scatter(
            x=raw_df.index,
            y=raw_df["Close"],
            name="HoverClose",
            mode="markers",
            marker=dict(size=10, opacity=0.0),
            hovertemplate="Price: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1
    )
    # EMAs
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["EMA20"], name="EMA 20", line=dict(color="#2ca02c", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["EMA50"], name="EMA 50", line=dict(color="#ef4444", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["EMA200"], name="EMA 200", line=dict(color="#7E57C2", width=2.5)), row=1, col=1)

    # TV-style pivot step S/R
    if "SR_R" in raw_df.columns:
        fig.add_trace(go.Scatter(
            x=raw_df.index, y=raw_df["SR_R"],
            name="Pivot Resistance",
            line=dict(color="#ef4444", width=3),
            connectgaps=False,
            line_shape="hv",
        ), row=1, col=1)
    if "SR_S" in raw_df.columns:
        fig.add_trace(go.Scatter(
            x=raw_df.index, y=raw_df["SR_S"],
            name="Pivot Support",
            line=dict(color="#06b6d4", width=3),
            connectgaps=False,
            line_shape="hv",
        ), row=1, col=1)

    # Volume
    fig.add_trace(
        go.Bar(
            x=raw_df.index,
            y=raw_df["Volume"],
            name="Volume",
            marker=dict(color=vol_colors),
            opacity=0.85,
        ),
        row=2, col=1
    )

    # Oscillator shading
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.10)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=20, y1=80, fillcolor="rgba(31,119,180,0.06)", line_width=0, row=3, col=1)

    # RSI + Stoch RSI
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["RSI"], name="RSI (14)", line=dict(color="purple", dash="dot", width=2.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["%K"], name="StochRSI %K", line=dict(color="#ff7f0e", width=2.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df["%D"], name="StochRSI %D", line=dict(color="#4c78a8", width=2.5)), row=3, col=1)

    # Guide lines
    guide_shapes = [
        dict(type="line", xref="paper", yref="y3", x0=0, x1=1, y0=50, y1=50,
             line=dict(color="rgba(255,255,255,0.45)", width=1.2, dash="dot")),
        dict(type="line", xref="paper", yref="y3", x0=0, x1=1, y0=70, y1=70,
             line=dict(color="rgba(0,153,255,0.45)", width=1.1, dash="dot")),
        dict(type="line", xref="paper", yref="y3", x0=0, x1=1, y0=30, y1=30,
             line=dict(color="rgba(0,153,255,0.45)", width=1.1, dash="dot")),
        dict(type="line", xref="paper", yref="y3", x0=0, x1=1, y0=80, y1=80,
             line=dict(color="rgba(220,38,38,0.55)", width=1.4)),
        dict(type="line", xref="paper", yref="y3", x0=0, x1=1, y0=20, y1=20,
             line=dict(color="rgba(220,38,38,0.55)", width=1.4)),
    ]

    # Visible range and y-range padding
    idx = raw_df.index
    x_end = idx[-1].strftime("%Y-%m-%d")
    # Common zoom presets
    range_3m = (idx[-1] - pd.Timedelta(days=93)).strftime("%Y-%m-%d")
    range_6m = (idx[-1] - pd.Timedelta(days=186)).strftime("%Y-%m-%d")
    range_1y = (idx[-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    range_2y = (idx[-1] - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    range_5y = (idx[-1] - pd.Timedelta(days=1825)).strftime("%Y-%m-%d")
    ytd_start = pd.Timestamp(year=idx[-1].year, month=1, day=1).strftime("%Y-%m-%d")

    vis = raw_df.iloc[-DEFAULT_BARS:].copy() if len(raw_df) > DEFAULT_BARS else raw_df.copy()
    close_last = float(raw_df["Close"].iloc[-1])
    y_lo = float(vis["Low"].min())
    y_hi = float(vis["High"].max())
    pad = (y_hi - y_lo) * 0.18 if y_hi > y_lo else (y_hi * 0.05 if y_hi else 1.0)
    y0_range = y_lo - pad
    y1_range = y_hi + pad

    # Right-edge labels for most recent pivot S/R (white text)
    annotations = []
    try:
        res_val = float(raw_df["SR_R"].dropna().iloc[-1]) if ("SR_R" in raw_df.columns and raw_df["SR_R"].notna().any()) else None
        sup_val = float(raw_df["SR_S"].dropna().iloc[-1]) if ("SR_S" in raw_df.columns and raw_df["SR_S"].notna().any()) else None
    except Exception:
        res_val, sup_val = None, None

    if res_val is not None or sup_val is not None:
        span = (y1_range - y0_range) if (y1_range > y0_range) else 1.0
        close_sr = (res_val is not None and sup_val is not None and abs(res_val - sup_val) < 0.04 * span)
        sep = 22 if close_sr else 12

        if res_val is not None:
            annotations.append(dict(
                x=0.995, xref="paper", xanchor="right", xshift=-4,
                y=res_val, yref="y", yanchor="bottom", yshift=-sep,
                text=f"R {res_val:.2f} ({((res_val/close_last)-1)*100:+.1f}%)", showarrow=False,
                bgcolor="rgba(239,68,68,0.55)", bordercolor="#ef4444", borderwidth=1,
                font=dict(size=12, color="white")
            ))
        if sup_val is not None:
            annotations.append(dict(
                x=0.995, xref="paper", xanchor="right", xshift=-4,
                y=sup_val, yref="y", yanchor="top", yshift=sep,
                text=f"S {sup_val:.2f} ({((sup_val/close_last)-1)*100:+.1f}%)", showarrow=False,
                bgcolor="rgba(6,182,212,0.55)", bordercolor="#06b6d4", borderwidth=1,
                font=dict(size=12, color="white")
            ))

    n_traces = len(fig.data)
    # Toggle only the candle traces (keep EMAs/levels/volume/oscillator visible)
    vis_regular = [True, False, True] + [True] * max(0, n_traces - 3)
    vis_heikin  = [False, True, True] + [True] * max(0, n_traces - 3)

    fig.update_layout(
        template="plotly_white",
        height=960,
        margin=dict(l=10, r=110, t=120, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode="pan",
        hovermode="x unified",
        hoverdistance=40,
        spikedistance=100000,
        font=dict(size=13),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font=dict(color="black")),
        xaxis_rangeslider_visible=False,
        shapes=guide_shapes,
        annotations=annotations,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0, y=1.05,
                xanchor="left", yanchor="top",
                pad=dict(r=6, t=0),
                bgcolor="rgba(245,245,245,0.95)",
                bordercolor="rgba(0,0,0,0.20)",
                borderwidth=1,
                font=dict(size=12, color="black"),
                active=1,
                buttons=[
                    dict(label="3M", method="relayout", args=[{"xaxis.autorange": False, "xaxis.range": [range_3m, x_end]}]),
                    dict(label="6M", method="relayout", args=[{"xaxis.autorange": False, "xaxis.range": [range_6m, x_end]}]),
                    dict(label="YTD", method="relayout", args=[{"xaxis.autorange": False, "xaxis.range": [ytd_start, x_end]}]),
                    dict(label="1Y", method="relayout", args=[{"xaxis.autorange": False, "xaxis.range": [range_1y, x_end]}]),
                    dict(label="2Y", method="relayout", args=[{"xaxis.autorange": False, "xaxis.range": [range_2y, x_end]}]),
                    dict(label="5Y", method="relayout", args=[{"xaxis.autorange": False, "xaxis.range": [range_5y, x_end]}]),
                    dict(label="All", method="relayout", args=[{"xaxis.autorange": True}]),
                ],
            ),
            dict(
                type="buttons",
                direction="left",
                x=0.32, y=1.05,
                xanchor="left", yanchor="top",
                pad=dict(r=6, t=0),
                bgcolor="rgba(245,245,245,0.95)",
                bordercolor="rgba(0,0,0,0.20)",
                borderwidth=1,
                font=dict(size=12, color="black"),
                active=0,
                buttons=[
                    dict(label="Regular", method="update", args=[{"visible": vis_regular}]),
                    dict(label="Heikin Ashi", method="update", args=[{"visible": vis_heikin}]),
                ],
            ),
        ],
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dot",
        spikecolor="white",
    )

    # Default zoom ~6M
    fig.update_layout(xaxis=dict(autorange=False, range=[range_6m, x_end]))
    for r in (1, 2, 3):
        fig.update_xaxes(range=[range_6m, x_end], row=r, col=1)

    # Constrain y-axis to visible window so candles fill the chart
    fig.update_yaxes(range=[y0_range, y1_range], row=1, col=1)
    fig.update_yaxes(
        title_text="Price",
        row=1, col=1,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="white"
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=True, gridcolor="rgba(0,0,0,0.08)", showticklabels=False)
    fig.update_yaxes(title_text="RSI / StochRSI", row=3, col=1, range=[0, 100], showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    # Optional price ruler line (hover-driven; snaps to hovered Close)
    if ruler_y is not None:
        try:
            yv = float(ruler_y)
            fig.add_hline(
                y=yv,
                line_width=1,
                line_dash="dot",
                line_color="white",
                opacity=0.9,
                annotation_text=f"{yv:,.2f}",
                annotation_position="right",
                annotation_font=dict(color="white", size=12),
                annotation_bgcolor="rgba(0,0,0,0.55)",
                annotation_bordercolor="rgba(255,255,255,0.25)",
                annotation_borderpad=3,
            )
        except Exception:
            pass

    # --- Layout polish: remove "gap" by placing legend INSIDE the chart area ---
    fig.update_layout(
        margin=dict(t=70, r=20, b=40, l=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.995,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            tracegroupgap=6,
        ),
        dragmode="pan",
    )

    if MOBILE:
        fig.update_layout(
            height=560,
            margin=dict(t=58, r=8, b=28, l=42),
            legend=dict(font=dict(size=10), y=0.99),
        )

    return fig

def scoring_chart(components: List[tuple]) -> go.Figure:
    names = [c[0] for c in components]
    pts = [int(c[1]) for c in components]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pts,
        y=names,
        orientation="h",
        text=[f"{p:+d}" for p in pts],
        textposition="outside",
        hovertemplate="%{y}: %{x:+d}<extra></extra>",
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="", zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.35)", dtick=1),
        yaxis=dict(title="", automargin=True),
        showlegend=False,
        template="plotly_white",
    )
    return fig

# =========================
# UI
# =========================
st.title(f"Daily Technical Dashboard — v{VERSION}")


if MOBILE:
    st.markdown("<div style='font-weight:600; margin-bottom:6px;'>Ticker</div>", unsafe_allow_html=True)
    ticker = st.text_input(
        "",
        value="",
        placeholder="Enter ticker",
        label_visibility="collapsed",
    ).strip().upper()
else:
    c1, c2, _ = st.columns([1, 2, 28])

    with c1:
        st.markdown("<div style='padding-top:6px; font-weight:600;'>Ticker:</div>", unsafe_allow_html=True)

    with c2:
        ticker = st.text_input(
            "",
            value="",
            placeholder="Enter ticker",
            label_visibility="collapsed",
        ).strip().upper()

# Ticker + current price (under the title)
last_q, chg_q, pct_q = get_quote(ticker)
if np.isfinite(last_q):
    color = "#00C853" if (np.isfinite(chg_q) and chg_q > 0) else ("#FF3D00" if (np.isfinite(chg_q) and chg_q < 0) else "#cfcfcf")
    arrow = "▲" if (np.isfinite(chg_q) and chg_q > 0) else ("▼" if (np.isfinite(chg_q) and chg_q < 0) else "")
    
try:
    df = load_daily(ticker)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if df.empty:
    st.error("No data returned for this ticker.")
    st.stop()





# Indicators (regular price)
df["EMA20"] = tv_ema(df["Close"], EMA_FAST)
df["EMA50"] = tv_ema(df["Close"], EMA_MID)
df["EMA200"] = tv_ema(df["Close"], EMA_SLOW)
df["RSI"] = tv_rsi(df["Close"], RSI_LEN)
df["%K"], df["%D"] = stoch_rsi(df["Close"], RSI_LEN, STOCH_RSI_LEN, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
df = ensure_sr_columns(df)

# Candle-derived S/R (nearest + major)
close_now = float(df["Close"].iloc[-1])
sr = compute_sr_from_candles(df, close_now=close_now, lookback=220, max_levels=8)
next_sup = sr.get("next_support")
next_res = sr.get("next_resistance")
major_sup = sr.get("major_support")
major_res = sr.get("major_resistance")

# Volume confluence uses candle-derived next levels
volc = volume_confluence(df, level_sup=next_sup, level_res=next_res)

# Slopes / prev values
ema20_slope = _slope(df["EMA20"], 5)
ema50_slope = _slope(df["EMA50"], 5)
prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
prev_ema20 = float(df["EMA20"].iloc[-2]) if len(df) >= 2 else None
prev_ema50 = float(df["EMA50"].iloc[-2]) if len(df) >= 2 else None

last = df.iloc[-1]
verdict = build_verdict(
    close=float(last["Close"]),
    ema20=float(last["EMA20"]),
    ema50=float(last["EMA50"]),
    ema200=float(last["EMA200"]),
    ema20_slope=ema20_slope,
    ema50_slope=ema50_slope,
    rsi=float(last["RSI"]),
    k=float(last["%K"]) if pd.notna(last["%K"]) else float("nan"),
    d=float(last["%D"]) if pd.notna(last["%D"]) else float("nan"),
    next_support=next_sup,
    next_resistance=next_res,
    volc=volc,
    prev_close=prev_close,
    prev_ema20=prev_ema20,
    prev_ema50=prev_ema50,
)


def render_chart_block():
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:14px;">
            <div style="
                padding:8px 14px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.12);
                font-size:{'16px' if MOBILE else '18px'};
                font-weight:600;
                max-width: 100%;
            ">
                {ticker} ${last_q:,.2f}
                <span style="color:{color}; margin-left:10px;">
                    {arrow} {chg_q:+.2f} ({pct_q:+.2%})
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    ruler_y = st.session_state.get("ruler_y")
    fig = make_chart(df, ruler_y=ruler_y)

    # Mobile-friendly sizing
    fig.update_layout(height=620 if MOBILE else 960)

    chart_config = {
        "scrollZoom": False if MOBILE else True,
        "displayModeBar": False if MOBILE else True,
        "displaylogo": False,
        "responsive": True,
        "doubleClick": False,
    }
    if MOBILE:
        chart_config["modeBarButtonsToRemove"] = ["zoom2d", "pan2d", "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]

    if plotly_events is None:
        st.plotly_chart(fig, use_container_width=True, config=chart_config)
        if not MOBILE:
            st.caption("Optional: `pip install streamlit-plotly-events` to enable hover price ruler (snaps to bar close).")
    else:
        hovered = plotly_events(
            fig,
            hover_event=True,
            click_event=False,
            select_event=False,
            key=f"hover_{ticker}",
        )
        if hovered:
            hy = hovered[0].get("y")
            if hy is not None:
                try:
                    st.session_state["ruler_y"] = float(hy)
                except Exception:
                    pass

        st.plotly_chart(fig, use_container_width=True, config=chart_config)

        if st.session_state.get("ruler_y") is not None and not MOBILE:
            st.markdown(
                f"<div style=\"margin-top:6px;padding:8px 10px;border-radius:10px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);font-weight:900;display:inline-block;\">Ruler price (hover close): {st.session_state['ruler_y']:,.2f}</div>",
                unsafe_allow_html=True,
            )

def render_right_panel():
    st.subheader("Snapshot")

    tile_grid(
        items=[
            {"label": "Next Support", "value": (f"{next_sup:,.2f}" if next_sup is not None else "—")},
            {"label": "Next Resistance", "value": (f"{next_res:,.2f}" if next_res is not None else "—")},
            {"label": "Major Support", "value": (f"{major_sup:,.2f}" if major_sup is not None else "—")},
            {"label": "Major Resistance", "value": (f"{major_res:,.2f}" if major_res is not None else "—")},
        ],
        cols=(2 if MOBILE else 4),
    )

    st.subheader(f"Analysis — {verdict.label}")
    if verdict.label == "Ready":
        st.success(verdict.headline)
    elif verdict.label == "Getting ready":
        st.warning(verdict.headline)
    else:
        st.error(verdict.headline)

    # Simple, beginner-friendly summary
    close_v = float(last["Close"])
    ema20_v = float(last["EMA20"])
    ema50_v = float(last["EMA50"])
    ema200_v = float(last["EMA200"])

    above20 = close_v > ema20_v
    above50 = close_v > ema50_v
    above200 = close_v > ema200_v
    trend_phrase = "up" if (ema20_v > ema50_v and ema50_v > ema200_v) else ("down" if (ema20_v < ema50_v and ema50_v < ema200_v) else "mixed")
    big_picture = "positive" if above200 else "still weak"
    short_term = "above" if (above20 and above50) else ("partly above" if above20 else "below")

    conf_color = "#22c55e" if verdict.confidence_label == "High" else ("#eab308" if verdict.confidence_label == "Moderate" else "#ef4444")

    summary_text = (
        f"Trend is **{trend_phrase}**. "
        f"Price is **{short_term}** the short‑term trend lines (EMA20/50), "
        f"and the bigger‑picture trend is **{big_picture}** (vs EMA200)."
    )
    st.markdown(
        f"""
        <div style="padding:12px 14px;border-radius:12px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);margin-bottom:10px;font-size:14px;line-height:1.4;">
        <div style="font-weight:900;">Quick Summary</div><div style="margin-top:6px;">{summary_text}</div><div style="margin-top:8px;opacity:0.9;font-weight:800;color:{conf_color};">{verdict.confidence_pct}% ({verdict.confidence_label})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MOBILE:
        with st.expander("Score breakdown", expanded=False):
            st.markdown("**Score breakdown (what the app is counting):**")
            fig_score = scoring_chart(verdict.components)
            st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})

        # Distance to nearby levels
        dist_items = []
        if next_res is not None:
            dist_r = (next_res / close_now - 1.0) * 100.0
            dist_items.append(("Distance to Resistance", f"{dist_r:+.2f}%"))
        if next_sup is not None:
            dist_s = (close_now / next_sup - 1.0) * 100.0
            dist_items.append(("Distance to Support", f"{dist_s:+.2f}%"))
        if dist_items:
            with st.expander("Distance to levels (room / risk)", expanded=False):
                for k, v in dist_items:
                    st.write(f"- {k}: {v}")

        with st.expander("Why this label", expanded=True):
            for line in verdict.explanations:
                st.write(f"- {line}")

        with st.expander("Guide", expanded=False):
            st.write("- EMAs: EMA20/50 = short-term; EMA200 = bigger trend. A reclaim is stronger when slopes turn up too.")
            st.write("- RSI: 50 is neutral; >55 supportive; <45 weak.")
            st.write("- Stoch RSI: 20/80 are oversold/overbought. Crosses near those zones can signal turns.")
            st.write("- Volume: above-average volume makes breakouts/breakdowns more believable.")
            st.write("- Support/Resistance: Next = nearest level; Major = highest-strength zone from clustering.")
    else:
        st.markdown("**Score breakdown (what the app is counting):**")
        fig_score = scoring_chart(verdict.components)
        st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})

        # Distance to nearby levels (simple risk/room gauge)
        dist_items = []
        if next_res is not None:
            dist_r = (next_res / close_now - 1.0) * 100.0
            dist_items.append(("Distance to Resistance", f"{dist_r:+.2f}%"))
        if next_sup is not None:
            dist_s = (close_now / next_sup - 1.0) * 100.0
            dist_items.append(("Distance to Support", f"{dist_s:+.2f}%"))
        if dist_items:
            st.markdown("**Distance to levels (room / risk):**")
            for k, v in dist_items:
                st.write(f"- {k}: {v}")

        with st.expander("How scoring & confidence work (simple)"):
            st.write("This app uses a **points system** to summarize the chart signals.")
            st.write("- **+ points** = signals lean bullish")
            st.write("- **0 points** = neutral / unclear")
            st.write("- **− points** = signals lean bearish")
            st.write("**Confidence** is based on the total score and reflects how **aligned** the signals are (not a guarantee).")

        st.markdown("**Why this label (plain English):**")
        for line in verdict.explanations:
            st.write(f"- {line}")

        with st.expander("How to read these indicators (simple guide)"):
            st.write("- EMAs: EMA20/50 = short-term; EMA200 = bigger trend. A reclaim is stronger when slopes turn up too.")
            st.write("- RSI: 50 is neutral; >55 supportive; <45 weak.")
            st.write("- Stoch RSI: 20/80 are oversold/overbought. Crosses near those zones can signal turns.")
            st.write("- Volume: above-average volume makes breakouts/breakdowns more believable.")
            st.write("- Support/Resistance: Next = nearest level; Major = highest-strength zone from clustering.")

# ---- layout ----
if MOBILE:
    render_chart_block()
    st.divider()
    render_right_panel()
else:
    left, right = st.columns([1.55, 1.05])
    with left:
        render_chart_block()
    with right:
        render_right_panel()

with st.expander("Debug (last 40 rows)"):
    cols = ["Open", "High", "Low", "Close", "Volume", "EMA20", "EMA50", "EMA200", "RSI", "%K", "%D", "SR_S", "SR_R"]
    st.dataframe(df[cols].tail(40), use_container_width=True)
