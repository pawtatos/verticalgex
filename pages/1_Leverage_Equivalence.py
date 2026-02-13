
import streamlit as st
import yfinance as yf
import textwrap
import math

st.set_page_config(
    page_title="Leverage Equivalence",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide sidebar & multipage nav chrome
st.markdown("""
<style>
  [data-testid="stSidebar"] { display: none; }
  [data-testid="stSidebarNav"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ======================
# NAV (match other pages)
# ======================
def top_nav(active: str = "lev"):
    st.markdown("""
    <style>
      .navwrap { max-width: 900px; margin: 0 auto; padding: 6px 0 14px 0; }
      .navwrap div[data-testid="stHorizontalBlock"] { gap: 0.65rem !important; }
      .navbtn button {
        border-radius: 12px !important;
        font-weight: 850 !important;
        height: 42px !important;
      }
      .navbtn.active button {
        border: 2px solid rgba(80,170,255,0.95) !important;
        background: rgba(80,170,255,0.20) !important;
        box-shadow: inset 0 -4px 0 rgba(80,170,255,0.95) !important;
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="navwrap">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")

    with c1:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="gex" else ""), unsafe_allow_html=True)
        if st.button("GEX", use_container_width=True):
            st.switch_page("app.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="lev" else ""), unsafe_allow_html=True)
        if st.button("Leveraged Converter", use_container_width=True):
            st.switch_page("pages/1_Leverage_Equivalence.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="dca" else ""), unsafe_allow_html=True)
        if st.button("Synthetic Put DCA", use_container_width=True):
            st.switch_page("pages/2_Synthetic_Put_DCA.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="cc" else ""), unsafe_allow_html=True)
        if st.button("CC / CSP", use_container_width=True):
            st.switch_page("pages/3_CC_CSP.py")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

top_nav(active="lev")

# ======================
# CSS (your original layout)
# ======================
st.markdown(
    """
    <style>
      div[data-baseweb="input"] input {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        font-size: 14px !important;
      }

      .mult-tabs { margin-top: 26px; }
      .mult-tabs div[data-testid="stSegmentedControl"] button {
        height: 56px !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        letter-spacing: 0.9px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background: rgba(255,255,255,0.03) !important;
      }
      .mult-tabs div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        border: 2px solid rgba(80,170,255,0.95) !important;
        background: rgba(80,170,255,0.20) !important;
        box-shadow: inset 0 -4px 0 rgba(80,170,255,0.95) !important;
      }

      .panel {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 12px;
        background: rgba(255,255,255,0.03);
      }
      .grid {
        display: grid;
        grid-template-columns: 0.95fr 1.25fr;
        gap: 12px;
        align-items: start;
      }
      .price-stack { display: grid; gap: 10px; }
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 12px 10px;
        background: rgba(255,255,255,0.02);
        text-align: center;
      }
      .tkr { font-size: 0.82rem; opacity: 0.80; font-weight: 800; letter-spacing: 0.4px; }
      .px  { font-size: 1.05rem; font-weight: 900; margin-top: 4px; }

      .muted { opacity: 0.75; font-size: 0.90rem; }
      .mono { font-variant-numeric: tabular-nums; }
      .out-title { font-size: 0.82rem; opacity: 0.80; font-weight: 800; }
      .out-eq    { font-size: 1.12rem; font-weight: 950; margin-top: 2px; }
      .divider   { height: 1px; background: rgba(255,255,255,0.10); margin: 10px 0; }
      .out-sub   { font-size: 0.90rem; opacity: 0.85; margin-top: 6px; }
      .out-num   { font-size: 0.98rem; font-weight: 900; margin-top: 2px; }

      .disclaimer {
        margin-top: 10px;
        margin-bottom: 14px;
        padding: 10px 12px 18px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.02);
        font-size: 0.82rem;
        opacity: 0.78;
        line-height: 1.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Helpers
# ======================
@st.cache_data(ttl=300)
def fetch_last_close(ticker: str) -> float:
    h = yf.Ticker(ticker).history(period="1d")
    if h is None or h.empty or "Close" not in h.columns:
        raise ValueError("No price history")
    return float(h["Close"].iloc[-1])

@st.cache_data(ttl=300)
def ticker_is_valid(ticker: str) -> bool:
    try:
        h = yf.Ticker(ticker).history(period="1d")
        return (h is not None) and (not h.empty) and ("Close" in h.columns)
    except Exception:
        return False

@st.cache_data(ttl=300)
def fetch_logret_var(ticker: str, lookback_days: int) -> float:
    period_days = max(lookback_days + 30, 60)
    h = yf.Ticker(ticker).history(period=f"{period_days}d")
    if h is None or h.empty or "Close" not in h.columns:
        raise ValueError("No history")

    closes = h["Close"].dropna()
    if len(closes) < lookback_days + 2:
        raise ValueError("Insufficient history")

    closes = closes.tail(lookback_days + 1)
    lr = (closes / closes.shift(1)).apply(lambda x: math.log(x) if x and x > 0 else float("nan")).dropna()
    var = float(lr.var(ddof=1))
    if not (var >= 0):
        raise ValueError("Bad variance")
    return var

def calc_equivalent(source_price, target_price, other_price, leverage, source_is_base: bool):
    pct_move = (target_price - source_price) / source_price
    adjusted_move = pct_move * leverage if source_is_base else pct_move / leverage
    eq_price = other_price * (1 + adjusted_move)
    return pct_move, adjusted_move, eq_price

def parse_float(s: str):
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# Short labels so they don't clip in a narrow column
LOOKBACK_MAP = {"20D": 20, "60D": 60, "120D": 120}

# ======================
# Defaults + State (same keys as your original for Clear All)
# ======================
DEFAULT_BASE = ""
DEFAULT_LEV = ""

if "base_in" not in st.session_state: st.session_state.base_in = DEFAULT_BASE
if "lev_in" not in st.session_state: st.session_state.lev_in = DEFAULT_LEV
if "target_in" not in st.session_state: st.session_state.target_in = ""
if "base_ph" not in st.session_state: st.session_state.base_ph = "Input stock ticker"
if "lev_ph" not in st.session_state: st.session_state.lev_ph = "Input leveraged equivalent"
if "mult_in" not in st.session_state: st.session_state.mult_in = "2X"
if st.session_state.mult_in not in ("2X", "3X"): st.session_state.mult_in = "2X"
if "target_src" not in st.session_state: st.session_state.target_src = DEFAULT_BASE

if "decay_on" not in st.session_state: st.session_state.decay_on = False
if "decay_horizon" not in st.session_state: st.session_state.decay_horizon = 21
if "decay_lookback_label" not in st.session_state: st.session_state.decay_lookback_label = "60D"
if st.session_state.decay_lookback_label not in LOOKBACK_MAP: st.session_state.decay_lookback_label = "60D"

# ======================
# Callbacks
# ======================
def on_clear_all():
    st.session_state.base_in = DEFAULT_BASE
    st.session_state.lev_in = DEFAULT_LEV
    st.session_state.target_in = ""
    st.session_state.mult_in = "2X"
    st.session_state.target_src = DEFAULT_BASE

    st.session_state.base_ph = "Input stock ticker"
    st.session_state.lev_ph = "Input leveraged equivalent"

    st.session_state.decay_on = False
    st.session_state.decay_horizon = 21
    st.session_state.decay_lookback_label = "60D"

    fetch_last_close.clear()
    ticker_is_valid.clear()
    fetch_logret_var.clear()

def on_validate_base():
    val = (st.session_state.get("base_in") or "").strip().upper()
    st.session_state.base_in = val
    if val and not ticker_is_valid(val):
        st.session_state.base_in = ""
        st.session_state.base_ph = "Ticker not found"
    else:
        st.session_state.base_ph = "Input stock ticker"

def on_validate_lev():
    val = (st.session_state.get("lev_in") or "").strip().upper()
    st.session_state.lev_in = val
    if val and not ticker_is_valid(val):
        st.session_state.lev_in = ""
        st.session_state.lev_ph = "Ticker not found"
    else:
        st.session_state.lev_ph = "Input leveraged equivalent"

# ======================
# UI
# ======================
st.header("📈 Leverage Equivalence Calculator")
st.caption("Set a target price on one ticker → see the equivalent target price on the leveraged peer.")

with st.container(border=True):

    c1, c2, c3 = st.columns([1.15, 1.15, 1.2])

    with c1:
        st.text_input("Stock", key="base_in", placeholder=st.session_state.base_ph, on_change=on_validate_base)

    with c2:
        st.text_input("Leveraged", key="lev_in", placeholder=st.session_state.lev_ph, on_change=on_validate_lev)

    with c3:
        st.markdown('<div class="mult-tabs">', unsafe_allow_html=True)
        st.segmented_control("Multiplier", options=["2X", "3X"], key="mult_in", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    leverage = 2.0 if st.session_state.mult_in == "2X" else 3.0

    st.divider()

    left, right = st.columns([1.05, 1.45], vertical_alignment="top")

    base_ticker = (st.session_state.base_in or DEFAULT_BASE).strip().upper()
    lev_ticker = (st.session_state.lev_in or DEFAULT_LEV).strip().upper()

    with left:
        opts = [base_ticker, lev_ticker]
        if st.session_state.target_src not in opts:
            st.session_state.target_src = base_ticker

        st.segmented_control("Set target for:", options=opts, key="target_src")

        st.text_input("Target price", key="target_in", placeholder="Enter target price")

        st.markdown("**Decay (optional)**")
        st.toggle("Decay-adjusted output", key="decay_on")

        with st.expander("Decay settings", expanded=False):
            # Put slider + lookback on one row (wider lookback column)
            d1, d2 = st.columns([0.95, 1.35], gap="medium")
            with d1:
                st.slider(
                    "Horizon (trading days)",
                    min_value=1,
                    max_value=252,
                    value=int(st.session_state.decay_horizon),
                    step=1,
                    key="decay_horizon",
                )
            with d2:
                st.radio(
                    "Vol lookback",
                    options=list(LOOKBACK_MAP.keys()),
                    key="decay_lookback_label",
                    horizontal=False
                )
                st.caption("20D = more reactive • 60D = baseline • 120D = smoother")

    # Prices
    try:
        base_px = fetch_last_close(base_ticker)
        lev_px = fetch_last_close(lev_ticker)
    except Exception:
        st.error("Could not fetch prices. Double-check tickers.")
        base_px = None
        lev_px = None

# ======================
# Compute + Output (restore your "data next to each other" panel)
# ======================
if base_px is not None and lev_px is not None:
    target_price = parse_float(st.session_state.target_in)
    err = None
    result = None
    decay_result = None

    if st.session_state.target_in.strip() != "" and target_price is None:
        err = "Target price must be a number (e.g., 95 or 95.50)."
    elif target_price is not None and target_price <= 0:
        err = "Target price must be greater than 0."
    elif target_price is not None:
        if st.session_state.target_src == base_ticker:
            pct_move, lev_move, eq_price = calc_equivalent(base_px, target_price, lev_px, leverage, True)
            result = {"mode": "base_to_lev", "src": base_ticker, "dst": lev_ticker, "pct": pct_move, "adj": lev_move, "eq": eq_price}
        else:
            pct_move, base_move, eq_price = calc_equivalent(lev_px, target_price, base_px, leverage, False)
            result = {"mode": "lev_to_base", "src": lev_ticker, "dst": base_ticker, "pct": pct_move, "adj": base_move, "eq": eq_price}

        if st.session_state.decay_on:
            N = int(st.session_state.decay_horizon)
            lookback_days = LOOKBACK_MAP.get(st.session_state.decay_lookback_label, 60)
            try:
                var_d = fetch_logret_var(base_ticker, lookback_days=lookback_days)
                drag_log = 0.5 * (leverage**2 - leverage) * var_d * N

                if st.session_state.target_src == base_ticker:
                    log_move_base = math.log(target_price / base_px)
                    eq_price_decay = lev_px * math.exp(leverage * log_move_base - drag_log)
                    decay_result = {"dst": lev_ticker, "eq": eq_price_decay, "N": N, "lookback": lookback_days, "drag_bps": drag_log * 10000.0}
                else:
                    log_move_lev = math.log(target_price / lev_px)
                    eq_price_decay = base_px * math.exp((log_move_lev + drag_log) / leverage)
                    decay_result = {"dst": base_ticker, "eq": eq_price_decay, "N": N, "lookback": lookback_days, "drag_bps": drag_log * 10000.0}
            except Exception:
                decay_result = None

    if err:
        out_html = f'<div class="muted">{esc(err)}</div>'
    elif result is None:
        out_html = '<div class="muted">Enter a target price to see the equivalent.</div>'
    else:
        label2 = f'{int(leverage)}X move' if result["mode"] == "base_to_lev" else f'De-leveraged (÷{int(leverage)}X)'

        # Row 1: Equivalent next to Decay-adjusted
        eq_head = textwrap.dedent(f"""
        <div>
          <div class="out-title mono">Equivalent {esc(result["dst"])}</div>
          <div class="out-eq mono">${result["eq"]:,.2f}</div>
        </div>
        """).strip()

        if st.session_state.decay_on:
            if decay_result is None:
                decay_head = textwrap.dedent("""
                <div>
                  <div class="out-title mono">Decay-adjusted</div>
                  <div class="muted">Unavailable</div>
                </div>
                """).strip()
                assumptions = ""
            else:
                decay_head = textwrap.dedent(f"""
                <div>
                  <div class="out-title mono">Decay-adjusted {esc(decay_result["dst"])}</div>
                  <div class="out-eq mono">${decay_result["eq"]:,.2f}</div>
                </div>
                """).strip()
                assumptions = textwrap.dedent(f"""
                <div>
                  <div class="out-sub mono">Assumptions</div>
                  <div class="out-num mono">Horizon: {decay_result["N"]}d</div>
                  <div class="out-num mono">Vol lookback: {decay_result["lookback"]}d</div>
                  <div class="out-num mono">Vol drag (est.): {decay_result["drag_bps"]:,.0f} bps</div>
                </div>
                """).strip()

            move_block = textwrap.dedent(f"""
            <div>
              <div class="out-sub mono">{esc(result["src"])} move</div>
              <div class="out-num mono">{result["pct"]*100:.2f}%</div>
              <div class="out-sub mono">{esc(label2)}</div>
              <div class="out-num mono">{result["adj"]*100:.2f}%</div>
            </div>
            """).strip()

            out_html = textwrap.dedent(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items:start;">
              <div>{eq_head}</div>
              <div>{decay_head}</div>
              <div style="border-top:1px solid rgba(255,255,255,0.10); padding-top:10px;">{move_block}</div>
              <div style="border-top:1px solid rgba(255,255,255,0.10); padding-top:10px;">{assumptions}</div>
            </div>
            """).strip()
        else:
            move_block = textwrap.dedent(f"""
            <div>
              <div class="divider"></div>
              <div class="out-sub mono">{esc(result["src"])} move</div>
              <div class="out-num mono">{result["pct"]*100:.2f}%</div>
              <div class="out-sub mono">{esc(label2)}</div>
              <div class="out-num mono">{result["adj"]*100:.2f}%</div>
            </div>
            """).strip()
            out_html = (eq_head + move_block)

    panel_html = textwrap.dedent(f"""
    <div class="panel">
      <div class="grid">
        <div class="price-stack">
          <div class="card">
            <div class="tkr mono">{esc(base_ticker)}</div>
            <div class="px mono">${base_px:,.2f}</div>
          </div>
          <div class="card">
            <div class="tkr mono">{esc(lev_ticker)}</div>
            <div class="px mono">${lev_px:,.2f}</div>
          </div>
        </div>
        <div>
          {out_html}
        </div>
      </div>
    </div>
    """).strip()

    # Render in the right column (inside the container scope)
    # If 'right' isn't in scope due to Streamlit reruns, just render normally.
    try:
        with right:
            st.markdown(panel_html, unsafe_allow_html=True)
    except Exception:
        st.markdown(panel_html, unsafe_allow_html=True)

    # Clear button centered under output
    _, center_col, _ = st.columns([3, 1.5, 3])
    with center_col:
        st.button("🧹 Clear", on_click=on_clear_all)

    st.markdown(
        """
        <div class="disclaimer">
          <strong>Disclaimer:</strong> Same-move equivalence only. Leveraged products can diverge over time due to daily reset and path dependency.<br>
          “Decay-adjusted” is a simple estimate using recent realized volatility.
        </div>
        """,
        unsafe_allow_html=True
    )
