import streamlit as st
import yfinance as yf
import textwrap

st.set_page_config(
    page_title="Leverage Equivalence",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* Hide the entire sidebar (container) */
  [data-testid="stSidebar"] { display: none; }

  /* Hide the multipage nav just in case */
  [data-testid="stSidebarNav"] { display: none; }

  /* Hide the little “>” / collapse control */
  [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

def top_nav(active: str = "lev"):
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
          }
          .navbtn.active button {
            border: 2px solid rgba(80,170,255,0.95) !important;
            background: rgba(80,170,255,0.20) !important;
            box-shadow: inset 0 -4px 0 rgba(80,170,255,0.95) !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1, 1], gap="small")

    with c1:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="gex" else ""), unsafe_allow_html=True)
        if st.button("GEX", use_container_width=True):
            st.switch_page("app.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="lev" else ""), unsafe_allow_html=True)
        if st.button("Leverage Equivalence", use_container_width=True):
            st.switch_page("pages/1_Leverage_Equivalence.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="navbtn {}">'.format("active" if active=="cc" else ""), unsafe_allow_html=True)
        if st.button("CC / CSP", use_container_width=True):
            st.switch_page("pages/2_CC_CSP.py")
        st.markdown("</div>", unsafe_allow_html=True)


top_nav(active="lev")

# ======================
# CSS
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

      .btnrow div[data-testid="stButton"] > button {
        height: 44px !important;
        border-radius: 14px !important;
        font-weight: 850 !important;
      }

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

# ======================
# Defaults + State
# ======================
DEFAULT_BASE = ""
DEFAULT_LEV = ""

# Widget values
if "base_in" not in st.session_state:
    st.session_state.base_in = DEFAULT_BASE
if "lev_in" not in st.session_state:
    st.session_state.lev_in = DEFAULT_LEV
if "target_in" not in st.session_state:
    st.session_state.target_in = ""

# Placeholders (inline in input boxes)
if "base_ph" not in st.session_state:
    st.session_state.base_ph = "Input stock ticker"
if "lev_ph" not in st.session_state:
    st.session_state.lev_ph = "Input leveraged equivalent"

# Multiplier widget key
if "mult_in" not in st.session_state:
    st.session_state.mult_in = "2X"
if st.session_state.mult_in not in ("2X", "3X"):
    st.session_state.mult_in = "2X"

# Target source widget key (MUST be a widget key to reset reliably)
if "target_src" not in st.session_state:
    st.session_state.target_src = DEFAULT_BASE

# ======================
# Callbacks
# ======================
def on_refresh():
    fetch_last_close.clear()
    ticker_is_valid.clear()

def on_clear_all():
    # Reset the ACTUAL widget keys
    st.session_state.base_in = DEFAULT_BASE
    st.session_state.lev_in = DEFAULT_LEV
    st.session_state.target_in = ""
    st.session_state.mult_in = "2X"
    st.session_state.target_src = DEFAULT_BASE

    # Reset placeholders
    st.session_state.base_ph = "Input stock ticker"
    st.session_state.lev_ph = "Input leveraged equivalent"

    fetch_last_close.clear()
    ticker_is_valid.clear()
    # No st.rerun() here — Streamlit reruns automatically after callbacks.

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
        st.text_input(
            "Stock",
            key="base_in",
            placeholder=st.session_state.base_ph,
            on_change=on_validate_base,
        )

    with c2:
        st.text_input(
            "Leveraged",
            key="lev_in",
            placeholder=st.session_state.lev_ph,
            on_change=on_validate_lev,
        )

    with c3:
        st.markdown('<div class="mult-tabs">', unsafe_allow_html=True)
        st.segmented_control(
            "Multiplier",
            options=["2X", "3X"],
            key="mult_in",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    leverage = 2.0 if st.session_state.mult_in == "2X" else 3.0

    st.divider()

    left, right = st.columns([1.05, 1.45], vertical_alignment="top")

    base_ticker = (st.session_state.base_in or DEFAULT_BASE).strip().upper()
    lev_ticker = (st.session_state.lev_in or DEFAULT_LEV).strip().upper()

    with left:
        opts = [base_ticker, lev_ticker]

        # Keep target_src valid if tickers changed
        if st.session_state.target_src not in opts:
            st.session_state.target_src = base_ticker

        st.segmented_control(
            "Set target for:",
            options=opts,
            key="target_src",          # <-- key makes Clear All work reliably
        )

        st.text_input(
            "Target price",
            key="target_in",
            placeholder="Enter target price",
        )

    # Prices
    try:
        base_px = fetch_last_close(base_ticker)
        lev_px = fetch_last_close(lev_ticker)
    except Exception:
        st.error("Could not fetch prices. Double-check tickers.")
        base_px = None 
        lev_px = None
  
if base_px is not None and lev_px is not None:

    # Compute
    
    target_price = parse_float(st.session_state.target_in)
    err = None
    result = None

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

    if err:
        out_html = f'<div class="muted">{esc(err)}</div>'
    elif result is None:
        out_html = '<div class="muted">Enter a target price to see the equivalent.</div>'
    else:
        label2 = f'{int(leverage)}X move' if result["mode"] == "base_to_lev" else f'De-leveraged (÷{int(leverage)}X)'
        out_html = textwrap.dedent(f"""
        <div>
          <div class="out-title mono">Equivalent {esc(result["dst"])}</div>
          <div class="out-eq mono">${result["eq"]:,.2f}</div>
          <div class="divider"></div>
          <div class="out-sub mono">{esc(result["src"])} move</div>
          <div class="out-num mono">{result["pct"]*100:.2f}%</div>
          <div class="out-sub mono">{esc(label2)}</div>
          <div class="out-num mono">{result["adj"]*100:.2f}%</div>
        </div>
        """).strip()

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

    with right:
        st.markdown(panel_html, unsafe_allow_html=True)

    # Buttons row
    left_spacer, center_col, right_spacer = st.columns([3, 1.5, 3])

    with center_col:
        st.button(
            "🧹 Clear",
            on_click=on_clear_all
        )

    st.markdown(
        """
        <div class="disclaimer">
          <strong>Disclaimer:</strong> Same-move equivalence only. Leveraged products can diverge over time due to daily reset and path dependency.
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ======================
# Synthetic DCA Put Calculator
# ======================

st.divider()
st.header("🧮 Synthetic Put DCA Calculator")
st.caption("Debit put spread + additional CSP to control dollar-cost averaging.")

def small_metric(label, value):
    st.markdown(
        f"""
        <div style="margin-top:4px">
            <div style="font-size:0.8rem; opacity:0.75;">{label}</div>
            <div style="font-size:1.05rem; font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def reset_dca_inputs():
    keys = [
        "long_strike_txt", "long_prem_txt",
        "short2_strike_txt", "short2_prem_txt",
        "short3_strike_txt", "short3_prem_txt",
    ]
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = ""  # reset back to empty so placeholders show

with st.container(border=True):

    c1, c2, c3 = st.columns(3)

    def money_or_none(x):
        return None if x is None else f"${x:,.2f}"

    with c1:
        st.subheader("Long Put")
        long_strike_txt = st.text_input("Strike", value="", placeholder="Input strike price", key="long_strike_txt")
        long_prem_txt   = st.text_input("Premium (paid)", value="", placeholder="Input Premium Paid", key="long_prem_txt")

    with c2:
        st.subheader("Short Put")
        short2_strike_txt = st.text_input("Strike ", value="", placeholder="Input strike price", key="short2_strike_txt")
        short2_prem_txt   = st.text_input("Premium (received)", value="", placeholder="Input premium received", key="short2_prem_txt")

    with c3:
        st.subheader("Short Put (CSP)")
        short3_strike_txt = st.text_input("Strike  ", value="", placeholder="Input strike price", key="short3_strike_txt")
        short3_prem_txt   = st.text_input("Premium (received)", value="", placeholder="Input premium received", key="short3_prem_txt")

    # Parse inputs (use your existing helper)
    long_strike  = parse_float(long_strike_txt)
    long_prem    = parse_float(long_prem_txt)
    short2_strike = parse_float(short2_strike_txt)
    short2_prem   = parse_float(short2_prem_txt)
    short3_strike = parse_float(short3_strike_txt)
    short3_prem   = parse_float(short3_prem_txt)

    st.divider()

    MULT = 100

    # Validate
    vals = [long_strike, long_prem, short2_strike, short2_prem, short3_strike, short3_prem]
    if any(v is None for v in vals):
        st.info("Enter valid numbers for all strikes and premiums (e.g., 50, 2.15).")
    else:
        # ---------- Scenario 1 ----------
        # Stock > long put strike: net premium collected = (opt2 + opt3) - opt1
        net_premium_s1 = (short2_prem + short3_prem - long_prem) * MULT

        # ---------- Scenario 2 ----------
        # Instead of a single value (max), show the range across S in [K3, K1]
        def total_pl_at_expiry(S: float) -> float:
            # premiums: +P2 +P3 -P1
            prem = (short2_prem + short3_prem - long_prem) * MULT
            # option payoffs at expiry (1 contract)
            long_pay = max(long_strike - S, 0.0) * MULT
            short2_pay = -max(short2_strike - S, 0.0) * MULT
            # option 3 payoff is 0 in Scenario 2 band (S >= K3), so omit safely
            return prem + long_pay + short2_pay

        lo_S = short3_strike
        hi_S = long_strike

        candidates = [lo_S, hi_S]
        if lo_S <= short2_strike <= hi_S:
            candidates.append(short2_strike)

        vals = [total_pl_at_expiry(s) for s in candidates]
        s2_low, s2_high = min(vals), max(vals)

        # Keep a representative "mid" value if you still want one (optional)
        # net_premium_s2_mid = total_pl_at_expiry((lo_S + hi_S) / 2.0)


        # ---------- Scenario 3 ----------
        # Assignment at strike3.
        # Include TOTAL "premium gained" from the DEBIT SPREAD (its max value) + CSP premium.
        debit_spread_max_value = (long_strike - short2_strike) * MULT - (long_prem - short2_prem) * MULT
        total_premiums_s3 = short3_prem * MULT + debit_spread_max_value

        assignment_cost = short3_strike * MULT
        net_cost_basis = assignment_cost - total_premiums_s3

                # ---------- Display ----------
        s1_label = f"Stock > ${long_strike:g}"
        s2_label = f"Stock between \\${short3_strike:g} and \\${long_strike:g}"
        s3_label = f"Assignment at ${short3_strike:g}"

        o1, o2, o3 = st.columns(3)

        with o1:
            st.markdown("### 📈 Scenario 1")
            st.markdown(f"**{s1_label}**")
            small_metric("Net Premium Collected", money_or_none(net_premium_s1))

        with o2:
            st.markdown("### ⚖️ Scenario 2")
            st.write(f"Stock between \\${short3_strike:g} and \\${long_strike:g}")
            small_metric("Total Premium Range", f"{money_or_none(s2_low)}  to  {money_or_none(s2_high)}")


        with o3:
            st.markdown("### 📉 Scenario 3")
            st.markdown(f"**{s3_label}**")
            small_metric("Total Premiums (Spread + CSP)", money_or_none(total_premiums_s3))
            small_metric("Net Cost Basis at Entry", money_or_none(net_cost_basis))

        st.divider()

        left_spacer, center_col, right_spacer = st.columns([3, 1.5, 3])

        with center_col:
            st.button(
            "🧹 Reset",
            use_container_width=True,
            on_click=reset_dca_inputs
        )

        st.markdown(
            f"""
            <div class="disclaimer">
            <strong>Breakdown (Scenario 3):</strong><br>
            • Debit spread max value: <span class="mono">{money_or_none(debit_spread_max_value)}</span><br>
            • CSP premium (Option 3): <span class="mono">{money_or_none(short3_prem * MULT)}</span><br>
            • Assigned shares cost (at {short3_strike:g}): <span class="mono">{money_or_none(assignment_cost)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )



