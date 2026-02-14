
import streamlit as st
import textwrap

st.set_page_config(
    page_title="Synthetic Put DCA",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { display: none; }
  [data-testid="stSidebarNav"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

def top_nav(active: str = "dca"):
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

top_nav(active="dca")

st.markdown(
    """
    <style>
      div[data-baseweb="input"] input {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        font-size: 14px !important;
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

def money_or_none(x):
    return None if x is None else f"${x:,.2f}"

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

def reset_inputs():
    for k in ("long_strike_txt", "long_prem_txt", "short_strike_txt", "short_prem_txt"):
        if k in st.session_state:
            st.session_state[k] = ""

st.header("🧮 Synthetic Put DCA Calculator")
st.caption("Debit put spread + additional CSP to control dollar-cost averaging (1 long put + 2 short puts at the same strike).")

with st.container(border=True):
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Long Put")
        long_strike_txt = st.text_input("Strike", value="", placeholder="Input strike price", key="long_strike_txt")
        long_prem_txt   = st.text_input("Premium (paid)", value="", placeholder="Input premium paid", key="long_prem_txt")

    with c2:
        st.subheader("Short Put (2x)")
        short_strike_txt = st.text_input("Strike", value="", placeholder="Input strike price", key="short_strike_txt")
        short_prem_txt   = st.text_input("Premium (received)", value="", placeholder="Input premium received", key="short_prem_txt")

    long_strike = parse_float(long_strike_txt)
    long_prem   = parse_float(long_prem_txt)
    short_strike = parse_float(short_strike_txt)
    short_prem   = parse_float(short_prem_txt)

    st.divider()

    MULT = 100
    SHORT_QTY = 2  # spread short + CSP short

    if any(v is None for v in (long_strike, long_prem, short_strike, short_prem)):
        st.info("Enter valid numbers for all strikes and premiums (e.g., 50, 2.15).")
    else:
        # Scenario 1: stock > long strike
        net_premium_s1 = ((SHORT_QTY * short_prem) - long_prem) * MULT

        # Scenario 2: stock between short strike and long strike -> show premium range
        def total_pl_at_expiry(S: float) -> float:
            prem = ((SHORT_QTY * short_prem) - long_prem) * MULT
            long_pay = max(long_strike - S, 0.0) * MULT
            return prem + long_pay

        lo_S = short_strike
        hi_S = long_strike
        s2_vals = [total_pl_at_expiry(lo_S), total_pl_at_expiry(hi_S)]
        s2_low, s2_high = min(s2_vals), max(s2_vals)

        # Scenario 3: assignment is still 100 shares (CSP short put)
        assignment_cost = short_strike * MULT

        debit_spread_max_value = (long_strike - short_strike) * MULT - (long_prem - short_prem) * MULT
        extra_short_premium = short_prem * MULT  # the second short put premium (CSP)

        total_premiums_s3 = debit_spread_max_value + extra_short_premium
        net_cost_basis_total = assignment_cost - total_premiums_s3
        net_cost_basis_per_share = net_cost_basis_total / MULT

        o1, o2, o3 = st.columns(3)
        with o1:
            st.markdown("### 📈 Scenario 1")
            st.markdown(f"**Stock > ${long_strike:g}**")
            small_metric("Net Premium Collected", money_or_none(net_premium_s1))

        with o2:
            st.markdown("### ⚖️ Scenario 2")
            st.write(f"Stock between \\${short_strike:g} and \\${long_strike:g}")
            small_metric("Total Premium Range", f"{money_or_none(s2_low)}  to  {money_or_none(s2_high)}")

        with o3:
            st.markdown("### 📉 Scenario 3")
            st.markdown(f"**Assignment at ${short_strike:g} (100 shares)**")
            small_metric("Total Premiums (Spread + Extra Short)", money_or_none(total_premiums_s3))
            small_metric("Net Cost Basis (total)", money_or_none(net_cost_basis_total))
            small_metric("Net Cost Basis (per share)", money_or_none(net_cost_basis_per_share))

        st.divider()
        left_spacer, center_col, right_spacer = st.columns([3, 1.5, 3])
        with center_col:
            st.button("🧹 Reset", use_container_width=True, on_click=reset_inputs)

        st.markdown(
            f"""
            <div class="disclaimer">
            <strong>Breakdown (Scenario 3):</strong><br>
            • Debit spread max value: <span class="mono">{money_or_none(debit_spread_max_value)}</span><br>
            • Extra short put premium (CSP): <span class="mono">{money_or_none(extra_short_premium)}</span><br>
            • Assigned shares cost (100 @ {short_strike:g}): <span class="mono">{money_or_none(assignment_cost)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
# ==============
# DCA Calculator
# ==============

st.divider()
st.header("💰 Dollar Cost Average (DCA) Calculator")
st.caption("Simple weighted-average update after a new buy.")

# --- Reset mechanism (safe) ---
if st.session_state.get("dca_reset_pending", False):
    for k in ("dca_total_shares_txt", "dca_avg_cost_txt", "dca_new_shares_txt", "dca_new_cost_txt"):
        st.session_state[k] = ""
    st.session_state["dca_reset_pending"] = False

with st.container(border=True):

    d1, d2 = st.columns(2)

    with d1:
        total_shares_txt = st.text_input(
            "Shares Owned",
            value=st.session_state.get("dca_total_shares_txt", ""),
            placeholder="Enter share quantity",
            key="dca_total_shares_txt",
        )
        avg_cost_txt = st.text_input(
            "Average Cost Basis (Cost/share)",
            value=st.session_state.get("dca_avg_cost_txt", ""),
            placeholder="Enter cost per share",
            key="dca_avg_cost_txt",
        )

    with d2:
        new_shares_txt = st.text_input(
            "New Shares Quantity",
            value=st.session_state.get("dca_new_shares_txt", ""),
            placeholder="Enter share quantity",
            key="dca_new_shares_txt",
        )
        new_cost_txt = st.text_input(
            "New share price",
            value=st.session_state.get("dca_new_cost_txt", ""),
            placeholder="Enter purchase price",
            key="dca_new_cost_txt",
        )

    st.divider()

    total_shares = parse_float(total_shares_txt)
    avg_cost = parse_float(avg_cost_txt)
    new_shares = parse_float(new_shares_txt)
    new_cost = parse_float(new_cost_txt)

    if any(v is None for v in (total_shares, avg_cost, new_shares, new_cost)):
        st.info("Enter valid numbers for all fields.")
    elif total_shares < 0 or new_shares < 0 or avg_cost < 0 or new_cost < 0:
        st.warning("Values must be non-negative.")
    else:
        old_total_cost = total_shares * avg_cost
        new_total_cost = new_shares * new_cost
        combined_shares = total_shares + new_shares

        if combined_shares == 0:
            st.warning("Total shares after purchase is 0. Nothing to calculate.")
        else:
            new_avg = (old_total_cost + new_total_cost) / combined_shares

            m1, m2, m3 = st.columns(3)
            with m1:
                small_metric("Current Cost Basis", money_or_none(old_total_cost))
            with m2:
                small_metric("New Additional Basis", money_or_none(new_total_cost))
            with m3:
                small_metric("New Average Cost Basis", money_or_none(new_avg))

            st.caption(f"New total shares quantity: **{combined_shares:,.0f}**")

            # Reset appears ONLY after valid calculation / metrics render
            sp1, mid, sp2 = st.columns([4, 1, 4])
            with mid:
                if st.button("Reset", key="dca_reset_btn", use_container_width=True):
                    st.session_state["dca_reset_pending"] = True
                    st.rerun()

