import streamlit as st
import pandas as pd
import numpy as np
from math import log, sqrt
from scipy.stats import norm
from options.data import get_options_view_df
from streamlit_javascript import st_javascript

screen_w = st_javascript("window.innerWidth")
is_mobile = False if screen_w is None else screen_w <= 768

# Optional (auto spot pull)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# Charting
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False


# =========================
# App config
# =========================
st.set_page_config(page_title="Gamma Exposure (GEX)", layout="wide")
st.title("Gamma Exposure (GEX)")


# =========================
# Hidden defaults
# =========================
DEFAULT_RISK_FREE = 0.05
DEFAULT_MULTIPLIER = 100
STRIKES_WINDOW_DEFAULT = 42  # only 30 strikes closest to spot

# Chart sizing + styling
CHART_BG = "#2b2b2b"         # dark gray background
GRID_COLOR = "#3a3a3a"       # subtle gridlines
AXIS_TEXT = "#cfcfcf"
TITLE_TEXT = "#e6e6e6"

# More background space so labels fit nicely
CHART_PADDING = {"left": 20, "right": 30, "top": 20, "bottom": 20}

# =========================
# Cache: options chain
# =========================
@st.cache_data(ttl=300)
def fetch_chain(symbol: str) -> pd.DataFrame:
    _, raw = get_options_view_df(symbol)
    df = raw.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


# =========================
# Cache: spot pull (robust)
# =========================
@st.cache_data(ttl=30)
def fetch_spot_yahoo(symbol: str) -> float:
    if not YF_OK:
        raise RuntimeError("yfinance not installed")

    t = yf.Ticker(symbol)

    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            for key in ["last_price", "lastPrice"]:
                if key in fi and fi[key] is not None:
                    v = float(fi[key])
                    if v > 0:
                        return v
    except Exception:
        pass

    # 1m intraday
    try:
        intraday = t.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty and "Close" in intraday.columns:
            v = float(intraday["Close"].dropna().iloc[-1])
            if v > 0:
                return v
    except Exception:
        pass

    # daily close fallback
    hist = t.history(period="5d", interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise RuntimeError("No price history returned (Yahoo may be blocking or symbol not found).")
    v = float(hist["Close"].dropna().iloc[-1])
    if v <= 0:
        raise RuntimeError("Invalid price returned from Yahoo.")
    return v


# =========================
# Black–Scholes Gamma
# =========================
def bs_gamma(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))


def to_years(expiry_str):
    exp_dt = pd.to_datetime(expiry_str, errors="coerce")
    if pd.isna(exp_dt):
        return 0.0
    now = pd.Timestamp.now()
    secs = (exp_dt - now).total_seconds()
    return max(secs, 0.0) / (365.0 * 24 * 3600)


# =========================
# Snapshot UI (INLINE HTML ONLY — reliable in Streamlit)
# =========================
def snap_cell(label: str, value: str, label_px=9, value_px=14, wrap_value=False):
    white_space = "normal" if wrap_value else "nowrap"
    return f"""
    <div style="
      padding:6px 8px;
      border-radius:10px;
      background: rgba(255,255,255,0.03);
      border:1px solid rgba(255,255,255,0.06);
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
      ">{label}</div>

      <div style="
        font-size:{value_px}px;
        font-weight:700;
        line-height:1.15;
        white-space:{white_space};
        overflow:hidden;
        text-overflow:ellipsis;
        word-break:break-word;
      ">{value}</div>
    </div>
    """


def tile_grid(items):
    html = """
    <div style="
      display:grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap:10px 14px;
      margin-top:6px;
      margin-bottom:8px;
    ">
    """
    for it in items:
        html += snap_cell(
            it["label"],
            it["value"],
            label_px=9,
            value_px=14,
            wrap_value=it.get("wrap", False),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Levels / stats
# =========================
def compute_levels_by_strike(gex_df):
    if gex_df.empty:
        return {}
    gex_df = gex_df.sort_values("strike").reset_index(drop=True)
    net_gex = float(gex_df["dealer_gex"].sum())
    put_wall = float(gex_df.loc[gex_df["dealer_gex"].idxmax(), "strike"])
    call_wall = float(gex_df.loc[gex_df["dealer_gex"].idxmin(), "strike"])
    return {"net_gex": net_gex, "put_wall": put_wall, "call_wall": call_wall}


def compute_zero_gamma_level(gex_by_strike: pd.DataFrame):
    if gex_by_strike is None or gex_by_strike.empty:
        return None

    g = gex_by_strike.sort_values("strike").reset_index(drop=True)
    y = g["dealer_gex"].values
    x = g["strike"].values

    zeros = np.where(y == 0)[0]
    if len(zeros) > 0:
        return float(x[zeros[0]])

    s = np.sign(y)
    for i in range(1, len(s)):
        if s[i] == 0:
            return float(x[i])
        if s[i - 1] == 0:
            return float(x[i - 1])
        if s[i] != s[i - 1]:
            x0, y0 = x[i - 1], y[i - 1]
            x1, y1 = x[i], y[i]
            if (y1 - y0) == 0:
                return float(x0)
            return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return None


def compute_gex_snapshot_stats(per_exp_parts: list[pd.DataFrame]):
    if not per_exp_parts:
        return {"total_call_gex": 0.0, "total_put_gex": 0.0, "gex_ratio": None}

    allp = pd.concat(per_exp_parts, ignore_index=True)
    total_call_gex = float(allp["call_gex"].sum()) if "call_gex" in allp.columns else 0.0
    total_put_gex = float(allp["put_gex"].sum()) if "put_gex" in allp.columns else 0.0
    denom = total_call_gex + total_put_gex
    gex_ratio = (total_call_gex / denom) if denom > 0 else None
    return {"total_call_gex": total_call_gex, "total_put_gex": total_put_gex, "gex_ratio": gex_ratio}


def compute_extreme_oi_for_summary(per_exp_parts: list[pd.DataFrame], gex_by_strike: pd.DataFrame):
    out = {
        "neg_strike": None,
        "pos_strike": None,
        "call_oi_at_neg": None,
        "put_oi_at_pos": None,
        "zero_gamma": None,
    }
    if not per_exp_parts or gex_by_strike is None or gex_by_strike.empty:
        return out

    allp = pd.concat(per_exp_parts, ignore_index=True)
    oi_by_strike = allp.groupby("strike", as_index=False).agg(
        call_oi=("call_oi", "sum"),
        put_oi=("put_oi", "sum"),
    )

    gg = gex_by_strike.copy().sort_values("strike").reset_index(drop=True)
    neg_strike = float(gg.loc[gg["dealer_gex"].idxmin(), "strike"])
    pos_strike = float(gg.loc[gg["dealer_gex"].idxmax(), "strike"])

    out["neg_strike"] = neg_strike
    out["pos_strike"] = pos_strike

    row_neg = oi_by_strike[oi_by_strike["strike"] == neg_strike]
    row_pos = oi_by_strike[oi_by_strike["strike"] == pos_strike]
    if not row_neg.empty:
        out["call_oi_at_neg"] = float(row_neg["call_oi"].iloc[0])
    if not row_pos.empty:
        out["put_oi_at_pos"] = float(row_pos["put_oi"].iloc[0])

    out["zero_gamma"] = compute_zero_gamma_level(gex_by_strike)
    return out


def format_big(n):
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "—"
    return f"{n:,.0f}"


# =========================
# Chart (horizontal bars like your screenshot)
# =========================
def render_chart(gex_all: pd.DataFrame, spot: float, chart_title: str):
    """
    Horizontal dealer hedge intensity by Strike.

    X-axis: dealer hedge (shares per $1 move), approximated as:
        shares_per_1d = dealer_gex / spot^2
    """
    if gex_all is None or gex_all.empty:
        st.warning("No GEX data to chart.")
        return

    if not ALTAIR_OK:
        st.warning("Altair not installed. Run: python -m pip install altair")
        g = gex_all.copy()
        g["dist"] = (g["strike"] - spot).abs()
        g = (
            g.sort_values("dist")
            .head(int(STRIKES_WINDOW_DEFAULT))
            .sort_values("strike")
            .reset_index(drop=True)
        )
        st.bar_chart(g.set_index("strike")[["dealer_gex"]])
        return

    g = gex_all.copy()
    g["strike"] = pd.to_numeric(g["strike"], errors="coerce")
    g["dealer_gex"] = pd.to_numeric(g["dealer_gex"], errors="coerce")
    g = g.dropna(subset=["strike", "dealer_gex"]).copy()

    # only N strikes closest to spot
    g["dist"] = (g["strike"] - spot).abs()
    g = (
        g.sort_values("dist")
        .head(int(STRIKES_WINDOW_DEFAULT))
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # Create an ordinal/banded strike label (prevents hairline bars)
    g["strike_lbl"] = g["strike"].map(lambda x: f"{x:.2f}")
    
    # Convert Dealer GEX -> shares per $1 move
    denom = max(float(spot) ** 2, 1e-9)
    g["shares_per_1d"] = (g["dealer_gex"] / denom).replace([np.inf, -np.inf], np.nan)

    g["bar_color"] = np.where(g["shares_per_1d"] >= 0, "pos", "neg")

    abs_max = float(np.abs(g["shares_per_1d"]).max()) if len(g) else 1.0
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0

    # X axis domain: positive on the LEFT, negative on the RIGHT (matches your screenshot)
    x_domain = [abs_max, -abs_max]

    # ---------- Spacing controls ----------
    n = len(g)
    # Give each strike more vertical room so bars don't cluster
    chart_h = int(np.clip(n * 22, 560, 1300))
    # Make bars a bit thinner so there's visible gap between rows
    bar_size = int(np.clip(12 - (n / 12), 6, 10))

    x_axis = alt.Axis(
        title="Dealer Hedge (Shares per $1 move)",
        grid=True,
        gridColor=GRID_COLOR,
        gridOpacity=0.55,
        tickColor=GRID_COLOR,
        labelColor=AXIS_TEXT,
        titleColor=AXIS_TEXT,
        domainColor=GRID_COLOR,
        labelPadding=6,
        ticks=False,
        labelExpr="replace(format(datum.value, '~s'), 'k', 'K')",
    )
    y_axis = alt.Axis(
        title="Strike",
        grid=False,
        labelColor=AXIS_TEXT,
        titleColor=AXIS_TEXT,
        tickColor=GRID_COLOR,
        domainColor=GRID_COLOR,
        ticks=False,
    )

    bars = alt.Chart(g).mark_bar(size=bar_size).encode(
        x=alt.X(
            "shares_per_1d:Q",
            title="Dealer Hedge(Shares per $1 move)",
            axis=x_axis,
            scale=alt.Scale(domain=x_domain),
        ),
        y=alt.Y(
            "strike_lbl:N",
            title="Strike",
            sort=alt.SortField(field="strike", order="descending"),
            axis=y_axis,
            # This is the KEY to more separation between bars
            scale=alt.Scale(
                paddingInner=0.55, 
                paddingOuter=0.30,
                reverse=True
            ),
        ),
        color=alt.Color(
            "bar_color:N",
            scale=alt.Scale(domain=["pos", "neg"], range=["#D00000", "#00A000"]),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("strike:Q", format=".2f", title="Strike"),
            alt.Tooltip("shares_per_1d:Q", format=",.0f", title="Shares per $1"),
            alt.Tooltip("dealer_gex:Q", format=",.0f", title="Dealer GEX (Raw)"),
        ],
    ).properties(height=chart_h)

    # Center zero line
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#8a8a8a", opacity=0.9, strokeWidth=2
    ).encode(x="x:Q")

    # Spot line: snap to nearest displayed strike and run across the chart
    spot_row = g.loc[(g["strike"] - spot).abs().idxmin()]
    spot_lbl = str(spot_row["strike_lbl"])

    spot_rule_df = pd.DataFrame({"x1": [abs_max * 1.00], "x2": [-abs_max * 1.00], "y": [spot_lbl]})
    spot_rule = alt.Chart(spot_rule_df).mark_rule(
        strokeDash=[6, 6],
        strokeWidth=2,
        opacity=0.9,
        color="white"
    ).encode(
        x="x1:Q",
        x2="x2:Q",
        y=alt.Y("y:N", sort=alt.SortField(field="y", order="descending")),
    )

    # Spot label on far right
    spot_label_mult = 1.50 if is_mobile else 1.12
    spot_label_df = pd.DataFrame({
        "x": [-abs_max * spot_label_mult], 
        "y": [spot_lbl], 
        "label": [f"{spot:.2f}"]
        })
    spot_label = alt.Chart(spot_label_df).mark_text(
        align="right",
        baseline="middle",
        dx=8,
        color="white",
        fontSize=12,
        fontWeight="bold"
    ).encode(
        x="x:Q",
        y=alt.Y("y:N", sort=alt.SortField(field="y", order="descending")),
        text="label:N"
    )

    # Title at the TOP of the chart (centered)
    title_df = pd.DataFrame({"x": [0.0], "t": [chart_title]})
    title_layer = alt.Chart(title_df).mark_text(
        align="center",
        baseline="top",
        dy=-30,
        color=TITLE_TEXT,
        fontSize=16,
        fontWeight="bold",
        opacity=0.95,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_domain)),
        y=alt.value(0),
        text="t:N",
    )

    chart = (bars + zero_line + spot_rule + spot_label + title_layer).properties(
        padding=CHART_PADDING
    ).configure_view(
        strokeWidth=0,
        fill=CHART_BG,
    ).configure(
        background=CHART_BG
    )

    st.altair_chart(chart, use_container_width=True)




# =========================
# Expiry selection helpers
# =========================
def key_for_expiry(ticker: str, exp: str) -> str:
    return f"expchk::{ticker}::{exp}"


def ensure_expiry_state(ticker: str, expiries: list[str], default_checked: bool):
    h = hash(tuple(expiries))
    hkey = f"expiry_hash::{ticker}"
    if st.session_state.get(hkey) != h:
        st.session_state[hkey] = h
        for e in expiries:
            st.session_state[key_for_expiry(ticker, e)] = default_checked

    for e in expiries:
        k = key_for_expiry(ticker, e)
        if k not in st.session_state:
            st.session_state[k] = default_checked


def set_all(ticker: str, expiries: list[str], value: bool):
    for e in expiries:
        st.session_state[key_for_expiry(ticker, e)] = value


# =========================
# Layout
# =========================
left, right = st.columns([1, 2])

with left:
    ticker = st.text_input("Ticker", "SPY").upper().strip()

    bucket = st.radio("Expiration", ["Nearest expiration date", "All expiries"], horizontal=False)

    if not YF_OK:
        st.error("Auto spot requires yfinance. Install: python -m pip install yfinance")
        st.stop()

    with st.spinner("Updating data…"):
        spot = fetch_spot_yahoo(ticker)
        raw = fetch_chain(ticker)

        need = ["option_type", "expire_date", "strike", "open_interest", "imp_vol"]
        missing = [c for c in need if c not in raw.columns]
        if missing:
            st.error(f"Missing columns from feed: {missing}")
            st.write("Columns found:", list(raw.columns))
            st.stop()

        df = raw[need].rename(columns={
            "expire_date": "expiry",
            "option_type": "type",
            "open_interest": "oi",
            "imp_vol": "iv"
        }).copy()

        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)

        df["iv"] = df["iv"].astype(str).str.replace("%", "", regex=False)
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce").fillna(0)
        if df["iv"].median() > 1:
            df["iv"] = df["iv"] / 100.0

        df = df.dropna(subset=["strike", "expiry"]).copy()
        df["expiry_str"] = df["expiry"].astype(str)
        df["type"] = df["type"].astype(str).str.lower()

        exp_tbl = (
            df[["expiry_str"]]
            .drop_duplicates()
            .assign(T=lambda x: x["expiry_str"].apply(to_years))
            .assign(DTE=lambda x: x["T"] * 365.0)
            .sort_values("T")
            .reset_index(drop=True)
        )

        dte_map = dict(zip(exp_tbl["expiry_str"].astype(str), exp_tbl["DTE"].astype(float)))
        future_exp = exp_tbl.loc[exp_tbl["T"] > 0, "expiry_str"].astype(str).tolist()
        all_expiries = future_exp if future_exp else exp_tbl["expiry_str"].astype(str).tolist()

        if not all_expiries:
            st.error("No expiries returned for this ticker.")
            st.stop()

    chosen_expiries = []
    next_expiry_used = ""

    if bucket == "Nearest expiration date":
        next_expiry_used = all_expiries[0]
        chosen_expiries = [next_expiry_used]
        dte0 = dte_map.get(next_expiry_used, np.nan)
        st.caption(
            f"Using nearest expiration: **{next_expiry_used} ({int(round(dte0))} DTE)**"
            if np.isfinite(dte0) else f"Using nearest expiration: **{next_expiry_used}**"
        )
    else:
        ensure_expiry_state(ticker, all_expiries, default_checked=False)

        st.markdown("**Expiries**")
        c1, c2 = st.columns(2)
        if c1.button("Select all", use_container_width=True):
            set_all(ticker, all_expiries, True)
        if c2.button("Clear all", use_container_width=True):
            set_all(ticker, all_expiries, False)

        with st.expander("Select expiries", expanded=False):
            q = st.text_input(
                "Search expiry (YYYY-MM-DD)",
                value="",
                placeholder="e.g., 2026-01",
                key=f"exp_search::{ticker}"
            )
            filtered = [e for e in all_expiries if q.strip() in e] if q.strip() else all_expiries

            box = st.container(height=260)
            for e in filtered:
                dte = dte_map.get(e, np.nan)
                label = f"{e} ({int(round(dte))} DTE)" if np.isfinite(dte) else e
                box.checkbox(label, key=key_for_expiry(ticker, e))

        chosen_expiries = [e for e in all_expiries if st.session_state.get(key_for_expiry(ticker, e), False)]
        st.caption(f"Selected: **{len(chosen_expiries)}** expiries")

        if not chosen_expiries:
            st.info("Pick at least one expiry to compute GEX.")
            st.stop()

    with st.spinner("Computing GEX…"):
        parts = []
        per_exp_parts = []

        for exp in chosen_expiries:
            dfe = df[df["expiry_str"] == exp].copy()
            T = to_years(exp)

            dfe["gamma"] = dfe.apply(
                lambda r0: bs_gamma(spot, r0["strike"], T, DEFAULT_RISK_FREE, r0["iv"]),
                axis=1
            )

            calls = dfe[dfe["type"].str.contains("call")].copy()
            puts = dfe[dfe["type"].str.contains("put")].copy()

            call_agg = calls.groupby("strike").apply(
                lambda x: pd.Series({
                    "call_oi": x["oi"].sum(),
                    "call_gamma": (x["gamma"] * x["oi"]).sum() / max(x["oi"].sum(), 1),
                })
            ).reset_index()

            put_agg = puts.groupby("strike").apply(
                lambda x: pd.Series({
                    "put_oi": x["oi"].sum(),
                    "put_gamma": (x["gamma"] * x["oi"]).sum() / max(x["oi"].sum(), 1),
                })
            ).reset_index()

            gex = pd.merge(call_agg, put_agg, on="strike", how="outer").fillna(0)

            gex["call_gex"] = (gex["call_gamma"] * gex["call_oi"]) * DEFAULT_MULTIPLIER * (spot ** 2)
            gex["put_gex"] = (gex["put_gamma"] * gex["put_oi"]) * DEFAULT_MULTIPLIER * (spot ** 2)
            gex["dealer_gex"] = gex["put_gex"] - gex["call_gex"]

            parts.append(gex[["strike", "dealer_gex"]])
            per_exp_parts.append(gex[["strike", "call_gex", "put_gex", "call_oi", "put_oi"]])

        gex_all = pd.concat(parts, ignore_index=True).groupby("strike", as_index=False)["dealer_gex"].sum()
        gex_all = gex_all.sort_values("strike").reset_index(drop=True)

        levels = compute_levels_by_strike(gex_all)
        snap = compute_gex_snapshot_stats(per_exp_parts)
        levels_oi = compute_extreme_oi_for_summary(per_exp_parts, gex_all)

    # =========================
    # Snapshot + Levels tiles
    # =========================
    st.markdown("### Snapshot")

    if bucket == "Nearest expiration date":
        dte_disp = dte_map.get(next_expiry_used, np.nan)
        expiry_display = (
            f"{next_expiry_used} ({int(round(dte_disp))} DTE)"
            if np.isfinite(dte_disp) else next_expiry_used
        )
    else:
        dtes = [dte_map.get(e, np.nan) for e in chosen_expiries]
        dtes = [d for d in dtes if np.isfinite(d)]
        if dtes:
            expiry_display = f"{len(chosen_expiries)} expiries (min {int(round(min(dtes)))} / max {int(round(max(dtes)))} DTE)"
        else:
            expiry_display = f"{len(chosen_expiries)} expiries"

    gex_ratio = snap.get("gex_ratio", None)
    gex_ratio_disp = f"{gex_ratio:.2%}" if gex_ratio is not None else "—"

    tile_grid(
        items=[
            {"label": "Ticker", "value": ticker},
            {"label": "Stock Price", "value": f"{spot:,.2f}"},
            {"label": "Expiry (DTE)", "value": expiry_display, "wrap": True},
            {"label": "GEX Ratio", "value": gex_ratio_disp},
            {"label": "Call Wall", "value": f"{levels['call_wall']:.2f}"},
            {"label": "Put Wall", "value": f"{levels['put_wall']:.2f}"},
        ]
    )

    zg = levels_oi.get("zero_gamma", None)
    zg_disp = f"{zg:.2f}" if zg is not None else "—"

    neg_strike = levels_oi.get("neg_strike", None)
    pos_strike = levels_oi.get("pos_strike", None)
    call_oi_neg = levels_oi.get("call_oi_at_neg", None)
    put_oi_pos = levels_oi.get("put_oi_at_pos", None)

    neg_k = f"{neg_strike:.2f}" if neg_strike is not None else "—"
    pos_k = f"{pos_strike:.2f}" if pos_strike is not None else "—"

    tile_grid(
        items=[
            {"label": f"Call OI @ {neg_k}", "value": format_big(call_oi_neg)},
            {"label": f"Put OI @ {pos_k}", "value": format_big(put_oi_pos)},
            {"label": "ZeroGamma", "value": zg_disp},
        ]
    )

with right:
    # Determine chart title correctly
    if bucket == "Nearest expiration date" and next_expiry_used:
        exp = next_expiry_used

    elif bucket == "All expiries" and len(chosen_expiries) == 1:
        exp = chosen_expiries[0]

    else:
        exp = None

    if exp:
        dte0 = dte_map.get(exp, np.nan)
        if np.isfinite(dte0):
            chart_title = f"{ticker} - {exp} ({int(round(dte0))} DTE)"
        else:
            chart_title = f"{ticker} - {exp}"
    else:
        chart_title = f"{ticker} - All expiries"

    render_chart(gex_all=gex_all, spot=spot, chart_title=chart_title)








