import streamlit as st
import pandas as pd
import numpy as np
from math import log, sqrt
from scipy.stats import norm
from options.data import get_options_view_df

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

DEFAULT_RISK_FREE = 0.05
DEFAULT_MULTIPLIER = 100
STRIKES_AROUND_SPOT = 30  # +/- 30 strikes (by count)


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
# Cache: spot pull
# =========================
@st.cache_data(ttl=30)
def fetch_spot_yahoo(symbol: str) -> float:
    if not YF_OK:
        raise RuntimeError("yfinance not installed")

    t = yf.Ticker(symbol)

    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            v = fi.get("last_price", None)
            if v is not None and float(v) > 0:
                return float(v)
    except Exception:
        pass

    # 1m intraday
    try:
        intraday = t.history(period="1d", interval="1m")
        if isinstance(intraday, pd.DataFrame) and (not intraday.empty) and ("Close" in intraday.columns):
            v = float(intraday["Close"].dropna().iloc[-1])
            if v > 0:
                return v
    except Exception:
        pass

    # daily
    hist = t.history(period="5d", interval="1d")
    if not isinstance(hist, pd.DataFrame) or hist.empty or "Close" not in hist.columns:
        raise RuntimeError("No price history returned.")
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
# Snapshot UI tiles
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
# Levels / stats
# =========================
def compute_levels_by_strike(gex_df: pd.DataFrame):
    if not isinstance(gex_df, pd.DataFrame) or gex_df.empty:
        return {}
    gex_df = gex_df.sort_values("strike").reset_index(drop=True)
    put_wall = float(gex_df.loc[gex_df["dealer_gex"].idxmax(), "strike"])
    call_wall = float(gex_df.loc[gex_df["dealer_gex"].idxmin(), "strike"])
    return {"put_wall": put_wall, "call_wall": call_wall}


def compute_zero_gamma_level(gex_by_strike: pd.DataFrame):
    if not isinstance(gex_by_strike, pd.DataFrame) or gex_by_strike.empty:
        return None
    g = gex_by_strike.sort_values("strike").reset_index(drop=True)
    y = g["dealer_gex"].values
    x = g["strike"].values
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


def compute_gex_ratio(per_exp_parts: list[pd.DataFrame]):
    if per_exp_parts is None or len(per_exp_parts) == 0:
        return None
    allp = pd.concat(per_exp_parts, ignore_index=True)
    total_call = float(allp["call_gex"].sum())
    total_put = float(allp["put_gex"].sum())
    denom = total_call + total_put
    if denom <= 0:
        return None
    return total_call / denom


def compute_extreme_oi(per_exp_parts: list[pd.DataFrame], gex_by_strike: pd.DataFrame):
    out = {
        "neg_strike": None,
        "pos_strike": None,
        "call_oi_at_neg": None,
        "put_oi_at_pos": None,
        "zero_gamma": None,
    }
    if per_exp_parts is None or len(per_exp_parts) == 0:
        return out
    if not isinstance(gex_by_strike, pd.DataFrame) or gex_by_strike.empty:
        return out

    allp = pd.concat(per_exp_parts, ignore_index=True)
    oi_by_strike = allp.groupby("strike", as_index=False).agg(
        call_oi=("call_oi", "sum"),
        put_oi=("put_oi", "sum"),
    )

    gg = gex_by_strike.sort_values("strike").reset_index(drop=True)
    out["neg_strike"] = float(gg.loc[gg["dealer_gex"].idxmin(), "strike"])
    out["pos_strike"] = float(gg.loc[gg["dealer_gex"].idxmax(), "strike"])

    rn = oi_by_strike[oi_by_strike["strike"] == out["neg_strike"]]
    rp = oi_by_strike[oi_by_strike["strike"] == out["pos_strike"]]
    if not rn.empty:
        out["call_oi_at_neg"] = float(rn["call_oi"].iloc[0])
    if not rp.empty:
        out["put_oi_at_pos"] = float(rp["put_oi"].iloc[0])

    out["zero_gamma"] = compute_zero_gamma_level(gex_by_strike)
    return out


def format_big(n):
    if n is None:
        return "—"
    try:
        if isinstance(n, float) and not np.isfinite(n):
            return "—"
        return f"{float(n):,.0f}"
    except Exception:
        return "—"


# =========================
# Chart
# - NO title
# - X: POS -> NEG (left -> right)
# - Y: small bottom -> large top
# - POS bars = RED, NEG bars = GREEN
# - Spot: dashed line across BOTH + and - sides (nearly full width) + label far right
# =========================
def render_chart(gex_all: pd.DataFrame, spot: float):
    if not ALTAIR_OK:
        st.warning("Altair not available.")
        return
    if gex_all is None or (not isinstance(gex_all, pd.DataFrame)) or gex_all.empty:
        st.warning("No GEX data to chart.")
        return

    g0 = gex_all.copy()
    g0["strike"] = pd.to_numeric(g0["strike"], errors="coerce")
    g0 = g0.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    if g0.empty:
        st.warning("No valid strikes.")
        return

    strikes = g0["strike"].to_numpy()
    idx = int(np.argmin(np.abs(strikes - spot)))
    lo = max(idx - STRIKES_AROUND_SPOT, 0)
    hi = min(idx + STRIKES_AROUND_SPOT + 1, len(strikes))
    g = g0.iloc[lo:hi].copy().sort_values("strike").reset_index(drop=True)

    abs_max = float(np.abs(g["dealer_gex"]).max()) if not g.empty else 1.0
    if abs_max <= 0:
        abs_max = 1.0

    # X axis: POS -> NEG (left -> right)
    x_domain = [abs_max, -abs_max]

    # POS=RED, NEG=GREEN
    g["bar_color"] = np.where(g["dealer_gex"] >= 0, "pos", "neg")

    bg = "#2a2a2a"
    grid = "#3c3c3c"
    axis_label = "#e6e6e6"

    bars = alt.Chart(g).mark_bar(size=9).encode(
        x=alt.X(
            "dealer_gex:Q",
            scale=alt.Scale(domain=x_domain),
            axis=alt.Axis(
                title="",
                grid=True,
                gridColor=grid,
                gridOpacity=0.8,
                ticks=False,
                labelColor=axis_label,
                labelExpr="replace(format(datum.value, '~s'), 'k', 'K')",
            ),
        ),
        y=alt.Y(
            "strike:O",
            sort="ascending",
            scale=alt.Scale(reverse=True),  # small bottom, large top
            axis=alt.Axis(title="", ticks=False, labelColor=axis_label, format=".2f"),
        ),
        color=alt.Color(
            "bar_color:N",
            scale=alt.Scale(domain=["pos", "neg"], range=["#D00000", "#00A000"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("strike:Q", format=".2f", title="Strike"),
            alt.Tooltip("dealer_gex:Q", format=",.0f", title="Dealer GEX"),
        ],
    ).properties(height=600)

    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#8a8a8a", opacity=0.9, strokeWidth=2
    ).encode(x="x:Q")

    # Spot line: snap to nearest displayed strike (Y is ordinal)
    spot_strike = float(g.loc[(g["strike"] - spot).abs().idxmin(), "strike"])

    # Dashed spot line segment spanning BOTH + and - (nearly full width, slightly inset)
    spot_line_df = pd.DataFrame({
        "x1": [ abs_max * 0.985],   # near left edge (positive side)
        "x2": [-abs_max * 0.985],   # near right edge (negative side)
        "y":  [spot_strike],
    })
    spot_line = alt.Chart(spot_line_df).mark_line(
        color="white",
        strokeDash=[6, 6],
        strokeWidth=1.6,
        opacity=0.9
    ).encode(
        x=alt.X("x1:Q"),
        x2="x2:Q",
        y=alt.Y("y:O", sort="ascending", scale=alt.Scale(reverse=True))
    )

    # Spot label: push past right edge so it hugs the border
    spot_text = alt.Chart(
        pd.DataFrame({
            "x": [-abs_max * 1.03],
            "y": [spot_strike],
            "t": [f"{spot:.2f}"]
        })
    ).mark_text(
        align="right",
        baseline="middle",
        dx=8,
        color="white",
        fontSize=12,
        fontWeight="bold",
    ).encode(
        x="x:Q",
        y=alt.Y("y:O", sort="ascending", scale=alt.Scale(reverse=True)),
        text="t:N",
    )

    chart = (
        alt.layer(bars, zero_line, spot_line, spot_text)
        .configure(background=bg)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)


# =========================
# Main UI
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
        if len(missing) > 0:
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
        all_expiries = future_exp if len(future_exp) > 0 else exp_tbl["expiry_str"].astype(str).tolist()

        if len(all_expiries) == 0:
            st.error("No expiries returned.")
            st.stop()

    # Choose expiries
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

        if len(chosen_expiries) == 0:
            st.info("Pick at least one expiry to compute GEX.")
            st.stop()

    # =========================
    # Compute GEX (sum across expiries)
    # =========================
    with st.spinner("Computing GEX…"):
        per_exp_parts = []
        dealer_parts = []

        for exp in chosen_expiries:
            dfe = df[df["expiry_str"] == exp].copy()
            T = to_years(exp)

            dfe["gamma"] = dfe.apply(
                lambda r0: bs_gamma(float(spot), float(r0["strike"]), float(T), DEFAULT_RISK_FREE, float(r0["iv"])),
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

            gex["call_gex"] = (gex["call_gamma"] * gex["call_oi"]) * DEFAULT_MULTIPLIER * (float(spot) ** 2)
            gex["put_gex"]  = (gex["put_gamma"]  * gex["put_oi"])  * DEFAULT_MULTIPLIER * (float(spot) ** 2)
            gex["dealer_gex"] = gex["put_gex"] - gex["call_gex"]

            per_exp_parts.append(gex[["strike", "call_gex", "put_gex", "call_oi", "put_oi"]])
            dealer_parts.append(gex[["strike", "dealer_gex"]])

        gex_all = pd.concat(dealer_parts, ignore_index=True).groupby("strike", as_index=False)["dealer_gex"].sum()
        gex_all = gex_all.sort_values("strike").reset_index(drop=True)

        levels = compute_levels_by_strike(gex_all)
        gex_ratio = compute_gex_ratio(per_exp_parts)
        ext = compute_extreme_oi(per_exp_parts, gex_all)

    # =========================
    # Snapshot
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
        if len(dtes) > 0:
            expiry_display = f"{len(chosen_expiries)} expiries (min {int(round(min(dtes)))} / max {int(round(max(dtes)))} DTE)"
        else:
            expiry_display = f"{len(chosen_expiries)} expiries"

    gex_ratio_disp = f"{gex_ratio:.2%}" if gex_ratio is not None else "—"

    tile_grid(
        items=[
            {"label": "Ticker", "value": ticker},
            {"label": "Stock Price", "value": f"{float(spot):,.2f}"},
            {"label": "Expiry (DTE)", "value": expiry_display, "wrap": True},
            {"label": "GEX Ratio", "value": gex_ratio_disp},
            {"label": "Call Wall", "value": f"{levels.get('call_wall', np.nan):.2f}" if levels else "—"},
            {"label": "Put Wall", "value": f"{levels.get('put_wall', np.nan):.2f}" if levels else "—"},
        ]
    )

    zg = ext.get("zero_gamma", None)
    zg_disp = f"{zg:.2f}" if zg is not None else "—"

    neg_strike = ext.get("neg_strike", None)
    pos_strike = ext.get("pos_strike", None)
    call_oi_neg = ext.get("call_oi_at_neg", None)
    put_oi_pos = ext.get("put_oi_at_pos", None)

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
    render_chart(gex_all=gex_all, spot=float(spot))
