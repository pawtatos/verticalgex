# options/data.py
import pandas as pd
import yfinance as yf


def get_options_view_df(symbol: str):
    """
    Fetch options chain from Yahoo Finance and return a unified DataFrame.

    Returns:
        (None, DataFrame)
    """
    t = yf.Ticker(symbol)
    expirations = t.options

    rows = []

    for exp in expirations:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue

        for side, df in [("call", chain.calls), ("put", chain.puts)]:
            if df is None or df.empty:
                continue

            tmp = df.copy()
            tmp["option_type"] = side
            tmp["expire_date"] = exp

            rows.append(
                tmp[[
                    "option_type",
                    "expire_date",
                    "strike",
                    "openInterest",
                    "impliedVolatility"
                ]]
            )

    if not rows:
        return None, pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # Normalize column names to what your app expects
    out = out.rename(columns={
        "openInterest": "open_interest",
        "impliedVolatility": "imp_vol",
    })

    return None, out
