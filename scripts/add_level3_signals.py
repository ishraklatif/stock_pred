#!/usr/bin/env python3
"""
add_level3_signals.py

Generates Level-3 signals on top of macro_level2:

- Rolling correlations between key macro assets
- Rolling betas (rolling linear regression)
- Regime features (volatility, commodity, FX, rates, dollar)

Input:
    data/enriched/macro_level2.parquet

Outputs:
    data/level3/rolling_correlations.parquet
    data/level3/rolling_betas.parquet
    data/level3/regime_features.parquet
    data/level3/level3_all.parquet
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SOURCE_FILE = "data/level2/macro_level2.parquet"
OUT_DIR = "data/level3"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Mapping: ASSET NAME -> return column in macro_level2
# ------------------------------------------------------------------
ASSET_RETURN_MAP = {
    "AXJO": "^AXJO_return",
    "SPY": "SPY_return",
    "GSPC": "^GSPC_return",
    "FTSE": "^FTSE_return",
    "N225": "^N225_return",
    "SSE": "000001.SS_return",      # Shanghai Composite
    "CSI300": "000300.SS_return",
    "HSI": "^HSI_return",
    "KS11": "^KS11_return",
    "TWII": "^TWII_return",
    "STI": "^STI_return",
    "NSEI": "^NSEI_return",

    "GOLD": "GC=F_return",
    "OIL": "BZ=F_return",
    "IRON": "TIO=F_return",
    "COPPER": "COPPER_COPPER_return",
    "COMMODITY": "COMMODITY_IDX_COMMODITY_IDX_return",

    "AUDUSD": "AUDUSD=X_return",
    "AUDJPY": "AUDJPY=X_return",
    "AUDCNY": "AUDCNY=X_return",

    "VIX": "VIX_VIX_return",
    "DXY": "DXY_DXY_return",
    "NDX": "NDX_NDX_return",

    "US10Y": "US10Y_US10Y_return",
    "US2Y": "US2Y_US2Y_return",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def rolling_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation."""
    return s1.rolling(window).corr(s2)


def rolling_beta(df: pd.DataFrame, target_col: str, factor_col: str, window: int) -> pd.Series:
    """
    Rolling beta of target vs factor using linear regression over a moving window.
    Beta_t = cov(target, factor) / var(factor)
    Implemented via sklearn LinearRegression for robustness.
    """
    lr = LinearRegression()
    betas = []

    for i in range(len(df)):
        start = max(0, i - window + 1)
        window_df = df.iloc[start:i + 1][[target_col, factor_col]]

        # Require full window
        if len(window_df) < window:
            betas.append(np.nan)
            continue

        # Drop rows with NaNs inside the window
        window_df = window_df.dropna()
        if len(window_df) < window // 2:  # too few valid points
            betas.append(np.nan)
            continue

        X = window_df[factor_col].values.reshape(-1, 1)
        y = window_df[target_col].values

        try:
            lr.fit(X, y)
            betas.append(lr.coef_[0])
        except Exception:
            betas.append(np.nan)

    return pd.Series(betas, index=df.index)


def quantile_regime(series: pd.Series, low_q: float = 0.33, high_q: float = 0.66) -> pd.Series:
    """
    0 = low regime, 1 = mid regime, 2 = high regime
    based on quantiles of the absolute series.
    """
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(np.nan, index=series.index)

    low = s.quantile(low_q)
    high = s.quantile(high_q)

    regimes = np.full(len(series), np.nan)
    for i, v in enumerate(series.values):
        if np.isnan(v):
            regimes[i] = np.nan
        elif abs(v) < low:
            regimes[i] = 0
        elif abs(v) < high:
            regimes[i] = 1
        else:
            regimes[i] = 2
    return pd.Series(regimes, index=series.index)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"[INFO] Loading Level-2 macro dataset from {SOURCE_FILE} ...")
    df = pd.read_parquet(SOURCE_FILE)
    df = df.sort_values("Date")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Check which mapped columns actually exist
    print("\n[INFO] Verifying mapped return columns present in dataframe...")
    valid_assets = {}
    for asset, col in ASSET_RETURN_MAP.items():
        if col in df.columns:
            valid_assets[asset] = col
            print(f"  [OK] {asset}: {col}")
        else:
            print(f"  [WARN] {asset}: missing column {col} → will be skipped in Level-3 features.")

    # ----------------------------------------------------------
    # 1) Rolling Correlations
    # ----------------------------------------------------------
    print("\n[INFO] Computing rolling correlations...")
    corr_df = pd.DataFrame(index=df.index)

    # Choose economically meaningful correlation pairs
    corr_pairs = [
        ("AXJO", "SPY"),
        ("AXJO", "N225"),
        ("AXJO", "DXY"),
        ("AXJO", "VIX"),
        ("SPY", "VIX"),
        ("GOLD", "DXY"),
        ("COPPER", "DXY"),
        ("COMMODITY", "DXY"),
        ("AUDUSD", "DXY"),
    ]

    corr_windows = [30, 60, 90]

    for a, b in corr_pairs:
        if a not in valid_assets or b not in valid_assets:
            print(f"  [SKIP] correlation {a} vs {b}: missing asset data.")
            continue

        col_a = valid_assets[a]
        col_b = valid_assets[b]
        s1 = df[col_a]
        s2 = df[col_b]

        for w in corr_windows:
            feature_name = f"corr_{a}_vs_{b}_{w}d"
            corr_df[feature_name] = rolling_corr(s1, s2, w)

    corr_path = os.path.join(OUT_DIR, "rolling_correlations.parquet")
    corr_df.to_parquet(corr_path)
    print(f"[OK] Rolling correlations saved → {corr_path}")

    # ----------------------------------------------------------
    # 2) Rolling Betas
    # ----------------------------------------------------------
    print("\n[INFO] Computing rolling betas...")
    beta_df = pd.DataFrame(index=df.index)

    # Target vs Factor pairs (AXJO vs global/commodity/FX)
    beta_pairs = [
        ("AXJO", "SPY"),
        ("AXJO", "DXY"),
        ("AXJO", "GOLD"),
        ("AXJO", "VIX"),
        ("AXJO", "COMMODITY"),
        ("GOLD", "DXY"),
        ("COPPER", "DXY"),
    ]

    beta_windows = [60, 90, 120]

    for target, factor in beta_pairs:
        if target not in valid_assets or factor not in valid_assets:
            print(f"  [SKIP] beta {target} vs {factor}: missing asset data.")
            continue

        t_col = valid_assets[target]
        f_col = valid_assets[factor]

        for w in beta_windows:
            feature_name = f"beta_{target}_vs_{factor}_{w}d"
            beta_df[feature_name] = rolling_beta(df, t_col, f_col, window=w)

    beta_path = os.path.join(OUT_DIR, "rolling_betas.parquet")
    beta_df.to_parquet(beta_path)
    print(f"[OK] Rolling betas saved → {beta_path}")

    # ----------------------------------------------------------
    # 3) Regime Features
    # ----------------------------------------------------------
    print("\n[INFO] Computing regime features...")
    regime_df = pd.DataFrame(index=df.index)

    # Volatility regime via VIX
    if "VIX" in valid_assets:
        vix_ret = df[valid_assets["VIX"]]
        regime_df["vol_regime_vix"] = quantile_regime(vix_ret.abs())
    else:
        print("  [SKIP] VIX regime: VIX return series missing.")

    # Commodity regime via COMMODITY index
    if "COMMODITY" in valid_assets:
        com_ret = df[valid_assets["COMMODITY"]]
        regime_df["commodity_regime"] = quantile_regime(com_ret.abs())
        regime_df["commodity_trend_90"] = com_ret.rolling(90).mean()
    else:
        print("  [SKIP] commodity regime: COMMODITY return series missing.")

    # FX regime via AUDUSD
    if "AUDUSD" in valid_assets:
        fx_ret = df[valid_assets["AUDUSD"]]
        regime_df["fx_regime_audusd"] = quantile_regime(fx_ret.abs())
        regime_df["fx_trend_90"] = fx_ret.rolling(90).mean()
    else:
        print("  [SKIP] FX regime: AUDUSD return series missing.")

    # Rates regime via US10Y
    if "US10Y" in valid_assets:
        r_ret = df[valid_assets["US10Y"]]
        regime_df["rates_regime_us10y"] = quantile_regime(r_ret.abs())
        regime_df["rates_trend_90"] = r_ret.rolling(90).mean()
    else:
        print("  [SKIP] rates regime: US10Y return series missing.")

    # Dollar regime via DXY
    if "DXY" in valid_assets:
        dxy_ret = df[valid_assets["DXY"]]
        regime_df["dxy_regime"] = quantile_regime(dxy_ret.abs())
        regime_df["dxy_trend_90"] = dxy_ret.rolling(90).mean()
    else:
        print("  [SKIP] DXY regime: DXY return series missing.")

    regime_path = os.path.join(OUT_DIR, "regime_features.parquet")
    regime_df.to_parquet(regime_path)
    print(f"[OK] Regime features saved → {regime_path}")

    # ----------------------------------------------------------
    # 4) Save combined Level-3 features
    # ----------------------------------------------------------
    level3_all = pd.concat([corr_df, beta_df, regime_df], axis=1)
    combined_path = os.path.join(OUT_DIR, "level3_all.parquet")
    level3_all.to_parquet(combined_path)
    print(f"\n[OK] All Level-3 features saved → {combined_path}")
    print("\n[ALL DONE] Level-3 signal generation complete.\n")


if __name__ == "__main__":
    main()

# python3 scripts/add_level3_signals.py