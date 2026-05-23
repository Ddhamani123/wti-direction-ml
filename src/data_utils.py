"""
data_utils.py
Recreates the exact same X_train/val/test feature matrices used in the
training notebooks, without re-running them.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Constants (must match all notebooks exactly)
# ─────────────────────────────────────────────

FEATURES = [
    "RPI", "W875RX1", "CMRMTSPLx", "IPFPNSS", "USWTRADE", "USTRADE",
    "BUSLOANS", "CONSPI", "S&P 500", "S&P PE ratio", "FEDFUNDS",
    "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA",
    "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM",
    "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx",
    "PPICMM", "UMCSENTx",
]

TCODE_MAP = {
    # T-code 5 — log first difference (growth rate)
    "RPI": 5, "W875RX1": 5, "CMRMTSPLx": 5,
    "IPFPNSS": 5, "USWTRADE": 5, "USTRADE": 5,
    "BUSLOANS": 5, "CONSPI": 5, "S&P 500": 5,
    "EXSZUSx": 5, "EXJPUSx": 5, "EXUSUKx": 5,
    "EXCAUSx": 5, "PPICMM": 5,
    # T-code 2 — first difference
    "FEDFUNDS": 2, "TB3MS": 2, "TB6MS": 2,
    "GS1": 2, "GS5": 2, "GS10": 2,
    "AAA": 2, "BAA": 2, "TB3SMFFM": 2,
    "TB6SMFFM": 2, "T1YFFM": 2, "T5YFFM": 2,
    "T10YFFM": 2, "AAAFFM": 2, "BAAFFM": 2,
    # T-code 1 — level (no transformation)
    "S&P PE ratio": 1, "UMCSENTx": 1,
}

LOOKBACK  = 30
TRAIN_END = "2014-12-31"
VAL_END   = "2024-12-31"

BASE_DIR   = Path(__file__).resolve().parent.parent  # repo root
MACRO_FILE = BASE_DIR / "data" / "fred_md_2025_12.csv"
PRICE_FILE = BASE_DIR / "data" / "price" / "Price.csv"


# ─────────────────────────────────────────────
# Helper functions (verbatim from notebooks)
# ─────────────────────────────────────────────

def apply_tcodes(df: pd.DataFrame, tcode_map: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        tc = tcode_map.get(col, 1)
        x = df[col].astype(float)
        if tc == 1:
            out[col] = x
        elif tc == 2:
            out[col] = x.diff()
        elif tc == 5:
            out[col] = np.log(x).diff()
        else:
            out[col] = x.diff()
    return out


def load_macro_monthly(path: Path, features: list) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1]).rename(columns={"sasdate": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Missing macro columns (skipped): {missing}")
    return df[keep].copy()


def load_price(path: Path) -> pd.Series:
    px = pd.read_csv(path, parse_dates=["Date"])
    return px.set_index("Date")["Price"].sort_index()


def make_price_lags(price: pd.Series, L: int) -> pd.DataFrame:
    px = price.astype(float)
    return pd.concat(
        [px.shift(l).rename(f"price_lag{l:02d}") for l in range(1, L + 1)],
        axis=1,
    )


def make_macro_lags(X: pd.DataFrame, L: int) -> pd.DataFrame:
    return pd.concat(
        [X.shift(l).add_suffix(f"_lag{l:02d}") for l in range(1, L + 1)],
        axis=1,
    )


def make_direction_target(price: pd.Series) -> pd.Series:
    price = price.astype(float)
    direction = np.sign(price.diff())
    direction[direction == 0] = -1
    return direction.rename("y")


# ─────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────

def build_dataset(
    macro_file: Path = MACRO_FILE,
    price_file: Path = PRICE_FILE,
    features: list = FEATURES,
    lookback: int = LOOKBACK,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
):
    """
    Returns X_train, y_train, X_val, y_val, X_test, y_test
    as pd.DataFrame / pd.Series with DatetimeIndex.
    Identical preprocessing to all training notebooks.
    """
    macro_m = load_macro_monthly(macro_file, features)
    macro_m = apply_tcodes(macro_m, TCODE_MAP)
    # One-month reporting delay: Jan data available Feb 1st
    macro_m.index = macro_m.index + pd.offsets.MonthBegin(1)

    price = load_price(price_file)
    macro_d = macro_m.reindex(price.index, method="ffill")

    X_macro = make_macro_lags(macro_d, lookback)
    X_price = make_price_lags(price, lookback)

    X = X_macro.join(X_price).dropna()
    y = make_direction_target(price).reindex(X.index).dropna()
    X = X.loc[y.index]

    tr = X.index <= pd.Timestamp(train_end)
    va = (X.index > pd.Timestamp(train_end)) & (X.index <= pd.Timestamp(val_end))
    te = X.index > pd.Timestamp(val_end)

    print(f"Dataset sizes  — Train: {tr.sum()}  Val: {va.sum()}  Test: {te.sum()}")
    print(f"Feature matrix — {X.shape[0]} rows x {X.shape[1]} cols")
    print(f"Class balance  — Up: {(y==1).sum()}  Down: {(y==-1).sum()}")
    print(f"Date range     — {X.index[0].date()} to {X.index[-1].date()}")

    return (
        X.loc[tr], y.loc[tr],
        X.loc[va], y.loc[va],
        X.loc[te], y.loc[te],
    )


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset()
    print("\nSplit date ranges:")
    print(f"  Train: {X_train.index[0].date()} to {X_train.index[-1].date()}")
    print(f"  Val:   {X_val.index[0].date()} to {X_val.index[-1].date()}")
    print(f"  Test:  {X_test.index[0].date()} to {X_test.index[-1].date()}")
    print("\n[OK] data_utils.py working correctly.")
