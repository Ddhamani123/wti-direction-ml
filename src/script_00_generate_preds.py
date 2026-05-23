"""
script_00_generate_preds.py
Load trained .pkl models and save out-of-sample predictions to CSV.
Covers both val (2015-2024) and test (2025+) periods.
Models: XGBoost, MLP+ElasticNet, LSTM, VAR.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import accuracy_score, roc_auc_score

from model_classes import XGBoostDirectionClassifier, MLPFeatureExtractor, ElasticNetClassifier
from data_utils import (build_dataset, load_macro_monthly, apply_tcodes,
                        load_price, make_direction_target,
                        MACRO_FILE, PRICE_FILE, FEATURES, TCODE_MAP)

BASE     = Path(__file__).parent
PRED_DIR = BASE / "predictions"
PRED_DIR.mkdir(exist_ok=True)

TRAIN_END = pd.Timestamp("2014-12-31")
VAL_END   = pd.Timestamp("2024-12-31")
LOOKBACK  = 30


def save_preds(index, y_true, y_pred, prob, path):
    pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "prob": prob},
        index=index,
    ).to_csv(path)
    print(f"  Saved: {path.name}  ({len(y_true)} rows)")


# ── 1. Build tabular feature matrices (for XGBoost + MLP+EN) ─────────────────
print("Building tabular feature matrices...")
X_train, y_train, X_val, y_val, X_test, y_test = build_dataset()

# ── 2. XGBoost predictions ────────────────────────────────────────────────────
print("\nLoading XGBoost model...")
xgb = XGBoostDirectionClassifier()
xgb.load(BASE / "xgboost_wti_direction.pkl")

pred_val  = xgb.predict(X_val)
prob_val  = xgb.predict_proba(X_val)
pred_test = xgb.predict(X_test)
prob_test = xgb.predict_proba(X_test)

save_preds(X_val.index,  y_val.values,  pred_val,  prob_val,  PRED_DIR / "xgb_val_preds.csv")
save_preds(X_test.index, y_test.values, pred_test, prob_test, PRED_DIR / "xgb_test_preds.csv")
print(f"  XGB Val  — Acc: {accuracy_score(y_val, pred_val):.4f}  AUC: {roc_auc_score(y_val, prob_val):.4f}")
print(f"  XGB Test — Acc: {accuracy_score(y_test, pred_test):.4f}  AUC: {roc_auc_score(y_test, prob_test):.4f}")

# ── 3. MLP + ElasticNet predictions ──────────────────────────────────────────
print("\nLoading MLP + ElasticNet models...")
mlp = MLPFeatureExtractor()
mlp.load(BASE / "mlp_extractor_wti.pkl")

en = ElasticNetClassifier()
en.load(BASE / "mlp_en_wti_direction.pkl")

E_val  = mlp.get_embeddings(X_val)
E_test = mlp.get_embeddings(X_test)

pred_val_en  = en.predict(E_val)
prob_val_en  = en.predict_proba(E_val)
pred_test_en = en.predict(E_test)
prob_test_en = en.predict_proba(E_test)

save_preds(X_val.index,  y_val.values,  pred_val_en,  prob_val_en,  PRED_DIR / "mlpen_val_preds.csv")
save_preds(X_test.index, y_test.values, pred_test_en, prob_test_en, PRED_DIR / "mlpen_test_preds.csv")
print(f"  MLPEN Val  — Acc: {accuracy_score(y_val, pred_val_en):.4f}  AUC: {roc_auc_score(y_val, prob_val_en):.4f}")
print(f"  MLPEN Test — Acc: {accuracy_score(y_test, pred_test_en):.4f}  AUC: {roc_auc_score(y_test, prob_test_en):.4f}")

# ── 4. LSTM predictions ───────────────────────────────────────────────────────
print("\nLoading LSTM model + building sequences...")

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """Exact architecture from LSTM_Direction_VIP.ipynb — must match saved state."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# Load PKL — state dict was saved separately from other attributes
with open(BASE / "lstm_wti_direction.pkl", "rb") as f:
    lstm_state = pickle.load(f)

p_best    = lstm_state["best_params_"]
scaler    = lstm_state["scaler_"]
feat_names = lstm_state["feature_names_"]   # 32 feature names in correct order

# Use CPU to avoid device-mapping issues; GPU was for training only
device = "cpu"

lstm_net = LSTMNet(
    input_size=len(feat_names),
    hidden_size=p_best["hidden_size"],
    num_layers=p_best["num_layers"],
    dropout=p_best["dropout"],
).to(device)

# Move state dict tensors to CPU before loading (handles case where saved on GPU)
model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v
               for k, v in lstm_state["model_state"].items()}
lstm_net.load_state_dict(model_state)
lstm_net.eval()

LSTM_THRESHOLD = 0.38   # tuned on val set (from notebook output)

# Build LSTM sequences using the same preprocessing as LSTM_Direction_VIP.ipynb:
# 31 t-code FRED-MD (with 1-month shift) + raw Price as 32nd feature
# 30-step sliding window: window [i-30, i) predicts direction at i
macro_m_lstm = load_macro_monthly(MACRO_FILE, FEATURES)
macro_m_lstm = apply_tcodes(macro_m_lstm, TCODE_MAP)
macro_m_lstm.index = macro_m_lstm.index + pd.offsets.MonthBegin(1)
price_s = load_price(PRICE_FILE)
macro_d_lstm = macro_m_lstm.reindex(price_s.index, method="ffill")

# feat_names from PKL gives the exact column order used during training
base_lstm = macro_d_lstm.join(price_s.rename("Price"), how="inner").dropna()
# Reindex to the stored feature order (ensures column alignment)
base_lstm = base_lstm[[c for c in feat_names if c in base_lstm.columns]]
feature_data = base_lstm.values.astype(np.float32)
dates_lstm   = base_lstm.index

direction_target = make_direction_target(price_s)

X_seq, y_seq, d_seq = [], [], []
for i in range(LOOKBACK, len(feature_data)):
    dt = dates_lstm[i]
    yt = direction_target.get(dt, np.nan)
    if np.isfinite(yt):
        X_seq.append(feature_data[i - LOOKBACK:i])
        y_seq.append(yt)
        d_seq.append(dt)

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
d_seq = pd.DatetimeIndex(d_seq)

val_mask  = (d_seq > TRAIN_END) & (d_seq <= VAL_END)
test_mask = d_seq > VAL_END

print(f"  LSTM sequences — Val: {val_mask.sum()}  Test: {test_mask.sum()}")


def lstm_predict_proba(X: np.ndarray) -> np.ndarray:
    N, T, F = X.shape
    X_sc = scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
    with torch.no_grad():
        logits = lstm_net(torch.tensor(X_sc).to(device)).cpu().numpy()
    return torch.sigmoid(torch.tensor(logits)).numpy().ravel()


prob_val_lstm  = lstm_predict_proba(X_seq[val_mask])
pred_val_lstm  = np.where(prob_val_lstm >= LSTM_THRESHOLD, 1, -1)
prob_test_lstm = lstm_predict_proba(X_seq[test_mask])
pred_test_lstm = np.where(prob_test_lstm >= LSTM_THRESHOLD, 1, -1)

save_preds(d_seq[val_mask],  y_seq[val_mask],  pred_val_lstm,  prob_val_lstm,  PRED_DIR / "lstm_val_preds.csv")
save_preds(d_seq[test_mask], y_seq[test_mask], pred_test_lstm, prob_test_lstm, PRED_DIR / "lstm_test_preds.csv")
print(f"  LSTM Val  — Acc: {accuracy_score(y_seq[val_mask], pred_val_lstm):.4f}  AUC: {roc_auc_score(y_seq[val_mask], prob_val_lstm):.4f}")
print(f"  LSTM Test — Acc: {accuracy_score(y_seq[test_mask], pred_test_lstm):.4f}  AUC: {roc_auc_score(y_seq[test_mask], prob_test_lstm):.4f}")

# ── 5. VAR predictions ────────────────────────────────────────────────────────
print("\nLoading VAR model + building Z-matrix...")

# pandas StringDtype pickle compatibility fix.
# The PKL was saved with pandas StringArray (StringDtype(storage='python',
# na_value=nan)) which raises NotImplementedError in NDArrayBacked.__setstate__
# on some pandas versions. We intercept the GLOBAL opcode for NDArrayBacked
# via a custom Unpickler and replace it with a plain Python list-like proxy.
class _CompatNDArrayBacked:
    """Proxy for StringArray — replaces NDArrayBacked during PKL load."""
    def __setstate__(self, state):
        # state is (dtype, data) for StringArray; use index 1 to be safe
        self._data = np.asarray(state[1], dtype=object)
    @property
    def dtype(self): return np.dtype("O")
    @property
    def ndim(self):  return 1
    @property
    def shape(self): return self._data.shape
    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)
    def __iter__(self):       return iter(self._data)
    def __len__(self):        return len(self._data)
    def __getitem__(self, i): return self._data[i]
    def tolist(self):         return self._data.tolist()


def _compat_pyx_unpickle_NDArrayBacked(cls, checksum, state):
    # Only proxy StringArray — other NDArrayBacked types (DatetimeArray etc.)
    # need their own __new__ so pickle can call __setstate__ normally.
    if cls.__name__ == "StringArray":
        return _CompatNDArrayBacked()   # pickle calls __setstate__ separately
    obj = cls.__new__(cls)
    return obj


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module == "pandas._libs.arrays"
                and name == "__pyx_unpickle_NDArrayBacked"):
            return _compat_pyx_unpickle_NDArrayBacked
        return super().find_class(module, name)


with open(BASE / "var_wti_direction.pkl", "rb") as f:
    var_state = _CompatUnpickler(f).load()

var_results       = var_state["model_"]
selected_features = var_state["selected_features_"]
best_p            = var_state["best_p_"]
var_threshold     = var_state["threshold_"]
var_sigma         = var_state["sigma_"]

print(f"  VAR lag order  : {best_p}")
print(f"  Selected macro : {selected_features}")
print(f"  Threshold      : {var_threshold:.5f}   Sigma: {var_sigma:.6f}")

# Build Z matrix: wti_ret (col 0) + 31 t-code FRED-MD, same 1-month macro shift
macro_m_var = load_macro_monthly(MACRO_FILE, FEATURES)
macro_m_var = apply_tcodes(macro_m_var, TCODE_MAP)
macro_m_var.index = macro_m_var.index + pd.offsets.MonthBegin(1)
macro_d_var = macro_m_var.reindex(price_s.index, method="ffill")

wti_logret = np.log(price_s).diff().rename("wti_ret")
Z = pd.concat([wti_logret, macro_d_var], axis=1).dropna()

Z_train    = Z.loc[:TRAIN_END]
Z_val_z    = Z.loc[(Z.index > TRAIN_END) & (Z.index <= VAL_END)]
Z_test_z   = Z.loc[Z.index > VAL_END]
Z_trainval = Z.loc[:VAL_END]

print(f"  Z matrix — Train: {len(Z_train)}  Val: {len(Z_val_z)}  Test: {len(Z_test_z)}")

# Only the reduced column set is used for VAR forecasting
cols_var = ["wti_ret"] + selected_features


def sequential_forecast_var(Z_history: pd.DataFrame,
                             Z_future: pd.DataFrame) -> np.ndarray:
    """1-step-ahead WTI log-return forecasts using actual observations only."""
    h = Z_history[cols_var].values
    f = Z_future[cols_var].values
    Z_all  = np.vstack([h, f])
    n_hist = len(h)
    forecasts = []
    for i in range(len(f)):
        window = Z_all[n_hist + i - best_p : n_hist + i]
        fc = var_results.forecast(y=window, steps=1)
        forecasts.append(fc[0, 0])   # wti_ret is col 0
    return np.array(forecasts)


fc_val_var  = sequential_forecast_var(Z_train,    Z_val_z)
fc_test_var = sequential_forecast_var(Z_trainval, Z_test_z)

prob_val_var  = norm.cdf((fc_val_var  - var_threshold) / var_sigma)
prob_test_var = norm.cdf((fc_test_var - var_threshold) / var_sigma)
pred_val_var  = np.where(fc_val_var  > var_threshold, 1, -1)
pred_test_var = np.where(fc_test_var > var_threshold, 1, -1)

# y_true from the shared direction target
y_true_val_var  = make_direction_target(price_s).reindex(Z_val_z.index).values
y_true_test_var = make_direction_target(price_s).reindex(Z_test_z.index).values

save_preds(Z_val_z.index,  y_true_val_var,  pred_val_var,  prob_val_var,  PRED_DIR / "var_val_preds.csv")
save_preds(Z_test_z.index, y_true_test_var, pred_test_var, prob_test_var, PRED_DIR / "var_test_preds.csv")
print(f"  VAR Val  — Acc: {accuracy_score(y_true_val_var, pred_val_var):.4f}  AUC: {roc_auc_score(y_true_val_var, prob_val_var):.4f}")
print(f"  VAR Test — Acc: {accuracy_score(y_true_test_var, pred_test_var):.4f}  AUC: {roc_auc_score(y_true_test_var, prob_test_var):.4f}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("All prediction CSVs saved to predictions/")
print("=" * 55)
for f in sorted(PRED_DIR.glob("*.csv")):
    rows = sum(1 for _ in open(f)) - 1
    print(f"  {f.name:<35s} {rows:>5} rows")
