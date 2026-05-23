"""
model_classes.py
Minimal class definitions needed to load the trained .pkl model files
without re-running the full training notebooks.
"""
import numpy as np
import pandas as pd
import pickle
import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from xgboost import XGBClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_classification(y_true, y_pred, y_prob=None, label=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, labels=[-1, 1], zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, labels=[-1, 1], zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, labels=[-1, 1], zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    if label:
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Down (-1)", "Up (+1)"], zero_division=0))
    print(f"  ROC-AUC : {auc:.4f}\n  F1 (wtd): {f1_w:.4f}")
    return {
        "accuracy": acc,
        "prec_down": prec[0], "prec_up": prec[1],
        "rec_down":  rec[0],  "rec_up":  rec[1],
        "f1_down":   f1[0],   "f1_up":   f1[1],
        "f1_weighted": f1_w,  "roc_auc": auc,
    }


class BaseForecastModel(ABC):
    def __init__(self, task_type: str):
        self.task_type = task_type

    @abstractmethod
    def fit(self, *args, **kwargs): pass

    @abstractmethod
    def predict(self, X): pass

    @abstractmethod
    def evaluate(self, X, y): pass

    @abstractmethod
    def save(self, filepath: str): pass

    @abstractmethod
    def load(self, filepath: str): pass


# ─────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────

class XGBoostDirectionClassifier(BaseForecastModel):
    def __init__(self, n_splits=5, num_boost_round=1000,
                 early_stopping_rounds=100, random_state=42, n_jobs=-1):
        super().__init__("classification")
        self.n_splits              = n_splits
        self.num_boost_round       = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state          = random_state
        self.n_jobs                = n_jobs
        self.param_grid            = []
        self.model                 = None
        self.best_params_          = None
        self.cv_results_           = []
        self.eval_results_         = None
        self.feature_names_        = None

    @staticmethod
    def _to_binary(y):
        return ((np.asarray(y) + 1) // 2).astype(int)

    @staticmethod
    def _to_signed(y):
        return np.where(np.asarray(y) == 1, 1, -1)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Use the trained pkl — fit() not needed here.")

    def predict(self, X, threshold: float = 0.5):
        proba = self.predict_proba(X)
        return np.where(proba >= threshold, 1, -1)

    def predict_proba(self, X):
        xv = X.values if hasattr(X, "values") else X
        return self.model.predict_proba(xv)[:, 1]

    def evaluate(self, X, y, label="", threshold: float = 0.5):
        yt = y.values if hasattr(y, "values") else y
        return evaluate_classification(yt, self.predict(X, threshold), self.predict_proba(X), label=label)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))


# ─────────────────────────────────────────────
# MLP Feature Extractor
# ─────────────────────────────────────────────

class _MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, dropout=0.1):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_layer_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev, 1)

    def forward(self, x):
        return self.output(self.hidden(x))

    def get_embedding(self, x):
        return self.hidden(x)


def _train_one_fold(*args, **kwargs):
    raise NotImplementedError("Use the trained pkl — _train_one_fold() not needed here.")


class MLPFeatureExtractor(BaseForecastModel):
    def __init__(self, hidden_layer_sizes=(256, 128), alpha=1e-4,
                 max_epochs=500, batch_size=256, device=None):
        super().__init__("classification")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha              = alpha
        self.max_epochs         = max_epochs
        self.batch_size         = batch_size
        self.device             = device or DEVICE
        self.scaler             = StandardScaler()
        self.net_               = None
        self.embedding_dim_     = hidden_layer_sizes[-1]
        self.train_losses_      = []
        self.val_losses_        = []

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Use the trained pkl — fit() not needed here.")

    def get_embeddings(self, X):
        Xs = self.scaler.transform(X.values if hasattr(X, "values") else X)
        self.net_.eval()
        with torch.no_grad():
            return self.net_.get_embedding(torch.FloatTensor(Xs)).numpy()

    def _proba(self, X_scaled):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.FloatTensor(X_scaled)).squeeze().numpy()
        return torch.sigmoid(torch.FloatTensor(logits)).numpy()

    def predict(self, X):
        Xs = self.scaler.transform(X.values if hasattr(X, "values") else X)
        return np.where(self._proba(Xs) >= 0.5, 1, -1)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X.values if hasattr(X, "values") else X)
        return self._proba(Xs)

    def evaluate(self, X, y, label=""):
        yt = y.values if hasattr(y, "values") else y
        return evaluate_classification(yt, self.predict(X), self.predict_proba(X), label=label)

    def save(self, path: str):
        payload = {k: v for k, v in self.__dict__.items() if k != "net_"}
        payload["_net_state"]     = self.net_.state_dict()
        payload["_net_input_dim"] = self.scaler.n_features_in_
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        ns  = payload.pop("_net_state")
        nid = payload.pop("_net_input_dim")
        self.__dict__.update(payload)
        self.net_ = _MLPNet(nid, self.hidden_layer_sizes)
        self.net_.load_state_dict(ns)
        self.net_.eval()


# ─────────────────────────────────────────────
# ElasticNet Classifier
# ─────────────────────────────────────────────

class ElasticNetClassifier(BaseForecastModel):
    def __init__(self, Cs=np.logspace(-2, 1, 5), l1_ratios=None,
                 n_splits=5, max_iter=1000):
        super().__init__("classification")
        self.Cs           = Cs
        self.l1_ratios    = l1_ratios or [0.2, 0.5, 0.8]
        self.n_splits     = n_splits
        self.max_iter     = max_iter
        self.scaler       = None
        self.model        = None
        self.best_params_ = None
        self.cv_results_  = []
        self.feature_names_ = None

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Use the trained pkl — fit() not needed here.")

    def _sc(self, X):
        return self.scaler.transform(X if isinstance(X, np.ndarray) else X.values)

    def predict(self, X):
        return self.model.predict(self._sc(X))

    def predict_proba(self, X):
        idx = list(self.model.classes_).index(1)
        return self.model.predict_proba(self._sc(X))[:, idx]

    def evaluate(self, X, y, label=""):
        yt = y if isinstance(y, np.ndarray) else y.values
        return evaluate_classification(yt, self.predict(X), self.predict_proba(X), label=label)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))


# ─────────────────────────────────────────────
# Quick smoke-test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    print("Loading XGBoost model...")
    xgb = XGBoostDirectionClassifier()
    xgb.load(os.path.join(base, "xgboost_wti_direction.pkl"))
    print(f"  [OK] XGBoost loaded | best params: {xgb.best_params_}")

    print("Loading MLP extractor...")
    mlp = MLPFeatureExtractor()
    mlp.load(os.path.join(base, "mlp_extractor_wti.pkl"))
    print(f"  [OK] MLP loaded | hidden layers: {mlp.hidden_layer_sizes} | emb dim: {mlp.embedding_dim_}")

    print("Loading ElasticNet classifier...")
    en = ElasticNetClassifier()
    en.load(os.path.join(base, "mlp_en_wti_direction.pkl"))
    print(f"  [OK] ElasticNet loaded | best params: {en.best_params_}")

    print("\nAll models loaded successfully.")
