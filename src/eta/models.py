"""B0 baseline, LightGBM quantile models, PyTorch embedding quantile model."""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from torch import nn

from . import config as C
from .features import CAT_FEATURES, FEATURES, N_ZONES

QUANTILES = (0.5, 0.9)


# ----------------------------------------------------------------------------- metrics
def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def metrics(y, p50, p90) -> dict:
    return {
        "n": int(len(y)),
        "mae_min": float(np.mean(np.abs(y - p50))),
        "pinball50": pinball(y, p50, 0.5),
        "pinball90": pinball(y, p90, 0.9),
        "p90_coverage": float(np.mean(y <= p90)),
        "bias_min": float(np.mean(p50 - y)),
    }


# ----------------------------------------------------------------------------- B0
class HistoricalMedian:
    """Zone x hour-of-week quantiles from training labels; zone, then global as fallback."""

    def fit(self, zone, how, y):
        df = pl.DataFrame({"zone": zone, "how": how, "y": y})
        agg = lambda g: df.group_by(g).agg(pl.col("y").quantile(0.5).alias("q50"),
                                           pl.col("y").quantile(0.9).alias("q90"),
                                           pl.len().alias("n"))
        self.global_ = (float(np.quantile(y, 0.5)), float(np.quantile(y, 0.9)))
        self.zone_ = np.tile(np.array(self.global_), (N_ZONES, 1))
        for z, q50, q90, n in agg(["zone"]).iter_rows():
            self.zone_[z] = (q50, q90)
        self.zh_ = np.repeat(self.zone_[:, None, :], 168, axis=1).copy()
        for z, h, q50, q90, n in agg(["zone", "how"]).iter_rows():
            if n >= 5:   # thin cells fall back to the zone quantile
                self.zh_[z, h] = (q50, q90)
        return self

    def predict(self, zone, how):
        out = self.zh_[np.asarray(zone, dtype=np.int64), np.asarray(how, dtype=np.int64)]
        return out[:, 0], out[:, 1]

    def predict_global(self, n):
        return np.full(n, self.global_[0]), np.full(n, self.global_[1])

    def save(self, path):
        np.savez(path, zh=self.zh_, zone=self.zone_, glob=np.array(self.global_))

    @classmethod
    def load(cls, path):
        m, d = cls(), np.load(path)
        m.zh_, m.zone_, m.global_ = d["zh"], d["zone"], tuple(d["glob"])
        return m


# ----------------------------------------------------------------------------- LightGBM
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=4,
                  cat_smooth=10, verbose=-1, num_threads=2, seed=C.SEED)


class LGBQuantile:
    def fit(self, X: np.ndarray, y, Xv, yv, rounds=1500, log=None):
        cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]
        self.boosters, self.best_iter = {}, {}
        for a in QUANTILES:
            t0 = time.time()
            dtr = lgb.Dataset(X, y, feature_name=FEATURES, categorical_feature=cat_idx, free_raw_data=False)
            dva = lgb.Dataset(Xv, yv, reference=dtr, categorical_feature=cat_idx)
            b = lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": a, "metric": "quantile"},
                          dtr, rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            self.boosters[a], self.best_iter[a] = b, b.best_iteration
            if log:
                log(f"lgb alpha={a}: best_iter={b.best_iteration} ({time.time()-t0:.0f}s)")
        return self

    def predict(self, X):
        p50 = self.boosters[0.5].predict(X, num_iteration=self.best_iter[0.5])
        p90 = self.boosters[0.9].predict(X, num_iteration=self.best_iter[0.9])
        return p50, np.maximum(p90, p50)   # no quantile crossing

    def save(self, prefix):
        for a, b in self.boosters.items():
            b.save_model(f"{prefix}_q{int(a*100)}.txt", num_iteration=self.best_iter[a])

    @classmethod
    def load(cls, prefix):
        m = cls()
        m.boosters = {a: lgb.Booster(model_file=f"{prefix}_q{int(a*100)}.txt") for a in QUANTILES}
        m.best_iter = {a: None for a in QUANTILES}
        return m


# ----------------------------------------------------------------------------- PyTorch


def torch_inputs(feat: dict | pl.DataFrame, b0_p50: np.ndarray):
    """Numeric block (log counts, imputed wait + missing flag, cyclic time) and categorical ids."""
    g = (lambda k: np.asarray(feat[k], dtype=np.float32))
    wm = g("wait_mean_30")
    miss = np.isnan(wm)
    wm = np.where(miss, b0_p50, wm).astype(np.float32)   # impute with the baseline estimate
    w10 = g("wait_mean_10")
    miss10 = np.isnan(w10)
    w10 = np.where(miss10, wm, w10).astype(np.float32)
    mod = g("minute_of_day") * (2 * np.pi / 1440)
    num = np.stack([
        np.log1p(g("req_5")), np.log1p(g("req_15")), np.log1p(g("req_60")),
        np.log1p(g("drop_15")), np.log1p(g("drop_60")), np.log(g("ds_ratio_15")),
        wm, miss.astype(np.float32), np.log1p(g("wait_n_30")), w10, miss10.astype(np.float32),
        np.log1p(g("open_reqs")), np.log1p(g("open_age_mean")), g("wav"), g("is_airport"),
        np.sin(mod), np.cos(mod),
    ], axis=1).astype(np.float32)
    cat = np.stack([g("zone"), g("how"), g("borough_id")], axis=1).astype(np.int64)
    return num, cat


class ETANet(nn.Module):
    """Embeddings for zone / hour-of-week / borough + MLP, two quantile heads (P50, P90 >= P50).
    Same family as Uber's DeepETA idea (embed categoricals, asymmetric loss), much smaller."""

    def __init__(self, n_num: int, d_zone=16, d_how=8, d_bor=3, hidden=(256, 128)):
        super().__init__()
        self.e_zone, self.e_how, self.e_bor = nn.Embedding(N_ZONES, d_zone), nn.Embedding(168, d_how), nn.Embedding(8, d_bor)
        layers, d = [], n_num + d_zone + d_how + d_bor
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.05)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2)

    def forward(self, num, cat):
        x = torch.cat([num, self.e_zone(cat[:, 0]), self.e_how(cat[:, 1]), self.e_bor(cat[:, 2])], 1)
        o = self.head(self.mlp(x))
        q50 = o[:, 0]
        q90 = q50 + nn.functional.softplus(o[:, 1])
        return q50, q90


def pinball_t(y, q, a):
    d = y - q
    return torch.mean(torch.maximum(a * d, (a - 1) * d))


class TorchQuantile:
    def fit(self, num, cat, y, numv, catv, yv, epochs=6, bs=4096, lr=2e-3, log=None):
        torch.manual_seed(C.SEED); np.random.seed(C.SEED)
        torch.set_num_threads(2)
        self.mu, self.sd = num.mean(0), num.std(0) + 1e-6
        nt, ct, yt = self._t(num), torch.from_numpy(cat), torch.from_numpy(y.astype(np.float32))
        self.net = ETANet(num.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best, best_state, self.history = np.inf, None, []
        for ep in range(epochs):
            t0 = time.time(); self.net.train()
            perm = torch.randperm(len(yt))
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                q50, q90 = self.net(nt[j], ct[j])
                loss = pinball_t(yt[j], q50, 0.5) + pinball_t(yt[j], q90, 0.9)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            p50, p90 = self.predict(numv, catv)
            vloss = pinball(yv, p50, 0.5) + pinball(yv, p90, 0.9)
            self.history.append({"epoch": ep, "val_pinball_sum": vloss, "sec": time.time() - t0})
            if log:
                log(f"torch epoch {ep}: val pinball50+90={vloss:.4f} ({time.time()-t0:.0f}s)")
            if vloss < best:
                best, best_state = vloss, {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        return self

    def _t(self, num):
        return torch.from_numpy(((num - self.mu) / self.sd).astype(np.float32))

    @torch.no_grad()
    def predict(self, num, cat, bs=65536):
        self.net.eval()
        out50, out90 = [], []
        for i in range(0, len(num), bs):
            a, b = self.net(self._t(num[i:i + bs]), torch.from_numpy(cat[i:i + bs]))
            out50.append(a.numpy()); out90.append(b.numpy())
        return np.concatenate(out50), np.concatenate(out90)

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        m = cls(); m.net = ETANet(len(d["mu"])); m.net.load_state_dict(d["state"]); m.mu, m.sd = d["mu"], d["sd"]
        return m
