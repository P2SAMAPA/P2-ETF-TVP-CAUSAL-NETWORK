"""
features.py  —  From prices (and macro) to the panel the switching model reads.

For ETF e on day t (everything known at the close of t):
  0-3   momentum in vol units: r_t/sigma_t, 5d, 21d and 63d returns scaled by sigma_t*sqrt(horizon)
  4     volatility regime: log(vol21 / vol126)
  5-8   60-day rolling betas of the ETF's returns to: the universe return, the VIX change, the 10y-2y curve change and
        the dollar (DXY) change — what the ETF is exposed to, so that macro moves can be linked to its return
  9-12  today's macro environment, common to every ETF: the universe return, VIX, curve and DXY shocks relative to their
        trailing 60-day volatility (history only)
ETF-specific features (0-8) are standardized across the universe each day; all features are clipped to +-5.
Target: next-day RELATIVE return (minus the universe mean), in units of the ETF's trailing volatility ("risk-adjusted").

A row (day, ETF) is usable only if every feature and its volatility are finite. Context rows for a query day d come from
days d-CTX_DAYS .. d-1 only — their targets (the return from t to t+1, t <= d-1) are known at d.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

import config

F = config.F_MAX
NF = config.N_FEAT


def _roll_beta(Z: np.ndarray, R: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS betas of each column of R (n,E) on the regressors Z (n,k) over a trailing window: (n,E,k); NaN until enough history."""
    n, k = Z.shape
    E = R.shape[1]
    zz = np.einsum("ti,tj->tij", Z, Z).reshape(n, k * k)
    c_zz = np.concatenate([np.zeros((1, k * k)), np.cumsum(zz, axis=0)], axis=0)
    zr = np.einsum("ti,te->tie", Z, np.nan_to_num(R)).reshape(n, k * E)
    c_zr = np.concatenate([np.zeros((1, k * E)), np.cumsum(zr, axis=0)], axis=0)
    out = np.full((n, E, k), np.nan)
    if n < window:
        return out
    S = (c_zz[window:] - c_zz[:-window]).reshape(-1, k, k) + 1e-6 * np.eye(k)[None]
    C = (c_zr[window:] - c_zr[:-window]).reshape(-1, k, E)
    out[window - 1:] = np.linalg.solve(S, C).transpose(0, 2, 1)
    return out


def _tz(x, win: int = 252):
    """History-only time z-score: today's value against the PREVIOUS window's mean and sd, clipped to +-5, NaN -> 0."""
    s = pd.Series(x)
    mu = s.rolling(win, min_periods=60).mean().shift(1)
    sd = s.rolling(win, min_periods=60).std().shift(1)
    z = ((s - mu) / sd.where(sd > 1e-12)).values
    return np.nan_to_num(np.clip(z, -5, 5))


def _avg_corr(R: np.ndarray, w: int) -> np.ndarray:
    n, E = R.shape
    out = np.full(n, np.nan)
    for t in range(w - 1, n):
        blk = R[t - w + 1:t + 1]
        ok = np.isfinite(blk).all(axis=0)
        if ok.sum() < 3:
            continue
        b = blk[:, ok]
        sd = b.std(axis=0, ddof=1)
        b = (b - b.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
        C = b.T @ b / (w - 1)
        k = C.shape[0]
        out[t] = (C.sum() - np.trace(C)) / (k * (k - 1))
    return out


def build_conditions(R: np.ndarray, mac: np.ndarray, macro_df, index, have_macro: bool) -> np.ndarray:
    """
    The U_DIM conditions u_t that drive the regime transitions, all known at the close of t (history-only z-scores):
    market/VIX/curve/DXY shocks (given), VIX level, curve level, DXY trend, average pairwise correlation (20d),
    cross-sectional dispersion (5d), breadth (5d), universe volatility ratio.
    """
    n, E = R.shape
    U = np.zeros((n, config.U_DIM))
    U[:, 0:4] = mac
    if have_macro:
        m = macro_df.reindex(index).ffill()
        U[:, 4] = _tz(np.log(m["VIX"].clip(lower=1e-6)).values)
        U[:, 5] = _tz(m["T10Y2Y"].values)
        U[:, 6] = _tz(np.log(m["DXY"].clip(lower=1e-6)).diff(60).values)
    U[:, 7] = _tz(_avg_corr(R, config.CORR_WINDOW))
    disp = pd.Series(np.nanstd(R, axis=1)).rolling(5, min_periods=5).mean().values
    U[:, 8] = _tz(disp)
    r5 = pd.DataFrame(R).rolling(5, min_periods=5).sum().values
    U[:, 9] = _tz(np.nanmean(np.where(np.isfinite(r5), (r5 > 0).astype(float), np.nan), axis=1))
    mk = pd.Series(np.nanmean(R, axis=1))
    v21, v126 = mk.rolling(21, min_periods=21).std(), mk.rolling(126, min_periods=100).std()
    U[:, 10] = _tz(np.log(v21 / v126).values)
    return np.clip(U, -5, 5)


def build_panel(prices: pd.DataFrame, macro: Optional[pd.DataFrame], tickers: List[str], name: str = "") -> Optional[Dict]:
    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < config.MIN_UNIVERSE:
        return None
    px = prices[tickers].astype(float)
    lr = np.log(px / px.shift(1))
    R = lr.values                                              # (n,E) log returns, row t = return from t-1 to t
    n, E = R.shape
    sig = pd.DataFrame(R).rolling(config.VOL_WINDOW, min_periods=40).std().values        # sigma_t uses returns up to t
    mkt = np.nanmean(R, axis=1)

    def z_macro(x):                                           # today's shock vs the PREVIOUS window's volatility (history only)
        s = pd.Series(x).rolling(config.MACRO_Z_WINDOW, min_periods=30).std().shift(1).values
        return np.clip(x / np.where(s > 1e-12, s, np.nan), -5, 5)

    have_macro = macro is not None and all(c in macro.columns for c in config.MACRO_COLS)
    if have_macro:
        m = macro.reindex(px.index).ffill()
        d_vix = np.log(m["VIX"].clip(lower=1e-6)).diff().values
        d_curve = m["T10Y2Y"].diff().values
        d_dxy = np.log(m["DXY"].clip(lower=1e-6)).diff().values
    else:
        d_vix = d_curve = d_dxy = np.zeros(n)
    regs = np.stack([mkt, d_vix, d_curve, d_dxy], axis=1)
    scale = pd.DataFrame(regs).rolling(252, min_periods=60).std().values
    Zs = np.nan_to_num(regs / np.where(scale > 1e-12, scale, np.nan))
    betas = _roll_beta(Zs, R, config.BETA_WINDOW)              # (n,E,4)

    cs = lambda a: pd.DataFrame(R).rolling(a, min_periods=a).sum().values
    feats = np.full((n, E, F), np.nan)
    sg = np.where(sig > 1e-9, sig, np.nan)
    feats[:, :, 0] = R / sg
    feats[:, :, 1] = cs(5) / (sg * np.sqrt(5))
    feats[:, :, 2] = cs(21) / (sg * np.sqrt(21))
    feats[:, :, 3] = cs(63) / (sg * np.sqrt(63))
    v21 = pd.DataFrame(R).rolling(21, min_periods=21).std().values
    v126 = pd.DataFrame(R).rolling(126, min_periods=100).std().values
    feats[:, :, 4] = np.log(np.where(v21 > 1e-9, v21, np.nan) / np.where(v126 > 1e-9, v126, np.nan))
    feats[:, :, 5:9] = betas
    if not have_macro:
        feats[:, :, 6:9] = 0.0                                 # no macro series -> no macro sensitivities
    base = np.isfinite(feats[:, :, :9]).all(axis=2) & np.isfinite(sg)
    for j in range(9):                                         # cross-sectional standardization of the ETF-specific features
        col = np.where(base, feats[:, :, j], np.nan)
        mu = np.nanmean(col, axis=1, keepdims=True); sd = np.nanstd(col, axis=1, keepdims=True)
        # a feature that is constant across the universe on a day carries no cross-sectional information: 0, not NaN
        feats[:, :, j] = np.where(base, (col - mu) / np.where(sd > 1e-9, sd, 1.0), np.nan)
    mac = np.stack([z_macro(mkt / np.where(np.nanmean(sg, axis=1) > 0, np.nanmean(sg, axis=1), np.nan)), z_macro(d_vix), z_macro(d_curve), z_macro(d_dxy)], axis=1)
    mac = np.nan_to_num(mac)                                   # missing macro shock -> 0 ("no information")
    if not have_macro:
        mac[:] = 0.0
    feats[:, :, 9:13] = mac[:, None, :]
    feats = np.clip(feats, -5, 5)
    valid = np.isfinite(feats[:, :, :NF]).all(axis=2)
    nvalid = valid.sum(axis=1)
    valid &= (nvalid >= config.MIN_ETFS_PER_DAY)[:, None]
    X = np.where(valid[:, :, None], feats, np.nan)
    X[:, :, NF:] = 0.0
    X = np.where(valid[:, :, None], np.nan_to_num(X), 0.0)

    ret_next = np.vstack([R[1:], np.full((1, E), np.nan)])                      # next-day log return (raw units)
    z = np.vstack([R[1:] / sg[:-1], np.full((1, E), np.nan)])                   # next-day return in units of TODAY's volatility
    z = np.clip(z, -8, 8)
    zv = np.where(valid & np.isfinite(z), z, np.nan)
    y = zv - np.nanmean(zv, axis=1, keepdims=True)                              # relative to the universe, per day
    y = np.where(valid & np.isfinite(y), y, np.nan)
    U = build_conditions(R, mac, macro, px.index, have_macro)
    # vol-scaled returns for the VAR: Z[t] = R[t] / sigma[t-1] (volatility known BEFORE day t), clipped; missing -> 0 with a mask
    Zr = np.vstack([np.full((1, E), np.nan), R[1:] / sg[:-1]])
    zmask = np.isfinite(Zr)
    Z = np.clip(np.where(zmask, Zr, 0.0), -config.Z_CLIP, config.Z_CLIP)
    return {"name": name, "dates": px.index, "etfs": tickers, "X": X.astype(np.float32), "valid": valid, "y": y, "ret_next": ret_next, "have_macro": have_macro,
            "U": U.astype(np.float32), "Z": Z.astype(np.float64), "zmask": zmask}




CODE_VERSION = config.CODE_VERSION
