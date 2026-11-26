"""
simworld.py  —  Synthetic ETF markets with a KNOWN directed network, a KNOWN partial-correlation network and known change times.

z_t = Lambda f_t + u_t           f_t: a common factor (contemporaneous correlation), u_t: idiosyncratic
u_t = A_t u_{t-1} + e_t          A_t: a sparse directed network (who predicts whom); e_t ~ N(0, Omega_t^{-1}): a sparse partial-correlation network
Worlds:
  null      no directed edges (A = 0); weak factor; fixed partial-correlation network
  confound  no directed edges, but a STRONG common factor: lots of correlation, no lead-lag (correlation without causality)
  static    a fixed set of directed edges
  tv        directed edges that switch on and off at RANDOM times, and a partial-correlation network that changes at other random times
  trigger   the same, but the switching is DRIVEN by observable conditions (edges follow the cross-ETF correlation and dispersion states, the
            partial-correlation network follows VIX level / dispersion / correlation): structure change that is predictable from conditions
The change times are known: when any edge's activity crosses one half. Panels carry the real layout (Z, y, ret_next, random features X, conditions U).
"""

import numpy as np
import pandas as pd
from typing import Dict

import config

F, NF, DU = config.F_MAX, config.N_FEAT, config.U_DIM


def _activity(n, rng, mean_len=200):
    """A smooth 0/1 activity path with random segment lengths."""
    s = np.zeros(n); t = 0; on = rng.random() < 0.5
    while t < n:
        L = int(rng.integers(int(0.6 * mean_len), int(1.6 * mean_len)))
        s[t:t + L] = 1.0 if on else 0.0
        t += L; on = not on
    k = np.exp(-np.arange(-30, 31) ** 2 / (2 * 10.0 ** 2)); k /= k.sum()
    return np.convolve(np.pad(s, 30, mode="edge"), k, mode="same")[30:-30]


def _smooth01(s):
    k = np.exp(-np.arange(-15, 16) ** 2 / (2 * 5.0 ** 2)); k /= k.sum()
    return np.convolve(np.pad(s.astype(float), 15, mode="edge"), k, mode="same")[15:-15]


def make_world(kind: str, n: int = 1000, E: int = 20, amp: float = 0.16, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    Uc = np.zeros((n, DU)); ue = rng.normal(size=(n, DU)); Uc[0] = ue[0]
    for t in range(1, n):
        Uc[t] = 0.97 * Uc[t - 1] + np.sqrt(1 - 0.97 ** 2) * ue[t]
    Uc = np.clip(Uc / Uc.std(axis=0), -4, 4)
    strong = kind == "confound"
    lam = rng.uniform(0.6, 0.95, E) if strong else rng.uniform(0.2, 0.5, E)
    pairs = [(int(a), int(b)) for a, b in (rng.choice(E, 2, replace=False) for _ in range(6))]
    pairs = list(dict.fromkeys(pairs))
    A_true = np.zeros((n, E, E)); acts = []
    for k_, (i, j) in enumerate(pairs):                                         # edge i -> j : A[j, i]
        sgn = rng.choice([-1.0, 1.0])
        if kind == "trigger":
            s = _smooth01(Uc[:, 7] > 0) if k_ % 2 == 0 else _smooth01(Uc[:, 8] > 0)
        else:
            s = np.ones(n) if kind == "static" else (_activity(n, rng) if kind == "tv" else np.zeros(n))
        acts.append(s); A_true[:, j, i] = sgn * amp * s
    if kind in ("null", "confound"):
        A_true[:] = 0.0
    # partial-correlation network: base edges, with a few toggled in the tv / trigger worlds
    adj_base = np.zeros((E, E))
    for _ in range(8):
        a, b = rng.choice(E, 2, replace=False); adj_base[a, b] = adj_base[b, a] = 1.0
    toggles = [tuple(rng.choice(E, 2, replace=False)) for _ in range(3)] if kind in ("tv", "trigger") else []
    if kind == "trigger":
        tog_act = [_smooth01(Uc[:, 4] > 0), _smooth01(Uc[:, 8] > 0), _smooth01(Uc[:, 7] < 0)]
    else:
        tog_act = [_activity(n, rng, 260) for _ in toggles]
    Pt = np.zeros((n, E, E), bool)
    u = np.zeros((n, E)); Ls = {}
    for t in range(1, n):
        adj = adj_base.copy()
        for (a, b), s in zip(toggles, tog_act):
            if s[t] > 0.5:
                adj[a, b] = adj[b, a] = 1.0
        Pt[t] = adj > 0
        key = adj.tobytes()
        if key not in Ls:
            Om = np.eye(E) + 0.3 * adj; Ls[key] = np.linalg.cholesky(np.linalg.inv(Om))
        u[t] = A_true[t] @ u[t - 1] + Ls[key] @ rng.normal(size=E)
    f = rng.normal(size=n)
    z = np.clip(lam[None, :] * f[:, None] + u, -config.Z_CLIP, config.Z_CLIP)
    y = z[1:] - z[1:].mean(axis=1, keepdims=True)
    Y = np.vstack([y, np.full((1, E), np.nan)])
    ret_next = np.vstack([z[1:] * 0.01, np.full((1, E), np.nan)])
    ar = np.zeros((n, E, NF)); e = rng.normal(size=(n, E, NF)); ar[0] = e[0]
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + np.sqrt(1 - 0.25) * e[t]
    X = np.zeros((n, E, F)); X[:, :, :NF] = (ar - ar.mean(axis=1, keepdims=True)) / np.maximum(ar.std(axis=1, keepdims=True), 1e-9)
    change = np.zeros(n, bool)
    for s in acts + tog_act:
        c = np.where(np.diff((s > 0.5).astype(int)) != 0)[0] + 1; change[c] = True
    return {"name": "SIM_" + kind, "dates": pd.bdate_range("2019-01-02", periods=n), "etfs": ["E%02d" % i for i in range(E)], "X": X.astype(np.float32),
            "valid": np.ones((n, E), bool), "y": Y, "ret_next": ret_next, "have_macro": True, "U": Uc.astype(np.float32),
            "Z": z, "zmask": np.ones((n, E), bool), "A_true": A_true, "P_true": Pt, "pairs": pairs, "change_times": np.where(change)[0], "kind": kind}
