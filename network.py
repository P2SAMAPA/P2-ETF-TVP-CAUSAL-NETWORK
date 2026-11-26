"""
network.py  —  Time-varying directed and undirected networks from a locally stationary VAR(1).

At a grid day g the estimate uses ONLY returns through the close of g. Pairs (x = Z[t-1], y = Z[t]) for t <= g are weighted by a ONE-sided
exponential kernel; the coefficients are LOCAL-LINEAR (level + slope in (t-g)/h), and for every predictor j and target i the pair (level, slope)
is one GROUP in an adaptively weighted group LASSO (weights from a ridge pilot), so an edge is kept or dropped as a whole — the "locally weighted
group sparsity" of the cited paper. The penalty is chosen on a time-ordered hold-out (the most recent VAL_FRAC of the window); "no edges" is a
candidate and is left only if the best candidate beats it by VAL_Z standard errors (then the sparsest penalty within one standard error is taken).
Selected edges are refit without shrinkage and kept only if their own t-statistic is >= EDGE_T. Highly correlated ETFs are handled by projecting
out the leading principal factors first.

The undirected network is the precision matrix of the VAR residuals (graphical lasso on the kernel-weighted residual correlation): partial
correlations, i.e. conditional dependence that remains after the directed edges and the factors are accounted for.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Dict, Optional

import config


def window_idx(g: int, h: float) -> np.ndarray:
    span = min(g, config.STATIC_WINDOW) if not np.isfinite(h) else int(config.MAX_WINDOW_MULT * h)
    lo = max(1, g - span + 1)
    return np.arange(lo, g + 1)


def factor_projector(Y: np.ndarray, w: np.ndarray, r: int):
    """Projector that removes the r leading principal factors of the weighted covariance of Y (m,E). Returns (Pi, L)."""
    E = Y.shape[1]
    C = (Y * w[:, None]).T @ Y / w.sum()
    _, vecs = eigh(C)
    L = vecs[:, -r:] if r > 0 else np.zeros((E, 0))
    return np.eye(E) - L @ L.T, L


def _prox(Bl, Bs, thr):
    nrm = np.sqrt(Bl ** 2 + (Bs ** 2 if Bs is not None else 0.0))
    sc = np.maximum(0.0, 1.0 - thr / np.maximum(nrm, 1e-12))
    return Bl * sc, (Bs * sc if Bs is not None else None)


def _fista(G, C, omega, lam, Lip, E, ll, iters, B0=None):
    """Group lasso for B (p x E), p = 2E (level; slope) if ll else E. Group (j,i) = (B[j,i], B[E+j,i])."""
    B = np.zeros_like(C) if B0 is None else B0.copy()
    Y = B.copy(); tk = 1.0
    thr = lam * omega / Lip
    for _ in range(iters):
        grad = G @ Y - C
        Z_ = Y - grad / Lip
        Bl, Bs = _prox(Z_[:E], Z_[E:] if ll else None, thr)
        Bn = np.vstack([Bl, Bs]) if ll else Bl
        t2 = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
        Y = Bn + ((tk - 1) / t2) * (Bn - B)
        B, tk = Bn, t2
    return B


def _gram(Xt, Y, w):
    Xw = Xt * w[:, None]
    m = Xt.shape[0]
    return Xt.T @ Xw / m, Xw.T @ Y / m


def _prep(Y0, X0, w, u, ll, factor, seg):
    """Project out the leading factors (estimated on `seg` only), build the local-linear design."""
    E = Y0.shape[1]
    Pi = np.eye(E)
    Y, X = Y0, X0
    if factor and config.N_FACTORS > 0:
        Pi, _ = factor_projector(Y0[seg], w[seg], config.N_FACTORS)
        Y, X = Y0 @ Pi, X0 @ Pi
    Xt = np.hstack([X, X * u[:, None]]) if ll else X
    return Pi, Y, Xt


def _weights_and_lam(G, C, E, ll, p):
    """Adaptive group weights from a ridge pilot, and the smallest penalty that zeroes every group."""
    B0 = np.linalg.solve(G + 0.05 * np.trace(G) / p * np.eye(p), C)
    pil = np.sqrt(B0[:E] ** 2 + (B0[E:] ** 2 if ll else 0.0))
    omega = 1.0 / (pil + 1e-2) ** config.ADAPT_GAMMA
    gnorm = np.sqrt(C[:E] ** 2 + (C[E:] ** 2 if ll else 0.0))
    return omega, float((gnorm / omega).max())


def fit_directed(Z: np.ndarray, g: int, h: float, factor: bool = True, local_linear: bool = True, lam_fracs=None) -> Optional[Dict]:
    """
    The directed network at grid day g (None if too little history). The penalty is chosen on a time-ordered hold-out; EVERYTHING that is
    data-dependent in that stage (factor projector, ridge pilot, adaptive weights, lambda_max) is computed from the TRAINING segment only —
    an earlier version computed the pilot on the whole window, which leaked the validation segment into the weights and made the model
    "beat no-edges" out of sample in a world with no edges at all.
    """
    lam_fracs = config.LAM_FRACS if lam_fracs is None else lam_fracs
    t = window_idx(g, h)
    if len(t) < config.MIN_OBS:
        return None
    E = Z.shape[1]
    Y0, X0 = Z[t].copy(), Z[t - 1].copy()
    w = np.exp(-(g - t) / h) if np.isfinite(h) else np.ones(len(t))
    w = w / w.mean()
    u = (t - g) / h if np.isfinite(h) else np.zeros(len(t))
    ll = bool(local_linear and np.isfinite(h))
    m = len(t); cut = int(m * (1 - config.VAL_FRAC))
    # --- stage 1: choose the penalty on a hold-out, using the training segment only for everything data-dependent
    _, Yv_all, Xtv_all = _prep(Y0, X0, w, u, ll, factor, slice(0, cut))
    p = Xtv_all.shape[1]
    Gtr, Ctr = _gram(Xtv_all[:cut], Yv_all[:cut], w[:cut])
    om_tr, lm_tr = _weights_and_lam(Gtr, Ctr, E, ll, p)
    Lip_tr = float(np.linalg.eigvalsh(Gtr)[-1]) + 1e-9
    Xv, Yv, wv = Xtv_all[cut:], Yv_all[cut:], w[cut:]
    loss0 = wv * (Yv ** 2).sum(axis=1)                                                   # per-observation hold-out loss of the empty network
    cand = []; Bw = None
    for f in sorted(lam_fracs, reverse=True):
        Bw = _fista(Gtr, Ctr, om_tr, f * lm_tr, Lip_tr, E, ll, config.FISTA_ITERS, Bw)
        d = loss0 - wv * ((Yv - Xv @ Bw) ** 2).sum(axis=1)                               # improvement over "no edges", per observation
        cand.append((f, float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))))
    # margin + one-standard-error rule: leave the empty network only if the best candidate beats it by VAL_Z standard errors; then take the
    # SPARSEST penalty whose improvement is within one standard error of the best (a bare "beats zero" is a coin flip between near-zero models)
    kbest = max(range(len(cand)), key=lambda k: cand[k][1])
    best_f = None
    if cand[kbest][2] > 0 and cand[kbest][1] / cand[kbest][2] >= config.VAL_Z:
        ok = [c for c in cand if c[1] >= cand[kbest][1] - cand[kbest][2]]
        best_f = max(c[0] for c in ok)
    Pi, Y, Xt = _prep(Y0, X0, w, u, ll, factor, slice(None))
    out = {"g": g, "lam_frac": best_f, "Pi": Pi, "A": np.zeros((E, E)), "mask": np.zeros((E, E), bool), "n_edges": 0, "nobs": m, "B": np.zeros((p, E)), "ll": ll}
    if best_f is None:
        out["_win"] = (Xt, Y, w)                                                        # residuals = the (factor-adjusted) returns themselves
        return out                                                                      # nothing beats zero out of sample: an empty network
    # --- stage 2: refit on the whole window with the chosen penalty fraction (pilot and weights now from the full window)
    G, C = _gram(Xt, Y, w)
    omega, lam_max = _weights_and_lam(G, C, E, ll, p)
    Lip = float(np.linalg.eigvalsh(G)[-1]) + 1e-9
    B = _fista(G, C, omega, best_f * lam_max, Lip, E, ll, config.FISTA_ITERS * 2)
    sel = (np.sqrt(B[:E] ** 2 + (B[E:] ** 2 if ll else 0.0)) > 0)
    Bd = np.zeros_like(B)
    keep = np.zeros((E, E), bool)                                                       # keep[i,j]: edge j -> i survives the t-test
    m_eff = (w.sum() ** 2) / (w ** 2).sum()
    for i in range(E):                                                                  # post-selection refit without shrinkage + t-test
        S = np.where(sel[:, i])[0]
        if len(S) == 0:
            continue
        cols = np.concatenate([S, S + E]) if ll else S
        Gs = G[np.ix_(cols, cols)] + 1e-3 * np.trace(G) / p * np.eye(len(cols))
        Gi = np.linalg.inv(Gs)
        Bd[cols, i] = Gi @ C[cols, i]
        res = Y[:, i] - Xt[:, cols] @ Bd[cols, i]
        s2 = float((w * res ** 2).sum() / w.sum())
        se = np.sqrt(np.maximum(s2 * np.diag(Gi)[:len(S)] / m_eff, 1e-18))
        keep[i, S] = np.abs(Bd[S, i] / se) >= config.EDGE_T
        for j in S[~keep[i, S]]:                                                        # drop edges that fail the t-test
            Bd[j, i] = 0.0
            if ll:
                Bd[j + E, i] = 0.0
    A = Bd[:E].T                                                                        # A[i,j]: effect of x_j on y_i at the end of the window
    mask = keep & (np.abs(A) > 0)
    A = np.where(mask, A, 0.0)
    out.update({"A": A, "mask": mask, "n_edges": int(mask.sum() - np.trace(mask)), "B": Bd, "_win": (Xt, Y, w)})
    return out


def fit_undirected(fit: Dict, alpha: float = None) -> Optional[Dict]:
    """Partial-correlation network from the kernel-weighted residual correlation of a directed fit (graphical lasso)."""
    from sklearn.covariance import graphical_lasso
    alpha = config.GLASSO_ALPHA if alpha is None else alpha
    if "_win" not in fit:
        return None
    Xt, Y, w = fit["_win"]
    eps = Y - Xt @ fit["B"]
    S = (eps * w[:, None]).T @ eps / w.sum()
    d = np.sqrt(np.maximum(np.diag(S), 1e-9))
    R = S / np.outer(d, d)
    R = 0.95 * R + 0.05 * np.eye(R.shape[0])
    try:
        _, prec = graphical_lasso(R, alpha=alpha, max_iter=60, tol=1e-3)
    except Exception:
        return None
    dd = np.sqrt(np.maximum(np.diag(prec), 1e-9))
    P = -prec / np.outer(dd, dd)
    np.fill_diagonal(P, 0.0)
    P[np.abs(P) < 1e-6] = 0.0
    return {"P": P, "mask": np.abs(P) > 0}


def predict(fit: Dict, z_t: np.ndarray) -> np.ndarray:
    """Next-day forecast (in the idiosyncratic space of the fit) from today's vol-scaled returns."""
    return fit["A"] @ (z_t @ fit["Pi"])


def ridge_var(Z: np.ndarray, g: int, window: int = None, lam: float = None, factor: bool = False):
    """Competitor without sparsity: a ridge VAR on the last `window` days."""
    window = config.ROLL_RIDGE_WINDOW if window is None else window
    lam = config.VAR_RIDGE if lam is None else lam
    t = np.arange(max(1, g - window + 1), g + 1)
    if len(t) < 30:
        return None
    E = Z.shape[1]
    Y, X = Z[t], Z[t - 1]
    Pi = np.eye(E)
    if factor and config.N_FACTORS > 0:
        Pi, _ = factor_projector(Y, np.ones(len(t)), config.N_FACTORS)
        Y, X = Y @ Pi, X @ Pi
    A = np.linalg.solve(X.T @ X + lam * np.eye(E), X.T @ Y).T
    return {"A": A, "Pi": Pi}


def change_distance(A1: np.ndarray, A2: np.ndarray) -> float:
    num = np.linalg.norm(A1 - A2); den = np.linalg.norm(A1) + np.linalg.norm(A2)
    return float(num / den) if den > 1e-12 else 0.0


CODE_VERSION = config.CODE_VERSION
