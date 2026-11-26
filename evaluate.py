"""
evaluate.py  —  Do time-varying directed networks forecast ETF returns, and is network change predictable?

Variants and competitors on identical (day, ETF) rows:
  tv_net       the time-varying, factor-adjusted directed network (one-sided local-linear adaptive group LASSO, hold-out penalty, edge t-test)
  tv_nofactor  the same without factor adjustment            static_net  the same with a fixed network (no time variation)
  roll_var     a ridge VAR on the last 60 days (no sparsity)  roll_ridge / fixed_ridge / lgbm / momentum: feature-based models of the earlier engines
The network variants ABSTAIN on days when they select no edges (IC counted as 0, no trade); the share of active days is reported.
Network CHANGE probability: P(the network moves materially over the next CHANGE_K sessions), a logistic model on the network's own recent
behaviour and market conditions (undirected target = primary, always defined; directed = secondary); scored by AUC and Brier skill against the base
rate, against a "recent change persists" score, and as a confidence gate for the forecasts.
"""

import numpy as np
from scipy.stats import rankdata
from typing import Dict, List

import config
import network as N

NF = config.N_FEAT
NETWORK_VARIANTS = ("tv_net", "tv_nofactor", "static_net")
COMPETITORS = ("tv_nofactor", "static_net", "roll_var", "roll_ridge", "fixed_ridge", "lgbm", "momentum")
HIST = 504


def tstat(x) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def daily_ic(score, y, min_etfs=None):
    min_etfs = config.MIN_ETFS_PER_DAY if min_etfs is None else min_etfs
    out = np.full(score.shape[0], np.nan)
    ok = np.isfinite(score) & np.isfinite(y)
    for t in np.where(ok.sum(axis=1) >= min_etfs)[0]:
        m = ok[t]
        a, b = rankdata(score[t, m]), rankdata(y[t, m])
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        out[t] = (a * b).sum() / den if den > 0 else np.nan
    return out


def topn_series(score, ret_next, top_n=None, cost_bps=None):
    top_n = config.TOP_N if top_n is None else top_n
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    n = score.shape[0]
    gross, net = np.full(n, np.nan), np.full(n, np.nan)
    prev = None
    for t in range(n):
        m = np.isfinite(score[t]) & np.isfinite(ret_next[t])
        if m.sum() < config.MIN_ETFS_PER_DAY:
            prev = None
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(-score[t, idx])][:top_n]
        g = ret_next[t, order].mean() - ret_next[t, idx].mean()
        repl = (top_n - len(set(order.tolist()) & prev)) / top_n if prev is not None else 0.0
        gross[t], net[t] = g, g - repl * cost_bps / 1e4
        prev = set(order.tolist())
    return gross, net


# ---------------------------------------------------------------------------
# Feature-based competitors (as in the earlier engines)
# ---------------------------------------------------------------------------
def _rows(panel, lo, hi):
    ok = panel["valid"][lo:hi] & np.isfinite(panel["y"][lo:hi])
    t, e = np.where(ok)
    return panel["X"][lo:hi][t, e][:, :NF], panel["y"][lo:hi][t, e]


def rolling_ridge(panel, days, K=config.ROLL_WINDOW, lam=config.ROLL_RIDGE_ALPHA):
    n, E = panel["valid"].shape
    mu = np.full((n, E), np.nan)
    for d in days:
        X, y = _rows(panel, max(d - K, 0), d)
        if len(y) < 40:
            continue
        beta = np.linalg.solve(X.T @ X + lam * np.eye(NF), X.T @ y)
        qi = np.where(panel["valid"][d])[0]
        mu[d, qi] = panel["X"][d, qi][:, :NF] @ beta
    return mu


def fixed_models(panel, starts, blocks, fit_lgbm=True):
    n, E = panel["valid"].shape
    mu_r = np.full((n, E), np.nan); mu_g = np.full((n, E), np.nan)
    for b0, days in zip(starts, blocks):
        X, y = _rows(panel, 0, b0)
        if len(y) < 500:
            continue
        beta = np.linalg.solve(X.T @ X + config.RIDGE_ALPHA * np.eye(NF), X.T @ y)
        gm = None
        if fit_lgbm:
            import lightgbm as lgb
            gm = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(X, y)
        for d in days:
            qi = np.where(panel["valid"][d])[0]
            Xq = panel["X"][d, qi][:, :NF]
            mu_r[d, qi] = Xq @ beta
            if gm is not None:
                mu_g[d, qi] = gm.predict(Xq)
    return mu_r, mu_g


# ---------------------------------------------------------------------------
# Networks over a grid of estimation days
# ---------------------------------------------------------------------------
def eval_window(panel, eval_days=None):
    n = len(panel["dates"])
    ed = int(config.EVAL_YEARS * 252) if eval_days is None else eval_days
    d1 = n - 1
    return max(config.MIN_HISTORY_DAYS, d1 - ed), d1


def estimate_grid(panel: Dict, grid: List[int], undirected: bool = True, variants=("tv_net", "tv_nofactor", "static_net", "roll_var"), log=None) -> Dict:
    Z = panel["Z"]
    out = {v: {} for v in variants}
    for k, g in enumerate(grid):
        if "tv_net" in variants:
            f = N.fit_directed(Z, g, config.HALF_LIFE, True, True)
            if f is not None:
                keep = {"A": f["A"], "Pi": f["Pi"], "mask": f["mask"], "n_edges": f["n_edges"], "lam_frac": f["lam_frac"]}
                if undirected and "_win" in f:
                    u = N.fit_undirected(f)
                    keep["P"] = u["P"] if u else None
                out["tv_net"][g] = keep
        if "tv_nofactor" in variants:
            f = N.fit_directed(Z, g, config.HALF_LIFE, False, True)
            if f is not None:
                out["tv_nofactor"][g] = {"A": f["A"], "Pi": f["Pi"], "mask": f["mask"], "n_edges": f["n_edges"]}
        if "static_net" in variants:
            f = N.fit_directed(Z, g, np.inf, True, False)
            if f is not None:
                out["static_net"][g] = {"A": f["A"], "Pi": f["Pi"], "mask": f["mask"], "n_edges": f["n_edges"]}
        if "roll_var" in variants:
            f = N.ridge_var(Z, g)
            if f is not None:
                out["roll_var"][g] = {"A": f["A"], "Pi": f["Pi"], "mask": np.ones_like(f["A"], bool), "n_edges": -1}
        if log and (k + 1) % 100 == 0:
            log("    network grid %d/%d" % (k + 1, len(grid)))
    return out


def forecast_scores(panel: Dict, fits: Dict, grid: List[int], days):
    Z = panel["Z"]
    n, E = Z.shape
    sc = np.full((n, E), np.nan); active = np.zeros(n, bool)
    ga = np.array(grid)
    for d in days:
        k = np.searchsorted(ga, d, side="right") - 1
        if k < 0:
            continue
        f = fits.get(int(ga[k]))
        if f is None or not np.any(f["A"] != 0):
            continue
        pred = f["A"] @ (Z[d] @ f["Pi"])
        sc[d] = np.where(panel["valid"][d], pred, np.nan)
        active[d] = True
    return sc, active


# ---------------------------------------------------------------------------
# Network-change probability
# ---------------------------------------------------------------------------
def change_dataset(panel: Dict, fits: Dict, grid: List[int]):
    """Per grid index m: features known at g_m and the forward distances (directed, undirected) over the next CHANGE_K sessions."""
    kk = max(config.CHANGE_K // config.GRID_STEP, 1)
    F = fits["tv_net"]
    ms = [m for m, g in enumerate(grid) if g in F]
    A = {m: F[grid[m]]["A"] for m in ms}
    P = {m: F[grid[m]].get("P") for m in ms}
    nE = {m: F[grid[m]]["n_edges"] for m in ms}
    U = panel["U"]
    rows = []
    for m in ms:
        g = grid[m]
        if (m - kk) not in A:
            continue
        dA = N.change_distance(A[m - kk], A[m])
        dP = N.change_distance(P[m - kk], P[m]) if (P.get(m - kk) is not None and P.get(m) is not None) else 0.0
        pm = P.get(m)
        feat = [dA, dP, nE[m], float(np.linalg.norm(A[m])), float(np.abs(pm).mean()) if pm is not None else 0.0, nE[m] - nE[m - kk],
                float(U[g, 4]), float(U[g, 7]), float(U[g, 8]), float(U[g, 10])]
        futA = N.change_distance(A[m], A[m + kk]) if (m + kk) in A else np.nan
        futP = N.change_distance(P[m], P[m + kk]) if (P.get(m) is not None and P.get(m + kk) is not None) else np.nan
        rows.append((m, g, feat, futA, futP))
    return rows


def change_walk_forward(rows, grid, eval_lo_g: int, k_ahead: int, which: int = 3, persist_col: int = 0):
    """Expanding logistic model; labels are known only k_ahead grid steps later. Returns p, label, base rate, persistence score (aligned to rows)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    m_arr = np.array([r[0] for r in rows]); X = np.array([r[2] for r in rows], float); fut = np.array([r[which] for r in rows], float)
    nrow = len(rows)
    thr = np.full(nrow, np.nan); label = np.full(nrow, np.nan)
    for i in range(nrow):                                   # expanding 75th percentile of distances that are KNOWN at row i's time
        known = fut[:i + 1][(m_arr[:i + 1] + k_ahead) <= m_arr[i]]
        known = known[np.isfinite(known)]
        if len(known) >= 20:
            thr[i] = np.quantile(known, config.CHANGE_QUANTILE)
            if np.isfinite(fut[i]):
                label[i] = float(fut[i] >= thr[i])
    p = np.full(nrow, np.nan); p_persist = np.full(nrow, np.nan); base = np.full(nrow, np.nan)
    last_fit = -10 ** 9; model = None; scaler = None
    for i in range(nrow):
        if not np.isfinite(thr[i]) or rows[i][1] < eval_lo_g:
            continue
        if i - last_fit >= config.CHANGE_REFIT_GRID or model is None:
            tr = [j for j in range(i) if np.isfinite(label[j]) and (m_arr[j] + k_ahead) <= m_arr[i]]
            ytr = label[tr] if tr else np.array([])
            if len(tr) >= config.CHANGE_MIN_TRAIN and 0 < ytr.sum() < len(ytr):
                scaler = StandardScaler().fit(X[tr]); model = LogisticRegression(C=0.5, max_iter=300).fit(scaler.transform(X[tr]), ytr)
                last_fit = i
        if model is not None:
            p[i] = float(model.predict_proba(scaler.transform(X[i:i + 1]))[0, 1])
        known = label[:i][np.isfinite(label[:i]) & ((m_arr[:i] + k_ahead) <= m_arr[i])]
        base[i] = float(known.mean()) if len(known) else np.nan
        p_persist[i] = float(X[i, persist_col]) if i > 20 else np.nan          # the same kind of change over the previous CHANGE_K sessions
    return p, label, base, p_persist


def auc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p); y, p = y[ok], p[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def change_summary(p, label, base, p_persist) -> Dict:
    ok = np.isfinite(p) & np.isfinite(label) & np.isfinite(base)
    if ok.sum() < 30 or len(np.unique(label[ok])) < 2:
        return {"n": int(ok.sum()), "note": "too few labelled grid points"}
    brier = float(np.mean((p[ok] - label[ok]) ** 2)); brier0 = float(np.mean((base[ok] - label[ok]) ** 2))
    okp = ok & np.isfinite(p_persist)
    return {"n": int(ok.sum()), "base_rate": float(label[ok].mean()), "auc": auc(label[ok], p[ok]), "auc_persistence": auc(label[okp], p_persist[okp]) if okp.sum() > 30 else None,
            "brier": brier, "brier_base": brier0, "brier_skill": float(1 - brier / brier0) if brier0 > 0 else None}


# ---------------------------------------------------------------------------
# Walk-forward and metrics
# ---------------------------------------------------------------------------
def run_walk_forward(panel: Dict, eval_days: int = None, fit_lgbm: bool = True, undirected: bool = True, log=None) -> Dict:
    d0, d1 = eval_window(panel, eval_days)
    start = max(config.MIN_OBS + 1, d0 - HIST)
    grid = list(range(start, d1, config.GRID_STEP))
    fits = estimate_grid(panel, grid, undirected=undirected, log=log)
    days = np.arange(d0, d1)
    scores, active = {}, {}
    for v in ("tv_net", "tv_nofactor", "static_net", "roll_var"):
        scores[v], active[v] = forecast_scores(panel, fits[v], grid, days)
    starts = list(range(d0, d1, config.REFIT_DAYS))
    blocks = [list(range(b, min(b + config.REFIT_DAYS, d1))) for b in starts]
    scores["roll_ridge"] = rolling_ridge(panel, days)
    scores["fixed_ridge"], scores["lgbm"] = fixed_models(panel, starts, blocks, fit_lgbm=fit_lgbm)
    scores["momentum"] = np.where(panel["valid"], panel["X"][:, :, 2], np.nan)
    res = {"scores": scores, "active": active, "days": days, "window": (d0, d1), "grid": grid, "fits": fits}
    if undirected:
        rows = change_dataset(panel, fits, grid)
        kk = max(config.CHANGE_K // config.GRID_STEP, 1)
        if len(rows) > 60:
            res["change"] = {"rows": [(r[0], r[1]) for r in rows], "feat": np.array([r[2] for r in rows], float)}
            for name, which, pc in (("und", 4, 1), ("dir", 3, 0)):             # undirected (always defined) is the primary target
                p, label, base, pp = change_walk_forward(rows, grid, d0, kk, which, pc)
                res["change"][name] = {"p": p, "label": label, "base": base, "p_persist": pp}
    return res


def summarize(panel: Dict, r: Dict) -> Dict:
    days = r["days"]
    mask = np.zeros(panel["valid"].shape[0], bool); mask[days] = True
    out = {"methods": {}, "paired": {}, "n_days": 0}
    ser = {}
    for k, sc in r["scores"].items():
        if k not in ("tv_net", "tv_nofactor", "static_net", "roll_var", "roll_ridge", "fixed_ridge", "lgbm", "momentum"):
            continue
        ic = daily_ic(sc, panel["y"]); gr, nt = topn_series(sc, panel["ret_next"])
        valid_day = np.isfinite(panel["y"]).sum(axis=1) >= config.MIN_ETFS_PER_DAY
        if k in NETWORK_VARIANTS:                                              # abstention: no forecast = IC 0, no trade
            sil = mask & valid_day & ~r["active"][k]
            ic = np.where(sil, 0.0, ic); gr = np.where(sil, 0.0, gr); nt = np.where(sil, 0.0, nt)
        ser[k] = (ic, gr, nt)
        if not np.isfinite(ic[mask]).any():
            continue
        out["methods"][k] = {"ic_mean": float(np.nanmean(ic[mask])), "ic_t": tstat(ic[mask]), "gross_bps": float(np.nanmean(gr[mask]) * 1e4), "net_bps": float(np.nanmean(nt[mask]) * 1e4),
                             "net_t": tstat(nt[mask]), "n_days": int(np.isfinite(ic[mask]).sum()), "active_share": float(r["active"][k][mask].mean()) if k in r["active"] else 1.0}
    out["n_days"] = out["methods"].get("tv_net", {}).get("n_days", 0)
    comp = [k for k in COMPETITORS if k in out["methods"]]
    out["best_competitor"] = max(comp, key=lambda k: out["methods"][k]["ic_mean"]) if comp else None
    for base in COMPETITORS:
        if "tv_net" in ser and base in ser:
            d = (ser["tv_net"][0] - ser[base][0])[mask]; dn = (ser["tv_net"][2] - ser[base][2])[mask]
            out["paired"][base] = {"d_ic": float(np.nanmean(d)), "t": tstat(d), "d_net_bps": float(np.nanmean(dn) * 1e4), "net_t": tstat(dn)}
    # the network-change probability (undirected = primary, directed = secondary) and its use as a confidence gate
    ch = r.get("change")
    if ch:
        for name in ("und", "dir"):
            out["change_" + name] = change_summary(ch[name]["p"], ch[name]["label"], ch[name]["base"], ch[name]["p_persist"])
        pp = ch["und"]["p"]
        ok = np.isfinite(pp)
        if "tv_net" in ser and ok.sum() > 30:
            cuts = np.quantile(pp[ok], [1 / 3, 2 / 3]); ic = ser["tv_net"][0]; act = r["active"]["tv_net"]
            gate = {}
            for name, sel in (("low", pp <= cuts[0]), ("mid", (pp > cuts[0]) & (pp <= cuts[1])), ("high", pp > cuts[1])):
                vals = []
                for (m, g), keep in zip(ch["rows"], sel & ok):
                    if keep:
                        vals += [ic[d] for d in range(g + 1, min(g + config.GRID_STEP + 1, len(ic))) if mask[d] and act[d] and np.isfinite(ic[d])]
                gate[name] = {"ic": float(np.mean(vals)) if vals else None, "n_days": len(vals)}
            out["change_und"]["gate"] = gate
    return out


def verdict(summary: Dict) -> Dict:
    m = summary["methods"].get("tv_net"); best = summary.get("best_competitor")
    pr = summary["paired"].get(best) if best else None
    if not m or not pr:
        return {"verdict": "Not enough data", "reason": "No walk-forward predictions."}
    act = m.get("active_share", 1.0)
    if m["ic_t"] >= config.IC_T_SUPPORTED and pr["t"] >= config.IC_T_SUPPORTED and m["net_bps"] > 0:
        v, why = "Supported", "Significant IC, beats the best competitor (%s), and the top-3 pays after costs." % best
    elif m["ic_t"] >= config.IC_T_SUGGESTIVE and pr["t"] >= config.IC_T_SUGGESTIVE:
        v, why = "Suggestive", "Some ranking information beyond the best competitor (%s)" % best + ("" if m["net_bps"] > 0 else ", but the daily top-3 does not pay after costs (%.1f bps/day)." % m["net_bps"])
    else:
        v, why = "Not supported", "The network does not rank ETFs better than the best competitor, %s (IC t = %.2f, paired dIC t = %.2f); it made a forecast on %.0f%% of days." % (best, m["ic_t"], pr["t"], 100 * act)
    return {"verdict": v, "reason": why, "best_competitor": best, "ic_mean": m["ic_mean"], "ic_t": m["ic_t"], "net_bps": m["net_bps"], "active_share": act, "vs_best": pr}


CODE_VERSION = config.CODE_VERSION
