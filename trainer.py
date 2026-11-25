"""
trainer.py  —  P2 TVP-CAUSAL-NETWORK

Each run, per universe:
  1. build the panel (vol-scaled returns for the VAR, 13 features for the feature-based competitors, conditions for the change model);
  2. walk-forward over the last EVAL_YEARS: every GRID_STEP days estimate the time-varying directed network (one-sided local-linear adaptive group
     LASSO, hold-out penalty with a margin rule, edge t-test, factor-adjusted), the static network, the no-factor network and a ridge VAR, plus the
     time-varying undirected (partial-correlation) network; forecast each day from the latest estimate; score against the feature-based competitors;
  3. fit the networks on all history, report today's forecast (or that the network is empty and abstains), the edges, the lead-without-dependence edges
     and the correlated-but-unlinked pairs, and the network-change probability with its evaluation.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import features as FE
import network as N
import evaluate as E

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

CODE_VERSION = config.CODE_VERSION


def check_versions():
    stale = [m.__name__ + ".py" for m in (config, FE, N, E) if getattr(m, "CODE_VERSION", None) != CODE_VERSION]
    if stale:
        raise RuntimeError("Files from different releases: " + ", ".join(stale) + " are not version " + CODE_VERSION + ". Replace ALL files with the latest zip.")


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


def network_report(panel: Dict, r: Dict, f_live: Dict, u_live, key: str = "tv_net") -> Dict:
    """The live directed and undirected networks and what distinguishes them."""
    names = panel["etfs"]
    E_ = len(names)
    A, mask = f_live["A"], f_live["mask"].copy()
    np.fill_diagonal(mask, False)
    grid = r["grid"]; fits = r["fits"][key]
    recent = [fits[g]["mask"] for g in grid[-24:] if g in fits]
    persist = np.mean([m for m in recent], axis=0) if recent else np.zeros((E_, E_))
    P = u_live["P"] if u_live else np.zeros((E_, E_))
    directed = [{"source": names[j], "target": names[i], "coef": float(A[i, j]), "recent_presence": float(persist[i, j]), "partial_corr": float(P[i, j])}
                for i, j in zip(*np.where(mask))]
    directed.sort(key=lambda d: -abs(d["coef"]))
    und = [{"a": names[i], "b": names[j], "partial_corr": float(P[i, j])} for i in range(E_) for j in range(i + 1, E_) if P[i, j] != 0]
    und.sort(key=lambda d: -abs(d["partial_corr"]))
    lead_first = [d for d in directed if abs(d["partial_corr"]) < 0.05]
    Zw = panel["Z"][-120:]
    C = np.corrcoef(Zw, rowvar=False)
    only_corr = []
    for i in range(E_):
        for j in range(i + 1, E_):
            if abs(C[i, j]) >= 0.7 and not mask[i, j] and not mask[j, i] and abs(P[i, j]) < 0.05:
                only_corr.append({"a": names[i], "b": names[j], "correlation": float(C[i, j]), "partial_corr": float(P[i, j])})
    only_corr.sort(key=lambda d: -abs(d["correlation"]))
    n_edges_series = {"dates": [str(panel["dates"][g].date()) for g in grid if g in fits], "n_edges": [int(fits[g]["n_edges"]) for g in grid if g in fits]}
    return {"directed": directed[:25], "n_directed": len(directed), "undirected": und[:25], "n_undirected": len(und), "lead_without_dependence": lead_first[:15],
            "correlated_not_linked": only_corr[:15], "n_edges_series": n_edges_series}


def run_pipeline(prices_df: pd.DataFrame, macro_df, fit_lgbm: bool = True, eval_days: int = None) -> Dict:
    out = {"status": "ok", "universes": {}}
    first = True
    for name, tickers in config.UNIVERSES.items():
        p = FE.build_panel(prices_df, macro_df, tickers, name)
        if p is None:
            continue
        n = len(p["dates"])
        if first:
            out["last_data_date"] = str(p["dates"][-1].date()); first = False
        logger.info("📊 %s: %d ETFs, %d days, macro=%s" % (name, len(p["etfs"]), n, p["have_macro"]))
        r = E.run_walk_forward(p, eval_days=eval_days, fit_lgbm=fit_lgbm, log=logger.info)
        sm = E.summarize(p, r)
        v = E.verdict(sm)
        last = n - 1
        f = N.fit_directed(p["Z"], last, config.HALF_LIFE, True, True)
        u = N.fit_undirected(f) if f is not None and "_win" in f else None
        fs = N.fit_directed(p["Z"], last, np.inf, True, False)
        rv = N.ridge_var(p["Z"], last)
        roll = E.rolling_ridge(p, [last])
        qi = np.where(p["valid"][last])[0]
        z_last = p["Z"][last]
        sc = {"tv_net": (f["A"] @ (z_last @ f["Pi"])) if f is not None else None, "static_net": (fs["A"] @ (z_last @ fs["Pi"])) if fs is not None else None,
              "roll_var": (rv["A"] @ z_last) if rv is not None else None}
        active = f is not None and bool(np.any(f["A"] != 0))
        static_active = fs is not None and bool(np.any(fs["A"] != 0))
        today = []
        for j in qi:
            today.append({"etf": p["etfs"][j], "tv_net": _f(sc["tv_net"][j]) if sc["tv_net"] is not None and active else None,
                          "static_net": _f(sc["static_net"][j]) if sc["static_net"] is not None and static_active else None,
                          "roll_var": _f(sc["roll_var"][j]) if sc["roll_var"] is not None else None, "roll_ridge": _f(roll[last, j])})
        key = "tv_net" if active else "roll_var"
        today.sort(key=lambda x: -(x[key] if x[key] is not None else -9))
        picks = today[:config.TOP_N] if active else []
        fallback = [x["etf"] for x in sorted(today, key=lambda x: -(x["roll_var"] if x["roll_var"] is not None else -9))[:config.TOP_N]]
        static_picks = [x["etf"] for x in sorted([t for t in today if t["static_net"] is not None], key=lambda x: -x["static_net"])[:config.TOP_N]]
        days = r["days"]
        ser = {"dates": [str(p["dates"][d].date()) for d in days[::config.SERIES_STEP]]}
        for k in ("tv_net", "static_net", "roll_var", "roll_ridge", "momentum"):
            scr = r["scores"].get(k)
            if scr is None:
                continue
            _, nt = E.topn_series(scr, p["ret_next"])
            if k in E.NETWORK_VARIANTS:
                nt = np.where(~r["active"][k] & np.isfinite(nt), 0.0, nt)
            ser[k] = [float(x) * 100 for x in np.cumsum(np.nan_to_num(nt[days]))[::config.SERIES_STEP]]
        ch = r.get("change")
        cur = {}
        if ch:
            for nm in ("und", "dir"):
                pp = ch[nm]["p"]; fin = np.where(np.isfinite(pp))[0]
                cur[nm] = _f(pp[fin[-1]]) if len(fin) else None
        out["universes"][name] = {"n_etfs": len(p["etfs"]), "have_macro": p["have_macro"], "window": [str(p["dates"][r["window"][0]].date()), str(p["dates"][r["window"][1] - 1].date())],
                                  "summary": sm, "verdict": v, "network_active_today": active, "picks": picks, "fallback_picks": fallback, "fallback_model": "roll_var", "static_picks": static_picks,
                                  "today": today, "series": ser, "network": network_report(p, r, f, u, "tv_net") if f is not None else None,
                                  "static_network": network_report(p, r, fs, u, "static_net") if fs is not None else None, "change_current": cur}
        logger.info("  %s: tv_net IC %.4f (t %.2f, active %.0f%%) | best competitor %s | vs best dIC %.4f (t %.2f) | %s" % (name, sm["methods"]["tv_net"]["ic_mean"], sm["methods"]["tv_net"]["ic_t"],
                    100 * sm["methods"]["tv_net"]["active_share"], sm["best_competitor"], sm["paired"][sm["best_competitor"]]["d_ic"], sm["paired"][sm["best_competitor"]]["t"], v["verdict"]))
    return _clean(out)


def run_trainer(prices_loader=None, push: bool = True) -> Dict:
    check_versions()
    logger.info("🔄 Loading data...")
    try:
        if prices_loader is None:
            from data_manager import load_master_data, validate_data
            prices_df, macro_df = load_master_data()
            validate_data(prices_df, macro_df)
        else:
            prices_df, macro_df = prices_loader()
    except Exception as e:
        import traceback
        logger.error(f"Failed to load data: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return {}
    results = run_pipeline(prices_df, macro_df)
    results.update({"run_date": datetime.now().strftime("%Y-%m-%d"), "algorithm": "TVP-CAUSAL-NETWORK",
                    "settings": {"eval_years": config.EVAL_YEARS, "half_life": config.HALF_LIFE, "grid_step": config.GRID_STEP, "factors": config.N_FACTORS, "glasso_alpha": config.GLASSO_ALPHA,
                                 "edge_t": config.EDGE_T, "val_z": config.VAL_Z, "change_k": config.CHANGE_K, "cost_bps": config.TRADING_COST_BPS}})
    vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
    if os.path.exists(vpath):
        try:
            results["validation"] = json.load(open(vpath))
        except Exception:
            pass
    out = f"tvp_results_{results['run_date']}.json"
    with open(out, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"💾 Saved: {out}")
    if push:
        try:
            from push_results import upload_results
            upload_results(out, hf_token=config.HF_TOKEN)
        except Exception as e:
            logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    out = run_trainer()
    if not out:
        logger.error("Run produced no results.")
        sys.exit(1)
