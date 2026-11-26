"""
validation.py  —  Does the pipeline find structure that is there, stay quiet when it is not, and distinguish correlation from causation?

Synthetic ETF markets with a KNOWN directed network, KNOWN partial-correlation network and known change times (simworld.py), several seeds each,
through the SAME walk-forward, abstention and verdict code as real data. Worlds: null / confound (strong common factor, NO directed edges) /
static / tv (edges switch at random times) / trigger (switching driven by observable conditions).
Reported: forecast IC of every method, the share of days the network speaks, directed-edge precision and recall against the truth for the
time-varying AND the static network, undirected (partial-correlation) edge precision and recall, and verdicts.

    python validation.py <seeds> <world 0-4>      # one world per call; saves validation_world_<i>.json
    python validation.py <seeds> merge            # -> validation_results.json
"""

import sys
import json
import time
import numpy as np

import config
import simworld as S
import evaluate as E

WORLDS = ["null", "confound", "static", "tv", "trigger"]
AMP, N_DAYS, EVAL_DAYS = 0.25, 1400, 500


def edge_scores(P, r, key):
    fits = r["fits"][key]; tp = fp = fn = empty = nfit = 0; utp = ufp = ufn = 0
    for g in r["grid"][::4]:
        f = fits.get(g)
        if f is None or g < r["window"][0]:
            continue
        nfit += 1
        est = f["mask"].copy(); np.fill_diagonal(est, False)
        truth = np.abs(P["A_true"][min(g + 1, len(P["A_true"]) - 1)]) > 0.5 * AMP; np.fill_diagonal(truth, False)
        empty += int(est.sum() == 0); tp += int((est & truth).sum()); fp += int((est & ~truth).sum()); fn += int((~est & truth).sum())
        if key == "tv_net" and f.get("P") is not None:
            ue = np.triu(np.abs(f["P"]) > 0, 1); ut = np.triu(P["P_true"][g], 1)
            utp += int((ue & ut).sum()); ufp += int((ue & ~ut).sum()); ufn += int((~ue & ut).sum())
    return {"fits": nfit, "empty_share": empty / max(nfit, 1), "tp": tp, "fp": fp, "fn": fn, "utp": utp, "ufp": ufp, "ufn": ufn}


def one_run(kind, seed):
    P = S.make_world(kind, n=N_DAYS, amp=AMP, seed=seed)
    r = E.run_walk_forward(P, eval_days=EVAL_DAYS, fit_lgbm=True)
    sm = E.summarize(P, r)
    return {"seed": seed, "methods": sm["methods"], "paired": sm["paired"], "best": sm["best_competitor"], "verdict": E.verdict(sm)["verdict"],
            "edges": edge_scores(P, r, "tv_net"), "edges_static": edge_scores(P, r, "static_net")}


def _agg(runs, key):
    ed = {k: float(np.sum([r[key][k] for r in runs])) for k in ("tp", "fp", "fn", "utp", "ufp", "ufn", "fits")}
    ed["empty_share"] = float(np.mean([r[key]["empty_share"] for r in runs]))
    ed["precision"] = ed["tp"] / (ed["tp"] + ed["fp"]) if ed["tp"] + ed["fp"] > 0 else None
    ed["recall"] = ed["tp"] / (ed["tp"] + ed["fn"]) if ed["tp"] + ed["fn"] > 0 else None
    ed["u_precision"] = ed["utp"] / (ed["utp"] + ed["ufp"]) if ed["utp"] + ed["ufp"] > 0 else None
    ed["u_recall"] = ed["utp"] / (ed["utp"] + ed["ufn"]) if ed["utp"] + ed["ufn"] > 0 else None
    return ed


def run_world(i, seeds):
    kind = WORLDS[i]
    runs = [one_run(kind, 100 + s) for s in range(seeds)]
    w = {"world": kind, "n_runs": seeds, "verdicts": {}, "methods": {}, "paired": {}}
    for r in runs:
        w["verdicts"][r["verdict"]] = w["verdicts"].get(r["verdict"], 0) + 1
    for k in runs[0]["methods"]:
        w["methods"][k] = {m: float(np.mean([r["methods"][k][m] for r in runs])) for m in ("ic_mean", "ic_t", "net_bps", "active_share")}
    for k in runs[0]["paired"]:
        ts = [r["paired"][k]["t"] for r in runs]
        w["paired"][k] = {"d_ic": float(np.mean([r["paired"][k]["d_ic"] for r in runs])), "mean_t": float(np.mean(ts)), "share_positive": float(np.mean([r["paired"][k]["d_ic"] > 0 for r in runs]))}
    w["edges"] = _agg(runs, "edges"); w["edges_static"] = _agg(runs, "edges_static")
    return w


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if len(sys.argv) > 2 and sys.argv[2] == "merge":
        out = {"seeds_per_world": seeds, "n_days": N_DAYS, "eval_days": EVAL_DAYS, "edge_amplitude": AMP, "worlds": [json.load(open("validation_world_%d.json" % i)) for i in range(len(WORLDS))]}
        json.dump(out, open("validation_results.json", "w"), indent=1); print("merged -> validation_results.json")
    else:
        for i in [int(a) for a in sys.argv[2:]]:
            t0 = time.time()
            w = run_world(i, seeds); w["seconds"] = round(time.time() - t0)
            json.dump(w, open("validation_world_%d.json" % i, "w"), indent=1)
            ed, es = w["edges"], w["edges_static"]
            f2 = lambda x: "n/a" if x is None else "%.2f" % x
            print(w["world"], w["verdicts"], "in %ds" % w["seconds"], flush=True)
            print("   IC: " + " | ".join("%s %+.4f%s" % (k, v["ic_mean"], "" if v["active_share"] > 0.999 else " (active %.0f%%)" % (100 * v["active_share"])) for k, v in w["methods"].items()))
            print("   directed edges vs truth — time-varying: empty %.0f%% | precision %s | recall %s ;  static: empty %.0f%% | precision %s | recall %s ;  undirected precision %s | recall %s" % (
                100 * ed["empty_share"], f2(ed["precision"]), f2(ed["recall"]), 100 * es["empty_share"], f2(es["precision"]), f2(es["recall"]), f2(ed["u_precision"]), f2(ed["u_recall"])), flush=True)
