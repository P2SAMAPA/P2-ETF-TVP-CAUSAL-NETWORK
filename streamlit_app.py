import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 TVP-Causal-Network", page_icon="🕸️", layout="wide", initial_sidebar_state="collapsed")

PRIMARY = "#0f766e"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
VCOL = {"Supported": "#16a34a", "Suggestive": "#d97706", "Not supported": "#dc2626", "Not enough data": "#64748b"}

CSS = """
<style>
    .stApp { background-color: __BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUB__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .banner { background: #ffffff; border: 1px solid __BD__; border-left: 6px solid var(--accent); border-radius: 10px;
        padding: 0.8rem 1.1rem; font-size: 0.92rem; color: __INK__; margin: 0.4rem 0 0.8rem 0; }
    .pick-card { background: #ffffff; border: 1px solid __BD__; border-radius: 14px; padding: 1rem 1.2rem; text-align: center; }
    .pick-rank { font-size: 0.72rem; color: __SUB__; text-transform: uppercase; letter-spacing: 0.05em; }
    .pick-ticker { font-size: 1.8rem; font-weight: 800; color: __INK__; }
    .pick-sub { font-size: 0.78rem; color: __SUB__; }
    .note { font-size: 0.82rem; color: __SUB__; margin: 0.4rem 0 0.9rem 0; }
    div[data-testid="stDataFrame"] { border: 1px solid __BD__; border-radius: 10px; overflow: hidden; }
</style>
"""
st.markdown(CSS.replace("__BG__", PAGE_BG).replace("__INK__", INK).replace("__SUB__", SUBTLE).replace("__BD__", CARD_BORDER), unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-tvp-network-results"


def _latest(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("tvp_results_") and f.endswith(".json"))
        return (names[-1], None) if names else (None, "No tvp_results_*.json files found in " + repo_id + ".")
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_k=""):
    notes = []
    local = glob.glob("tvp_results_*.json")
    if local:
        f = sorted(local)[-1]
        try:
            return json.load(open(f)), ["Loaded local file: " + f]
        except Exception as e:
            notes.append("Could not parse " + f + ": " + str(e))
    name, err = _latest(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            h = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
            r = requests.get("https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name, headers=h, timeout=30)
            if r.status_code == 200:
                return r.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(r.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def fnum(x, fmt="%.2f", na="n/a"):
    return na if x is None else fmt % x


def render_today(name, u):
    v = u["verdict"]
    st.markdown('<div class="banner" style="--accent: ' + VCOL.get(v["verdict"], "#64748b") + ';"><b>' + name.replace("_", " ").title() + ': ' + v["verdict"] + '.</b> ' + str(v.get("reason", "")) + '</div>', unsafe_allow_html=True)
    picks = u.get("picks", [])
    if picks:
        cols = st.columns(len(picks))
        for i, (c, p) in enumerate(zip(cols, picks)):
            with c:
                st.markdown('<div class="pick-card"><div class="pick-rank">#' + str(i + 1) + '</div><div class="pick-ticker">' + p["etf"] + '</div><div class="pick-sub">network forecast ' + fnum(p["tv_net"], "%+.3f") + '</div></div>', unsafe_allow_html=True)
    else:
        extra = ""
        if u.get("static_picks"):
            extra = " The static network (no time variation) does forecast today: " + ", ".join(u["static_picks"]) + "."
        st.info("The time-varying directed network selected no edges today, so it makes no forecast (abstains)." + extra + " For reference the ridge VAR (dense, no sparsity) ranks: " + ", ".join(u.get("fallback_picks", [])) + ".")
    today = pd.DataFrame(u.get("today", []))
    if not today.empty:
        today = today.rename(columns={"etf": "ETF", "tv_net": "Time-varying network", "static_net": "Static network", "roll_var": "Ridge VAR", "roll_ridge": "Rolling ridge (features)"})
        with st.expander("All " + str(len(today)) + " ETFs: forecasts by method"):
            st.dataframe(today.style.format({c: "{:+.4f}" for c in today.columns if c != "ETF"}, na_rep="—"), use_container_width=True, hide_index=True)


def _edges_table(rows, cols, rename):
    if not rows:
        st.caption("None.")
        return
    df = pd.DataFrame(rows)[cols].rename(columns=rename)
    st.dataframe(df.style.format({c: "{:+.3f}" for c in df.columns if df[c].dtype == float}), use_container_width=True, hide_index=True)


def render_network(data):
    st.markdown('<div class="note"><b>Directed</b> edges say who PREDICTS whom (a Granger-type lead, estimated on factor-adjusted vol-scaled returns); <b>undirected</b> edges are partial correlations: conditional dependence that remains '
                'after the directed edges and the common factors. An edge can be directed without partial correlation (a lead that appears before the two become dependent) or two ETFs can be highly correlated '
                'with no direct link (a common factor or a third variable). Validation shows the time-varying network is nearly silent when edges switch (see the Validation tab): expect it to be empty on most days; '
                'the static network is the one that finds persistent edges.</div>', unsafe_allow_html=True)
    names = list(data["universes"].keys())
    pick = st.selectbox("Universe", names, index=len(names) - 1)
    u = data["universes"][pick]
    for label, key in (("Time-varying network (today)", "network"), ("Static network (all history, no time variation)", "static_network")):
        nw = u.get(key)
        st.markdown("###### " + label)
        if not nw:
            st.caption("Not available.")
            continue
        st.write("Directed edges: **" + str(nw["n_directed"]) + "** | undirected (partial-correlation) edges: **" + str(nw["n_undirected"]) + "**")
        _edges_table(nw["directed"], ["source", "target", "coef", "recent_presence", "partial_corr"], {"source": "Leads", "target": "Follows", "coef": "Coefficient", "recent_presence": "Present in recent fits", "partial_corr": "Partial corr."})
        if key == "network":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Leads without conditional dependence** (directed edge, |partial corr| < 0.05)")
                _edges_table(nw["lead_without_dependence"], ["source", "target", "coef"], {"source": "Leads", "target": "Follows", "coef": "Coefficient"})
            with c2:
                st.markdown("**Correlated but not linked** (|corr| ≥ 0.7, no directed edge, |partial corr| < 0.05)")
                _edges_table(nw["correlated_not_linked"], ["a", "b", "correlation", "partial_corr"], {"a": "ETF A", "b": "ETF B", "correlation": "Correlation", "partial_corr": "Partial corr."})
            st.markdown("**Strongest conditional dependences (undirected)**")
            _edges_table(nw["undirected"][:12], ["a", "b", "partial_corr"], {"a": "ETF A", "b": "ETF B", "partial_corr": "Partial corr."})
            ser = nw.get("n_edges_series", {})
            if ser.get("dates"):
                fig = go.Figure(go.Scatter(x=ser["dates"], y=ser["n_edges"], mode="lines", line=dict(color=PRIMARY, width=1.5)))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(title="Directed edges selected", gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"))
                st.plotly_chart(fig, use_container_width=True, key="ne_" + pick)


def render_change(data):
    st.markdown('<div class="note"><b>Network-change probability</b> (an addition to the cited method, not part of it): the probability that the network moves materially over the next ' + str(data["settings"]["change_k"])
                + ' sessions, where "materially" means a distance above the expanding 75th percentile of past distances. The primary target is the undirected (partial-correlation) network, which always exists; the directed one is '
                'undefined when it is empty. Judge the model by the evaluation, not by the number: validation found it barely above chance even where structure change is driven by observable conditions.</div>', unsafe_allow_html=True)
    rows = []
    for name, u in data["universes"].items():
        cu = u["summary"].get("change_und", {}); cd = u["summary"].get("change_dir", {}); cc = u.get("change_current", {})
        rows.append({"Universe": name, "P(change), undirected": cc.get("und"), "AUC": cu.get("auc"), "Persistence-score AUC": cu.get("auc_persistence"), "Brier skill vs base rate": cu.get("brier_skill"),
                     "Base rate": cu.get("base_rate"), "Labelled points": cu.get("n"), "Directed: AUC": cd.get("auc"), "Directed: P(change)": cc.get("dir")})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ("P(change), undirected", "AUC", "Persistence-score AUC", "Brier skill vs base rate", "Base rate", "Directed: AUC", "Directed: P(change)")}, na_rep="n/a"), use_container_width=True, hide_index=True)
    st.caption("A Brier skill below 0 means the model is worse than always predicting the base rate. The persistence score (recent change predicts future change) is inflated by estimation noise: consecutive distances share the same noisy estimate, so it also scores above 0.5 on pure noise.")
    g = []
    for name, u in data["universes"].items():
        gate = u["summary"].get("change_und", {}).get("gate")
        if gate:
            g.append({"Universe": name, "IC when change prob. LOW": gate["low"]["ic"], "MID": gate["mid"]["ic"], "HIGH": gate["high"]["ic"], "days (low/mid/high)": "%d / %d / %d" % (gate["low"]["n_days"], gate["mid"]["n_days"], gate["high"]["n_days"])})
    if g:
        st.markdown("###### Does the change probability work as a confidence gate? (network forecast IC by predicted-change tercile, active days only)")
        st.dataframe(pd.DataFrame(g).style.format({"IC when change prob. LOW": "{:+.4f}", "MID": "{:+.4f}", "HIGH": "{:+.4f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)


def render_evidence(name, u):
    sm = u["summary"]
    st.markdown("**" + name.replace("_", " ").title() + "** — walk-forward " + u["window"][0] + " to " + u["window"][1] + " (best competitor: " + str(sm.get("best_competitor")) + ")")
    rows = [{"Method": k, "IC": x["ic_mean"], "t": x["ic_t"], "Top-3 gross bps/day": x["gross_bps"], "Top-3 net bps/day": x["net_bps"], "Active days": x["active_share"]} for k, x in sm["methods"].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"IC": "{:+.4f}", "t": "{:+.2f}", "Top-3 gross bps/day": "{:+.1f}", "Top-3 net bps/day": "{:+.1f}", "Active days": "{:.0%}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
    pr = [{"tv_net minus": k, "IC difference": x["d_ic"], "paired t": x["t"], "net bps/day difference": x["d_net_bps"]} for k, x in sm["paired"].items()]
    st.dataframe(pd.DataFrame(pr).style.format({"IC difference": "{:+.4f}", "paired t": "{:+.2f}", "net bps/day difference": "{:+.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
    st.caption("Network methods abstain on days when they select no edges (counted as IC 0, no trade); 'Active days' is the share on which they spoke. 'tv_net minus static_net' isolates the value of time variation; 'minus tv_nofactor' the value of factor adjustment; 'minus roll_var' the value of sparsity.")
    ser = u.get("series", {})
    if ser.get("dates"):
        fig = go.Figure()
        for k, col in (("tv_net", PRIMARY), ("static_net", "#6366f1"), ("roll_var", "#f59e0b"), ("roll_ridge", "#94a3b8"), ("momentum", "#dc2626")):
            if k in ser:
                fig.add_trace(go.Scatter(x=ser["dates"], y=ser[k], mode="lines", name=k, line=dict(color=col, width=2 if k == "tv_net" else 1.3)))
        fig.update_layout(height=270, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified", yaxis=dict(title="Cumulative net top-3 vs universe (%)", gridcolor="#eef2f7"),
                          xaxis=dict(gridcolor="#eef2f7"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True, key="cum_" + name)


def render_validation(data):
    v = data.get("validation")
    if not v:
        st.info("No validation_results.json shipped with this run.")
        return
    st.markdown('<div class="note">Synthetic ETF markets with a known directed network, known partial-correlation network and known change times; ' + str(v.get("seeds_per_world")) + ' seeds each, through the same walk-forward, abstention '
                'and verdict code as real data (directed edge amplitude ' + str(v.get("edge_amplitude")) + ').</div>', unsafe_allow_html=True)
    rows = []
    for w in v["worlds"]:
        m, ed, es = w["methods"], w["edges"], w["edges_static"]
        rows.append({"World": w["world"], "Verdicts": json.dumps(w["verdicts"]), "tv_net IC": m["tv_net"]["ic_mean"], "tv_net active": m["tv_net"]["active_share"], "static_net IC": m["static_net"]["ic_mean"], "static active": m["static_net"]["active_share"],
                     "Ridge VAR IC": m["roll_var"]["ic_mean"], "tv precision": ed["precision"], "tv recall": ed["recall"], "static precision": es["precision"], "static recall": es["recall"],
                     "Undirected precision": ed["u_precision"], "Undirected recall": ed["u_recall"]})
    st.dataframe(pd.DataFrame(rows).style.format({"tv_net IC": "{:+.4f}", "static_net IC": "{:+.4f}", "Ridge VAR IC": "{:+.4f}", "tv_net active": "{:.0%}", "static active": "{:.0%}", "tv precision": "{:.2f}", "tv recall": "{:.2f}",
                                                  "static precision": "{:.2f}", "static recall": "{:.2f}", "Undirected precision": "{:.2f}", "Undirected recall": "{:.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)


def main():
    if "kb" not in st.session_state:
        st.session_state.kb = 0
    a, b = st.columns([5, 1])
    with a:
        st.markdown('<div class="app-title">🕸️ P2 TVP-Causal-Network</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Who predicts whom (directed) and who is conditionally dependent on whom (undirected), both time-varying, with factor adjustment and a network-change probability.</div>', unsafe_allow_html=True)
    with b:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear(); st.session_state.kb += 1; st.rerun()
    data, notes = load_data(str(st.session_state.kb))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
        return
    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "** | data through " + str(data.get("last_data_date")))
    if data.get("status") != "ok":
        st.warning(str(data.get("message", "The run did not produce a ranking."))); return
    t1, t2, t3, t4, t5 = st.tabs(["🎯 Today", "🕸️ Networks", "🔔 Change probability", "📊 Evidence", "🧪 Validation"])
    with t1:
        for name, u in data["universes"].items():
            render_today(name, u)
    with t2:
        render_network(data)
    with t3:
        render_change(data)
    with t4:
        for name, u in data["universes"].items():
            render_evidence(name, u)
    with t5:
        render_validation(data)


if __name__ == "__main__":
    main()
