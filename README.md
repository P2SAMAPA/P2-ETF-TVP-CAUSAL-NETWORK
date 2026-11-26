# P2-ETF-TVP-CAUSAL-NETWORK — Time-Varying Probabilistic Causal Network

Two networks over the ETFs of a universe, both time-varying, estimated from a locally stationary VAR(1) on vol-scaled returns:
* a **directed** network — who *predicts* whom (Granger-type edges) — which produces the next-day forecasts;
* an **undirected** network — who is *conditionally dependent* on whom (partial correlations of the VAR residuals).
The distinction matters: a lead can appear before two ETFs become correlated, and two ETFs can be highly correlated through a common factor or a third
variable with no direct link. The dashboard shows both, the "leads without conditional dependence" and the "correlated but not linked" pairs. A
**network-change probability** estimates how likely the structure is to move over the next 10 sessions.

## The cited paper
Chen, Li, Li & Linton, *Estimating Time-Varying Networks for High-Dimensional Time Series*, **Journal of Econometrics 249 (2025) 105941** (arXiv 2302.02476) —
**verified**: a locally stationary high-dimensional VAR, penalised local-linear estimation with time-varying weighted group LASSO for the transition matrices,
a time-varying CLIME precision matrix, and a factor-adjusted extension. Two things to know: its application is a US **macroeconomic** panel, not return
forecasting; and the paper smooths with a **two-sided** kernel, whereas forecasting can only use the past, so the kernel here is **one-sided** (which costs power).
The **network-change probability is an addition**, not part of the paper.

## Method
* **Directed network** at every 5th day, using only returns through that day's close: one-sided exponential kernel (half-life 120 days), **local-linear** coefficients, and an
  **adaptively weighted group LASSO** where each edge's (level, slope) pair is one group, weights from a ridge pilot. The penalty is chosen on a **time-ordered hold-out**
  with "no edges" as a candidate, a **2-standard-error margin** and the **sparsest penalty within one standard error**; selected edges are refit without shrinkage and kept only if
  their own t-statistic is ≥ 2.5. The two leading principal factors are projected out first (**factor adjustment**). If nothing beats zero out of sample the network is empty and the engine
  **abstains** (counted as IC 0, no trade). A **static** version (equal weights over all history, no slope terms) runs alongside.
* **Undirected network**: graphical lasso (penalty 0.25) on the kernel-weighted residual correlation → partial correlations.
* **Change probability**: label = the directed (or undirected) network moves by more than the expanding 75th percentile of past distances over the next 10 sessions; a logistic model on the
  network's recent behaviour and market conditions, refit as labels become known. Scored by AUC and Brier skill against the base rate, against a "recent change persists" score, and as a confidence gate.
* **Competitors on identical rows** (the verdict compares with the best): the same network without factor adjustment (`tv_nofactor`), the static network, a ridge VAR on the last
  60 days (no sparsity), and the feature-based rolling ridge / fixed ridge / LightGBM / momentum of the earlier engines.

## Bugs found by testing (all fixed)
1. **Validation leak.** The adaptive weights came from a ridge pilot on the *whole* window, including the validation segment, so groups that happened to match validation noise were penalised
   less and then "predicted" it. In a world with no edges the model beat "no edges" out of sample in 72–78% of fits. All data-dependent quantities in the validation stage now use the training segment only.
2. **Coin-flip selection.** "Beats zero by any margin" between near-identical models. Fixed with the margin and one-standard-error rule.
3. **Over-selection.** Even after 1–2, many false edges: added the post-selection t-test.
4. **Graphical-lasso penalty** set by hand gave ~28 edges for 8 true (precision 0.20); calibrated on known graphs (0.25 → precision 0.47, recall 0.42).
5. **Change model undefined for an empty network**, so the undirected network is now its primary target.
6. A "recent change persists" baseline that used the wrong (directed) distance column; and, once fixed, it showed AUC 0.57–0.65 even on pure noise, because consecutive distances share the same noisy estimate.

## Validation (synthetic markets with known networks; 4 seeds each, 1,400 days, 500 evaluated; directed-edge amplitude 0.25; same walk-forward, abstention and verdict code as real data)
| world | verdicts | tv_net IC (active) | static_net IC (active) | ridge VAR IC | tv network: precision / recall | static network: precision / recall | undirected precision / recall |
|---|---|---|---|---|---|---|---|
| null | Not supported ×4 | 0.000 (0%) | 0.000 (0%) | −0.008 | – (always empty) | – (always empty) | 0.43 / 0.44 |
| **confound** (strong common factor, no directed edges) | Not supported ×4 | 0.000 (0%) | 0.000 (0%) | −0.007 | – (always empty) | – (always empty) | 0.46 / 0.49 |
| static edges | Not supported ×4 | +0.024 (59%) | **+0.089 (98%)** | +0.018 | 0.33 / 0.23 | **0.97 / 0.94** | 0.44 / 0.43 |
| edges switch at random times | Not supported ×4 | −0.000 (2%) | +0.029 (61%) | +0.007 | 1.00 / 0.00 (1 edge) | 0.41 / 0.43 | 0.45 / 0.34 |
| edges driven by conditions | Not supported ×4 | 0.000 (0%) | +0.004 (15%) | +0.013 | – / 0.00 | 0.33 / 0.07 | 0.38 / 0.33 |

What this says:
* **It does not mistake correlation for causation**: with a strong common factor and no directed edges both directed networks are empty in 100% of fits and make no forecast.
* **A static network recovers persistent edges accurately** (precision 0.97, recall 0.94 when edges are constant; IC +0.089).
* **The time-varying directed network is nearly silent when edges switch** (active 0–2%, recall 0.00), and loses to the static network in every world with edges (tv_net minus static_net: −0.065, t −5.9
  when edges are static; −0.030, t −3.6 when they switch at random; −0.004 when they follow conditions). The one-sided local estimator lacks the power for edges that change every ~200 days; the cited paper's
  two-sided smoothing is what gives it power, and a forecaster cannot use it. **Time variation did not pay in any world tested.** Factor adjustment made no detectable difference (tv_net minus tv_nofactor −0.011 to +0.000, |t| ≤ 1.3).
* The undirected network is only moderately accurate (precision ~0.4–0.5, recall ~0.3–0.5).
* **Change probability** (2 seeds each, 3,000 days, ~450 labelled points, tested separately during development): where structure change is driven by observable conditions AUC was 0.53–0.56 (null world 0.51) and Brier skill
  was negative in both (−0.01 to −0.07), i.e. it does not beat simply predicting the base rate. The "recent change persists" score reached 0.57–0.65 **including the null world** (estimation-noise artefact).
  With ~140 labelled points AUC varied by ±0.1 on the null alone, so small real-data samples can't distinguish skill from luck.

## Verdict (walk-forward period)
Supported = IC t ≥ 1.65 **and** beats the **best** competitor (paired t ≥ 1.65) **and** the daily top-3 pays after costs (15 bps round trip); Suggestive = IC t ≥ 1 and beats it with t ≥ 1;
otherwise Not supported. Days with an empty network count as IC 0, no trade. Because the static network is among the competitors, a time-varying network must beat it to count.

## Limits and what is not verified
* **Not run on real data.** Runtime was measured on synthetic prices at production scale: the largest universe (43 ETFs, 4,900 days, 5-year walk-forward) takes about 1 minute; all three about 2–3 minutes.
* Real daily lead-lag among ETFs is far weaker than the synthetic edges (amplitude 0.25 ≈ 6% R² per target); expect empty networks most of the time. Genuine lead-lag does exist where timing differs (e.g. US equities → internationally listed ETFs), which the **static** network could find.
* The network-change probability has no demonstrated skill in simulation; treat any number it shows as unvalidated.
* No new dependencies beyond the earlier engines (numpy, scipy, scikit-learn, LightGBM).

## Files
`config.py` · `features.py` (panel: vol-scaled returns, features, conditions) · `network.py` (directed and undirected estimators) · `evaluate.py` (grid estimation, forecasts, abstention,
competitors, change model, verdict) · `simworld.py` · `validation.py` · `trainer.py` · `streamlit_app.py` · `validation_results.json` · `.github/workflows/daily_run.yml`.
Every module carries a `CODE_VERSION`; `trainer.py` refuses a mixed set — **replace all files together**.

Run: `python trainer.py` (needs `HF_TOKEN`; results go to `P2SAMAPA/p2-etf-tvp-network-results`). Not financial advice.
