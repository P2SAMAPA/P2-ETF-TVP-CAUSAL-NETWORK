"""
config.py  —  Configuration for P2 TVP-CAUSAL-NETWORK

Two time-varying networks over the ETFs of a universe, estimated at once from a locally stationary VAR(1) on vol-scaled returns:
  directed    A_t  the transition matrix: who PREDICTS whom (Granger-type edges), estimated by a one-sided local-linear fit with an adaptively
              weighted GROUP LASSO (each edge's level and slope are one group, selected together)
  undirected  Omega_t  the precision matrix of the VAR residuals: who is conditionally DEPENDENT on whom (partial correlations), by a graphical lasso
Highly correlated ETFs are handled by removing the leading principal factors first (factor adjustment). The directed network produces the next-day
forecasts; a logistic model estimates a NETWORK-CHANGE PROBABILITY (how likely the directed structure is to move materially over the next sessions).

Foundation: Chen, Li, Li & Linton, "Estimating Time-Varying Networks for High-Dimensional Time Series", Journal of Econometrics 249 (2025) 105941
(arXiv 2302.02476). Its application is a US macroeconomic panel, not return forecasting; the change probability is an addition tested here, not
something the paper provides. The paper smooths with a TWO-sided kernel; forecasting can only use the past, so the kernel here is ONE-sided.
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-tvp-network-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
}
MIN_UNIVERSE = 6
TOP_N = 3
TRADING_COST_BPS = 15

# Features for the feature-based competitors (13 per ETF, as in the previous engines) and the conditions used by the change model.
F_MAX = 16
N_FEAT = 13
VOL_WINDOW = 63
BETA_WINDOW = 60
MACRO_Z_WINDOW = 60
MACRO_COLS = ["VIX", "T10Y2Y", "DXY"]
MIN_ETFS_PER_DAY = 6
U_DIM = 11
CORR_WINDOW = 20
E_MAX = 44
Z_CLIP = 6.0                 # vol-scaled returns are clipped to +-Z_CLIP

# Network estimation (all one-sided: the estimate at grid day g uses only returns through the close of g)
GRID_STEP = 5                # networks are re-estimated every GRID_STEP days and applied until the next estimate
HALF_LIFE = 120              # kernel half-life in days for the time-varying estimator
STATIC_WINDOW = 1500         # the "static" network uses all history up to this many days, equally weighted
MAX_WINDOW_MULT = 5          # observations older than MAX_WINDOW_MULT * HALF_LIFE are ignored
MIN_OBS = 250
LAM_FRACS = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)   # candidate penalties as fractions of lambda_max, chosen on a time-ordered hold-out
VAL_FRAC = 0.2
EDGE_T = 2.5                 # after the unshrunk refit an edge must have |t| >= EDGE_T to be kept (precision)
VAL_Z = 2.0                  # the best candidate must beat the empty network by this many standard errors on the hold-out
ADAPT_GAMMA = 1.0            # adaptive group weights = 1 / (pilot group norm)^gamma
FISTA_ITERS = 150
N_FACTORS = 2                # leading principal factors removed before the VAR (factor adjustment)
GLASSO_ALPHA = 0.25          # graphical-lasso penalty on the residual correlation matrix (calibrated on synthetic graphs: best precision/recall balance)
ROLL_RIDGE_WINDOW = 60       # competitor: ridge VAR on the last 60 days
VAR_RIDGE = 5.0

# Network-change probability
CHANGE_K = 10                # "the next few sessions": the structure is compared CHANGE_K sessions ahead
CHANGE_QUANTILE = 0.75       # a change is a distance above the expanding 75th percentile of past distances (base rate ~25%)
CHANGE_REFIT_GRID = 12       # refit the logistic model every 12 grid steps (~ 60 days)
CHANGE_MIN_TRAIN = 40

# Walk-forward evaluation
EVAL_YEARS = 5.0
MIN_HISTORY_DAYS = 756
REFIT_DAYS = 252
RIDGE_ALPHA = 100.0
ROLL_RIDGE_ALPHA = 100.0
ROLL_WINDOW = 60
LGBM_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=200,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=2)

# Verdict: Supported = IC t >= 1.65 AND beats the BEST competitor (paired t >= 1.65) AND the daily top-3 pays after costs;
# Suggestive = IC t >= 1 and beats the best competitor with t >= 1; otherwise Not supported. Days on which the network selects no edges make
# no forecast; they count as IC 0 and no trade (abstention), and the share of such days is reported.
IC_T_SUPPORTED = 1.65
IC_T_SUGGESTIVE = 1.0
SERIES_STEP = 3

CODE_VERSION = "2026-11-16.1"
