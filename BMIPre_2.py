import os
import math
import json
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import shap

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from catboost import CatBoostRegressor
import joblib

DATA_PATH = r"../data/BMIPre_1_2.csv"
TARGET_COL = "Target"
ID_COLS: List[str] = ["Number"]

EXCLUDE_FROM_MODEL: List[str] = []

SHAP_SUBDIR = "./BMIPre_SHAP_2"
SHAP_TOPK = 8
SHAP_FORCE_N = 40

ANCOVA_COVARIATES: List[str] = ["Baseline BMI"]
CONTINUOUS_PEARSON_COLS: List[str] = ["Baseline BMI", "Baseline BMR", "age"]

ALPHA = 0.05
OUTER_FOLDS = 5
INNER_FOLDS = 5

RANDOM_STATE = 42
SEED_OUTER = 42
SEED_INNER = 42

BOOT_B = 1000

OUT_DIR = "BMIPre_2"

ORDINAL_COLS: Dict[str, List[str]] = {
    "Paternal educational level": ["Lower", "Middle", "Higher"],
    "Maternal educational level": ["Lower", "Middle", "Higher"],
    "Family income": ["Lowest", "Lower middle", "Upper middle", "Highest"],
    "Daily sleep duration": ["< 6 h/day", "6–8 h/day", "> 8 h/day"],
    "Frequency of staying up late": ["Never", "Sometimes", "Often", "Always"],
    "Sedentariness duration on weekends": ["3–5 h/day", "5–7 h/day", "7–9 h/day", "> 9 h/day"],
    "Schoolwork burden": ["Minimal", "Manageable", "High", "Overwhelming"],
    "Frequency of high-protein food intake": ["Never", "Sometimes", "Often", "Always"],
    "Frequency of midnight snack intake": ["Never", "Sometimes", "Often", "Always"],
    "Frequency of high-calorie foods intake": ["Never", "Sometimes", "Often", "Always"],
    "Frequency of participation in physical activities": ["0 times/week", "1–2 times/week", "2–3 times/week",
                                                          "> 3 times/week"],
    "Post-exercise sensations": ["Relaxed", "Slightly tired", "Fairly tired", "Extremely tired"],
    "Physical activities duration on weekends": ["< 1 h/day", "1–2 h/day", "2–3 h/day", "> 3 h/day"],
    "Parental support for sports involvement": ["Low support", "Moderate support", "High support"],
    "Level of health literacy": ["Lowest", "Lower middle", "Upper middle", "Highest"],
}
UNORDERED_CATEGORICAL_COLS: List[str] = [
    "Gender",
    "Paternal overweight/obesity status",
    "Maternal overweight/obesity status",
    "Paternal occupation",
    "Maternal occupation",
    "Family residence location",
    "On-campus residence",
    "Participation in professional sports training",
    "Recognize self-weight status correctly",
    "Satisfaction with body size",
    "Considered changing body size",
]

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_point(y_true, y_pred) -> Dict[str, float]:
    _mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(_mse),
        "rmse": float(np.sqrt(_mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }


def bootstrap_metrics(y_true, y_pred, B=BOOT_B, seed=42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    mse, rmse_, mae, r2 = [], [], [], []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mse.append(mean_squared_error(yt, yp))
        rmse_.append(np.sqrt(mean_squared_error(yt, yp)))
        mae.append(mean_absolute_error(yt, yp))
        try:
            r2.append(r2_score(yt, yp))
        except Exception:
            r2.append(np.nan)

    def ci(a):
        a = np.asarray(a, dtype=float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        return {"mean": float(np.nanmean(a)), "lo": float(lo), "hi": float(hi)}

    return {
        "point": metrics_point(y_true, y_pred),
        "bootstrap": {
            "mse": ci(mse), "rmse": ci(rmse_), "mae": ci(mae), "r2": ci(r2), "B": B
        }
    }


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def allowed_feature_cols(df_cols: List[str]) -> List[str]:
    keep = []
    for c in df_cols:
        if c in ID_COLS:
            continue
        if c in EXCLUDE_FROM_MODEL:
            continue
        if c == TARGET_COL:
            continue
        keep.append(c)
    return keep


def build_preprocessor(feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    ordinal_cols = [c for c in feature_cols if c in ORDINAL_COLS]
    unordered_cols = [c for c in feature_cols if (c in UNORDERED_CATEGORICAL_COLS and c not in ordinal_cols)]
    num_cols = [c for c in feature_cols if (c not in ordinal_cols and c not in unordered_cols)]

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", StandardScaler(), num_cols))
    if len(ordinal_cols) > 0:
        ord_categories = [ORDINAL_COLS[c] for c in ordinal_cols]
        transformers.append(("ord", OrdinalEncoder(categories=ord_categories, dtype=float), ordinal_cols))
    if len(unordered_cols) > 0:
        transformers.append(("cat_ohe", make_ohe(), unordered_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, ordinal_cols, unordered_cols


def safe_pearson_p(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return 1.0
    _, p = stats.pearsonr(x[mask], y[mask])
    return float(p)


def ancova_p_value(y: np.ndarray, feature: np.ndarray, covariates: np.ndarray) -> float:
    yv = y.astype(float)
    if feature.ndim == 1:
        f = feature.reshape(-1, 1)
    else:
        f = feature

    if hasattr(feature, "dtype") and getattr(feature, "dtype", None) is not None and feature.dtype.kind in (
    "O", "U", "S"):
        f = pd.get_dummies(pd.Series(feature), drop_first=True).to_numpy()
    Xc = covariates if covariates is not None and covariates.size > 0 else None
    all_mat = f.astype(float) if Xc is None else np.concatenate([f.astype(float), Xc.astype(float)], axis=1)
    mask = ~np.isnan(yv) & ~np.isnan(all_mat).any(axis=1)
    if mask.sum() < (all_mat.shape[1] + 2):
        return 1.0
    X = np.column_stack([np.ones(mask.sum()), all_mat[mask]])
    yy = yv[mask]
    beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return 1.0
    sigma2 = resid.T @ resid / dof
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    k_f = f.shape[1]
    se = np.sqrt(np.diag(cov_beta))[1:1 + k_f]
    tvals = beta[1:1 + k_f] / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=dof))
    pvals_sorted = np.sort(pvals);
    m = len(pvals_sorted)
    simes = np.min((m * pvals_sorted) / (np.arange(1, m + 1)))
    return float(min(1.0, simes))


def compute_univariate_pvals(X_df: pd.DataFrame, y: np.ndarray, covariate_cols: List[str]) -> Dict[str, float]:
    cols = allowed_feature_cols(list(X_df.columns))
    X_df = X_df[cols].copy()
    pvals: Dict[str, float] = {}
    covars = None
    if covariate_cols:
        cov_keep = [c for c in covariate_cols if c in X_df.columns]
        covars = X_df[cov_keep].to_numpy(dtype=float) if len(cov_keep) > 0 else None
    for col in X_df.columns:
        s = X_df[col]
        if col in CONTINUOUS_PEARSON_COLS:
            try:
                x_num = pd.to_numeric(s, errors="coerce").to_numpy()
                p = safe_pearson_p(x_num, y)
            except Exception:
                p = 1.0
        else:
            p = ancova_p_value(y, s.to_numpy(), covars)
        pvals[col] = float(p) if p is not None else 1.0
    return pvals


def select_features_by_p(pvals: Dict[str, float], alpha: float) -> List[str]:
    return [c for c, p in pvals.items() if (p is not None and p < alpha)]


def consensus_from_sets(list_of_sets: List[set], min_count: int) -> List[str]:
    freq = {}
    for s in list_of_sets:
        for f in s:
            freq[f] = freq.get(f, 0) + 1
    return [f for f, c in freq.items() if c >= min_count]


def nested_cv_catboost(X_train_pool: pd.DataFrame,
                       y_train_pool: np.ndarray,
                       covariates: List[str],
                       rng: np.random.RandomState) -> Tuple[Dict[str, Any], List[str]]:
    outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED_OUTER)
    outer_metrics = {"mse": [], "rmse": [], "mae": [], "r2": []}
    inner_best_params_per_outer = []
    outer_consensus_features_per_fold = []

    all_feature_names = allowed_feature_cols(list(X_train_pool.columns))

    grid = {
        "model__iterations": [200, 500],
        "model__depth": [4, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__l2_leaf_reg": [3, 5],
        "model__bagging_temperature": [0, 0.5]
    }

    from itertools import product
    fold_idx = 0
    for tr_idx, te_idx in outer_kf.split(X_train_pool, y_train_pool):
        fold_idx += 1
        X_tr_outer = X_train_pool.iloc[tr_idx].copy()
        y_tr_outer = y_train_pool[tr_idx].copy()
        X_te_outer = X_train_pool.iloc[te_idx].copy()
        y_te_outer = y_train_pool[te_idx].copy()

        inner_kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED_INNER + fold_idx)
        selected_per_inner = []

        grid_keys = list(grid.keys())
        grid_vals = [grid[k] for k in grid_keys]
        param_combos = [dict(zip(grid_keys, combo)) for combo in product(*grid_vals)] if grid_keys else [{}]
        combo_scores = []

        for inner_tr_idx, inner_val_idx in inner_kf.split(X_tr_outer, y_tr_outer):
            X_tr_inner = X_tr_outer.iloc[inner_tr_idx].copy()
            y_tr_inner = y_tr_outer[inner_tr_idx].copy()
            X_val_inner = X_tr_outer.iloc[inner_val_idx].copy()
            y_val_inner = y_tr_outer[inner_val_idx].copy()

            pvals = compute_univariate_pvals(X_tr_inner, y_tr_inner, covariates)
            selected_feats = select_features_by_p(pvals, ALPHA)
            selected_per_inner.append(set(selected_feats))
            if not selected_feats:
                selected_feats = allowed_feature_cols(list(X_tr_inner.columns))

            pre, *_ = build_preprocessor(selected_feats)

            for pc in param_combos:
                model = CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE")
                pipe = Pipeline([("pre", pre), ("model", model)])
                if pc:
                    pipe.set_params(**pc)
                pipe.fit(X_tr_inner[selected_feats], y_tr_inner)
                preds = pipe.predict(X_val_inner[selected_feats])
                score_rmse = rmse(y_val_inner, preds)
                key = tuple(sorted(pc.items())) if pc else ()
                found = False
                for rec in combo_scores:
                    if tuple(sorted(rec["params"].items())) == key:
                        rec["rmse_vals"].append(score_rmse)
                        found = True
                        break
                if not found:
                    combo_scores.append({"params": pc if pc else {}, "rmse_vals": [score_rmse]})

        for rec in combo_scores:
            vals = rec["rmse_vals"]
            rec["rmse_mean"] = float(np.mean(vals))
            rec["rmse_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        combo_scores.sort(key=lambda d: d["rmse_mean"])
        best_params_this_outer = combo_scores[0]["params"] if combo_scores else {}

        outer_consensus = consensus_from_sets(selected_per_inner, (INNER_FOLDS // 2) + 1)
        if not outer_consensus:
            outer_consensus = [c for c in all_feature_names if c not in ID_COLS]
        outer_consensus_features_per_fold.append(outer_consensus)
        inner_best_params_per_outer.append(best_params_this_outer)

        pre, *_ = build_preprocessor(outer_consensus)
        model = CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE")
        pipe = Pipeline([("pre", pre), ("model", model)])
        if best_params_this_outer:
            pipe.set_params(**best_params_this_outer)
        pipe.fit(X_tr_outer[outer_consensus], y_tr_outer)
        y_pred_outer = pipe.predict(X_te_outer[outer_consensus])

        met = metrics_point(y_te_outer, y_pred_outer)
        for k in outer_metrics:
            outer_metrics[k].append(met[k])

        print(f"[CB] out fold {fold_idx}: RMSE={met['rmse']:.4f}, MAE={met['mae']:.4f}, R2={met['r2']:.4f}")

    summary = {"algo": "CB", "outer_folds": OUTER_FOLDS, "metrics": {}}
    for k, vals in outer_metrics.items():
        summary["metrics"][k] = {
            "per_fold": [float(v) for v in vals],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        }
    summary["outer_consensus_features_per_fold"] = outer_consensus_features_per_fold
    return summary


def _get_final_estimator(obj):
    est = obj
    while isinstance(est, Pipeline):
        est = est.steps[-1][1]
    return est


def _get_preprocessor(ppl):
    pre = None
    if isinstance(ppl, Pipeline):
        if hasattr(ppl, "named_steps") and "preprocess" in ppl.named_steps:
            pre = ppl.named_steps["preprocess"]
        if pre is None:
            for _, step in ppl.steps:
                if isinstance(step, ColumnTransformer):
                    pre = step
                    break
    return pre


def _design_matrix(cols: List[np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.ones(len(cols[0]))] + [c.reshape(-1) for c in cols])
    return X.astype(float)


def _lm_r2(y: np.ndarray, Z: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    yhat = Z @ beta
    ssr = float(((yhat - y.mean()) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return max(0.0, min(1.0, ssr / sst)) if sst > 0 else 0.0


def breusch_pagan_on_bmi(resid: np.ndarray, bmi: np.ndarray) -> Dict[str, float]:
    e2 = resid ** 2
    Z = _design_matrix([bmi])
    r2 = _lm_r2(e2, Z)
    n, k = Z.shape
    LM = n * r2
    p = 1.0 - stats.chi2.cdf(LM, df=k - 1)
    return {"stat": float(LM), "df": int(k - 1), "p": float(p), "r2": float(r2)}


def fit_variance_model(y_true: np.ndarray, y_pred: np.ndarray, bmi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    resid = y_true - y_pred
    e2 = (resid ** 2).clip(min=1e-8)
    loge2 = np.log(e2)
    X = np.column_stack([np.ones_like(bmi), bmi, y_pred])
    gamma = np.linalg.lstsq(X, loge2, rcond=None)[0]
    log_sigma2 = X @ gamma
    return gamma, log_sigma2


def fit_wls_calibrator(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:

    W = np.diag(weights)
    X = np.column_stack([np.ones_like(y_pred), y_pred])
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y_true
    beta = np.linalg.solve(XtWX, XtWy)
    a, b = float(beta[0]), float(beta[1])
    return a, b


def apply_linear_map(yhat: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * yhat


def rtm_summary(y1_baseline: np.ndarray, y2_followup: np.ndarray, B: int = 1000, seed: int = 42) -> dict:

    y1 = np.asarray(y1_baseline, dtype=float)
    y2 = np.asarray(y2_followup, dtype=float)
    mask = ~np.isnan(y1) & ~np.isnan(y2)
    y1, y2 = y1[mask], y2[mask]
    n = len(y1)
    if n < 3:
        return {"n": int(n), "note": "not enough samples"}

    delta = y2 - y1
    X = np.column_stack([np.ones(n), y1])

    beta, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
    yhat = X @ beta
    resid = delta - yhat
    dof = n - X.shape[1]
    sigma2 = float((resid @ resid) / max(dof, 1))
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(cov_beta[1, 1]))
    b = float(beta[1])

    tcrit = stats.t.ppf(0.975, df=max(dof, 1))
    b_lo, b_hi = b - tcrit * se_b, b + tcrit * se_b

    tval = b / (se_b + 1e-12)
    p_b = 2 * (1 - stats.t.cdf(abs(tval), df=max(dof, 1)))

    sst = float(((delta - delta.mean()) ** 2).sum())
    ssr = float(((yhat - delta.mean()) ** 2).sum())
    r2 = 0.0 if sst <= 0 else ssr / sst

    rng = np.random.RandomState(seed)
    r2_bs = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        Xb = np.column_stack([np.ones(n), y1[idx]])
        deltab = delta[idx]
        betab, _, _, _ = np.linalg.lstsq(Xb, deltab, rcond=None)
        yhatb = Xb @ betab
        sstb = float(((deltab - deltab.mean()) ** 2).sum())
        ssrb = float(((yhatb - deltab.mean()) ** 2).sum())
        r2_bs.append(0.0 if sstb <= 0 else ssrb / sstb)
    lo, hi = np.percentile(r2_bs, [2.5, 97.5])

    return {
        "n": int(n),
        "slope_b": b, "slope_b_lo": float(b_lo), "slope_b_hi": float(b_hi), "slope_b_p": float(p_b),
        "r2": float(r2), "r2_lo": float(lo), "r2_hi": float(hi), "B": int(B)
    }


def error_distribution_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    e = y_true - y_pred
    abs_e = np.abs(e)
    return {
        "n": int(len(e)),
        "bias_mean": float(np.mean(e)),
        "sd": float(np.std(e, ddof=1)) if len(e) > 1 else 0.0,
        "mad": float(np.median(np.abs(e - np.median(e)))),
        "iqr_abs_e": float(np.percentile(abs_e, 75) - np.percentile(abs_e, 25)),
        "p90_abs_e": float(np.percentile(abs_e, 90)),
        "p95_abs_e": float(np.percentile(abs_e, 95)),
    }


def bootstrap_group_metrics(df: pd.DataFrame, group_col: str,
                            y_col: str, yhat_col: str,
                            B: int = BOOT_B, seed: int = 42) -> Dict[str, Any]:

    rng = np.random.RandomState(seed)
    out = {"groups": {}, "delta": {}}
    groups = [g for g in df[group_col].dropna().unique()]

    for g in groups:
        sub = df[df[group_col] == g]
        y = sub[y_col].to_numpy()
        yp = sub[yhat_col].to_numpy()
        out["groups"][str(g)] = {
            "metrics": bootstrap_metrics(y, yp, B=B, seed=seed),
            "errdist": error_distribution_table(y, yp),
            "n": int(len(sub))
        }

    def stat_rmse(d: pd.DataFrame, col_pred: str):
        vals = []
        for g in groups:
            s = d[d[group_col] == g]
            vals.append(rmse(s[y_col].to_numpy(), s[col_pred].to_numpy()))
        return np.max(vals) - np.min(vals)

    def stat_mae(d, col_pred):
        vals = []
        for g in groups:
            s = d[d[group_col] == g]
            vals.append(mean_absolute_error(s[y_col], s[col_pred]))
        return np.max(vals) - np.min(vals)

    def stat_r2(d, col_pred):
        vals = []
        for g in groups:
            s = d[d[group_col] == g]
            try:
                vals.append(r2_score(s[y_col], s[col_pred]))
            except Exception:
                vals.append(np.nan)
        vals = np.array(vals, dtype=float)
        return np.nanmax(vals) - np.nanmin(vals)

    def stat_mse(d, col_pred):
        vals = []
        for g in groups:
            s = d[d[group_col] == g]
            e = s[y_col].to_numpy() - s[col_pred].to_numpy()
            vals.append(float(np.mean(e ** 2)))
        return np.max(vals) - np.min(vals)

    n = len(df)
    rmse_bs, mae_bs, r2_bs, mse_bs = [], [], [], []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        d = df.iloc[idx]
        rmse_bs.append(stat_rmse(d, yhat_col))
        mae_bs.append(stat_mae(d, yhat_col))
        r2_bs.append(stat_r2(d, yhat_col))
        mse_bs.append(stat_mse(d, yhat_col))

    def ci(a):
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        return {"mean": float(np.nanmean(a)), "lo": float(lo), "hi": float(hi)}

    out["delta"]["rmse"] = ci(rmse_bs)
    out["delta"]["mae"] = ci(mae_bs)
    out["delta"]["r2"] = ci(r2_bs)
    out["delta"]["mse"] = ci(mse_bs)

    def perm_p(stat_func, R=2000):
        true = stat_func(df, yhat_col)
        cnt = 0
        sizes = [len(df[df[group_col] == g]) for g in groups]
        for _ in range(R):
            perm = df.copy()

            perm[group_col] = np.random.permutation(perm[group_col].values)
            s = stat_func(perm, yhat_col)
            if s >= true - 1e-12:
                cnt += 1
        return float((cnt + 1) / (R + 1))

    out["delta_p"] = {
        "rmse": perm_p(stat_rmse),
        "mae": perm_p(stat_mae),
        "r2": perm_p(stat_r2),
        "mse": perm_p(stat_mse),
    }
    return out


def plot_pred_vs_obs(y_true, y_pred, out_png):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, y_true, s=18, alpha=0.7)
    sm = lowess(y_true, y_pred, frac=0.3, it=0, return_sorted=True)
    plt.plot(sm[:, 0], sm[:, 1], linewidth=2)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title("Predicted vs Observed (Test)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_bland_altman(y_true, y_pred, out_png):
    diff = y_true - y_pred
    mean_ = (y_true + y_pred) / 2.0
    md = np.mean(diff)
    sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    loa_lo = md - 1.96 * sd
    loa_hi = md + 1.96 * sd

    plt.figure(figsize=(7, 5))
    plt.scatter(mean_, diff, s=18, alpha=0.7)

    plt.axhline(md, color='r', linestyle='--', linewidth=1, label=f"Mean={md:.2f}")
    plt.axhline(loa_lo, color='g', linestyle='--', linewidth=1, label=f"LoA-={loa_lo:.2f}")
    plt.axhline(loa_hi, color='g', linestyle='--', linewidth=1, label=f"LoA+={loa_hi:.2f}")
    plt.xlabel("(Observed + Predicted) / 2")
    plt.ylabel("Observed - Predicted")
    plt.title("Bland–Altman (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_residuals_vs_pred(y_true, y_pred, out_png):
    resid = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, s=18, alpha=0.7)

    plt.axhline(0.0, color='r', linestyle='--', linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals (y - yhat)")
    plt.title("Residuals vs Predicted (Test)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


BMI_THRESH = {
    7: {"M": (17.4, 19.2), "F": (17.2, 18.9)},
    8: {"M": (18.1, 20.3), "F": (18.1, 19.9)},
    9: {"M": (19.0, 21.4), "F": (19.0, 21.0)},
    10: {"M": (19.6, 22.5), "F": (20.0, 22.1)},
    11: {"M": (20.3, 23.6), "F": (21.1, 23.3)},
    12: {"M": (21.0, 24.7), "F": (22.1, 24.5)},
    13: {"M": (21.9, 25.7), "F": (22.6, 25.6)},
    14: {"M": (22.6, 26.6), "F": (23.0, 26.3)},
    15: {"M": (23.1, 26.9), "F": (23.4, 26.9)},
    16: {"M": (23.5, 27.4), "F": (23.7, 27.4)},
    17: {"M": (23.8, 27.8), "F": (23.8, 27.7)},
    18: {"M": (24.0, 28.0), "F": (24.0, 28.0)},
}


def sex_to_code(s):
    s = str(s).strip().lower()
    if s in ["male", "m", "1", "boy"]:
        return "M"
    return "F"


def classify_bmi(age_years: float, gender: str, bmi: float) -> str:

    if pd.isna(age_years) or pd.isna(bmi):
        return np.nan
    a = int(np.clip(int(np.floor(age_years)), 7, 18))
    g = sex_to_code(gender)
    over, obese = BMI_THRESH.get(a, {"M": (np.inf, np.inf), "F": (np.inf, np.inf)}).get(g, (np.inf, np.inf))
    if bmi >= obese:
        return "obese"
    elif bmi >= over:
        return "overweight"
    else:
        return "normal"


def make_age_group(age):
    if pd.isna(age):
        return np.nan
    age = float(age)
    if 14 <= age <= 15:
        return "14-15"
    elif 16 <= age <= 17:
        return "16-17"
    else:
        return "other"


def main():
    ensure_dir(OUT_DIR)

    lower = DATA_PATH.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(DATA_PATH)
    else:
        try:
            df = pd.read_csv(DATA_PATH)
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATH, encoding="gb18030")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column：{TARGET_COL}")

    if "Gender" not in df.columns:
        raise ValueError("Missing Gender column")
    if "Age" not in df.columns and "age" in df.columns:
        df.rename(columns={"age": "Age"}, inplace=True)
    if "Age" not in df.columns:
        raise ValueError("Missing Age column (integer years)")
    if "Baseline BMI" not in df.columns:
        raise ValueError("Missing Baseline BMI column")

    for c in ID_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors="ignore")

    y = df[TARGET_COL].to_numpy(dtype=float)
    X = df.drop(columns=[TARGET_COL]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("=== Nested Cross-Validation: CatBoost ===")
    rng = np.random.RandomState(RANDOM_STATE)
    cb_report = nested_cv_catboost(X_train, y_train, ANCOVA_COVARIATES, rng)
    with open(os.path.join(OUT_DIR, "cb_outer_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cb_report, f, indent=2, ensure_ascii=False)

    fold_sets = cb_report["outer_consensus_features_per_fold"]

    def consensus(list_of_sets, min_count):
        freq = {}
        for s in list_of_sets:
            for it in s:
                freq[it] = freq.get(it, 0) + 1
        return [k for k, v in freq.items() if v >= min_count]

    final_feats = consensus([set(s) for s in fold_sets], (OUTER_FOLDS // 2) + 1)
    if not final_feats:
        final_feats = sorted(list(set().union(*[set(s) for s in fold_sets])))
    if not final_feats:
        final_feats = [c for c in X_train.columns if c not in EXCLUDE_FROM_MODEL and c not in ID_COLS]

    pre, *_ = build_preprocessor(final_feats)
    model = CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE")
    pipe = Pipeline([("pre", pre), ("model", model)])

    grid = {"model__iterations": [200, 500],
            "model__depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__l2_leaf_reg": [3, 5],
            "model__bagging_temperature": [0, 0.5]}

    scoring = make_scorer(lambda yt, yp: -rmse(yt, yp))
    cv = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED_OUTER)
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train[final_feats], y_train)

    best_cb = gs.best_estimator_
    best_params = gs.best_params_
    best_rmse_mean = -float(gs.best_score_)
    joblib.dump(best_cb, os.path.join(OUT_DIR, "final_refit_CB.joblib"))
    with open(os.path.join(OUT_DIR, "final_refit_CB_params.json"), "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "cv_rmse_mean": best_rmse_mean,
                   "final_features": final_feats}, f, indent=2, ensure_ascii=False)
    print(f"[Final CV] CB: best RMSE={best_rmse_mean:.4f} params={best_params}")
    print("Number of final features:", len(final_feats))

    kf_oof = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED_OUTER)
    oof_pred = np.zeros_like(y_train, dtype=float)

    for tr_idx, val_idx in kf_oof.split(X_train, y_train):

        pre_oof, *_ = build_preprocessor(final_feats)
        model_oof = CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE")
        pipe_oof = Pipeline([("pre", pre_oof), ("model", model_oof)])

        if best_params:
            pipe_oof.set_params(**best_params)

        pipe_oof.fit(X_train.iloc[tr_idx][final_feats], y_train[tr_idx])

        oof_pred[val_idx] = pipe_oof.predict(X_train.iloc[val_idx][final_feats])

    resid_tr_oof = y_train - oof_pred
    bmi_tr_all = X_train["Baseline BMI"].to_numpy(dtype=float)

    bp_train_oof = breusch_pagan_on_bmi(resid_tr_oof, bmi_tr_all)
    with open(os.path.join(OUT_DIR, "bp_train_oof.json"), "w", encoding="utf-8") as f:
        json.dump(bp_train_oof, f, indent=2, ensure_ascii=False)
    print("BP(train OOF, e^2~BMI):", bp_train_oof)

    yhat_test = best_cb.predict(X_test[final_feats])
    uncal = bootstrap_metrics(y_test, yhat_test, B=BOOT_B, seed=RANDOM_STATE)
    with open(os.path.join(OUT_DIR, "uncalibrated_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(uncal, f, indent=2, ensure_ascii=False)

    NON_ADJUSTABLE_FEATURES = {
        'Baseline BMI', 'Baseline BMR', 'Age', 'Gender',
        'Paternal overweight/obesity status', 'Paternal educational level', 'Paternal occupation',
        'Maternal overweight/obesity status', 'Maternal educational level', 'Maternal occupation',
        'Family income', 'Family residence location', 'On-campus residence',
        'Post-exercise sensations', 'Satisfaction with body size', 'Considered changing body size'
    }

    SHAP_NAME_ABBR = {
        'Daily sleep duration': 'DSD',
        'Frequency of staying up late': 'FSUL',
        'Sedentariness duration on weekends': 'SDOW',
        'Schoolwork burden': 'SB',
        'Frequency of high-protein food intake': 'FHPFI',
        'Frequency of midnight snack intake': 'FMSI',
        'Frequency of high-calorie foods intake': 'FHCFI',
        'Frequency of participation in physical activities': 'FPPA',
        'Physical activities duration on weekends': 'PADOW',
        'Participation in professional sports training': 'PPST',
        'Parental support for sports involvement': 'PSFSI',
        'Recognize self-weight status correctly': 'RSWSC',
        'Level of health literacy': 'LHL',
    }

    print("\n=== SHAP (Un-calibrated final model) ===")
    out_shap_dir = os.path.join(OUT_DIR, SHAP_SUBDIR)
    os.makedirs(out_shap_dir, exist_ok=True)

    cb_model = _get_final_estimator(best_cb)
    preproc = _get_preprocessor(best_cb)

    X_shap_raw = X_test[final_feats].copy()
    if len(X_shap_raw) > 3000:
        X_shap_raw = X_shap_raw.sample(n=3000, random_state=RANDOM_STATE, replace=False)

    if preproc is not None:
        Xtx = preproc.transform(X_shap_raw)
        if hasattr(preproc, "get_feature_names_out"):
            try:
                tx_cols = preproc.get_feature_names_out(final_feats).tolist()
            except Exception:
                tx_cols = [f"f{i}" for i in range(Xtx.shape[1])]
        else:
            tx_cols = [f"f{i}" for i in range(Xtx.shape[1])]
        if hasattr(Xtx, "toarray"):
            Xtx = Xtx.toarray()
        X_shap_df = pd.DataFrame(Xtx, columns=tx_cols, index=X_shap_raw.index)
    else:
        X_shap_df = X_shap_raw

    try:
        explainer = shap.TreeExplainer(cb_model)
        shap_values = explainer.shap_values(X_shap_df)
        expected_value = explainer.expected_value
    except Exception as e:
        print("TreeExplainer not supported for this object, falling back to generic Explainer:", e)
        exp = shap.Explainer(cb_model.predict, X_shap_df)
        sv = exp(X_shap_df)
        shap_values = sv.values
        expected_value = float(np.array(sv.base_values).reshape(-1)[0])

    _PREFIXES = ("num__", "ord__", "cat__", "cat_ohe__", "ohe__")

    def strip_prefix(col: str) -> str:
        for pf in _PREFIXES:
            if col.startswith(pf):
                return col[len(pf):]
        return col

    def raw_base_name(col: str) -> str:
        tail = strip_prefix(col)
        if "=" in tail:
            tail = tail.split("=", 1)[0]
        if "_" in tail:
            return tail.split("_", 1)[0]
        return tail

    def abbr_name(base: str) -> str:
        return SHAP_NAME_ABBR.get(base, base)

    def make_unique(names):
        seen, out = {}, []
        for n in names:
            if n in seen:
                seen[n] += 1
                out.append(f"{n}_{seen[n]}")
            else:
                seen[n] = 0
                out.append(n)
        return out

    def _slug(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)

    def save_html_for_png(png_path: str, title: str, extra_html: str = ""):
        html_path = os.path.splitext(png_path)[0] + ".html"
        rel = os.path.basename(png_path)
        html = f"""<!doctype html><meta charset="utf-8">
    <title>{title}</title>
    <h3 style="font-family:Arial;margin:6px 0;">{title}</h3>
    <div><img src="{rel}" style="max-width:100%;height:auto;border:0;"/></div>
    {extra_html}
    """
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    saved_imgs = []

    tx_cols = X_shap_df.columns.tolist()
    base_names = [raw_base_name(c) for c in tx_cols]
    from collections import defaultdict
    group_indices_all = defaultdict(list)
    for j, b in enumerate(base_names):
        group_indices_all[b].append(j)

    kept_features = [f for f in group_indices_all.keys() if f not in NON_ADJUSTABLE_FEATURES]
    if len(kept_features) == 0:
        print("Warning: no displayable features after filtering. Please check NON_ADJUSTABLE_FEATURES.")

    shap_grouped_list = []
    for f in kept_features:
        idxs = group_indices_all[f]
        sv_f = shap_values[:, idxs].sum(axis=1)
        shap_grouped_list.append(sv_f.reshape(-1, 1))
    shap_grouped = np.hstack(shap_grouped_list) if shap_grouped_list else np.zeros((len(X_shap_df), 0))

    X_grouped = pd.DataFrame(index=X_shap_raw.index)
    for f in kept_features:
        if f in X_shap_raw.columns:
            X_grouped[f] = X_shap_raw[f].astype(str)
        else:
            any_col = group_indices_all[f][0]
            cand = strip_prefix(tx_cols[any_col])
            if "=" in cand:
                X_grouped[f] = cand.split("=", 1)[-1]
            elif "_" in cand:
                X_grouped[f] = cand.split("_", 1)[-1]
            else:
                X_grouped[f] = cand

    X_grouped_color = pd.DataFrame(index=X_grouped.index)
    for c in X_grouped.columns:
        X_grouped_color[c] = pd.Categorical(X_grouped[c]).codes

    plt.figure(figsize=(9, max(4, int(0.35 * max(1, shap_grouped.shape[1])))))
    shap.summary_plot(
        shap_grouped, X_grouped_color, show=False, plot_type="dot",
        feature_names=list(X_grouped.columns), color_bar=True
    )
    fig = plt.gcf()
    ax = plt.gca()

    fig.set_size_inches(9.8, 6)
    plt.subplots_adjust(left=0.37, right=0.88, top=0.96, bottom=0.12)

    ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12, color="#333")
    ax.tick_params(axis="x", labelsize=11, colors="#333")

    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=11)
    cbar_ax.set_ylabel("Feature value", fontsize=12)

    plt.tight_layout()
    png = os.path.join(out_shap_dir, "summary_beeswarm.png")
    plt.savefig(png, dpi=300);
    plt.close()
    saved_imgs.append(("SHAP summary (beeswarm, colored)", png))

    mean_abs_grouped = np.abs(shap_grouped).mean(axis=0) if shap_grouped.size > 0 else np.array([])
    imp_group = pd.DataFrame({"feature": list(X_grouped.columns), "mean_abs_shap": mean_abs_grouped})
    imp_group = imp_group.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_group.to_json(os.path.join(out_shap_dir, "summary_importance_grouped.json"),
                      force_ascii=False, orient="records", indent=2)

    order_idx = np.argsort(-mean_abs_grouped) if shap_grouped.size > 0 else np.array([])
    plot_features = [X_grouped.columns[i] for i in order_idx[:SHAP_TOPK]]

    Fnames = list(X_grouped.columns)

    for f in plot_features:
        try:
            i = Fnames.index(f)

            X_codes = pd.DataFrame(index=X_grouped.index)
            for c in Fnames:
                X_codes[c] = pd.Categorical(X_grouped[c]).codes

            j_cands = shap.approximate_interactions(i, shap_grouped, X_codes.values)
            if isinstance(j_cands, (list, np.ndarray)) and len(j_cands) > 0 and int(j_cands[0]) != i:
                j = int(j_cands[0])
            else:
                j = None

            shap.dependence_plot(
                i, shap_grouped, X_codes.values,
                interaction_index=j, show=False, feature_names=Fnames
            )
            ax = plt.gca()

            cats_x = list(pd.Categorical(X_grouped[f]).categories)
            xticks = sorted(np.unique(X_codes[f]))
            ax.set_xticks(xticks)
            ax.set_xticklabels(cats_x, rotation=0)
            ax.set_xlabel(f)
            ax.set_ylabel(f"SHAP value for\n{f}")

            cb_ax = plt.gcf().axes[-1]
            if j is not None:
                fj = Fnames[j]
                cats_c = list(pd.Categorical(X_grouped[fj]).categories)
                cticks = sorted(np.unique(X_codes[fj]))
                cb_ax.set_yticks(cticks)
                cb_ax.set_yticklabels(cats_c)
                cb_ax.set_ylabel(fj)
            else:
                cticks = sorted(np.unique(X_codes[f]))
                cb_ax.set_yticks(cticks)
                cb_ax.set_yticklabels(cats_x)
                cb_ax.set_ylabel(f)

                fig = plt.gcf()
                ax = plt.gca()

                fig.set_size_inches(8.5, 5.5)
                plt.subplots_adjust(left=0.22, right=0.86, bottom=0.24, top=0.90)

                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
                ax.tick_params(axis="x", labelsize=11)
                ax.tick_params(axis="y", labelsize=11)

                cbar_ax = plt.gcf().axes[-1]
                cbar_ax.tick_params(labelsize=10, pad=6)
                cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=12)

            plt.title(f"Dependence: {f}")
            plt.tight_layout()
            png = os.path.join(out_shap_dir, f"dependence_{_slug(f)}.png")
            plt.savefig(png, dpi=300)
            plt.close()

        except Exception as e:
            print(f"SHAP dependence_plot({f}) plotting failed:", e)

    inter_vals = None
    if hasattr(explainer, "shap_interaction_values"):
        try:
            inter_vals = explainer.shap_interaction_values(X_shap_df)
        except Exception as ee:
            print("Interaction SHAP computation failed (TreeExplainer):", ee)
    if isinstance(inter_vals, list):
        inter_vals = inter_vals[0]

    if inter_vals is not None:

        S_cols = np.abs(inter_vals).mean(axis=0)
        all_features = list(group_indices_all.keys())
        F_all = len(all_features)
        H_all = np.zeros((F_all, F_all), dtype=float)
        for i, fi in enumerate(all_features):
            Ii = group_indices_all[fi]
            for j, fj in enumerate(all_features):
                Ij = group_indices_all[fj]
                block = S_cols[np.ix_(Ii, Ij)]
                H_all[i, j] = float(np.mean(block)) if block.size > 0 else 0.0

        df_H_all = pd.DataFrame(H_all, index=all_features, columns=all_features)
        df_H_all.to_csv(os.path.join(out_shap_dir, "interaction_matrix_full.csv"), encoding="utf-8-sig")
        abbr_all = make_unique([abbr_name(f) for f in all_features])
        df_H_all_abbr = pd.DataFrame(H_all, index=abbr_all, columns=abbr_all)
        df_H_all_abbr.to_csv(os.path.join(out_shap_dir, "interaction_matrix_full_abbr.csv"), encoding="utf-8-sig")

        upper_all = []
        for i in range(F_all):
            for j in range(i + 1, F_all):
                upper_all.append((all_features[i], all_features[j], H_all[i, j]))
        upper_all.sort(key=lambda x: x[2], reverse=True)
        pd.DataFrame(upper_all, columns=["Feature_A", "Feature_B", "mean_|int_SHAP|"]) \
            .to_csv(os.path.join(out_shap_dir, "interaction_top_pairs_full.csv"), index=False, encoding="utf-8-sig")

        mod_feats = [f for f in all_features if f not in NON_ADJUSTABLE_FEATURES]
        idx_mod = [all_features.index(f) for f in mod_feats]
        upper_mod = []
        for ii, i in enumerate(idx_mod):
            for jj, j in enumerate(idx_mod):
                if j <= i:
                    continue
                upper_mod.append((mod_feats[ii], mod_feats[jj], H_all[i, j]))
        upper_mod.sort(key=lambda x: x[2], reverse=True)
        pd.DataFrame(upper_mod, columns=["Feature_A", "Feature_B", "mean_|int_SHAP|"]) \
            .to_csv(os.path.join(out_shap_dir, "interaction_top_pairs_modifiable.csv"), index=False,
                    encoding="utf-8-sig")

        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform

            H_mod = df_H_all.loc[mod_feats, mod_feats].values if len(mod_feats) > 0 else np.zeros((0, 0))
            if H_mod.size == 0:
                print("Interaction heatmap: no modifiable×modifiable combinations to display.")
            else:

                Hn = H_mod.copy()

                vmax = Hn.max() if Hn.max() > 0 else 1.0
                D = 1.0 - (Hn / vmax)
                np.fill_diagonal(D, 0.0)
                Z = linkage(squareform(D, checks=False), method="average")
                order = leaves_list(Z)
                H_show = Hn[np.ix_(order, order)]
                labels_order_full = [mod_feats[i] for i in order]
                labels_order_abbr = make_unique([abbr_name(x) for x in labels_order_full])

                fig, ax = plt.subplots(figsize=(1.0 + 0.45 * H_show.shape[0], 1.0 + 0.45 * H_show.shape[1]))
                im = ax.imshow(H_show, cmap="coolwarm", vmin=0.0, vmax=H_show.max() if H_show.max() > 0 else 1)
                ax.set_xticks(range(H_show.shape[1]))
                ax.set_yticks(range(H_show.shape[0]))
                ax.set_xticklabels(labels_order_abbr, rotation=60, ha="right")
                ax.set_yticklabels(labels_order_abbr)
                ax.set_title("SHAP interaction heatmap (mean |interaction SHAP|)", pad=8)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                title_size = 10
                tick_size = 10
                cbar_size = 10
                ax.set_title("SHAP interaction heatmap (mean |interaction SHAP|)",
                             fontsize=title_size, color="#222", pad=10)

                ax.tick_params(axis="both", which="major", labelsize=tick_size, colors="#333")

                ax.set_xlabel(ax.get_xlabel(), fontsize=tick_size, color="#333")
                ax.set_ylabel(ax.get_ylabel(), fontsize=tick_size, color="#333")

                cbar.ax.tick_params(labelsize=cbar_size, colors="#333")
                cbar.set_label("mean |interaction SHAP|", fontsize=title_size - 2, color="#333")

                plt.tight_layout()

                cbar.set_label("mean |interaction SHAP|")
                plt.tight_layout()
                png = os.path.join(out_shap_dir, "interaction_heatmap_modifiable.png")
                plt.savefig(png, dpi=300);
                plt.close()
                saved_imgs.append(("Interaction heatmap (modifiable)", png))
        except Exception as e:
            print("Interaction heatmap generation failed:", e)
    else:
        print("Interaction SHAP not available (explainer does not provide shap_interaction_values).")

    try:
        abbr_names = make_unique([abbr_name(f) for f in X_grouped.columns])

        FORCE_IDS = None
        if FORCE_IDS is None:
            idx_list = list(X_grouped.index)[:SHAP_FORCE_N]
        else:
            idx_list = [i for i in FORCE_IDS if i in X_grouped.index][:SHAP_FORCE_N]

        base_val = float(np.array(expected_value).reshape(-1)[0])

        for k, idx in enumerate(idx_list, 1):
            row_pos = X_grouped.index.get_loc(idx)
            sv_row = shap_grouped[row_pos, :].astype(float)

            row_vals = X_grouped.iloc[row_pos, :].astype(str).values

            order = np.argsort(-np.abs(sv_row))
            sv_ord = sv_row[order]
            names_ord = [abbr_names[i] for i in order]
            data_ord = np.array([row_vals[i] for i in order], dtype=object)

            max_display = min(12, len(sv_ord))

            try:
                exp = shap.Explanation(
                    values=sv_ord,
                    base_values=base_val,
                    data=data_ord,
                    feature_names=names_ord
                )
                shap.plots.waterfall(exp, max_display=max_display, show=False)
            except Exception:
                shap.waterfall_plot(
                    expected_value=base_val,
                    shap_values=sv_ord,
                    feature_names=names_ord,
                    max_display=max_display,
                    show=False
                )

            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            plt.title(f"Waterfall: instance #{k} (idx={idx})", fontsize=12)
            plt.tight_layout()

            png = os.path.join(out_shap_dir, f"waterfall_{k:02d}_idx{idx}.png")
            plt.savefig(png, dpi=300)
            plt.close()

            lis = "\n".join([f"<li><b>{n}</b> = {v}</li>" for n, v in zip(names_ord, data_ord)])
            extra = f"<div style='font-family:Arial;margin-top:8px;'><b>Feature values (sorted by |SHAP|):</b><ul>{lis}</ul></div>"
            saved_imgs.append((f"Waterfall plot #{k} (idx={idx})", png, extra))

    except Exception as e:
        print("SHAP waterfall plots generation failed:", e)

    for item in saved_imgs:
        title = item[0]
        png_path = item[1]
        extra_html = item[2] if len(item) > 2 else ""
        save_html_for_png(png_path, title, extra_html)

    meta = {
        "n_shap_samples": int(len(X_shap_df)),
        "n_display_features": int(shap_grouped.shape[1]),
        "exclude_features_for_display": sorted(list(NON_ADJUSTABLE_FEATURES)),
        "top_features_after_filter": plot_features if len(plot_features) > 0 else [],
        "exports": {
            "interaction_matrix_full.csv": "Full feature×feature mean(|interaction SHAP|), includes non-adjustable (diagonal = main effects)",
            "interaction_matrix_full_abbr.csv": "Same as above; rows/columns use abbreviations",
            "interaction_top_pairs_full.csv": "Ranking of all pairs (including non-adjustable) by strength",
            "interaction_top_pairs_modifiable.csv": "Ranking of modifiable×modifiable pairs by strength",
            "summary_importance_grouped.json": "Grouped global importance (visualization set only)"
        }
    }
    with open(os.path.join(out_shap_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    y1_test = X_test["Baseline BMI"].to_numpy(dtype=float)
    y2_test = y_test
    rtm = rtm_summary(y1_test, y2_test, B=BOOT_B, seed=RANDOM_STATE)
    with open(os.path.join(OUT_DIR, "rtm_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rtm, f, indent=2, ensure_ascii=False)

    need_cal = (bp_train_oof["p"] < 0.05)
    yhat_tr = best_cb.predict(X_train[final_feats])
    bmi_tr = X_train["Baseline BMI"].to_numpy(dtype=float)

    if need_cal:

        gamma, log_sigma2_tr = fit_variance_model(y_train, yhat_tr, bmi_tr)
        sigma2_tr = np.exp(log_sigma2_tr)
        w_tr = 1.0 / np.clip(sigma2_tr, 1e-6, None)

        a, b = fit_wls_calibrator(y_train, yhat_tr, w_tr)

        bmi_te = X_test["Baseline BMI"].to_numpy(dtype=float)
        Xv_te = np.column_stack([np.ones_like(bmi_te), bmi_te, yhat_test])
        sigma2_te = np.exp(Xv_te @ gamma)

        yhat_cal = apply_linear_map(yhat_test, a, b)

        cal = bootstrap_metrics(y_test, yhat_cal, B=BOOT_B, seed=RANDOM_STATE)
        with open(os.path.join(OUT_DIR, "calibrated_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"wls": {"a": a, "b": b, "gamma": [float(g) for g in gamma]},
                       "metrics": cal}, f, indent=2, ensure_ascii=False)

        plot_pred_vs_obs(y_test, yhat_cal, os.path.join(OUT_DIR, "pred_vs_obs_calibrated.png"))
        plot_bland_altman(y_test, yhat_cal, os.path.join(OUT_DIR, "bland_altman_calibrated.png"))
        plot_residuals_vs_pred(y_test, yhat_cal, os.path.join(OUT_DIR, "residuals_vs_pred_calibrated.png"))

        test_df = X_test.copy()
        test_df["y"] = y_test
        test_df["yhat_cal"] = yhat_cal

        test_df["SexG"] = test_df["Gender"].apply(sex_to_code)

        test_df["age_group"] = test_df["Age"].apply(make_age_group)

        test_df["bmi_class"] = test_df.apply(
            lambda r: classify_bmi(r["Age"], r["SexG"], r["Baseline BMI"]), axis=1
        )

        test_df["bmi_2group"] = test_df["bmi_class"].map(
            lambda z: ("normal" if z == "normal" else ("ow_ob" if z in ["overweight", "obese"] else np.nan))
        )

        ed_overall = error_distribution_table(test_df["y"].to_numpy(), test_df["yhat_cal"].to_numpy())
        by_sex = bootstrap_group_metrics(test_df, "SexG", "y", "yhat_cal", B=BOOT_B, seed=RANDOM_STATE)
        by_age = bootstrap_group_metrics(test_df[test_df["age_group"].isin(["14-15", "16-17"])].copy(),
                                         "age_group", "y", "yhat_cal", B=BOOT_B, seed=RANDOM_STATE)
        by_bmi2 = bootstrap_group_metrics(
            test_df[test_df["bmi_2group"].isin(["normal", "ow_ob"])].copy(),
            "bmi_2group", "y", "yhat_cal", B=BOOT_B, seed=RANDOM_STATE
        )

        with open(os.path.join(OUT_DIR, "error_distributions_calibrated.json"), "w", encoding="utf-8") as f:
            json.dump({"overall": ed_overall,
                       "by_sex": by_sex, "by_age": by_age, "by_bmi_2group": by_bmi2},
                      f, indent=2, ensure_ascii=False)

        print("\n=== Calibrated CatBoost: test set point estimates ===");
        print(cal["point"])
        print("(Stratified error distributions and performance have been saved as JSON)")

    else:
        print(
            "\nBP not significant: WLS calibration not triggered. Uncalibrated test performance and BP results have been saved.")

    print("\nDone. Outputs at:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
