import json, math, os, warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.utils import check_random_state

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = r"../data/BMIPre_1_2.csv"
SHEET_NAME = 0
TARGET_COL = "Target"
ID_COLS: List[str] = ["Number"]

ANCOVA_COVARIATES: List[str] = ["Baseline BMI"]

CONTINUOUS_PEARSON_COLS: List[str] = ["Baseline BMI", "Baseline BMR", "age"]

ALPHA = 0.05
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_JOBS = -1

RANDOM_STATE = 42
SEED_OUTER = 42
SEED_INNER = 42

INNER_CONSENSUS_MIN_COUNT = (INNER_FOLDS // 2) + 1
OUTER_CONSENSUS_MIN_COUNT = (OUTER_FOLDS // 2) + 1

OUT_DIR = "./outputs BMIPre"

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
    "Frequency of participation in physical activities": ["0 times/week", "1–2 times/week", "2–3 times/week", "> 3 times/week"],
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

NATIVE_CATEGORICAL_MODELS = set()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {"mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}

def bootstrap_ci_on_test(estimator, X_test: pd.DataFrame, y_test: np.ndarray, feats: List[str],
                         B: int = 1000, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(random_state)
    n = len(y_test)
    rmse_vals, mae_vals, mse_vals, r2_vals = [], [], [], []

    X_t = X_test[feats].reset_index(drop=True)
    y_t = pd.Series(y_test).reset_index(drop=True)

    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        y_true_b = y_t.iloc[idx].to_numpy()
        y_pred_b = estimator.predict(X_t.iloc[idx])
        mse_b = mean_squared_error(y_true_b, y_pred_b)
        mse_vals.append(mse_b)
        rmse_vals.append(math.sqrt(mse_b))
        mae_vals.append(mean_absolute_error(y_true_b, y_pred_b))
        r2_vals.append(r2_score(y_true_b, y_pred_b))

    def pct_ci(arr, lo=2.5, hi=97.5):
        a = np.asarray(arr, dtype=float)
        return float(np.percentile(a, lo)), float(np.percentile(a, hi))

    mse_lo, mse_hi = pct_ci(mse_vals)
    rmse_lo, rmse_hi = pct_ci(rmse_vals)
    mae_lo, mae_hi = pct_ci(mae_vals)
    r2_lo, r2_hi = pct_ci(r2_vals)

    return {
        "mse":  {"lo": mse_lo,  "hi": mse_hi},
        "rmse": {"lo": rmse_lo, "hi": rmse_hi},
        "mae":  {"lo": mae_lo,  "hi": mae_hi},
        "r2":   {"lo": r2_lo,   "hi": r2_hi},
        "B": B
    }

def paired_baseline_vs_model_bootstrap(best_estimator,
                                       X_test: pd.DataFrame,
                                       y_test: np.ndarray,
                                       feats_for_best: List[str],
                                       baseline_col: str = "Baseline BMI",
                                       B: int = 1000,
                                       random_state: int = 42) -> Dict[str, Any]:
    if baseline_col not in X_test.columns:
        raise ValueError(f"Cannot find the baseline column '{baseline_col}' ImpossibleToConstruct")

    y_pred_best = best_estimator.predict(X_test[feats_for_best])
    y_pred_base = X_test[baseline_col].to_numpy().astype(float)

    best_point = metrics_dict(y_test, y_pred_best)
    base_point = metrics_dict(y_test, y_pred_base)

    rng = np.random.RandomState(random_state)
    n = len(y_test)
    diff_mse, diff_rmse, diff_mae, diff_r2 = [], [], [], []

    Xb = X_test[feats_for_best].reset_index(drop=True)
    y_series = pd.Series(y_test).reset_index(drop=True)
    base_series = pd.Series(y_pred_base).reset_index(drop=True)

    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yt = y_series.iloc[idx].to_numpy()
        yp_best = best_estimator.predict(Xb.iloc[idx])
        yp_base = base_series.iloc[idx].to_numpy()

        mse_best = mean_squared_error(yt, yp_best)
        mse_base = mean_squared_error(yt, yp_base)
        rmse_best = math.sqrt(mse_best)
        rmse_base = math.sqrt(mse_base)
        mae_best = mean_absolute_error(yt, yp_best)
        mae_base = mean_absolute_error(yt, yp_base)
        r2_best = r2_score(yt, yp_best)
        r2_base = r2_score(yt, yp_base)

        diff_mse.append(mse_base - mse_best)
        diff_rmse.append(rmse_base - rmse_best)
        diff_mae.append(mae_base - mae_best)
        diff_r2.append(r2_best - r2_base)

    def pct_ci(a, lo=2.5, hi=97.5):
        return float(np.percentile(a, lo)), float(np.percentile(a, hi))

    out = {
        "baseline_point_metrics": base_point,
        "best_model_point_metrics": best_point,
        "paired_improvement": {
            "mse":  {"mean": float(np.mean(diff_mse)),  "lo": pct_ci(diff_mse)[0],  "hi": pct_ci(diff_mse)[1]},
            "rmse": {"mean": float(np.mean(diff_rmse)), "lo": pct_ci(diff_rmse)[0], "hi": pct_ci(diff_rmse)[1]},
            "mae":  {"mean": float(np.mean(diff_mae)),  "lo": pct_ci(diff_mae)[0],  "hi": pct_ci(diff_mae)[1]},
            "r2":   {"mean": float(np.mean(diff_r2)),   "lo": pct_ci(diff_r2)[0],   "hi": pct_ci(diff_r2)[1]},
            "B": B
        }
    }
    return out

def paired_model_vs_model_bootstrap(model_A, feats_A: List[str],
                                    model_B, feats_B: List[str],
                                    X_test: pd.DataFrame, y_test: np.ndarray,
                                    B: int = 1000, random_state: int = 42) -> Dict[str, Any]:
    yA = model_A.predict(X_test[feats_A])
    yB = model_B.predict(X_test[feats_B])

    A_point = metrics_dict(y_test, yA)
    B_point = metrics_dict(y_test, yB)

    rng = np.random.RandomState(random_state)
    n = len(y_test)
    diff_mse, diff_rmse, diff_mae, diff_r2 = [], [], [], []

    XA = X_test[feats_A].reset_index(drop=True)
    XB = X_test[feats_B].reset_index(drop=True)
    y_series = pd.Series(y_test).reset_index(drop=True)

    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yt = y_series.iloc[idx].to_numpy()
        yA_b = model_A.predict(XA.iloc[idx])
        yB_b = model_B.predict(XB.iloc[idx])

        mse_A = mean_squared_error(yt, yA_b)
        mse_B = mean_squared_error(yt, yB_b)
        rmse_A = math.sqrt(mse_A)
        rmse_B = math.sqrt(mse_B)
        mae_A = mean_absolute_error(yt, yA_b)
        mae_B = mean_absolute_error(yt, yB_b)
        r2_A = r2_score(yt, yA_b)
        r2_B = r2_score(yt, yB_b)

        diff_mse.append(mse_B - mse_A)
        diff_rmse.append(rmse_B - rmse_A)
        diff_mae.append(mae_B - mae_A)
        diff_r2.append(r2_A - r2_B)

    def pct_ci(a, lo=2.5, hi=97.5):
        return float(np.percentile(a, lo)), float(np.percentile(a, hi))

    return {
        "A_point_metrics": A_point,
        "B_point_metrics": B_point,
        "paired_difference": {
            "mse":  {"mean": float(np.mean(diff_mse)),  "lo": pct_ci(diff_mse)[0],  "hi": pct_ci(diff_mse)[1]},
            "rmse": {"mean": float(np.mean(diff_rmse)), "lo": pct_ci(diff_rmse)[0], "hi": pct_ci(diff_rmse)[1]},
            "mae":  {"mean": float(np.mean(diff_mae)),  "lo": pct_ci(diff_mae)[0],  "hi": pct_ci(diff_mae)[1]},
            "r2":   {"mean": float(np.mean(diff_r2)),   "lo": pct_ci(diff_r2)[0],   "hi": pct_ci(diff_r2)[1]},
            "B": B
        }
    }

def safe_pearson_p(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return 1.0
    _, p = stats.pearsonr(x[mask], y[mask])
    return float(p)

def ancova_p_value(y: np.ndarray, feature: np.ndarray, covariates: np.ndarray) -> float:
    yv = y.astype(float)
    mask = ~np.isnan(yv)

    if feature.ndim == 1:
        f = feature.reshape(-1, 1)
    else:
        f = feature
    if hasattr(feature, "dtype") and getattr(feature, "dtype", None) is not None and feature.dtype.kind in ("O","U","S"):
        f = pd.get_dummies(pd.Series(feature), drop_first=True).to_numpy()

    Xc = covariates if covariates is not None and covariates.size > 0 else None
    all_mat = f.astype(float) if Xc is None else np.concatenate([f.astype(float), Xc.astype(float)], axis=1)

    full_mask = mask & ~np.isnan(all_mat).any(axis=1)
    if full_mask.sum() < (all_mat.shape[1] + 2):
        return 1.0

    X = np.column_stack([np.ones(full_mask.sum()), all_mat[full_mask]])
    yy = yv[full_mask]

    beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return 1.0
    sigma2 = resid.T @ resid / dof
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)

    k_f = f.shape[1]
    se = np.sqrt(np.diag(cov_beta))[1:1+k_f]
    tvals = beta[1:1+k_f] / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=dof))

    pvals_sorted = np.sort(pvals)
    m = len(pvals_sorted)
    simes = np.min((m * pvals_sorted) / (np.arange(1, m+1)))
    return float(min(1.0, simes))

def compute_univariate_pvals(X_df: pd.DataFrame, y: np.ndarray, covariate_cols: List[str]) -> Dict[str, float]:
    pvals: Dict[str, float] = {}
    covars = None
    if covariate_cols:
        covars = X_df[covariate_cols].to_numpy(dtype=float)

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

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor_for_model(model_key: str,
                                 feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:

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
        if model_key in NATIVE_CATEGORICAL_MODELS:
            transformers.append(("cat_native", "passthrough", unordered_cols))
        else:
            transformers.append(("cat_ohe", make_ohe(), unordered_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, ordinal_cols, unordered_cols

def get_algorithms_and_grids(random_state: int) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    algos = {
        "MLP": (
            MLPRegressor(max_iter=200, random_state=42, early_stopping=True,
                         n_iter_no_change=10, validation_fraction=0.1),
            {
                "model__hidden_layer_sizes": [(100, 50), (64, 32)],
                "model__activation": ["relu"],
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [5e-3, 1e-2]
            }
        ),
        "LGBM": (
            LGBMRegressor(random_state=42, n_estimators=400),
            {
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.05, 0.1],
                "model__min_child_samples": [10, 20]
            }
        ),
        "CB": (
            CatBoostRegressor(verbose=0, random_state=42, loss_function="RMSE"),
            {
                "model__iterations": [200, 500],
                "model__depth": [4, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__l2_leaf_reg": [3, 5],
                "model__bagging_temperature": [0, 0.5]
            }
        ),
        "SVR": (
            SVR(),
            {
                "model__C": [1, 10],
                "model__epsilon": [0.01, 0.1],
                "model__kernel": ["linear", "rbf"]
            }
        ),
        "KNN": (
            KNeighborsRegressor(),
            {
                "model__n_neighbors": [5, 10],
                "model__weights": ["uniform", "distance"]
            }
        ),
        "DT": (
            DecisionTreeRegressor(random_state=42),
            {
                "model__max_depth": [5, 10, 15],
                "model__min_samples_split": [2, 5]
            }
        ),
    }
    return algos

def consensus_from_sets(list_of_sets: List[set], min_count: int) -> List[str]:
    freq = {}
    for s in list_of_sets:
        for f in s:
            freq[f] = freq.get(f, 0) + 1
    return [f for f, c in freq.items() if c >= min_count]

def nested_cv_for_algorithm(algo_name: str,
                            base_estimator,
                            param_grid: Dict[str, List[Any]],
                            X_train_pool: pd.DataFrame,
                            y_train_pool: np.ndarray,
                            covariates: List[str],
                            rng: np.random.RandomState) -> Tuple[Dict[str, Any], List[str]]:
    outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED_OUTER)
    outer_metrics = {"mse": [], "rmse": [], "mae": [], "r2": []}
    inner_best_params_per_outer = []
    outer_consensus_features_per_fold = []
    all_feature_names = [c for c in X_train_pool.columns if c != TARGET_COL]

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
        best_params_this_outer = None

        grid_keys = list(param_grid.keys())
        grid_vals = [param_grid[k] for k in grid_keys]
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
                selected_feats = [c for c in X_tr_inner.columns if c != TARGET_COL and c not in ID_COLS]

            pre, num_cols, ord_cols, unordered_cols = build_preprocessor_for_model(algo_name, selected_feats)

            for pc in param_combos:
                model = clone(base_estimator)
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
            rec["rmse_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        combo_scores.sort(key=lambda d: d["rmse_mean"])
        best_params_this_outer = combo_scores[0]["params"] if combo_scores else {}

        outer_consensus = consensus_from_sets(selected_per_inner, INNER_CONSENSUS_MIN_COUNT)
        if not outer_consensus:
            outer_consensus = [c for c in all_feature_names if c not in ID_COLS]
        outer_consensus_features_per_fold.append(outer_consensus)
        inner_best_params_per_outer.append(best_params_this_outer)

        pre, num_cols, ord_cols, unordered_cols = build_preprocessor_for_model(algo_name, outer_consensus)
        model = clone(base_estimator)
        pipe = Pipeline([("pre", pre), ("model", model)])
        if best_params_this_outer:
            pipe.set_params(**best_params_this_outer)
        pipe.fit(X_tr_outer[outer_consensus], y_tr_outer)
        y_pred_outer = pipe.predict(X_te_outer[outer_consensus])

        met = metrics_dict(y_te_outer, y_pred_outer)
        for k in outer_metrics:
            outer_metrics[k].append(met[k])

        print(f"[{algo_name}] Outer fold {fold_idx}: RMSE={met['rmse']:.4f}, MAE={met['mae']:.4f}, R2={met['r2']:.4f}")

    summary = {"algo": algo_name, "outer_folds": OUTER_FOLDS, "metrics": {}}
    for k, vals in outer_metrics.items():
        summary["metrics"][k] = {
            "per_fold": [float(v) for v in vals],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        }
    summary["inner_best_params_per_outer"] = inner_best_params_per_outer
    summary["outer_consensus_features_per_fold"] = outer_consensus_features_per_fold

    union_features = sorted(list(set().union(*[set(s) for s in outer_consensus_features_per_fold])))
    return summary, union_features

def main():
    ensure_dir(OUT_DIR)

    lower = str(DATA_PATH).lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    elif lower.endswith(".csv"):
        try:
            df = pd.read_csv(DATA_PATH)
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATH, encoding="gb18030")
    else:
        raise ValueError("DATA_PATH must .xlsx/.xls or .csv")

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' Not in the data column. The actual columns are (partially): {list(df.columns)[:12]} ...")

    if ID_COLS:
        for c in ID_COLS:
            if c in df.columns:
                df.drop(columns=[c], inplace=True, errors="ignore")

    y = df[TARGET_COL].to_numpy(dtype=float)
    X = df.drop(columns=[TARGET_COL]).copy()

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
    )

    rng = check_random_state(RANDOM_STATE)

    algos = get_algorithms_and_grids(RANDOM_STATE)

    outer_reports = {}
    outer_feature_sets_per_algo = {}

    for name, (est, grid) in algos.items():
        print(f"\n=== Nested cross-validation: {name} ===")
        rep, _ = nested_cv_for_algorithm(
            name, est, grid, X_train_pool, y_train_pool, ANCOVA_COVARIATES, rng
        )
        outer_reports[name] = rep
        outer_feature_sets_per_algo[name] = rep["outer_consensus_features_per_fold"]

    ensure_dir(OUT_DIR)
    with open(os.path.join(OUT_DIR, "outer_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(outer_reports, f, indent=2, ensure_ascii=False)

    final_consensus_by_algo = {}
    for name, fold_sets in outer_feature_sets_per_algo.items():
        fold_sets_as_sets = [set(s) for s in fold_sets]
        final_feats = consensus_from_sets(fold_sets_as_sets, OUTER_CONSENSUS_MIN_COUNT)
        if not final_feats and len(fold_sets_as_sets) > 0:
            final_feats = sorted(list(set().union(*fold_sets_as_sets)))
        final_consensus_by_algo[name] = final_feats

    with open(os.path.join(OUT_DIR, "final_consensus_features.json"), "w", encoding="utf-8") as f:
        json.dump(final_consensus_by_algo, f, indent=2, ensure_ascii=False)

    final_cv_best_params = {}
    final_cv_scores = {}
    per_algo_test_metrics = {}
    per_algo_test_ci = {}

    scoring = make_scorer(rmse, greater_is_better=False)
    cv = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_estimators = {}

    for name, (est, grid) in algos.items():
        feats = final_consensus_by_algo[name]
        if not feats:
            feats = [c for c in X_train_pool.columns if c not in ID_COLS]

        pre, num_cols, ord_cols, unordered_cols = build_preprocessor_for_model(name, feats)

        pipe = Pipeline([("pre", pre), ("model", est)])
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=N_JOBS, refit=True)
        gs.fit(X_train_pool[feats], y_train_pool)

        final_cv_best_params[name] = gs.best_params_
        best_rmse_mean = -float(gs.best_score_)
        final_cv_scores[name] = best_rmse_mean
        best_estimators[name] = (gs.best_estimator_, feats)

        joblib.dump(gs.best_estimator_, os.path.join(OUT_DIR, f"final_refit_{name}.joblib"))

        y_pred_test = gs.best_estimator_.predict(X_test[feats])
        per_algo_test_metrics[name] = metrics_dict(y_test, y_pred_test)
        ci = bootstrap_ci_on_test(gs.best_estimator_, X_test, y_test, feats, B=1000, random_state=RANDOM_STATE)
        per_algo_test_ci[name] = ci

        print(f"[Final CV] {name}: best RMSE={final_cv_scores[name]:.4f} params={final_cv_best_params[name]}")

    with open(os.path.join(OUT_DIR, "final_cv_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(final_cv_best_params, f, indent=2, ensure_ascii=False)

    baseline_col = "Baseline BMI"
    if baseline_col in X_test.columns:
        y_pred_base = X_test[baseline_col].to_numpy().astype(float)
        baseline_point_metrics = metrics_dict(y_test, y_pred_base)
        with open(os.path.join(OUT_DIR, "baseline_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(baseline_point_metrics, f, indent=2, ensure_ascii=False)

        class _BaselinePredictor:
            def predict(self, X):
                return X[baseline_col].to_numpy().astype(float)

        baseline_ci = bootstrap_ci_on_test(_BaselinePredictor(), X_test, y_test, feats=[baseline_col],
                                           B=1000, random_state=RANDOM_STATE)
        with open(os.path.join(OUT_DIR, "baseline_test_bootstrap_ci.json"), "w", encoding="utf-8") as f:
            json.dump(baseline_ci, f, indent=2, ensure_ascii=False)
    else:
        print(f"[Warning] Column '{baseline_col}' not found in test set; cannot compute simple baseline.")

    if "CB" in best_estimators and baseline_col in X_test.columns:
        cb_estimator, cb_feats = best_estimators["CB"]
        paired_result = paired_baseline_vs_model_bootstrap(
            best_estimator=cb_estimator,
            X_test=X_test,
            y_test=y_test,
            feats_for_best=cb_feats,
            baseline_col=baseline_col,
            B=1000,
            random_state=RANDOM_STATE
        )
        with open(os.path.join(OUT_DIR, "paired_incremental_benefit_CB_vs_baseline.json"), "w", encoding="utf-8") as f:
            json.dump(paired_result, f, indent=2, ensure_ascii=False)
        print("\n[Paired incremental benefit] Written: paired_incremental_benefit_CB_vs_baseline.json")
    else:
        print("[Info] CatBoost best model not found or missing Baseline BMI; skip baseline paired analysis.")

    paired_summary_all = {}
    if "CB" in best_estimators:
        cb_estimator, cb_feats = best_estimators["CB"]
        for algo_name, (estimator, feats) in best_estimators.items():
            if algo_name == "CB":
                continue
            result_pair = paired_model_vs_model_bootstrap(
                model_A=cb_estimator, feats_A=cb_feats,
                model_B=estimator,    feats_B=feats,
                X_test=X_test, y_test=y_test,
                B=1000, random_state=RANDOM_STATE
            )
            out_name = f"paired_incremental_benefit_CB_vs_{algo_name}.json"
            with open(os.path.join(OUT_DIR, out_name), "w", encoding="utf-8") as f:
                json.dump(result_pair, f, indent=2, ensure_ascii=False)
            paired_summary_all[f"CB_vs_{algo_name}"] = result_pair
            print(f"[Paired incremental benefit] Written: {out_name}")

        with open(os.path.join(OUT_DIR, "paired_incremental_benefit_CB_vs_all.json"), "w", encoding="utf-8") as f:
            json.dump(paired_summary_all, f, indent=2, ensure_ascii=False)
        print("[Paired incremental benefit] Summary written: paired_incremental_benefit_CB_vs_all.json")
    else:
        print("[Info] CatBoost best model not found; skip 'CB vs others' paired analysis.")

    report = {
        "final_features_by_algo": {k: v for k, v in final_consensus_by_algo.items()},
        "final_cv_rmse_by_algo": final_cv_scores,
        "independent_test_metrics_by_algo": per_algo_test_metrics,
        "independent_test_bootstrap_ci_by_algo": per_algo_test_ci,
        "config": {
            "data_path": DATA_PATH,
            "sheet": SHEET_NAME,
            "target": TARGET_COL,
            "ancova_covariates": ANCOVA_COVARIATES,
            "continuous_pearson_cols": CONTINUOUS_PEARSON_COLS,
            "alpha": ALPHA,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "random_state": RANDOM_STATE,
            "native_models": list(NATIVE_CATEGORICAL_MODELS),
            "ordinal_cols": ORDINAL_COLS,
            "unordered_categorical_cols": UNORDERED_CATEGORICAL_COLS
        }
    }
    with open(os.path.join(OUT_DIR, "final_model_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Training complete (includes baseline and all-model paired analyses) ===")
    print("Output directory:", os.path.abspath(OUT_DIR))
    print("Per-algorithm test metrics (point estimates):", per_algo_test_metrics)
    print("Per-algorithm test set bootstrap 95% CI:", per_algo_test_ci)

if __name__ == "__main__":
    main()
