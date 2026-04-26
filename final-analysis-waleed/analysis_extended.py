"""
Ketosis Prediction — Extended Model Comparison
BMI/CS 776: Advanced Bioinformatics — Spring 2026
Team: Seonyeong Heo, Waleed Arshad, Ignacio Azola Ortiz

Adds to the original 4-model analysis:
  Ridge, ElasticNet, SVR (RBF), ExtraTrees
All 8 models evaluated under the same Nested 5x3 CV framework.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import (
    LinearRegression, LassoCV, Ridge, ElasticNet
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUT  = Path(__file__).parent / "figures"
RES  = Path(__file__).parent
OUT.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("tab10", 10)
MODEL_ORDER = [
    "Linear Model", "Ridge", "LASSO", "ElasticNet",
    "Extra Trees", "Random Forest", "XGBoost", "SVR (RBF)"
]
MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_ORDER)}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1.  DATA LOADING")
print("=" * 70)

df = pd.read_csv(BASE / "trajectory_cleaned2.csv")

NUMERIC_FEATS = [
    "DIM", "acetone", "urea",
    "fat", "casein", "protein", "lactose",
    "sfa", "ufa", "newly_fa", "mixed_fa", "preformed_fa",
    "c18_0", "c18_1", "scc_log",
    "ph", "freezing", "rennet", "k20", "a30", "iac",
]
TARGET = "bhba"

md = df[NUMERIC_FEATS + ["parity_group", TARGET]].dropna().copy()
parity_dummies = pd.get_dummies(md["parity_group"], prefix="parity", drop_first=True)
X_df = pd.concat([md[NUMERIC_FEATS], parity_dummies], axis=1)
feature_names = list(X_df.columns)
X = X_df.values.astype(float)
y = md[TARGET].values

print(f"N={len(md)}, features={X.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NESTED 5x3 CV — ALL 8 MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2.  NESTED 5x3 CV — ALL 8 MODELS")
print("=" * 70)

np.random.seed(42)
LASSO_ALPHAS = np.logspace(-5, 1, 60)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
# feature importance collectors
lasso_coef_list  = []
ridge_coef_list  = []
en_coef_list     = []
rf_imp_list      = []
et_imp_list      = []
xgb_shap_list    = []
xgb_lc_list      = []

for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Outer Fold {fold_i+1}/5 ---")
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=fold_i * 100)

    # Standardised versions (for linear models + SVR)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    def eval_model(pred):
        return (np.sqrt(mean_squared_error(y_te, pred)),
                r2_score(y_te, pred))

    # ── A. Linear Model ───────────────────────────────────────────────────────
    lm = LinearRegression().fit(X_tr_sc, y_tr)
    lm_rmse, lm_r2 = eval_model(lm.predict(X_te_sc))
    print(f"  LM        RMSE={lm_rmse:.4f}  R2={lm_r2:.4f}")

    # ── B. Ridge ──────────────────────────────────────────────────────────────
    ridge_gs = GridSearchCV(
        Ridge(max_iter=5000),
        {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]},
        cv=inner_cv, scoring="neg_root_mean_squared_error"
    ).fit(X_tr_sc, y_tr)
    ridge_pred = ridge_gs.best_estimator_.predict(X_te_sc)
    ridge_rmse, ridge_r2 = eval_model(ridge_pred)
    ridge_coef_list.append(ridge_gs.best_estimator_.coef_.copy())
    print(f"  Ridge     RMSE={ridge_rmse:.4f}  R2={ridge_r2:.4f}  "
          f"alpha={ridge_gs.best_params_['alpha']}")

    # ── C. LASSO ──────────────────────────────────────────────────────────────
    lasso_cv = LassoCV(alphas=LASSO_ALPHAS, cv=inner_cv,
                       max_iter=5000, random_state=42).fit(X_tr_sc, y_tr)
    lasso_rmse, lasso_r2 = eval_model(lasso_cv.predict(X_te_sc))
    lasso_coef_list.append(lasso_cv.coef_.copy())
    print(f"  LASSO     RMSE={lasso_rmse:.4f}  R2={lasso_r2:.4f}  "
          f"alpha={lasso_cv.alpha_:.5f}")

    # ── D. ElasticNet ─────────────────────────────────────────────────────────
    en_gs = GridSearchCV(
        ElasticNet(max_iter=5000, random_state=42),
        {"alpha":    [0.0001, 0.001, 0.01, 0.1],
         "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        cv=inner_cv, scoring="neg_root_mean_squared_error"
    ).fit(X_tr_sc, y_tr)
    en_pred = en_gs.best_estimator_.predict(X_te_sc)
    en_rmse, en_r2 = eval_model(en_pred)
    en_coef_list.append(en_gs.best_estimator_.coef_.copy())
    print(f"  ElasticNet RMSE={en_rmse:.4f}  R2={en_r2:.4f}  "
          f"alpha={en_gs.best_params_['alpha']}  "
          f"l1={en_gs.best_params_['l1_ratio']}")

    # ── E. Extra Trees ────────────────────────────────────────────────────────
    et_gs = GridSearchCV(
        ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        {"max_features":     [3, 5, 7, 10, 14],
         "min_samples_leaf": [3, 5, 10]},
        cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).fit(X_tr, y_tr)
    et_pred = et_gs.best_estimator_.predict(X_te)
    et_rmse, et_r2 = eval_model(et_pred)
    et_imp_list.append(et_gs.best_estimator_.feature_importances_.copy())
    print(f"  ExtraTrees RMSE={et_rmse:.4f}  R2={et_r2:.4f}  "
          f"{et_gs.best_params_}")

    # ── F. Random Forest ──────────────────────────────────────────────────────
    rf_gs = GridSearchCV(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        {"max_features":     [3, 5, 7, 10, 14],
         "min_samples_leaf": [3, 5, 10]},
        cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).fit(X_tr, y_tr)
    rf_pred = rf_gs.best_estimator_.predict(X_te)
    rf_rmse, rf_r2 = eval_model(rf_pred)
    rf_imp_list.append(rf_gs.best_estimator_.feature_importances_.copy())
    print(f"  RF        RMSE={rf_rmse:.4f}  R2={rf_r2:.4f}  "
          f"{rf_gs.best_params_}")

    # ── G. XGBoost ────────────────────────────────────────────────────────────
    XGB_GRID = [
        {"max_depth": md_, "learning_rate": lr, "min_child_weight": mcw}
        for md_ in [3, 4, 5, 6]
        for lr in [0.01, 0.05, 0.1]
        for mcw in [3, 5, 10]
    ]
    xgb_inner_rmse, xgb_best_nr = [], []
    for params_ in XGB_GRID:
        fold_r, fold_nr = [], []
        for itr_i, ite_i in inner_cv.split(X_tr):
            dtrain_i = xgb.DMatrix(X_tr[itr_i], label=y_tr[itr_i],
                                   feature_names=feature_names)
            dval_i   = xgb.DMatrix(X_tr[ite_i], label=y_tr[ite_i],
                                   feature_names=feature_names)
            er = {}
            m = xgb.train(
                {**params_, "objective": "reg:squarederror",
                 "subsample": 0.8, "colsample_bytree": 0.8,
                 "eval_metric": "rmse", "seed": 42},
                dtrain_i, num_boost_round=500,
                evals=[(dtrain_i, "train"), (dval_i, "val")],
                early_stopping_rounds=20, evals_result=er, verbose_eval=False
            )
            preds = m.predict(dval_i)
            fold_r.append(np.sqrt(mean_squared_error(y_tr[ite_i], preds)))
            fold_nr.append(m.best_iteration + 1)
        xgb_inner_rmse.append(np.mean(fold_r))
        xgb_best_nr.append(int(np.mean(fold_nr)))
    best_g = int(np.argmin(xgb_inner_rmse))
    best_xp = XGB_GRID[best_g]
    best_nr = xgb_best_nr[best_g]

    dtrain_out = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dtest_out  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_names)
    er_out = {}
    xgb_fit = xgb.train(
        {**best_xp, "objective": "reg:squarederror",
         "subsample": 0.8, "colsample_bytree": 0.8,
         "eval_metric": "rmse", "seed": 42},
        dtrain_out, num_boost_round=best_nr,
        evals=[(dtrain_out, "train"), (dtest_out, "val")],
        evals_result=er_out, verbose_eval=False
    )
    xgb_pred = xgb_fit.predict(dtest_out)
    xgb_rmse, xgb_r2 = eval_model(xgb_pred)
    xgb_lc_list.append({"train": er_out["train"]["rmse"],
                         "val":   er_out["val"]["rmse"]})
    exp = shap.TreeExplainer(xgb_fit)
    xgb_shap_list.append(np.abs(exp.shap_values(X_te)).mean(axis=0))
    print(f"  XGBoost   RMSE={xgb_rmse:.4f}  R2={xgb_r2:.4f}  "
          f"{best_xp}  nr={best_nr}")

    # ── H. SVR (RBF) ──────────────────────────────────────────────────────────
    svr_gs = GridSearchCV(
        SVR(kernel="rbf"),
        {"C":       [0.1, 1, 10, 100],
         "epsilon": [0.005, 0.01, 0.05, 0.1],
         "gamma":   ["scale"]},
        cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).fit(X_tr_sc, y_tr)
    svr_pred = svr_gs.best_estimator_.predict(X_te_sc)
    svr_rmse, svr_r2 = eval_model(svr_pred)
    print(f"  SVR(RBF)  RMSE={svr_rmse:.4f}  R2={svr_r2:.4f}  "
          f"C={svr_gs.best_params_['C']}  "
          f"eps={svr_gs.best_params_['epsilon']}")

    fold_results.append({
        "fold":       fold_i + 1,
        "lm_rmse":    lm_rmse,    "lm_r2":    lm_r2,
        "ridge_rmse": ridge_rmse, "ridge_r2": ridge_r2,
        "lasso_rmse": lasso_rmse, "lasso_r2": lasso_r2,
        "en_rmse":    en_rmse,    "en_r2":    en_r2,
        "et_rmse":    et_rmse,    "et_r2":    et_r2,
        "rf_rmse":    rf_rmse,    "rf_r2":    rf_r2,
        "xgb_rmse":   xgb_rmse,  "xgb_r2":   xgb_r2,
        "svr_rmse":   svr_rmse,  "svr_r2":   svr_r2,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.  RESULTS SUMMARY")
print("=" * 70)

rdf = pd.DataFrame(fold_results)
keys = ["lm","ridge","lasso","en","et","rf","xgb","svr"]
labels = MODEL_ORDER

summary = pd.DataFrame({
    "Model":     labels,
    "Mean_RMSE": [rdf[f"{k}_rmse"].mean() for k in keys],
    "SD_RMSE":   [rdf[f"{k}_rmse"].std()  for k in keys],
    "Mean_R2":   [rdf[f"{k}_r2"].mean()   for k in keys],
    "SD_R2":     [rdf[f"{k}_r2"].std()    for k in keys],
}).round(4).sort_values("Mean_RMSE").reset_index(drop=True)

print("\n=== Nested 5×3 CV — All 8 Models ===")
print(summary.to_string(index=False))

rdf.to_csv(RES / "cv_fold_results_extended.csv", index=False)
summary.to_csv(RES / "cv_summary_extended.csv",  index=False)
print("\nSaved: cv_fold_results_extended.csv, cv_summary_extended.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  UPDATED FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4.  FIGURES")
print("=" * 70)

bar_colors = [MODEL_COLORS[m] for m in summary["Model"]]

# ── 4a. RMSE comparison (all 8 models, sorted) ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(summary["Model"], summary["Mean_RMSE"],
              yerr=summary["SD_RMSE"], capsize=5,
              color=bar_colors, edgecolor="white", width=0.6, alpha=0.88,
              error_kw=dict(elinewidth=1.5, ecolor="black"))
for bar, val in zip(bars, summary["Mean_RMSE"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + summary["SD_RMSE"].max() * 0.08,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylabel("RMSE (mmol/L)")
ax.set_title("Nested 5×3 CV — Mean RMSE, All Models (sorted)",
             fontsize=12, fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT / "ext_01_rmse_all_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_01_rmse_all_models.png")

# ── 4b. R² comparison (all 8 models) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(summary["Model"], summary["Mean_R2"],
              yerr=summary["SD_R2"], capsize=5,
              color=bar_colors, edgecolor="white", width=0.6, alpha=0.88,
              error_kw=dict(elinewidth=1.5, ecolor="black"))
for bar, val in zip(bars, summary["Mean_R2"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            max(0, bar.get_height()) + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("R²")
ax.set_title("Nested 5×3 CV — Mean R², All Models (sorted by RMSE)",
             fontsize=12, fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT / "ext_02_r2_all_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_02_r2_all_models.png")

# ── 4c. Per-fold RMSE trace (all 8 models) ────────────────────────────────────
long_rows = []
for r in fold_results:
    for k, lbl in zip(keys, labels):
        long_rows.append({"Fold": r["fold"], "Model": lbl, "RMSE": r[f"{k}_rmse"]})
long_df = pd.DataFrame(long_rows)

# Split into two groups for readability
linear_models = ["Linear Model", "Ridge", "LASSO", "ElasticNet"]
tree_models   = ["Extra Trees", "Random Forest", "XGBoost", "SVR (RBF)"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, group, title in zip(axes,
                             [linear_models, tree_models],
                             ["Linear / Regularised Models",
                              "Ensemble / Kernel Models"]):
    for m in group:
        sub = long_df[long_df["Model"] == m]
        ax.plot(sub["Fold"], sub["RMSE"], marker="o", lw=2, ms=6,
                label=m, color=MODEL_COLORS[m])
    ax.set_xlabel("Outer Fold")
    ax.set_ylabel("RMSE (mmol/L)")
    ax.set_xticks(range(1, 6))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
fig.suptitle("Per-Fold RMSE Trace — All Models", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "ext_03_rmse_per_fold_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_03_rmse_per_fold_all.png")

# ── 4d. RMSE vs R² scatter (model comparison bubble) ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for _, row in summary.iterrows():
    ax.scatter(row["Mean_RMSE"], row["Mean_R2"],
               color=MODEL_COLORS[row["Model"]], s=120, zorder=3)
    ax.annotate(row["Model"],
                xy=(row["Mean_RMSE"], row["Mean_R2"]),
                xytext=(4, 3), textcoords="offset points", fontsize=8.5)
ax.set_xlabel("Mean RMSE (mmol/L)  ← better")
ax.set_ylabel("Mean R²  better →")
ax.set_title("RMSE vs R² — Model Comparison", fontsize=12, fontweight="bold")
ax.invert_xaxis()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / "ext_04_rmse_vs_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_04_rmse_vs_r2.png")

# ── 4e. XGBoost learning curve (unchanged from original) ─────────────────────
min_len = min(len(lc["train"]) for lc in xgb_lc_list)
train_m = np.array([lc["train"][:min_len] for lc in xgb_lc_list]).mean(0)
val_m   = np.array([lc["val"][:min_len]   for lc in xgb_lc_list]).mean(0)
train_s = np.array([lc["train"][:min_len] for lc in xgb_lc_list]).std(0)
val_s   = np.array([lc["val"][:min_len]   for lc in xgb_lc_list]).std(0)
rounds  = np.arange(1, min_len + 1)
best_r  = int(val_m.argmin()) + 1

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds, train_m, color=PALETTE[6], lw=2, label="Train RMSE")
ax.fill_between(rounds, train_m - train_s, train_m + train_s,
                color=PALETTE[6], alpha=0.15)
ax.plot(rounds, val_m, color=PALETTE[0], lw=2, ls="--", label="Validation RMSE")
ax.fill_between(rounds, val_m - val_s, val_m + val_s,
                color=PALETTE[0], alpha=0.15)
ax.axvline(best_r, color="grey", ls=":", lw=1.5, label=f"Best round ≈ {best_r}")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("RMSE (mmol/L)")
ax.set_title("XGBoost Learning Curve — Mean ± SD across 5 Outer Folds",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "ext_05_xgb_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_05_xgb_learning_curve.png")

# ── 4f. Coefficient magnitude — linear models ────────────────────────────────
coef_data = {}
for name, coef_list in [("LM",        [None]),   # LM not collected
                         ("Ridge",     ridge_coef_list),
                         ("LASSO",     lasso_coef_list),
                         ("ElasticNet",en_coef_list)]:
    if coef_list[0] is not None:
        mat = np.vstack(coef_list)
        coef_data[name] = np.abs(mat).mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (name, vals) in zip(axes, coef_data.items()):
    top_idx = np.argsort(vals)[-12:]
    ax.barh([feature_names[i] for i in top_idx], vals[top_idx],
            color=MODEL_COLORS.get(name, PALETTE[0]), alpha=0.85)
    ax.set_title(f"{name} — Mean |Coef|", fontweight="bold")
    ax.set_xlabel("Mean |Coefficient|")
plt.suptitle("Linear Model Coefficient Magnitudes (Top 12)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "ext_06_linear_coef_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_06_linear_coef_comparison.png")

# ── 4g. Tree ensemble importance — RF vs ExtraTrees ──────────────────────────
rf_avg = np.vstack(rf_imp_list).mean(0)
et_avg = np.vstack(et_imp_list).mean(0)

top15_idx = np.argsort(rf_avg)[-15:]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (avg, name, color) in zip(axes,
    [(rf_avg, "Random Forest", MODEL_COLORS["Random Forest"]),
     (et_avg, "Extra Trees",   MODEL_COLORS["Extra Trees"])]):
    sorted_idx = top15_idx[np.argsort(avg[top15_idx])]
    ax.barh([feature_names[i] for i in sorted_idx], avg[sorted_idx],
            color=color, alpha=0.85)
    ax.set_title(f"{name} — MDI Importance", fontweight="bold")
    ax.set_xlabel("Mean Decrease Impurity")
plt.suptitle("Tree Ensemble Feature Importance (Top 15, RF order)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "ext_07_tree_importance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_07_tree_importance_comparison.png")

# ── 4h. XGBoost SHAP (updated from new CV run) ───────────────────────────────
shap_avg = np.vstack(xgb_shap_list).mean(0)
shap_df  = pd.DataFrame({"Feature": feature_names, "SHAP": shap_avg})\
             .sort_values("SHAP").tail(15)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(shap_df["Feature"], shap_df["SHAP"],
        color=MODEL_COLORS["XGBoost"], alpha=0.85)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("XGBoost SHAP Feature Importance (Top 15)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "ext_08_xgb_shap_updated.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ext_08_xgb_shap_updated.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FINAL PRINT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL RESULTS — Nested 5×3 CV, All 8 Models (ranked by RMSE)")
print("=" * 70)
print(summary.to_string(index=False))

print("\nTop 5 SHAP features (XGBoost):")
print(pd.DataFrame({"Feature": feature_names, "SHAP": shap_avg})
      .sort_values("SHAP", ascending=False).head(5).to_string(index=False))

print(f"\nAll outputs saved to: {OUT}")
print("=== Done! ===")
