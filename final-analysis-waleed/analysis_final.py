"""
Ketosis Prediction from Milk Biomarkers
BMI/CS 776: Advanced Bioinformatics — Spring 2026
Team: Seonyeong Heo, Waleed Arshad, Ignacio Azola Ortiz

Target  : Blood BHBA (beta-hydroxybutyrate) — ketosis indicator
Data    : trajectory_cleaned2.csv  (N=1612, one observation per cow)
Models  : Linear Model, LASSO, Random Forest, XGBoost
CV      : Nested 5-fold outer / 3-fold inner grid search
Metrics : RMSE, R^2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import shap

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
PALETTE = sns.color_palette("Set2", 6)
MODEL_COLORS = {
    "Linear Model": PALETTE[0],
    "LASSO":        PALETTE[1],
    "Random Forest":PALETTE[2],
    "XGBoost":      PALETTE[3],
}

BASE  = Path(__file__).parent.parent          # CS776-Project/
OUT   = Path(__file__).parent / "figures"     # project_progress/figures/
RES   = Path(__file__).parent                 # project_progress/
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1.  DATA LOADING & PREPROCESSING")
print("=" * 70)

df = pd.read_csv(BASE / "trajectory_cleaned2.csv")
print(f"Dataset shape: {df.shape}")

NUMERIC_FEATS = [
    "DIM", "acetone", "urea",
    "fat", "casein", "protein", "lactose",
    "sfa", "ufa", "newly_fa", "mixed_fa", "preformed_fa",
    "c18_0", "c18_1", "scc_log",
    "ph", "freezing", "rennet", "k20", "a30", "iac",
]
CAT_FEATS  = ["parity_group"]
TARGET     = "bhba"

md = df[NUMERIC_FEATS + CAT_FEATS + [TARGET, "lactation_stage"]].dropna().copy()
print(f"Complete modelling rows: {len(md)}")
print(f"\nBHBA summary:\n{md[TARGET].describe().round(4)}")

# One-hot encode parity_group (drop first → mature_multiparous is reference)
parity_dummies = pd.get_dummies(md["parity_group"], prefix="parity", drop_first=True)
X_df = pd.concat([md[NUMERIC_FEATS], parity_dummies], axis=1)
feature_names = list(X_df.columns)
X = X_df.values.astype(float)
y = md[TARGET].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature names: {feature_names}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2.  EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ── 2a. BHBA distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(md[TARGET], bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].axvline(md[TARGET].quantile(0.75), color="crimson", ls="--", lw=1.5,
                label=f"Q3 = {md[TARGET].quantile(0.75):.3f}")
axes[0].axvline(md[TARGET].quantile(0.90), color="darkorange", ls=":", lw=1.5,
                label=f"Q90 = {md[TARGET].quantile(0.90):.3f}")
axes[0].set_xlabel("Blood BHBA (mmol/L)")
axes[0].set_ylabel("Count")
axes[0].set_title("BHBA Distribution")
axes[0].legend(fontsize=9)

axes[1].hist(np.log1p(md[TARGET]), bins=50, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[1].set_xlabel("log(1 + BHBA)")
axes[1].set_ylabel("Count")
axes[1].set_title("log(1+BHBA) Distribution")

plt.tight_layout()
plt.savefig(OUT / "eda_01_bhba_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_01_bhba_distribution.png")

# ── 2b. Feature correlations with BHBA ───────────────────────────────────────
corr_with_bhba = (
    md[NUMERIC_FEATS + [TARGET]]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [PALETTE[3] if v > 0 else PALETTE[0] for v in corr_with_bhba.values]
bars = ax.barh(corr_with_bhba.index, corr_with_bhba.values, color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson Correlation with Blood BHBA")
ax.set_title("Feature Correlations with BHBA")
pos_patch = mpatches.Patch(color=PALETTE[3], label="Positive")
neg_patch = mpatches.Patch(color=PALETTE[0], label="Negative")
ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "eda_02_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_02_feature_correlations.png")

# ── 2c. BHBA by parity group ──────────────────────────────────────────────────
parity_order = ["primiparous", "second_parity", "mature_multiparous"]
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=md, x="parity_group", y=TARGET, order=parity_order,
            palette="Set2", width=0.5, ax=ax, flierprops=dict(ms=3, alpha=0.4))
sns.stripplot(data=md, x="parity_group", y=TARGET, order=parity_order,
              color="grey", alpha=0.25, size=2.5, jitter=True, ax=ax)
ax.set_xlabel("Parity Group")
ax.set_ylabel("Blood BHBA (mmol/L)")
ax.set_title("BHBA by Parity Group")
plt.tight_layout()
plt.savefig(OUT / "eda_03_bhba_by_parity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_03_bhba_by_parity.png")

# ── 2d. BHBA by lactation stage ───────────────────────────────────────────────
stage_order = ["early", "mid", "late"]
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=md, x="lactation_stage", y=TARGET, order=stage_order,
            palette="Set2", width=0.5, ax=ax, flierprops=dict(ms=3, alpha=0.4))
sns.stripplot(data=md, x="lactation_stage", y=TARGET, order=stage_order,
              color="grey", alpha=0.25, size=2.5, jitter=True, ax=ax)
ax.set_xlabel("Lactation Stage")
ax.set_ylabel("Blood BHBA (mmol/L)")
ax.set_title("BHBA by Lactation Stage")
plt.tight_layout()
plt.savefig(OUT / "eda_04_bhba_by_stage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_04_bhba_by_stage.png")

# ── 2e. Acetone vs BHBA ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(md["acetone"], md[TARGET],
                c=md["DIM"], cmap="viridis", alpha=0.45, s=18, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Days in Milk (DIM)")
ax.set_xlabel("Milk Acetone (mmol/L)")
ax.set_ylabel("Blood BHBA (mmol/L)")
ax.set_title("Acetone vs BHBA — colored by DIM")
plt.tight_layout()
plt.savefig(OUT / "eda_05_acetone_bhba.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_05_acetone_bhba.png")

# ── 2f. Fatty acid profile boxplots ──────────────────────────────────────────
fa_cols = ["sfa", "ufa", "newly_fa", "mixed_fa", "preformed_fa", "c18_0", "c18_1"]
fig, axes = plt.subplots(1, len(fa_cols), figsize=(16, 4), sharey=False)
for i, col in enumerate(fa_cols):
    axes[i].boxplot(md[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7),
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(ms=2, alpha=0.3))
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_xticklabels([])
axes[0].set_ylabel("g / 100 g milk")
fig.suptitle("Fatty Acid Profile Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "eda_06_fa_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_06_fa_profile.png")

# ── 2g. Correlation heatmap (key numeric features) ────────────────────────────
key_feats = ["bhba", "acetone", "urea", "scc_log", "fat", "protein",
             "sfa", "ufa", "newly_fa", "preformed_fa", "DIM"]
corr_mat = md[key_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.4,
            annot_kws={"size": 7.5}, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Pairwise Pearson Correlations — Key Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "eda_07_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_07_correlation_matrix.png")

# ── EDA summary stats ─────────────────────────────────────────────────────────
print("\nDescriptive statistics (key features):")
print(md[["bhba", "acetone", "urea", "scc_log", "fat", "protein",
          "sfa", "ufa", "DIM"]].describe().round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  NESTED 5x3 CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.  NESTED 5x3 CROSS-VALIDATION")
print("=" * 70)

np.random.seed(42)

# Hyperparameter grids
LASSO_ALPHAS = np.logspace(-5, 1, 60)   # 60 values in [1e-5, 10]

RF_GRID = [
    {"max_features": mf, "min_samples_leaf": ml}
    for mf in [3, 5, 7, 10, 14]
    for ml in [3, 5, 10]
]

XGB_GRID = [
    {"max_depth": md_, "learning_rate": lr, "min_child_weight": mcw}
    for md_ in [3, 4, 5, 6]
    for lr in [0.01, 0.05, 0.1]
    for mcw in [3, 5, 10]
]

print(f"Grid sizes — LASSO: {len(LASSO_ALPHAS)} | RF: {len(RF_GRID)} | XGBoost: {len(XGB_GRID)}")

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
lasso_coef_list = []
rf_imp_list     = []
xgb_shap_list   = []
xgb_lc_list     = []   # learning curves per outer fold

for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Outer Fold {fold_i+1}/5 ---")
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=fold_i * 100)

    # ── A.  Linear Model (baseline, no tuning) ────────────────────────────────
    lm = LinearRegression()
    lm.fit(X_tr, y_tr)
    lm_pred = lm.predict(X_te)
    lm_rmse = np.sqrt(mean_squared_error(y_te, lm_pred))
    lm_r2   = r2_score(y_te, lm_pred)
    print(f"  LM done.  RMSE={lm_rmse:.4f}  R2={lm_r2:.4f}")

    # ── B.  LASSO — standardize on outer-train, LassoCV on inner ─────────────
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    lasso_cv = LassoCV(alphas=LASSO_ALPHAS, cv=inner_cv, max_iter=5000, random_state=42)
    lasso_cv.fit(X_tr_sc, y_tr)
    lasso_pred  = lasso_cv.predict(X_te_sc)
    lasso_rmse  = np.sqrt(mean_squared_error(y_te, lasso_pred))
    lasso_r2    = r2_score(y_te, lasso_pred)
    lasso_coef_list.append(lasso_cv.coef_.copy())
    print(f"  LASSO done. Best alpha={lasso_cv.alpha_:.5f}  RMSE={lasso_rmse:.4f}  R2={lasso_r2:.4f}")

    # ── C.  Random Forest — GridSearchCV with inner 3-fold ────────────────────
    rf_gs = GridSearchCV(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        param_grid={
            "max_features":       [3, 5, 7, 10, 14],
            "min_samples_leaf":   [3, 5, 10],
        },
        cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rf_gs.fit(X_tr, y_tr)
    rf_pred  = rf_gs.best_estimator_.predict(X_te)
    rf_rmse  = np.sqrt(mean_squared_error(y_te, rf_pred))
    rf_r2    = r2_score(y_te, rf_pred)
    rf_final = rf_gs.best_estimator_
    rf_imp_list.append(rf_final.feature_importances_.copy())
    print(f"  RF done.  Best {rf_gs.best_params_}  RMSE={rf_rmse:.4f}  R2={rf_r2:.4f}")

    # ── D.  XGBoost — manual inner CV for (max_depth, lr, mcw) + nrounds ─────
    xgb_inner_rmse = []
    xgb_best_nrounds_per_combo = []

    for params_ in XGB_GRID:
        combo_rmses = []
        combo_nrounds = []
        for itr_idx, ite_idx in inner_cv.split(X_tr):
            X_itr, X_ite = X_tr[itr_idx], X_tr[ite_idx]
            y_itr, y_ite = y_tr[itr_idx], y_tr[ite_idx]
            dtrain_in = xgb.DMatrix(X_itr, label=y_itr, feature_names=feature_names)
            dval_in   = xgb.DMatrix(X_ite, label=y_ite, feature_names=feature_names)
            er = {}
            m = xgb.train(
                {**params_,
                 "objective": "reg:squarederror",
                 "subsample": 0.8, "colsample_bytree": 0.8,
                 "eval_metric": "rmse", "seed": 42},
                dtrain_in, num_boost_round=500,
                evals=[(dtrain_in, "train"), (dval_in, "val")],
                early_stopping_rounds=20,
                evals_result=er, verbose_eval=False
            )
            best_nr = m.best_iteration + 1
            combo_nrounds.append(best_nr)
            pred_ = m.predict(dval_in)
            combo_rmses.append(np.sqrt(mean_squared_error(y_ite, pred_)))
        xgb_inner_rmse.append(np.mean(combo_rmses))
        xgb_best_nrounds_per_combo.append(int(np.mean(combo_nrounds)))

    best_g   = int(np.argmin(xgb_inner_rmse))
    best_par = XGB_GRID[best_g]
    best_nr  = xgb_best_nrounds_per_combo[best_g]

    # Refit on full outer-train with best params + learning curve captured
    dtrain_out = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dtest_out  = xgb.DMatrix(X_te, label=y_te, feature_names=feature_names)
    er_out = {}
    xgb_final_fold = xgb.train(
        {**best_par,
         "objective": "reg:squarederror",
         "subsample": 0.8, "colsample_bytree": 0.8,
         "eval_metric": "rmse", "seed": 42},
        dtrain_out, num_boost_round=best_nr,
        evals=[(dtrain_out, "train"), (dtest_out, "val")],
        evals_result=er_out, verbose_eval=False
    )
    xgb_pred = xgb_final_fold.predict(dtest_out)
    xgb_rmse = np.sqrt(mean_squared_error(y_te, xgb_pred))
    xgb_r2   = r2_score(y_te, xgb_pred)

    # Store learning curve
    xgb_lc_list.append({
        "train": er_out["train"]["rmse"],
        "val":   er_out["val"]["rmse"]
    })

    # SHAP values for this fold
    explainer_fold = shap.TreeExplainer(xgb_final_fold)
    shap_vals_fold = explainer_fold.shap_values(X_te)
    xgb_shap_list.append(np.abs(shap_vals_fold).mean(axis=0))

    print(f"  XGB done.  Best {best_par}  nrounds={best_nr}  RMSE={xgb_rmse:.4f}  R2={xgb_r2:.4f}")

    fold_results.append({
        "fold":       fold_i + 1,
        "lm_rmse":    lm_rmse,    "lm_r2":    lm_r2,
        "lasso_rmse": lasso_rmse, "lasso_r2": lasso_r2,
        "rf_rmse":    rf_rmse,    "rf_r2":    rf_r2,
        "xgb_rmse":   xgb_rmse,  "xgb_r2":   xgb_r2,
    })

# ── 3a. Aggregate results ─────────────────────────────────────────────────────
results_df = pd.DataFrame(fold_results)
summary = pd.DataFrame({
    "Model":     ["Linear Model", "LASSO", "Random Forest", "XGBoost"],
    "Mean_RMSE": [results_df["lm_rmse"].mean(),    results_df["lasso_rmse"].mean(),
                  results_df["rf_rmse"].mean(),     results_df["xgb_rmse"].mean()],
    "SD_RMSE":   [results_df["lm_rmse"].std(),     results_df["lasso_rmse"].std(),
                  results_df["rf_rmse"].std(),      results_df["xgb_rmse"].std()],
    "Mean_R2":   [results_df["lm_r2"].mean(),      results_df["lasso_r2"].mean(),
                  results_df["rf_r2"].mean(),       results_df["xgb_r2"].mean()],
    "SD_R2":     [results_df["lm_r2"].std(),       results_df["lasso_r2"].std(),
                  results_df["rf_r2"].std(),        results_df["xgb_r2"].std()],
})
summary = summary.round(4)

print("\n=== Nested 5x3 CV Performance Summary ===")
print(summary.to_string(index=False))

results_df.to_csv(RES / "cv_fold_results.csv", index=False)
summary.to_csv(RES / "cv_summary.csv", index=False)
print("\nSaved: cv_fold_results.csv, cv_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ML FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4.  ML RESULT FIGURES")
print("=" * 70)

# ── 4a. RMSE comparison bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [MODEL_COLORS[m] for m in summary["Model"]]
bars = ax.bar(summary["Model"], summary["Mean_RMSE"],
              yerr=summary["SD_RMSE"], capsize=5,
              color=bar_colors, edgecolor="white", width=0.55, alpha=0.88,
              error_kw=dict(elinewidth=1.5, ecolor="black"))
for bar, val in zip(bars, summary["Mean_RMSE"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + summary["SD_RMSE"].max() * 0.1,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("RMSE (mmol/L)")
ax.set_title("Nested 5×3 CV — Mean RMSE by Model",
             fontsize=12, fontweight="bold")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(OUT / "ml_01_rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ml_01_rmse_comparison.png")

# ── 4b. Per-fold RMSE trace ───────────────────────────────────────────────────
long_rmse = []
for r in fold_results:
    for m_key, m_label in [("lm", "Linear Model"), ("lasso", "LASSO"),
                            ("rf", "Random Forest"), ("xgb", "XGBoost")]:
        long_rmse.append({"Fold": r["fold"], "Model": m_label, "RMSE": r[f"{m_key}_rmse"]})
long_df = pd.DataFrame(long_rmse)

fig, ax = plt.subplots(figsize=(8, 5))
for model_name, grp in long_df.groupby("Model"):
    ax.plot(grp["Fold"], grp["RMSE"],
            marker="o", linewidth=2, markersize=6,
            label=model_name, color=MODEL_COLORS[model_name])
ax.set_xlabel("Outer Fold")
ax.set_ylabel("RMSE (mmol/L)")
ax.set_xticks(range(1, 6))
ax.set_title("RMSE per Outer Fold — All Models", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.8)
plt.tight_layout()
plt.savefig(OUT / "ml_02_rmse_per_fold.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ml_02_rmse_per_fold.png")

# ── 4c. XGBoost learning curve (mean ± SD across outer folds) ────────────────
min_len = min(len(lc["train"]) for lc in xgb_lc_list)
train_mat = np.array([lc["train"][:min_len] for lc in xgb_lc_list])
val_mat   = np.array([lc["val"][:min_len]   for lc in xgb_lc_list])
rounds    = np.arange(1, min_len + 1)

train_mean, train_sd = train_mat.mean(0), train_mat.std(0)
val_mean,   val_sd   = val_mat.mean(0),   val_mat.std(0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds, train_mean, color=PALETTE[3], lw=2, label="Train RMSE")
ax.fill_between(rounds, train_mean - train_sd, train_mean + train_sd,
                color=PALETTE[3], alpha=0.15)
ax.plot(rounds, val_mean, color=PALETTE[0], lw=2, ls="--", label="Validation RMSE")
ax.fill_between(rounds, val_mean - val_sd, val_mean + val_sd,
                color=PALETTE[0], alpha=0.15)
best_round = int(val_mean.argmin()) + 1
ax.axvline(best_round, color="grey", ls=":", lw=1.5,
           label=f"Best round ≈ {best_round}")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("RMSE (mmol/L)")
ax.set_title("XGBoost Learning Curve — Mean ± SD across 5 Outer Folds",
             fontsize=12, fontweight="bold")
ax.legend(framealpha=0.8)
plt.tight_layout()
plt.savefig(OUT / "ml_03_xgb_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ml_03_xgb_learning_curve.png")

# ── 4d. R² comparison bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(summary["Model"], summary["Mean_R2"],
              yerr=summary["SD_R2"], capsize=5,
              color=bar_colors, edgecolor="white", width=0.55, alpha=0.88,
              error_kw=dict(elinewidth=1.5, ecolor="black"))
for bar, val in zip(bars, summary["Mean_R2"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            max(0, bar.get_height()) + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("R²")
ax.set_title("Nested 5×3 CV — Mean R² by Model", fontsize=12, fontweight="bold")
ax.axhline(0, color="black", lw=0.8, ls="--")
plt.tight_layout()
plt.savefig(OUT / "ml_04_r2_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ml_04_r2_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5.  FEATURE IMPORTANCE")
print("=" * 70)

# ── 5a. LASSO — mean |coefficient| across outer folds ────────────────────────
lasso_coef_mat = np.vstack(lasso_coef_list)  # (5, n_features)
lasso_avg_abs  = np.abs(lasso_coef_mat).mean(axis=0)
lasso_imp_df   = (
    pd.DataFrame({"Feature": feature_names, "Importance": lasso_avg_abs})
    .query("Importance > 0")
    .sort_values("Importance", ascending=True)
)
print(f"LASSO non-zero features: {len(lasso_imp_df)}")

fig, ax = plt.subplots(figsize=(8, max(4, len(lasso_imp_df) * 0.33)))
ax.barh(lasso_imp_df["Feature"], lasso_imp_df["Importance"],
        color=PALETTE[1], edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean |Coefficient| (standardised scale)")
ax.set_title("LASSO Feature Importance\n(Mean |coef| across 5 outer folds, non-zero only)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fi_01_lasso_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fi_01_lasso_importance.png")

# ── 5b. Random Forest — mean feature importances ─────────────────────────────
rf_imp_mat = np.vstack(rf_imp_list)  # (5, n_features)
rf_avg_imp = rf_imp_mat.mean(axis=0)
rf_imp_df  = (
    pd.DataFrame({"Feature": feature_names, "Importance": rf_avg_imp})
    .sort_values("Importance", ascending=True)
    .tail(15)
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(rf_imp_df["Feature"], rf_imp_df["Importance"],
        color=PALETTE[2], edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean Impurity Decrease (MDI)")
ax.set_title("Random Forest Feature Importance\n(Mean MDI across 5 outer folds, Top 15)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fi_02_rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fi_02_rf_importance.png")

# ── 5c. XGBoost — mean SHAP across outer folds ───────────────────────────────
shap_mat = np.vstack(xgb_shap_list)  # (5, n_features)
shap_avg = shap_mat.mean(axis=0)
shap_df  = (
    pd.DataFrame({"Feature": feature_names, "SHAP": shap_avg})
    .sort_values("SHAP", ascending=True)
    .tail(15)
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(shap_df["Feature"], shap_df["SHAP"],
        color=PALETTE[3], edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("XGBoost SHAP Feature Importance\n(Mean |SHAP| across 5 outer folds, Top 15)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fi_03_xgb_shap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fi_03_xgb_shap.png")

# ── 5d. Combined top-10 feature comparison ───────────────────────────────────
top10 = set(lasso_imp_df["Feature"].tail(10)) | \
        set(rf_imp_df["Feature"].tail(10)) | \
        set(shap_df["Feature"].tail(10))
top10 = sorted(top10)

lasso_vals = dict(zip(feature_names, lasso_avg_abs / (lasso_avg_abs.max() + 1e-10)))
rf_vals    = dict(zip(feature_names, rf_avg_imp   / (rf_avg_imp.max()    + 1e-10)))
shap_vals  = dict(zip(feature_names, shap_avg     / (shap_avg.max()      + 1e-10)))

combined = pd.DataFrame({
    "Feature":       top10,
    "LASSO":         [lasso_vals.get(f, 0) for f in top10],
    "Random Forest": [rf_vals.get(f, 0) for f in top10],
    "XGBoost SHAP":  [shap_vals.get(f, 0) for f in top10],
}).set_index("Feature").sort_values("XGBoost SHAP", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(combined))
w = 0.25
ax.barh(x_pos - w,   combined["LASSO"],         w, label="LASSO",        color=PALETTE[1], alpha=0.85)
ax.barh(x_pos,       combined["Random Forest"], w, label="Random Forest", color=PALETTE[2], alpha=0.85)
ax.barh(x_pos + w,   combined["XGBoost SHAP"],  w, label="XGBoost SHAP", color=PALETTE[3], alpha=0.85)
ax.set_yticks(x_pos)
ax.set_yticklabels(combined.index)
ax.set_xlabel("Normalised Importance (model-specific max = 1)")
ax.set_title("Feature Importance Comparison Across Models\n(Normalised, Top Features)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "fi_04_combined_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fi_04_combined_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FINAL MODEL REFIT + SHAP BEESWARM (XGBoost on full data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6.  FINAL REFIT + SHAP BEESWARM")
print("=" * 70)

# Final XGBoost: use median best hyperparams from CV
# Identify the best overall params from CV (the combo that most often won)
final_xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "seed": 42,
}

dtrain_all = xgb.DMatrix(X, label=y, feature_names=feature_names)

# Find best nrounds via 3-fold CV on full data
cv_res = xgb.cv(
    final_xgb_params, dtrain_all,
    num_boost_round=1000, nfold=3,
    early_stopping_rounds=30,
    metrics="rmse", verbose_eval=False
)
final_nrounds = int(cv_res["test-rmse-mean"].idxmin()) + 1
print(f"Final XGBoost nrounds: {final_nrounds}")

xgb_all = xgb.train(
    final_xgb_params, dtrain_all,
    num_boost_round=final_nrounds,
    verbose_eval=False
)

# SHAP beeswarm plot
explainer_all = shap.TreeExplainer(xgb_all)
shap_values_all = explainer_all.shap_values(X)

fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_values_all, X, feature_names=feature_names,
                  max_display=15, show=False, plot_size=None)
plt.title("XGBoost SHAP Summary (Beeswarm) — Full Dataset, Top 15 Features",
          fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(OUT / "fi_05_xgb_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fi_05_xgb_shap_beeswarm.png")

# ── Save feature importance CSVs ─────────────────────────────────────────────
lasso_imp_df.to_csv(RES / "lasso_importance.csv", index=False)
pd.DataFrame({"Feature": feature_names, "MDI": rf_avg_imp})\
  .sort_values("MDI", ascending=False).to_csv(RES / "rf_importance.csv", index=False)
pd.DataFrame({"Feature": feature_names, "SHAP": shap_avg})\
  .sort_values("SHAP", ascending=False).to_csv(RES / "xgb_shap.csv", index=False)

print("\nSaved: lasso_importance.csv, rf_importance.csv, xgb_shap.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL SUMMARY PRINT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print("\nTop 5 features by XGBoost SHAP:")
print(pd.DataFrame({"Feature": feature_names, "SHAP": shap_avg})
      .sort_values("SHAP", ascending=False).head(5).to_string(index=False))

print("\n=== Done! ===")
print(f"All figures saved to: {OUT}")
print(f"All CSVs saved to:    {RES}")
