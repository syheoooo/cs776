"""
Milk Fatty Acid Profile as an Early Biomarker of Subclinical Mastitis
BMI/CS 776: Advanced Bioinformatics — Spring 2026
Team: Seonyeong Heo, Waleed Arshad, Ignacio Azola Ortiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# statsmodels for LMM
import statsmodels.formula.api as smf

# sklearn for ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, f1_score, classification_report
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# =============================================================================
# 1. LOAD & PREPROCESS
# =============================================================================

print("=" * 60)
print("STEP 1: Loading and Preprocessing Data")
print("=" * 60)

df = pd.read_csv("long_format_trajectory.csv")
print(f"Raw shape: {df.shape}")

# Fix dates
df["measurement_date"] = pd.to_datetime(df["measurement_date"], errors="coerce")
df["calving_date"]     = pd.to_datetime(df["calving_date"],     errors="coerce")

# Remove biologically impossible SCC values
df = df[df["scc"] > 0].copy()
print(f"After removing SCC <= 0: {df.shape}")

# Recompute scc_log cleanly
df["scc_log"] = np.log10(df["scc"])

# Recompute high_scc: standard threshold is SCC > 200 (×1000 cells/mL)
df["high_scc"] = (df["scc"] > 200).astype(int)
print(f"High SCC prevalence: {df['high_scc'].mean():.2%}")

# Fatty acid (FA) features available in this dataset
FA_COLS = ["sfa", "ufa", "newly_fa", "mixed_fa", "preformed_fa", "c18_0", "c18_1"]
MILK_COMP = ["fat", "protein", "casein", "lactose"]
MILK_QUAL = ["ph", "freezing", "k20", "a30", "ec"]
METABOLIC = ["bhba", "urea", "acetone"]

ALL_FEATURES = FA_COLS + MILK_COMP + MILK_QUAL + METABOLIC

# Subset to rows with at least SFA and UFA (core FA columns)
df_fa = df.dropna(subset=["sfa", "ufa", "scc_log"]).copy()
print(f"Rows with core FA data: {df_fa.shape}")

# Rename lactation → parity for clarity (matches proposal notation)
df_fa = df_fa.rename(columns={"lactation": "parity"})

print("\nClass balance in df_fa:")
print(df_fa["high_scc"].value_counts())

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# ── 2a. SCC distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_fa["scc"].clip(upper=3000), bins=80, color="steelblue", edgecolor="white")
axes[0].set_xlabel("SCC (×1000 cells/mL)")
axes[0].set_ylabel("Count")
axes[0].set_title("SCC Distribution (clipped at 3000)")

axes[1].hist(df_fa["scc_log"], bins=60, color="coral", edgecolor="white")
axes[1].axvline(np.log10(200), color="black", linestyle="--", label="Threshold (200)")
axes[1].set_xlabel("log₁₀(SCC)")
axes[1].set_ylabel("Count")
axes[1].set_title("log₁₀(SCC) Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT / "scc_distribution.png", dpi=150)
plt.close()
print("Saved: figures/scc_distribution.png")

# ── 2b. FA correlations with SCC_log ─────────────────────────────────────────
corr_features = [c for c in FA_COLS + MILK_COMP if c in df_fa.columns]
corr = df_fa[corr_features + ["scc_log"]].corr()["scc_log"].drop("scc_log").sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d73027" if v > 0 else "#4575b4" for v in corr]
corr.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with log₁₀(SCC)")
ax.set_title("Feature Correlations with SCC_log")
plt.tight_layout()
plt.savefig(OUT / "feature_correlations.png", dpi=150)
plt.close()
print("Saved: figures/feature_correlations.png")

# ── 2c. SCC_log by lactation stage ───────────────────────────────────────────
stage_order = ["early", "mid", "late"]
fig, ax = plt.subplots(figsize=(7, 4))
stage_data = [df_fa[df_fa["lactation_stage"] == s]["scc_log"].dropna().values
              for s in stage_order]
ax.boxplot(stage_data, labels=stage_order, patch_artist=True)
ax.set_xlabel("Lactation Stage")
ax.set_ylabel("log₁₀(SCC)")
ax.set_title("SCC_log by Lactation Stage")
plt.tight_layout()
plt.savefig(OUT / "scc_by_stage.png", dpi=150)
plt.close()
print("Saved: figures/scc_by_stage.png")

# ── 2d. Missing data heatmap ──────────────────────────────────────────────────
missing_pct = df[ALL_FEATURES].isnull().mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(10, 4))
missing_pct.plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Missing (%)")
ax.set_title("Missing Data by Feature")
ax.axhline(50, color="red", linestyle="--", label="50% threshold")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "missing_data.png", dpi=150)
plt.close()
print("Saved: figures/missing_data.png")

# ── 2e. SFA vs UFA scatter colored by high_scc ───────────────────────────────
sample = df_fa.sample(n=min(5000, len(df_fa)), random_state=42)
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(
    sample["sfa"], sample["ufa"],
    c=sample["scc_log"], cmap="RdYlBu_r", alpha=0.4, s=8
)
plt.colorbar(scatter, ax=ax, label="log₁₀(SCC)")
ax.set_xlabel("SFA (g/100g milk)")
ax.set_ylabel("UFA (g/100g milk)")
ax.set_title("SFA vs UFA colored by SCC_log")
plt.tight_layout()
plt.savefig(OUT / "sfa_ufa_scatter.png", dpi=150)
plt.close()
print("Saved: figures/sfa_ufa_scatter.png")

print("\nDescriptive statistics (FA + SCC):")
print(df_fa[FA_COLS + ["scc_log"]].describe().round(3))

# =============================================================================
# 3. PHASE 1 — ASSOCIATIVE ANALYSIS VIA LINEAR MIXED-EFFECTS MODELS (LMM)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Phase 1 — Linear Mixed-Effects Models")
print("=" * 60)

# Subset to rows with all LMM variables available
lmm_cols = ["scc_log", "sfa", "ufa", "parity", "DIM", "idAnimale"]
df_lmm = df_fa.dropna(subset=lmm_cols).copy()
print(f"LMM dataset size: {df_lmm.shape}")

# Standardize continuous predictors for interpretable coefficients
for col in ["sfa", "ufa", "DIM"]:
    df_lmm[f"{col}_z"] = (df_lmm[col] - df_lmm[col].mean()) / df_lmm[col].std()

# Model: scc_log ~ sfa + ufa + parity + DIM | random intercept per cow
# Uses animal ID as the grouping variable (repeated measures)
print("\nFitting LMM: scc_log ~ sfa_z + ufa_z + parity + DIM_z (cow random intercept)...")
try:
    lmm_model = smf.mixedlm(
        "scc_log ~ sfa_z + ufa_z + C(parity) + DIM_z",
        data=df_lmm,
        groups=df_lmm["idAnimale"]
    )
    lmm_result = lmm_model.fit(method="lbfgs", maxiter=500)
    print(lmm_result.summary())

    # Save coefficient table
    coef_df = pd.DataFrame({
        "coef":   lmm_result.fe_params,
        "se":     lmm_result.bse_fe,
        "pvalue": lmm_result.pvalues
    })
    coef_df.to_csv("lmm_coefficients.csv")
    print("\nSaved: lmm_coefficients.csv")

except Exception as e:
    print(f"LMM fitting error: {e}")
    print("Falling back to OLS for coefficient inspection...")
    import statsmodels.api as sm
    X_ols = df_lmm[["sfa_z", "ufa_z", "DIM_z", "parity"]].copy()
    X_ols = sm.add_constant(X_ols)
    ols_model = sm.OLS(df_lmm["scc_log"], X_ols).fit()
    print(ols_model.summary())

# =============================================================================
# 4. PHASE 2 — PREDICTIVE MODELING (MACHINE LEARNING)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Phase 2 — Machine Learning Models")
print("=" * 60)

# Use a broader feature set — rows must have scc_log + at least FA core features
ml_features = ["sfa", "ufa", "fat", "protein", "casein", "lactose",
               "parity", "DIM", "ufa_sfa_ratio"]
df_ml = df_fa.dropna(subset=ml_features + ["scc_log", "high_scc"]).copy()
print(f"ML dataset size: {df_ml.shape}")
print(f"High SCC prevalence: {df_ml['high_scc'].mean():.2%}")

X = df_ml[ml_features].values
y_reg = df_ml["scc_log"].values
y_cls = df_ml["high_scc"].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4a. Regression: predict SCC_log ──────────────────────────────────────────
print("\n--- Regression: Predicting log₁₀(SCC) ---")

# Linear Regression (baseline)
lr = LinearRegression()
cv_rmse_lr = np.sqrt(-cross_val_score(
    lr, X_scaled, y_reg, cv=5,
    scoring="neg_mean_squared_error"
))
cv_r2_lr = cross_val_score(lr, X_scaled, y_reg, cv=5, scoring="r2")
print(f"Linear Regression  | RMSE: {cv_rmse_lr.mean():.4f} ± {cv_rmse_lr.std():.4f} | "
      f"R²: {cv_r2_lr.mean():.4f} ± {cv_r2_lr.std():.4f}")

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
cv_rmse_rf = np.sqrt(-cross_val_score(
    rf_reg, X_scaled, y_reg, cv=5,
    scoring="neg_mean_squared_error"
))
cv_r2_rf = cross_val_score(rf_reg, X_scaled, y_reg, cv=5, scoring="r2")
print(f"Random Forest Reg  | RMSE: {cv_rmse_rf.mean():.4f} ± {cv_rmse_rf.std():.4f} | "
      f"R²: {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}")

# ── 4b. Classification: predict high_scc ─────────────────────────────────────
print("\n--- Classification: Predicting High SCC (>200) ---")

rf_cls = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight="balanced", n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores, f1_scores = [], []
for train_idx, test_idx in skf.split(X_scaled, y_cls):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y_cls[train_idx], y_cls[test_idx]
    rf_cls.fit(X_tr, y_tr)
    y_prob = rf_cls.predict_proba(X_te)[:, 1]
    y_pred = rf_cls.predict(X_te)
    auc_scores.append(roc_auc_score(y_te, y_prob))
    f1_scores.append(f1_score(y_te, y_pred, zero_division=0))

print(f"Random Forest Cls  | AUROC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f} | "
      f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# ── 4c. Feature Importance ────────────────────────────────────────────────────
rf_reg.fit(X_scaled, y_reg)
importances = pd.Series(rf_reg.feature_importances_, index=ml_features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Feature Importance (MDI)")
ax.set_title("Random Forest Feature Importance for SCC_log")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()
print("Saved: figures/feature_importance.png")

# Save regression results
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "Task": ["Regression", "Regression"],
    "RMSE_mean": [cv_rmse_lr.mean(), cv_rmse_rf.mean()],
    "RMSE_std":  [cv_rmse_lr.std(),  cv_rmse_rf.std()],
    "R2_mean":   [cv_r2_lr.mean(),   cv_r2_rf.mean()],
    "R2_std":    [cv_r2_lr.std(),    cv_r2_rf.std()],
})
results_df.to_csv("ml_results.csv", index=False)
print("\nSaved: ml_results.csv")

# =============================================================================
# 5. PHASE 3 — CLUSTERING / DIMENSIONALITY REDUCTION (PCA)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Phase 3 — PCA on FA Profile")
print("=" * 60)

pca_features = ["sfa", "ufa", "newly_fa", "mixed_fa", "preformed_fa"]
df_pca = df_fa.dropna(subset=pca_features + ["scc_log"]).copy()
print(f"PCA dataset size: {df_pca.shape}")

X_pca = StandardScaler().fit_transform(df_pca[pca_features].values)

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_pca)

print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")

# Plot PCA colored by SCC_log
sample_mask = np.random.choice(len(pcs), size=min(5000, len(pcs)), replace=False)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    pcs[sample_mask, 0], pcs[sample_mask, 1],
    c=df_pca["scc_log"].values[sample_mask],
    cmap="RdYlBu_r", alpha=0.4, s=6
)
plt.colorbar(sc, ax=ax, label="log₁₀(SCC)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA of Fatty Acid Profile — colored by SCC_log")
plt.tight_layout()
plt.savefig(OUT / "pca_fa_profile.png", dpi=150)
plt.close()
print("Saved: figures/pca_fa_profile.png")

# PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_features,
    columns=["PC1", "PC2"]
).round(4)
print("\nPCA Loadings:")
print(loadings)
loadings.to_csv("pca_loadings.csv")
print("Saved: pca_loadings.csv")

print("\n" + "=" * 60)
print("Analysis complete. All outputs saved.")
print("=" * 60)
