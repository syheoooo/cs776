# Session Notes — BMI/CS 776 Project

**Project:** Milk Fatty Acid Profile as an Early Biomarker of Subclinical Mastitis
**Team:** Seonyeong Heo, Waleed Arshad, Ignacio Azola Ortiz
**Report due:** Mon Apr 6, 2026 11:59 PM

---

## What was built

### `analysis.py`
Full analysis pipeline:
- Data loading & preprocessing (387,713 rows, 36 features; fixed `high_scc` label using SCC > 200 threshold; recomputed `scc_log`)
- EDA: SCC distribution, feature correlations, SCC by lactation stage, missing data audit, SFA vs UFA scatter
- Phase 1: LMM via `statsmodels.mixedlm` — `scc_log ~ sfa_z + ufa_z + C(parity) + DIM_z`, random intercept per cow (`idAnimale`). Converged on 315,514 obs / 19,172 cows.
- Phase 2: 5-fold CV Linear Regression and Random Forest (regression + classification). AUROC=0.71 for high SCC classification.
- Phase 3: PCA on 5 FA features — PC1=73.1%, PC2=23.7% variance.
- Outputs: `figures/`, `lmm_coefficients.csv`, `ml_results.csv`, `pca_loadings.csv`

### `project_progress_report.tex`
LaTeX progress report with all results populated. Updated to include all 7 figures from `figures/`:

| Figure | Placement |
|---|---|
| `scc_distribution.png` | After EDA bullet (full width) |
| `missing_data.png` + `feature_correlations.png` | Side by side after EDA |
| `scc_by_stage.png` + `sfa_ufa_scatter.png` | Side by side after EDA |
| `feature_importance.png` | After Phase 2 ML table |
| `pca_fa_profile.png` | After Phase 3 PCA bullet |

---

## Next steps
- SMOTE oversampling + XGBoost for classification improvement
- t-SNE / UMAP on stratified subsample (consider HTCondor for full dataset)
- Multiple imputation or two-cohort strategy for missing FA columns (`newly_fa`, `mixed_fa`, `preformed_fa`, `c18_0/c18_1`)
