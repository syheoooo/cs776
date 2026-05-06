# Predicting Bovine Ketosis from Milk Biomarkers

**CS/BMI 776 — Advanced Bioinformatics | UW-Madison**

Waleed Arshad · Seonyeong Heo · Ignacio Azola Ortiz

---

## Overview

Ketosis is a common metabolic disorder in dairy cows that significantly impacts milk yield and animal welfare. Early detection via non-invasive milk biomarkers can reduce herd-wide losses. This project builds and evaluates machine learning models to predict blood BHBA (beta-hydroxybutyric acid) concentration — the clinical marker for ketosis — from routinely collected milk composition measurements.

We compare four regression approaches (Linear Model, LASSO, Random Forest, and XGBoost) under nested cross-validation and interpret predictions using SHAP values and feature importance metrics.

---

## Repository Structure

```
cs776/
├── python-analysis-waleed/      # Exploratory analysis: data cleaning, PCA, t-SNE
│   ├── data-cleaning.py
│   ├── analysis.py
│   ├── cleaned_dairy_data.csv
│   ├── long_format_trajectory.csv
│   ├── PCA.png
│   └── tSNE.png
│
├── modeling/                    # R-based ketosis prediction model
│   └── ketosis_pred3.R
│
├── final-analysis-waleed/       # Final ML pipeline and results
│   ├── analysis_final.py        # Complete pipeline: EDA + nested CV + feature importance
│   ├── analysis_extended.py     # Extended analysis variant
│   ├── cv_summary.csv           # Cross-validation summary statistics
│   ├── cv_fold_results.csv      # Per-fold CV results
│   ├── cv_summary_extended.csv
│   ├── cv_fold_results_extended.csv
│   ├── lasso_importance.csv     # LASSO feature coefficients
│   ├── rf_importance.csv        # Random Forest MDI feature importance
│   ├── xgb_shap.csv             # XGBoost SHAP values
│   └── figures/                 # Generated plots (EDA, model results, feature importance)
│
└── project-progress/            # Reports, LaTeX source, and session notes
    ├── project_progress_report.tex
    ├── project_progress_report.pdf
    ├── CS776.pdf
    ├── analysis.py
    ├── ketosis_pred3.R
    ├── trajectory_cleaned2.csv  # Final dataset (1,612 cows, no missing values)
    ├── lmm_coefficients.csv
    ├── ml_results.csv
    ├── pca_loadings.csv
    ├── session_notes.md
    └── figures_scc/
```

---

## Dataset

- **Source:** Dairy cow milk composition measurements
- **Size:** 1,612 unique cows, one observation per cow
- **Target:** Blood BHBA concentration (continuous, mmol/L)
- **Features:** Milk acetone, milk fat, protein, lactose, and other routine test-day measures
- **Preprocessing:** Zero missing values after cleaning; trajectory data reshaped to wide format

---

## Methods

### Models
| Model | Description |
|---|---|
| Linear Model (LM) | OLS baseline |
| LASSO | L1-regularized regression; α tuned via inner CV |
| Random Forest (RF) | 100-tree ensemble; hyperparameters tuned via inner CV |
| XGBoost | Gradient-boosted trees; hyperparameters tuned via inner CV |

### Evaluation
- **Nested 5 × 3 cross-validation** (5 outer folds, 3 inner folds for hyperparameter selection)
- Metrics: RMSE, R²

### Interpretability
- SHAP values for XGBoost
- Mean Decrease in Impurity (MDI) for Random Forest
- Coefficient magnitudes for LASSO

---

## Results

| Model | RMSE | R² |
|---|---|---|
| Linear Model | 0.0322 | 0.59 |
| LASSO | 0.0322 | 0.59 |
| XGBoost | 0.0348 | 0.52 |
| Random Forest | 0.0357 | 0.50 |

Linear Model and LASSO tied for best performance (LASSO's optimal α ≈ 3–7 × 10⁻⁵ effectively collapses to OLS), suggesting that the BHBA–biomarker relationship is largely linear.

**Top predictive feature:** Milk acetone (XGBoost SHAP = 0.020; RF MDI = 0.46), consistent with its biochemical co-production with BHBA via ketogenesis.

---

## Setup & Reproduction

### Python pipeline (final analysis)

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn
python final-analysis-waleed/analysis_final.py
```

Outputs CSV result files and figures to `final-analysis-waleed/figures/`.

### R pipeline

```r
# Requires: lme4, tidyverse, caret
Rscript modeling/ketosis_pred3.R
```

### Exploratory analysis

```bash
python python-analysis-waleed/data-cleaning.py
python python-analysis-waleed/analysis.py
```

---

## Key Findings

1. Milk acetone is the strongest predictor of blood BHBA — both models and biochemistry agree.
2. Linear models match or outperform tree-based methods, suggesting the signal is additive and well-conditioned.
3. Feature importance is consistent across LASSO, RF (MDI), and XGBoost (SHAP), lending confidence to the selected biomarkers.

---

## Future Work

- Longitudinal / mixed-effects models using repeated test-day measurements
- Multi-herd validation across different farms and breeds
- Binary ketosis classification (BHBA ≥ 0.10 mmol/L threshold) for clinical decision support
