import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 1. Load data
df = pd.read_csv("long_format_trajectory.csv")
print("Initial shape:", df.shape)

# 2. Convert date columns
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 3. Separate numeric and non-numeric
df_numeric = df.select_dtypes(include=[np.number]).copy()
df_non_numeric = df.select_dtypes(exclude=[np.number]).copy()

print("Numeric columns:", df_numeric.shape[1])
print("Non-numeric columns:", df_non_numeric.shape[1])

# 4. Remove impossible negatives (exclude ID column)
if "idAnimale" in df_numeric.columns:
    id_series = df_numeric["idAnimale"]
    df_numeric = df_numeric.drop(columns=["idAnimale"])
else:
    id_series = None

for col in df_numeric.columns:
    df_numeric[col] = df_numeric[col].mask(df_numeric[col] < 0, np.nan)

# 5. Replace infinite values
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

# 6. Drop fully empty numeric columns only
fully_empty_cols = df_numeric.columns[df_numeric.isna().all()].tolist()
df_numeric = df_numeric.dropna(axis=1, how="all")

print("Dropped fully empty numeric columns:", fully_empty_cols)

# 7. Impute numeric columns
imputer = SimpleImputer(strategy="median")
df_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)

# 8. Reattach ID column if exists
if id_series is not None:
    df_numeric_imputed.insert(0, "idAnimale", id_series.values)

# 9. Recombine numeric + non-numeric
df_cleaned = pd.concat(
    [df_numeric_imputed.reset_index(drop=True),
     df_non_numeric.reset_index(drop=True)],
    axis=1
)

# 10. Keep original column order
df_cleaned = df_cleaned[df.columns]

print("Final cleaned shape:", df_cleaned.shape)

# 11. Save cleaned dataset
df_cleaned.to_csv("cleaned_dairy_data.csv", index=False)
print("Saved cleaned_dairy_data.csv")