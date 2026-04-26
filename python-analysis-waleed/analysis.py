import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Load cleaned dataset
# ===============================
df = pd.read_csv("cleaned_dairy_data.csv")

# ===============================
# 2. Select numeric features
# ===============================
df_numeric = df.select_dtypes(include=[np.number]).copy()

# Remove ID
if "idAnimale" in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=["idAnimale"])

# ===============================
# 3. Scale data
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ===============================
# 4. Create AnnData object
# ===============================
adata = sc.AnnData(X_scaled)

adata.var_names = df_numeric.columns.astype(str)
adata.obs_names = df.index.astype(str)

# ===============================
# 5. PCA
# ===============================
sc.tl.pca(adata, svd_solver="arpack")
sc.pl.pca(adata, show=True)

# ===============================
# 6. Compute neighbors graph
# ===============================
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10)

# ===============================
# 7. Leiden clustering
# ===============================
sc.tl.leiden(adata, resolution=0.5)

# ===============================
# 8. UMAP embedding
# ===============================
sc.tl.umap(adata)

# ===============================
# 9. Plot clusters
# ===============================
sc.pl.umap(adata, color="leiden", show=True)

# ===============================
# 10. Save cluster labels
# ===============================
df["cluster"] = adata.obs["leiden"].values
df.to_csv("clustered_dairy_data.csv", index=False)

print("Clustering complete.")
print("Number of clusters:", df["cluster"].nunique())