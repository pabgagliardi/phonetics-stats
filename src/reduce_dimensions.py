"""
Stage 7 — reduce_dimensions.py
Applies PCA and UMAP to reduce neural representations to:
  - 2 dimensions for visualisation
  - 50 dimensions for clustering

Outputs:
  - data/features_whisper_layer{N}_pca2.npz
  - data/features_whisper_layer{N}_pca50.npz
  - data/features_whisper_layer{N}_umap2.npz
  - data/features_xlsr_layer{N}_pca2.npz
  - data/features_xlsr_layer{N}_pca50.npz
  - data/features_xlsr_layer{N}_umap2.npz
"""

import os
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_CSV  = "data/phonemes.csv"
OUTPUT_DIR = "data"
N_VIS      = params["pca"]["n_components_visualisation"]   # 2
N_CLUST    = params["pca"]["n_components_clustering"]       # 50

# ── load metadata ──────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} phoneme tokens")

# ── define which files to process ─────────────────────────────────
FILES = {
    "whisper_layer4":  "data/features_whisper_layer4.npz",
    "whisper_layer20": "data/features_whisper_layer20.npz",
    "xlsr_layer4":     "data/features_xlsr_layer4.npz",
    "xlsr_layer10":    "data/features_xlsr_layer10.npz",
    "xlsr_layer20":    "data/features_xlsr_layer20.npz",
}

# ── try importing UMAP ─────────────────────────────────────────────
try:
    from umap import UMAP
    HAS_UMAP = True
    print("UMAP available ✓")
except ImportError:
    HAS_UMAP = False
    print("UMAP not available — skipping UMAP reduction")
    print("Install with: pip install umap-learn")

print()

# ── process each representation ────────────────────────────────────
for name, path in FILES.items():
    print(f"Processing {name}...")

    data      = np.load(path)
    X         = data["features"]      # (N, 1024)
    token_ids = data["token_ids"]

    # standardise before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA to 2D (visualisation) ──────────────────────────────────
    pca2   = PCA(n_components=N_VIS, random_state=42)
    X_pca2 = pca2.fit_transform(X_scaled)
    var_explained = pca2.explained_variance_ratio_.sum() * 100

    out_path = f"{OUTPUT_DIR}/features_{name}_pca2.npz"
    np.savez(out_path, features=X_pca2, token_ids=token_ids)
    print(f"  PCA-2D: variance explained = {var_explained:.1f}% → {out_path}")

    # ── PCA to 50D (clustering) ────────────────────────────────────
    n_clust = min(N_CLUST, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca50   = PCA(n_components=n_clust, random_state=42)
    X_pca50 = pca50.fit_transform(X_scaled)
    var_explained_50 = pca50.explained_variance_ratio_.sum() * 100

    out_path = f"{OUTPUT_DIR}/features_{name}_pca50.npz"
    np.savez(out_path, features=X_pca50, token_ids=token_ids)
    print(f"  PCA-50D: variance explained = {var_explained_50:.1f}% → {out_path}")

    # ── UMAP to 2D (visualisation) ─────────────────────────────────
    if HAS_UMAP:
        print(f"  Running UMAP (this may take a few minutes)...")
        umap  = UMAP(n_components=N_VIS, random_state=42,
                     n_neighbors=30, min_dist=0.1)
        X_umap = umap.fit_transform(X_pca50)  # fit on PCA-50 for speed

        out_path = f"{OUTPUT_DIR}/features_{name}_umap2.npz"
        np.savez(out_path, features=X_umap, token_ids=token_ids)
        print(f"  UMAP-2D → {out_path}")

    print()

print("Done.")