"""
Stage 10b — neural_l1l2_test.py
Section 6.1: L1 vs L2 on Neural Representations.

For each phoneme, tests whether the mean neural representation
differs between L1 and L2 speakers using a permutation test
on cosine distance between centroids.

Procedure:
1. Compute observed cosine distance between L1 and L2 centroids
2. Permute speaker labels 5000 times → null distribution
3. Report permutation p-value + BH correction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cosine
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

OUTPUT_DIR    = "results"
FRENCH_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"}
N_PERM        = 5000
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load metadata ──────────────────────────────────────────────────
meta = pd.read_csv("data/phonemes.csv")
meta = meta[meta["phoneme"].isin(FRENCH_VOWELS)].copy()
meta = meta.reset_index(drop=True)
print(f"Loaded {len(meta)} vowel tokens")

# ── BH correction ──────────────────────────────────────────────────
def bh_correction(pvalues):
    n          = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p   = np.array(pvalues)[sorted_idx]
    adjusted   = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[sorted_idx[i]] = sorted_p[i]
        else:
            adjusted[sorted_idx[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted[sorted_idx[i + 1]]
            )
    return np.minimum(adjusted, 1.0)


# ── cosine distance between two sets of vectors ────────────────────
def centroid_cosine_distance(X1, X2):
    c1 = X1.mean(axis=0)
    c2 = X2.mean(axis=0)
    return cosine(c1, c2)


# ── permutation test ───────────────────────────────────────────────
def permutation_test(X, labels, n_perm=N_PERM):
    """
    Tests whether L1 and L2 centroids differ in cosine distance.
    labels: array of "L1" or "L2" per token

    Returns observed distance and p-value.
    """
    l1_mask = labels == "L1"
    l2_mask = labels == "L2"

    if l1_mask.sum() < 2 or l2_mask.sum() < 2:
        return np.nan, np.nan

    # observed distance
    d_obs = centroid_cosine_distance(X[l1_mask], X[l2_mask])

    # null distribution via permutation
    null_dists = []
    for _ in range(n_perm):
        perm        = np.random.permutation(labels)
        d_perm      = centroid_cosine_distance(
            X[perm == "L1"], X[perm == "L2"]
        )
        null_dists.append(d_perm)

    # p-value: proportion of null distances >= observed
    p_val = np.mean(np.array(null_dists) >= d_obs)
    return d_obs, p_val


# ── run for each model ─────────────────────────────────────────────
MODELS = {
    "Whisper L4":  "data/features_whisper_layer4_pca50.npz",
    "Whisper L20": "data/features_whisper_layer20_pca50.npz",
    "XLS-R L4":    "data/features_xlsr_layer4_pca50.npz",
    "XLS-R L20":   "data/features_xlsr_layer20_pca50.npz",
}

all_results = []

for model_name, path in MODELS.items():
    print(f"\nProcessing {model_name}...")

    data      = np.load(path)
    X_full    = data["features"]
    token_ids = data["token_ids"]

    # align with meta
    X    = X_full
    tids = token_ids

    results = []
    for phoneme in sorted(FRENCH_VOWELS):
        # get tokens for this phoneme
        ph_mask = np.array([
            i < len(meta) and meta.iloc[i]["phoneme"] == phoneme
            for i in range(len(tids))
        ])

        if ph_mask.sum() < 10:
            continue

        X_ph     = X[ph_mask]
        l1_status = np.array([
            meta.iloc[i]["l1_status"]
            for i in range(len(tids))
            if i < len(meta) and meta.iloc[i]["phoneme"] == phoneme
        ])

        d_obs, p_val = permutation_test(X_ph, l1_status)

        results.append({
            "model":    model_name,
            "phoneme":  phoneme,
            "n_L1":     (l1_status == "L1").sum(),
            "n_L2":     (l1_status == "L2").sum(),
            "cosine_d": round(d_obs, 4) if not np.isnan(d_obs) else np.nan,
            "p_value":  p_val,
        })
        print(f"  /{phoneme}/: d={d_obs:.4f}  p={p_val:.4f}")

    results_df              = pd.DataFrame(results)
    results_df["p_adjusted"] = bh_correction(
        results_df["p_value"].tolist()
    )
    results_df["significant"] = results_df["p_adjusted"] < 0.05
    all_results.append(results_df)

# ── save combined results ──────────────────────────────────────────
combined = pd.concat(all_results)
combined.to_csv(f"{OUTPUT_DIR}/neural_l1l2_tests.csv", index=False)

print("\n── Summary ───────────────────────────────────────────────────")
print(combined[["model", "phoneme", "cosine_d",
                "p_value", "p_adjusted",
                "significant"]].to_string())

# ── forest plot ────────────────────────────────────────────────────
models    = list(MODELS.keys())
fig, axes = plt.subplots(1, len(models), figsize=(16, 6), sharey=True)

for ax, model_name in zip(axes, models):
    sub      = combined[combined["model"] == model_name]
    phonemes = sub["phoneme"].tolist()
    dists    = sub["cosine_d"].tolist()
    sig      = sub["significant"].tolist()
    colors   = ["#C62828" if s else "#90A4AE" for s in sig]

    ax.barh(phonemes, dists, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(model_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Cosine distance\n(L1 vs L2 centroid)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

legend = [
    mpatches.Patch(color="#C62828",
                   label="Significant (p_adj < 0.05)"),
    mpatches.Patch(color="#90A4AE", label="Not significant"),
]
axes[0].legend(handles=legend, fontsize=8)
fig.suptitle(
    "L1 vs L2 cosine distance per phoneme — permutation test\n"
    "(red = significant after BH correction)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_neural_l1l2.png", dpi=150)
plt.close()
print("\nSaved plot_neural_l1l2.png")
print("Done.")