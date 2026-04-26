"""
Stage 8 — visualise_neural.py
Produces visualisations for neural representations:

1. PCA and UMAP 2D projections coloured by:
   a. phoneme label
   b. L1 status
   c. gender
2. Between-class variance ratio
3. Within vs between phoneme cosine similarity ratio
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from sklearn.metrics.pairwise import cosine_similarity

with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_CSV  = "data/phonemes.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load metadata ──────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV).reset_index(drop=True)

# keep only French vowels of interest
FRENCH_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"}
MIN_TOKENS    = 10
phoneme_counts = df.groupby("phoneme").size()
valid_phonemes = phoneme_counts[phoneme_counts >= MIN_TOKENS].index
df = df[df["phoneme"].isin(valid_phonemes) &
        df["phoneme"].isin(FRENCH_VOWELS)].copy()
print(f"Using {len(df)} tokens, {df['phoneme'].nunique()} phonemes: "
      f"{sorted(df['phoneme'].unique())}")

# ── define representations to visualise ───────────────────────────
REPS = {
    "Whisper layer 4":  {
        "pca2":  "data/features_whisper_layer4_pca2.npz",
        "umap2": "data/features_whisper_layer4_umap2.npz",
        "pca50": "data/features_whisper_layer4_pca50.npz",
    },
    "Whisper layer 20": {
        "pca2":  "data/features_whisper_layer20_pca2.npz",
        "umap2": "data/features_whisper_layer20_umap2.npz",
        "pca50": "data/features_whisper_layer20_pca50.npz",
    },
    "XLS-R layer 4":  {
        "pca2":  "data/features_xlsr_layer4_pca2.npz",
        "umap2": "data/features_xlsr_layer4_umap2.npz",
        "pca50": "data/features_xlsr_layer4_pca50.npz",
    },
    "XLS-R layer 10": {
        "pca2":  "data/features_xlsr_layer10_pca2.npz",
        "umap2": "data/features_xlsr_layer10_umap2.npz",
        "pca50": "data/features_xlsr_layer10_pca50.npz",
    },
    "XLS-R layer 20": {
        "pca2":  "data/features_xlsr_layer20_pca2.npz",
        "umap2": "data/features_xlsr_layer20_umap2.npz",
        "pca50": "data/features_xlsr_layer20_pca50.npz",
    },
}

# ── colour maps ────────────────────────────────────────────────────
PHONEMES = sorted(df["phoneme"].unique())
PHONEME_COLORS = plt.cm.tab20(np.linspace(0, 1, len(PHONEMES)))
PHONEME_CMAP   = dict(zip(PHONEMES, PHONEME_COLORS))

L1_COLORS  = {"L1": "#1565C0", "L2": "#C62828"}
GEN_COLORS = {"f": "#E91E63", "m": "#1976D2"}


# ══════════════════════════════════════════════════════════════════
# Helper: 2D scatter plot
# ══════════════════════════════════════════════════════════════════
def plot_2d(X, labels, color_map, title, out_path, alpha=0.3, s=8):
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in sorted(set(labels)):
        mask = np.array(labels) == label
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[color_map[label]],
            label=label, alpha=alpha, s=s
        )
    ax.legend(fontsize=8, markerscale=2,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════
variance_ratios  = []
similarity_ratios = []

for rep_name, paths in REPS.items():
    print(f"\nProcessing {rep_name}...")
    safe_name = rep_name.replace(" ", "_").lower()

    # load 2D representations
    for method, key in [("PCA", "pca2"), ("UMAP", "umap2")]:
        data      = np.load(paths[key])
        X_full    = data["features"]
        token_ids = data["token_ids"]

        # align with filtered df
        mask = np.isin(token_ids, df.index)
        X    = X_full[mask]
        tids = token_ids[mask]
        meta = df.loc[tids]

        phoneme_labels = meta["phoneme"].tolist()
        l1_labels      = meta["l1_status"].tolist()
        gender_labels  = meta["gender"].tolist()

        # plot by phoneme
        plot_2d(
            X, phoneme_labels, PHONEME_CMAP,
            f"{rep_name} — {method} — by phoneme",
            f"{OUTPUT_DIR}/neural_{safe_name}_{method.lower()}_phoneme.png"
        )

        # plot by L1 status
        plot_2d(
            X, l1_labels, L1_COLORS,
            f"{rep_name} — {method} — by L1 status",
            f"{OUTPUT_DIR}/neural_{safe_name}_{method.lower()}_l1.png",
            alpha=0.4, s=10
        )

        # plot by gender
        plot_2d(
            X, gender_labels, GEN_COLORS,
            f"{rep_name} — {method} — by gender",
            f"{OUTPUT_DIR}/neural_{safe_name}_{method.lower()}_gender.png",
            alpha=0.4, s=10
        )

        print(f"  {method}: 3 plots saved")

# ── between-class variance ratio (PCA-2D and UMAP-2D) ─────────
    for method_name, method_key in [("PCA-2D", "pca2"), ("UMAP-2D", "umap2")]:
        data      = np.load(paths[method_key])
        X_full    = data["features"]
        token_ids = data["token_ids"]
        mask      = np.isin(token_ids, df.index)
        X         = X_full[mask]
        tids      = token_ids[mask]
        meta      = df.loc[tids]

        total_var     = X.var(axis=0).sum()
        phoneme_means = np.array([
            X[meta["phoneme"].values == p].mean(axis=0)
            for p in PHONEMES
            if (meta["phoneme"].values == p).sum() > 0
        ])
        between_var = phoneme_means.var(axis=0).sum()
        ratio       = between_var / total_var

        variance_ratios.append({
            "representation":         rep_name,
            "method":                 method_name,
            "between_class_var_ratio": round(ratio, 4),
        })
        print(f"  Between-class variance ratio ({method_name}): {ratio:.4f}")

    # ── cosine similarity ratio (PCA-50D) ─────────────────────────
    data50    = np.load(paths["pca50"])
    X50_full  = data50["features"]
    token_ids50 = data50["token_ids"]
    mask50    = np.isin(token_ids50, df.index)
    X50       = X50_full[mask50]
    tids50    = token_ids50[mask50]
    meta50    = df.loc[tids50]

    # sample for speed (cosine similarity is O(N²))
    MAX_SAMPLE = 2000
    if len(X50) > MAX_SAMPLE:
        idx = np.random.RandomState(42).choice(
            len(X50), MAX_SAMPLE, replace=False
        )
        X50_s   = X50[idx]
        meta50_s = meta50.iloc[idx]
    else:
        X50_s    = X50
        meta50_s = meta50

    S = cosine_similarity(X50_s)
    ph = meta50_s["phoneme"].values

    within  = []
    between = []
    n = len(ph)
    for i in range(n):
        for j in range(i + 1, n):
            if ph[i] == ph[j]:
                within.append(S[i, j])
            else:
                between.append(S[i, j])

    sim_ratio = np.mean(within) / np.mean(between) if np.mean(between) > 0 else np.nan
    similarity_ratios.append({
        "representation":        rep_name,
        "within_phoneme_sim":    round(np.mean(within), 4),
        "between_phoneme_sim":   round(np.mean(between), 4),
        "ratio_within_between":  round(sim_ratio, 4),
    })
    print(f"  Cosine sim — within: {np.mean(within):.4f}  "
          f"between: {np.mean(between):.4f}  "
          f"ratio: {sim_ratio:.4f}")

# ── save summary tables ────────────────────────────────────────────
var_df = pd.DataFrame(variance_ratios)
var_df.to_csv(f"{OUTPUT_DIR}/neural_variance_ratios.csv", index=False)
print(f"\nSaved neural_variance_ratios.csv")

sim_df = pd.DataFrame(similarity_ratios)
sim_df.to_csv(f"{OUTPUT_DIR}/neural_similarity_ratios.csv", index=False)
print(f"Saved neural_similarity_ratios.csv")

# ══════════════════════════════════════════════════════════════════
# Extra: per-speaker plots for high inter-speaker variability vowels
# ══════════════════════════════════════════════════════════════════
TARGET_PHONEMES = ["ø", "o", "i", "a"]  # high vs low inter-speaker

speakers       = sorted(df["speaker_id"].unique())
n_spk          = len(speakers)
SPEAKER_COLORS = plt.cm.tab20(np.linspace(0, 1, n_spk))
SPK_CMAP       = dict(zip(speakers, SPEAKER_COLORS))

for model_name, umap_path in [
    ("XLS-R layer 20",  "data/features_xlsr_layer20_umap2.npz"),
    ("Whisper layer 20", "data/features_whisper_layer20_umap2.npz"),
]:
    data      = np.load(umap_path)
    X_full    = data["features"]
    token_ids = data["token_ids"]
    mask      = np.isin(token_ids, df.index)
    X         = X_full[mask]
    tids      = token_ids[mask]
    meta_u    = df.loc[tids]

    fig, axes = plt.subplots(1, len(TARGET_PHONEMES),
                              figsize=(16, 4))

    for ax, phoneme in zip(axes, TARGET_PHONEMES):
        ph_mask = meta_u["phoneme"].values == phoneme
        X_ph    = X[ph_mask].copy()
        spk_ph  = meta_u["speaker_id"].values[ph_mask]

        # normalise each dimension to [0,1] for fair comparison
        for dim in range(2):
            min_val = X_ph[:, dim].min()
            max_val = X_ph[:, dim].max()
            if max_val > min_val:
                X_ph[:, dim] = (X_ph[:, dim] - min_val) / (max_val - min_val)

        for spk in speakers:
            spk_mask = spk_ph == spk
            if spk_mask.sum() == 0:
                continue
            ax.scatter(
                X_ph[spk_mask, 0], X_ph[spk_mask, 1],
                c=[SPK_CMAP[spk]], label=spk,
                s=20, alpha=0.7
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"/{phoneme}/", fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP Dim 1 (normalised)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("UMAP Dim 2 (normalised)")
    axes[0].legend(fontsize=6, markerscale=1.5,
                   bbox_to_anchor=(1.01, 1), loc="upper left")

    safe = model_name.replace(" ", "_").lower()
    fig.suptitle(
        f"{model_name} UMAP — coloured by speaker\n"
        f"(coordinates normalised per phoneme for fair comparison)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/neural_{safe}_umap_by_speaker.png",
                dpi=150)
    plt.close()
    print(f"Saved neural_{safe}_umap_by_speaker.png")
    
print("\nAll neural visualisations done!")