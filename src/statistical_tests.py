"""
Stage 9 — statistical_tests.py
Section 6 of the lab: formal hypothesis testing.

6.1 Group Comparisons:
  - L1 vs L2 on F1 AND F2 (t-test or Mann-Whitney U)
  - Normality check (Shapiro-Wilk + Q-Q plots)
  - Homogeneity of variances (Levene)
  - BH FDR correction
  - Gender differences at speaker level

6.2 Inter-phoneme distances:
  - Acoustic distance matrix (Euclidean + Mahalanobis)
  - Neural distance matrices (cosine)
  - Mantel test between distance matrices
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_NORM  = "data/features_acoustic_norm.csv"
INPUT_META  = "data/phonemes.csv"
OUTPUT_DIR  = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRENCH_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"}

# ── load data ──────────────────────────────────────────────────────
df = pd.read_csv(INPUT_NORM)
df = df[df["phoneme"].isin(FRENCH_VOWELS)].copy()
print(f"Loaded {len(df)} vowel tokens, {df['phoneme'].nunique()} phonemes")

phonemes = sorted(df["phoneme"].unique())

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


# ══════════════════════════════════════════════════════════════════
# 6.1a — Normality check + Q-Q plots
# ══════════════════════════════════════════════════════════════════
print("\n── Normality checks (Shapiro-Wilk + Q-Q plots) ──────────────")

for formant, col in [("F1", "f1_norm"), ("F2", "f2_norm")]:
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes      = axes.flatten()
    ax_idx    = 0

    normality_rows = []
    for phoneme in phonemes:
        sub = df[df["phoneme"] == phoneme][col].dropna()
        if len(sub) < 5:
            continue

        l1 = df[(df["phoneme"] == phoneme) &
                (df["l1_status"] == "L1")][col].dropna()
        l2 = df[(df["phoneme"] == phoneme) &
                (df["l1_status"] == "L2")][col].dropna()

        _, p_l1 = stats.shapiro(l1) if len(l1) <= 5000 else (None, 1.0)
        _, p_l2 = stats.shapiro(l2) if len(l2) <= 5000 else (None, 1.0)

        normality_rows.append({
            "phoneme":      phoneme,
            "formant":      formant,
            "n_L1":         len(l1),
            "n_L2":         len(l2),
            "shapiro_p_L1": round(p_l1, 4),
            "shapiro_p_L2": round(p_l2, 4),
            "normal_L1":    p_l1 > 0.05,
            "normal_L2":    p_l2 > 0.05,
        })

        # Q-Q plot
        if ax_idx < len(axes):
            ax = axes[ax_idx]
            stats.probplot(sub, dist="norm", plot=ax)
            ax.set_title(f"/{phoneme}/ {formant}",
                         fontsize=10, fontweight="bold")
            ax.get_lines()[0].set(markersize=2, alpha=0.5)
            ax_idx += 1

    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Q-Q plots — {formant} (Lobanov normalised)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_qq_{formant.lower()}.png", dpi=150)
    plt.close()
    print(f"  Saved plot_qq_{formant.lower()}.png")

    norm_df = pd.DataFrame(normality_rows)
    norm_df.to_csv(
        f"{OUTPUT_DIR}/normality_{formant.lower()}.csv", index=False
    )
    print(norm_df[["phoneme", "shapiro_p_L1", "shapiro_p_L2",
                   "normal_L1", "normal_L2"]].to_string())


# ══════════════════════════════════════════════════════════════════
# 6.1b — L1 vs L2 tests on F1 AND F2
# ══════════════════════════════════════════════════════════════════
print("\n── 6.1b L1 vs L2 group comparisons (F1 and F2) ──────────────")

all_results = []

for formant, col in [("F1", "f1_norm"), ("F2", "f2_norm")]:
    results = []
    for phoneme in phonemes:
        l1 = df[(df["phoneme"] == phoneme) &
                (df["l1_status"] == "L1")][col].dropna()
        l2 = df[(df["phoneme"] == phoneme) &
                (df["l1_status"] == "L2")][col].dropna()

        if len(l1) < 5 or len(l2) < 5:
            continue

        # normality
        _, p_norm_l1 = stats.shapiro(l1) if len(l1) <= 5000 \
                       else (None, 1.0)
        _, p_norm_l2 = stats.shapiro(l2) if len(l2) <= 5000 \
                       else (None, 1.0)
        normal = p_norm_l1 > 0.05 and p_norm_l2 > 0.05

        # homogeneity of variances
        _, p_levene = stats.levene(l1, l2)
        equal_var   = p_levene > 0.05

        # test
        if normal:
            stat, p_val = stats.ttest_ind(
                l1, l2, equal_var=equal_var
            )
            test_used = "t-test"
        else:
            stat, p_val = stats.mannwhitneyu(
                l1, l2, alternative="two-sided"
            )
            test_used = "Mann-Whitney"

        # effect size
        pooled_std = np.sqrt((l1.std()**2 + l2.std()**2) / 2)
        cohens_d   = (l1.mean() - l2.mean()) / pooled_std \
                     if pooled_std > 0 else np.nan

        results.append({
            "formant":      formant,
            "phoneme":      phoneme,
            "n_L1":         len(l1),
            "n_L2":         len(l2),
            "mean_L1":      round(l1.mean(), 3),
            "mean_L2":      round(l2.mean(), 3),
            "levene_p":     round(p_levene, 4),
            "equal_var":    equal_var,
            "test":         test_used,
            "stat":         round(stat, 3),
            "p_value":      p_val,
            "cohens_d":     round(cohens_d, 3),
        })

    results_df              = pd.DataFrame(results)
    results_df["p_adjusted"] = bh_correction(
        results_df["p_value"].tolist()
    )
    results_df["significant"] = results_df["p_adjusted"] < 0.05
    all_results.append(results_df)

    print(f"\n{formant}:")
    print(results_df[["phoneme", "mean_L1", "mean_L2", "levene_p",
                       "test", "p_value", "p_adjusted",
                       "significant", "cohens_d"]].to_string())

combined = pd.concat(all_results)
combined.to_csv(f"{OUTPUT_DIR}/acoustic_l1l2_tests.csv", index=False)

# ── forest plot for both formants ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, formant in zip(axes, ["F1", "F2"]):
    sub        = combined[combined["formant"] == formant]
    phonemes_f = sub["phoneme"].tolist()
    diffs      = (sub["mean_L1"] - sub["mean_L2"]).tolist()
    sig        = sub["significant"].tolist()
    colors     = ["#C62828" if s else "#90A4AE" for s in sig]

    ax.barh(phonemes_f, diffs, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel(f"Mean {formant}_norm difference (L1 − L2)",
                  fontsize=10)
    ax.set_title(f"L1 vs L2 {formant} differences\n"
                 f"(red = significant after BH correction)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

legend = [
    mpatches.Patch(color="#C62828",
                   label="Significant (p_adj < 0.05)"),
    mpatches.Patch(color="#90A4AE", label="Not significant"),
]
axes[0].legend(handles=legend, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_l1l2_forest.png", dpi=150)
plt.close()
print("\nSaved plot_l1l2_forest.png")


# ══════════════════════════════════════════════════════════════════
# 6.1c — Gender differences at the SPEAKER level
# ══════════════════════════════════════════════════════════════════
print("\n── 6.1c Gender differences (speaker-level, after Lobanov) ───")

gender_results = []

for formant, col in [("F1", "f1_norm"), ("F2", "f2_norm")]:
    for phoneme in phonemes:
        sub = df[df["phoneme"] == phoneme]

        # one value per speaker
        spk_means = sub.groupby(
            ["speaker_id", "gender"]
        )[col].mean().reset_index()

        f_means = spk_means[
            spk_means["gender"] == "f"][col].dropna()
        m_means = spk_means[
            spk_means["gender"] == "m"][col].dropna()

        if len(f_means) < 3 or len(m_means) < 3:
            continue

        _, p_norm_f = stats.shapiro(f_means)
        _, p_norm_m = stats.shapiro(m_means)
        normal      = p_norm_f > 0.05 and p_norm_m > 0.05
        _, p_levene = stats.levene(f_means, m_means)

        if normal:
            stat, p_val = stats.ttest_ind(
                f_means, m_means,
                equal_var=p_levene > 0.05
            )
            test_used = "t-test (speaker level)"
        else:
            stat, p_val = stats.mannwhitneyu(
                f_means, m_means, alternative="two-sided"
            )
            test_used = "Mann-Whitney (speaker level)"

        gender_results.append({
            "formant":  formant,
            "phoneme":  phoneme,
            "n_F_spk":  len(f_means),
            "n_M_spk":  len(m_means),
            "mean_F":   round(f_means.mean(), 3),
            "mean_M":   round(m_means.mean(), 3),
            "test":     test_used,
            "p_value":  p_val,
        })

gender_df               = pd.DataFrame(gender_results)
gender_df["p_adjusted"] = bh_correction(
    gender_df["p_value"].tolist()
)
gender_df["significant"] = gender_df["p_adjusted"] < 0.05
gender_df.to_csv(f"{OUTPUT_DIR}/gender_tests.csv", index=False)

print(gender_df[["formant", "phoneme", "mean_F", "mean_M",
                  "n_F_spk", "n_M_spk", "test",
                  "p_value", "p_adjusted",
                  "significant"]].to_string())


# ══════════════════════════════════════════════════════════════════
# 6.2 Inter-phoneme distance matrices
# ══════════════════════════════════════════════════════════════════
print("\n── 6.2 Inter-phoneme distance matrices ──────────────────────")

phoneme_list = sorted(df["phoneme"].unique())
n_ph         = len(phoneme_list)

centroids = np.array([
    df[df["phoneme"] == p][["f1_norm", "f2_norm"]].mean().values
    for p in phoneme_list
])

# Euclidean
D_eucl = np.zeros((n_ph, n_ph))
for i in range(n_ph):
    for j in range(n_ph):
        D_eucl[i, j] = np.linalg.norm(centroids[i] - centroids[j])

# Mahalanobis
pooled_cov = np.cov(df[["f1_norm", "f2_norm"]].dropna().T)
try:
    pooled_cov_inv = np.linalg.inv(pooled_cov)
    D_maha         = np.zeros((n_ph, n_ph))
    for i in range(n_ph):
        for j in range(n_ph):
            diff           = centroids[i] - centroids[j]
            D_maha[i, j]   = np.sqrt(diff @ pooled_cov_inv @ diff)
except np.linalg.LinAlgError:
    D_maha = D_eucl.copy()
    print("  WARNING: Mahalanobis failed, using Euclidean")

# neural distance matrices
neural_files = {
    "Whisper L20": "data/features_whisper_layer20_pca50.npz",
    "XLS-R L20":   "data/features_xlsr_layer20_pca50.npz",
}

D_neural = {}
meta     = pd.read_csv(INPUT_META)
meta     = meta[meta["phoneme"].isin(FRENCH_VOWELS)].copy()

for model_name, path in neural_files.items():
    data      = np.load(path)
    X         = data["features"]
    token_ids = data["token_ids"]

    neural_centroids = []
    for p in phoneme_list:
        ph_mask = np.array([
            i < len(meta) and meta.iloc[i]["phoneme"] == p
            for i in range(len(token_ids))
        ])
        if ph_mask.sum() > 0:
            neural_centroids.append(X[ph_mask].mean(axis=0))
        else:
            neural_centroids.append(np.zeros(X.shape[1]))

    D_neural[model_name] = cosine_distances(
        np.array(neural_centroids)
    )

# Mantel test
def mantel_test(D1, D2, n_permutations=999):
    idx      = np.triu_indices(D1.shape[0], k=1)
    v1, v2   = D1[idx], D2[idx]
    r_obs, _ = stats.spearmanr(v1, v2)
    count    = 0
    for _ in range(n_permutations):
        perm      = np.random.permutation(D1.shape[0])
        D1_perm   = D1[np.ix_(perm, perm)]
        r_perm, _ = stats.spearmanr(D1_perm[idx], v2)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_permutations + 1)

print("\nMantel test results:")
mantel_results = []
pairs = [
    ("Acoustic (Eucl)", D_eucl,
     "Whisper L20",     D_neural["Whisper L20"]),
    ("Acoustic (Eucl)", D_eucl,
     "XLS-R L20",       D_neural["XLS-R L20"]),
    ("Whisper L20",     D_neural["Whisper L20"],
     "XLS-R L20",       D_neural["XLS-R L20"]),
]

for name1, D1, name2, D2 in pairs:
    r, p = mantel_test(D1, D2)
    print(f"  {name1} vs {name2}: r={r:.3f}  p={p:.3f}")
    mantel_results.append({
        "D1": name1, "D2": name2,
        "mantel_r": round(r, 3),
        "p_value":  round(p, 3),
    })

pd.DataFrame(mantel_results).to_csv(
    f"{OUTPUT_DIR}/mantel_results.csv", index=False
)

# heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, D, title in zip(
    axes,
    [D_eucl, D_maha],
    ["Euclidean distance", "Mahalanobis distance"]
):
    im = ax.imshow(D, cmap="YlOrRd")
    ax.set_xticks(range(n_ph))
    ax.set_yticks(range(n_ph))
    ax.set_xticklabels(phoneme_list, fontsize=9)
    ax.set_yticklabels(phoneme_list, fontsize=9)
    ax.set_title(f"Acoustic {title}",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_distance_matrices.png", dpi=150)
plt.close()
print("\nSaved plot_distance_matrices.png")
print("\nAll statistical tests done!")