"""
Stage 4 — descriptive_stats.py
Computes descriptive statistics for acoustic features.

Produces:
1. Summary table: mean, median, std, IQR, CV per phoneme per group
2. Variance decomposition: inter-speaker, intra-speaker, residual
3. Vowel chart (F1 vs F2, IPA convention)
4. Box plots of F1 and F2 per phoneme
5. Violin plots of intra-speaker variability
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_CSV  = "data/features_acoustic_norm.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load normalised features ───────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} normalised vowel tokens")

# keep only phonemes with enough tokens for analysis
MIN_TOKENS = 10
phoneme_counts = df.groupby("phoneme").size()
valid_phonemes = phoneme_counts[phoneme_counts >= MIN_TOKENS].index
df = df[df["phoneme"].isin(valid_phonemes)]
print(f"Keeping {df['phoneme'].nunique()} phonemes with >={MIN_TOKENS} tokens")

# define groups
df["group"] = df["l1_status"] + "/" + df["gender"].str.upper()

GROUPS  = ["L1/F", "L1/M", "L2/F", "L2/M"]
COLORS  = {
    "L1/F": "#1565C0",
    "L1/M": "#42A5F5",
    "L2/F": "#C62828",
    "L2/M": "#EF9A9A",
}

# ══════════════════════════════════════════════════════════════════
# 1. Summary table
# ══════════════════════════════════════════════════════════════════
print("\n── Summary table (F1_norm per phoneme per group) ──")

rows = []
for phoneme in sorted(df["phoneme"].unique()):
    for group in GROUPS:
        sub = df[(df["phoneme"] == phoneme) & (df["group"] == group)]["f1_norm"]
        if len(sub) < 3:
            continue
        rows.append({
            "phoneme": phoneme,
            "group":   group,
            "n":       len(sub),
            "mean":    sub.mean(),
            "median":  sub.median(),
            "std":     sub.std(),
            "IQR":     sub.quantile(0.75) - sub.quantile(0.25),
            "CV":      sub.std() / abs(sub.mean()) if sub.mean() != 0 else np.nan,
        })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(f"{OUTPUT_DIR}/summary_table.csv", index=False)
print(summary_df.round(3).to_string())

# ══════════════════════════════════════════════════════════════════
# 2. Variance decomposition for F1
# ══════════════════════════════════════════════════════════════════
print("\n── Variance decomposition (F1_norm) ──")

var_rows = []
for phoneme in sorted(df["phoneme"].unique()):
    sub = df[df["phoneme"] == phoneme]

    # total variance
    var_total = sub["f1_norm"].var()

    # inter-speaker: variance of per-speaker means
    spk_means  = sub.groupby("speaker_id")["f1_norm"].mean()
    var_inter  = spk_means.var()

    # intra-speaker: mean of per-speaker variances
    spk_vars   = sub.groupby("speaker_id")["f1_norm"].var()
    var_intra  = spk_vars.mean()

    # residual
    var_resid  = max(var_total - var_inter - var_intra, 0)

    var_rows.append({
        "phoneme":    phoneme,
        "var_total":  var_total,
        "var_inter":  var_inter,
        "var_intra":  var_intra,
        "var_resid":  var_resid,
        "pct_inter":  var_inter / var_total * 100,
        "pct_intra":  var_intra / var_total * 100,
        "pct_resid":  var_resid / var_total * 100,
    })

var_df = pd.DataFrame(var_rows)
var_df.to_csv(f"{OUTPUT_DIR}/variance_decomposition.csv", index=False)
print(var_df.round(3).to_string())

# ══════════════════════════════════════════════════════════════════
# 3. Vowel chart (F1 vs F2, IPA convention)
# ══════════════════════════════════════════════════════════════════

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a confidence ellipse for points x, y."""
    if len(x) < 3:
        return
    cov  = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])
    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width, height=height,
        angle=angle, **kwargs
    )
    ax.add_patch(ellipse)

fig, ax = plt.subplots(figsize=(10, 8))

for group in GROUPS:
    sub_g = df[df["group"] == group]
    for phoneme in sorted(df["phoneme"].unique()):
        sub = sub_g[sub_g["phoneme"] == phoneme]
        if len(sub) < 5:
            continue
        x = sub["f2_norm"].values
        y = sub["f1_norm"].values
        # centroid
        ax.scatter(x.mean(), y.mean(),
                   color=COLORS[group], s=60, zorder=5)
        ax.annotate(
            phoneme,
            (x.mean(), y.mean()),
            fontsize=8, ha="center", va="bottom"
        )
        # confidence ellipse
        confidence_ellipse(
            x, y, ax,
            n_std=1.96,
            edgecolor=COLORS[group],
            facecolor=COLORS[group],
            alpha=0.08,
            linewidth=1.2
        )

# IPA convention: invert both axes
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("F2 (Lobanov normalised)", fontsize=12)
ax.set_ylabel("F1 (Lobanov normalised)", fontsize=12)
ax.set_title("Vowel chart — per-phoneme centroids by speaker group\n"
             "with 95% confidence ellipses", fontsize=13, fontweight="bold")

legend = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUPS]
ax.legend(handles=legend, fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_vowel_chart.png", dpi=150)
plt.close()
print("\nSaved plot_vowel_chart.png")

# ══════════════════════════════════════════════════════════════════
# 4. Box plots of F1 and F2 per phoneme
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

for ax, formant, col in zip(axes, ["f1_norm", "f2_norm"], ["F1", "F2"]):
    phonemes = sorted(df["phoneme"].unique())
    positions = []
    data      = []
    colors    = []
    xticks    = []
    xticklabels = []

    n_groups = len(GROUPS)
    gap      = 0.5
    width    = 0.6

    for i, phoneme in enumerate(phonemes):
        base_pos = i * (n_groups * width + gap)
        xticks.append(base_pos + (n_groups * width) / 2)
        xticklabels.append(phoneme)
        for j, group in enumerate(GROUPS):
            sub = df[(df["phoneme"] == phoneme) &
                     (df["group"] == group)][formant].dropna()
            if len(sub) < 3:
                continue
            pos = base_pos + j * width
            positions.append(pos)
            data.append(sub.values)
            colors.append(COLORS[group])

    bp = ax.boxplot(data, positions=positions, widths=width * 0.8,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel(f"{col} (Lobanov normalised)", fontsize=11)
    ax.set_title(f"{col} per phoneme by speaker group", fontsize=12,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

legend = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUPS]
axes[0].legend(handles=legend, fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_boxplots.png", dpi=150)
plt.close()
print("Saved plot_boxplots.png")

# ══════════════════════════════════════════════════════════════════
# 5. Violin plots of intra-speaker variability
# ══════════════════════════════════════════════════════════════════
# select 4 most frequent vowels
top_vowels = (
    df.groupby("phoneme").size()
    .sort_values(ascending=False)
    .head(4).index.tolist()
)

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)

for ax, phoneme in zip(axes, top_vowels):
    speakers = sorted(df[df["phoneme"] == phoneme]["speaker_id"].unique())
    data     = [
        df[(df["phoneme"] == phoneme) &
           (df["speaker_id"] == spk)]["f1_norm"].dropna().values
        for spk in speakers
    ]
    data = [d for d in data if len(d) >= 3]

    if data:
        parts = ax.violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#42A5F5")
            pc.set_alpha(0.7)

    ax.set_title(f"/{phoneme}/", fontsize=12, fontweight="bold")
    ax.set_xlabel("Speaker", fontsize=9)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(
        [f"S{i+1}" for i in range(len(data))],
        fontsize=7, rotation=45
    )
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("F1 (Lobanov normalised)", fontsize=11)
fig.suptitle("Intra-speaker variability across repetitions\n"
             "(top 4 most frequent vowels)",
             fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_violin.png", dpi=150)
plt.close()
print("Saved plot_violin.png")

print("\nAll descriptive statistics done!")