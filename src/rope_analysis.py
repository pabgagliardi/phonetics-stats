"""
Stage 13 — rope_analysis.py
Section 8: Confidence Intervals and ROPE analysis.

8.1 CIs on acoustic contrasts (from LME fixed effects)
8.2 CIs on neural contrasts (bootstrap, speaker-level)
8.3 ROPE definition
8.4 ROPE classification
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
FRENCH_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 8.1 CIs on acoustic contrasts
# from lme4 output — we extract beta_L2 and compute CIs
# using the standard error from the LME output
# ══════════════════════════════════════════════════════════════════
print("\n── 8.1 Acoustic CIs ─────────────────────────────────────────")

# We use the beta_L2 estimates from lme4 output
# and compute approximate 95% CIs as beta ± 1.96 * SE
# We need to re-extract SEs — add this to mixed_effects.R
# For now we use the values from the output file

# Load acoustic normalised data to compute bootstrap CIs directly
df_ac = pd.read_csv("data/features_acoustic_norm.csv")
df_ac = df_ac[df_ac["phoneme"].isin(FRENCH_VOWELS)].dropna(
    subset=["f1_norm", "f2_norm"]
).copy()

speakers = sorted(df_ac["speaker_id"].unique())
N_BOOT   = 2000

print("Computing bootstrap CIs on acoustic L1/L2 contrast...")

ac_ci_rows = []
for phoneme in FRENCH_VOWELS:
    sub = df_ac[df_ac["phoneme"] == phoneme]
    if len(sub) < 10:
        continue

    for formant, col in [("F1", "f1_norm"), ("F2", "f2_norm")]:
        l1_obs = sub[sub["l1_status"] == "L1"][col].mean()
        l2_obs = sub[sub["l1_status"] == "L2"][col].mean()
        obs_diff = l1_obs - l2_obs

        # bootstrap at speaker level
        boot_diffs = []
        for _ in range(N_BOOT):
            boot_spks = np.random.choice(
                speakers, len(speakers), replace=True
            )
            boot_sub = pd.concat([
                sub[sub["speaker_id"] == s]
                for s in boot_spks
            ])
            b_l1 = boot_sub[
                boot_sub["l1_status"] == "L1"
            ][col].mean()
            b_l2 = boot_sub[
                boot_sub["l1_status"] == "L2"
            ][col].mean()
            if not np.isnan(b_l1) and not np.isnan(b_l2):
                boot_diffs.append(b_l1 - b_l2)

        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

        ac_ci_rows.append({
            "phoneme":   phoneme,
            "formant":   formant,
            "obs_diff":  round(obs_diff, 4),
            "ci_lo":     round(ci_lo, 4),
            "ci_hi":     round(ci_hi, 4),
        })
        print(f"  /{phoneme}/ {formant}: "
              f"diff={obs_diff:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

ac_ci_df = pd.DataFrame(ac_ci_rows)
ac_ci_df.to_csv(f"{OUTPUT_DIR}/acoustic_cis.csv", index=False)

# ── forest plot — acoustic ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

for ax, formant in zip(axes, ["F1", "F2"]):
    sub     = ac_ci_df[ac_ci_df["formant"] == formant]
    ph_list = sub["phoneme"].tolist()
    diffs   = sub["obs_diff"].tolist()
    lo      = sub["ci_lo"].tolist()
    hi      = sub["ci_hi"].tolist()
    y       = np.arange(len(ph_list))

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    for i in range(len(ph_list)):
        color = "#C62828" if (lo[i] > 0 or hi[i] < 0) else "#90A4AE"
        ax.plot([lo[i], hi[i]], [y[i], y[i]],
                color=color, linewidth=2)
        ax.scatter(diffs[i], y[i],
                   color=color, s=50, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(ph_list, fontsize=10)
    ax.set_xlabel(f"L1 − L2 difference ({formant}_norm)",
                  fontsize=10)
    ax.set_title(f"95% CI on L1/L2 contrast — {formant}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

legend = [
    mpatches.Patch(color="#C62828", label="CI excludes zero"),
    mpatches.Patch(color="#90A4AE", label="CI includes zero"),
]
axes[0].legend(handles=legend, fontsize=9)
plt.suptitle("Forest plot — acoustic L1/L2 contrasts\n"
             "with 95% bootstrap CIs (speaker-level resampling)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_forest_acoustic.png", dpi=150)
plt.close()
print("Saved plot_forest_acoustic.png")


# ══════════════════════════════════════════════════════════════════
# 8.2 CIs on neural contrasts
# ══════════════════════════════════════════════════════════════════
print("\n── 8.2 Neural CIs ───────────────────────────────────────────")

meta = pd.read_csv("data/phonemes.csv")
meta = meta[meta["phoneme"].isin(FRENCH_VOWELS)].copy()
meta = meta.reset_index(drop=True)

NEURAL_MODELS = {
    "Whisper L20": "data/features_whisper_layer20_pca50.npz",
    "XLS-R L20":   "data/features_xlsr_layer20_pca50.npz",
}

# precompute per-speaker per-phoneme means
print("Precomputing per-speaker neural means...")
neural_spk_means = {}
for model_name, path in NEURAL_MODELS.items():
    data      = np.load(path)
    X         = data["features"]
    token_ids = data["token_ids"]
    neural_spk_means[model_name] = {}

    for spk in speakers:
        for ph in FRENCH_VOWELS:
            mask = np.array([
                i < len(meta) and
                meta.iloc[i]["phoneme"] == ph and
                meta.iloc[i]["speaker_id"] == spk
                for i in range(len(token_ids))
            ])
            if mask.sum() > 0:
                neural_spk_means[model_name][(spk, ph)] = \
                    X[mask].mean(axis=0)

# also compute intra-speaker distances for ROPE
print("Computing intra-speaker distances for ROPE...")
intra_dists = {m: [] for m in NEURAL_MODELS}

for model_name, path in NEURAL_MODELS.items():
    data      = np.load(path)
    X         = data["features"]
    token_ids = data["token_ids"]

    for spk in speakers:
        for ph in FRENCH_VOWELS:
            mask = np.array([
                i < len(meta) and
                meta.iloc[i]["phoneme"] == ph and
                meta.iloc[i]["speaker_id"] == spk
                for i in range(len(token_ids))
            ])
            if mask.sum() < 2:
                continue
            X_spk = X[mask]
            # pairwise cosine distances within speaker
            n = len(X_spk)
            for i in range(n):
                for j in range(i + 1, min(i + 5, n)):
                    d = cosine(X_spk[i], X_spk[j])
                    intra_dists[model_name].append(d)

neural_rope = {}
for model_name in NEURAL_MODELS:
    delta0 = np.mean(intra_dists[model_name])
    neural_rope[model_name] = delta0
    print(f"  {model_name} ROPE delta0 "
          f"(mean intra-speaker dist): {delta0:.4f}")

# bootstrap CIs on cosine distance
print("\nComputing bootstrap CIs on neural L1/L2 contrast...")
neural_ci_rows = []

for model_name in NEURAL_MODELS:
    for phoneme in FRENCH_VOWELS:
        # observed cosine distance
        l1_spks = [
            s for s in speakers
            if (s, phoneme) in neural_spk_means[model_name] and
            meta[meta["speaker_id"] == s]["l1_status"].values[0] == "L1"
        ]
        l2_spks = [
            s for s in speakers
            if (s, phoneme) in neural_spk_means[model_name] and
            meta[meta["speaker_id"] == s]["l1_status"].values[0] == "L2"
        ]

        if len(l1_spks) < 2 or len(l2_spks) < 2:
            continue

        l1_vecs = np.array([
            neural_spk_means[model_name][(s, phoneme)]
            for s in l1_spks
        ])
        l2_vecs = np.array([
            neural_spk_means[model_name][(s, phoneme)]
            for s in l2_spks
        ])

        obs_dist = cosine(
            l1_vecs.mean(axis=0), l2_vecs.mean(axis=0)
        )

        # bootstrap
        boot_dists = []
        all_spks_ph = l1_spks + l2_spks
        for _ in range(N_BOOT):
            boot_l1 = np.random.choice(
                l1_spks, len(l1_spks), replace=True
            )
            boot_l2 = np.random.choice(
                l2_spks, len(l2_spks), replace=True
            )
            c1 = np.mean([
                neural_spk_means[model_name][(s, phoneme)]
                for s in boot_l1
            ], axis=0)
            c2 = np.mean([
                neural_spk_means[model_name][(s, phoneme)]
                for s in boot_l2
            ], axis=0)
            boot_dists.append(cosine(c1, c2))

        ci_lo, ci_hi = np.percentile(boot_dists, [2.5, 97.5])

        neural_ci_rows.append({
            "model":    model_name,
            "phoneme":  phoneme,
            "obs_dist": round(obs_dist, 4),
            "ci_lo":    round(ci_lo, 4),
            "ci_hi":    round(ci_hi, 4),
        })
        print(f"  {model_name} /{phoneme}/: "
              f"d={obs_dist:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

neural_ci_df = pd.DataFrame(neural_ci_rows)
neural_ci_df.to_csv(f"{OUTPUT_DIR}/neural_cis.csv", index=False)

# ── forest plot — neural ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

for ax, model_name in zip(axes, NEURAL_MODELS.keys()):
    sub     = neural_ci_df[neural_ci_df["model"] == model_name]
    ph_list = sub["phoneme"].tolist()
    dists   = sub["obs_dist"].tolist()
    lo      = sub["ci_lo"].tolist()
    hi      = sub["ci_hi"].tolist()
    y       = np.arange(len(ph_list))
    rope    = neural_rope[model_name]

    # shade ROPE region
    ax.axvspan(0, rope, alpha=0.15,
               color="green", label=f"ROPE [0, {rope:.3f}]")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")

    for i in range(len(ph_list)):
        if hi[i] < rope:
            color = "#2E7D32"   # equivalent
        elif lo[i] > rope:
            color = "#C62828"   # non-equivalent
        else:
            color = "#FF8F00"   # indeterminate
        ax.plot([lo[i], hi[i]], [y[i], y[i]],
                color=color, linewidth=2)
        ax.scatter(dists[i], y[i],
                   color=color, s=50, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(ph_list, fontsize=10)
    ax.set_xlabel("Cosine distance (L1 vs L2 centroid)",
                  fontsize=10)
    ax.set_title(f"95% CI — {model_name}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("Forest plot — neural L1/L2 contrasts\n"
             "with 95% bootstrap CIs (speaker-level resampling)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_forest_neural.png", dpi=150)
plt.close()
print("Saved plot_forest_neural.png")


# ══════════════════════════════════════════════════════════════════
# 8.3 + 8.4 ROPE definition and classification
# ══════════════════════════════════════════════════════════════════
print("\n── 8.3-8.4 ROPE classification ──────────────────────────────")

# Acoustic ROPE
# JND for formant frequencies ≈ 3-5% of formant value
# For Lobanov-normalised values, we convert:
# typical F1 ≈ 500 Hz, JND ≈ 20 Hz → 20/500 = 0.04 Lobanov units
# We use ±0.04 as the acoustic ROPE
ACOUSTIC_ROPE = 0.04
print(f"Acoustic ROPE: [{-ACOUSTIC_ROPE}, {ACOUSTIC_ROPE}] "
      f"(Lobanov units, ≈ ±20 Hz)")

rope_rows = []

# acoustic ROPE classification
for _, row in ac_ci_df.iterrows():
    lo = row["ci_lo"]
    hi = row["ci_hi"]

    if hi < ACOUSTIC_ROPE and lo > -ACOUSTIC_ROPE:
        classification = "Equivalent"
    elif lo > ACOUSTIC_ROPE or hi < -ACOUSTIC_ROPE:
        classification = "Non-equivalent"
    else:
        classification = "Indeterminate"

    rope_rows.append({
        "representation": f"Acoustic {row['formant']}",
        "phoneme":        row["phoneme"],
        "point_estimate": row["obs_diff"],
        "ci_lo":          lo,
        "ci_hi":          hi,
        "rope_lo":        -ACOUSTIC_ROPE,
        "rope_hi":        ACOUSTIC_ROPE,
        "classification": classification,
    })

# neural ROPE classification
for _, row in neural_ci_df.iterrows():
    model_name = row["model"]
    rope_hi    = neural_rope[model_name]
    lo         = row["ci_lo"]
    hi         = row["ci_hi"]

    if hi < rope_hi:
        classification = "Equivalent"
    elif lo > rope_hi:
        classification = "Non-equivalent"
    else:
        classification = "Indeterminate"

    rope_rows.append({
        "representation": model_name,
        "phoneme":        row["phoneme"],
        "point_estimate": row["obs_dist"],
        "ci_lo":          lo,
        "ci_hi":          hi,
        "rope_lo":        0,
        "rope_hi":        round(rope_hi, 4),
        "classification": classification,
    })

rope_df = pd.DataFrame(rope_rows)
rope_df.to_csv(f"{OUTPUT_DIR}/rope_classification.csv", index=False)

# ── summary table ──────────────────────────────────────────────────
print("\nROPE classification summary:")
print(rope_df[["representation", "phoneme",
               "point_estimate", "ci_lo", "ci_hi",
               "classification"]].to_string())

# ── proportion non-equivalent per representation ───────────────────
print("\nProportion non-equivalent per representation:")
props = rope_df.groupby("representation")["classification"].apply(
    lambda x: (x == "Non-equivalent").mean()
).round(3)
print(props.to_string())

# ── visualise ROPE classification ──────────────────────────────────
reps      = rope_df["representation"].unique()
fig, axes = plt.subplots(
    1, len(reps), figsize=(5 * len(reps), 7), sharey=True
)
if len(reps) == 1:
    axes = [axes]

colors_map = {
    "Equivalent":     "#2E7D32",
    "Non-equivalent": "#C62828",
    "Indeterminate":  "#FF8F00",
}

for ax, rep in zip(axes, reps):
    sub     = rope_df[rope_df["representation"] == rep]
    ph_list = sub["phoneme"].tolist()
    pts     = sub["point_estimate"].tolist()
    lo      = sub["ci_lo"].tolist()
    hi      = sub["ci_hi"].tolist()
    cls     = sub["classification"].tolist()
    rope_lo = sub["rope_lo"].iloc[0]
    rope_hi = sub["rope_hi"].iloc[0]
    y       = np.arange(len(ph_list))

    # shade ROPE
    ax.axvspan(rope_lo, rope_hi, alpha=0.15,
               color="green", label="ROPE")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")

    for i in range(len(ph_list)):
        color = colors_map[cls[i]]
        ax.plot([lo[i], hi[i]], [y[i], y[i]],
                color=color, linewidth=2.5)
        ax.scatter(pts[i], y[i],
                   color=color, s=60, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(ph_list, fontsize=9)
    ax.set_title(rep, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7)

legend_patches = [
    mpatches.Patch(color=c, label=l)
    for l, c in colors_map.items()
]
axes[0].legend(handles=legend_patches, fontsize=8)

plt.suptitle("ROPE classification per phoneme per representation",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_rope_classification.png", dpi=150)
plt.close()
print("Saved plot_rope_classification.png")
print("\nDone.")