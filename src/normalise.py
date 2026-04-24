"""
Stage 3 — normalise.py
Applies Lobanov normalisation to F1/F2 for each speaker.

Lobanov formula (per speaker, per formant):
    F*_j,s = (F_j,s - mean(F_j,s)) / std(F_j,s)

where mean and std are computed over ALL vowel tokens
produced by speaker s (not per phoneme).

Outputs: data/features_acoustic_norm.csv
"""

import os
import numpy as np
import pandas as pd
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_CSV  = "data/features_acoustic.csv"
OUTPUT_CSV = "data/features_acoustic_norm.csv"

# ── load acoustic features ─────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} tokens")

# ── keep only vowel tokens for normalisation ───────────────────────
# as per the lab spec: "consult only vowel tokens for this computation"
vowels_df = df[df["is_vowel"] == True].copy()
print(f"Vowel tokens: {len(vowels_df)}")

# ── drop the 2 tokens with missing F1/F2 ──────────────────────────
vowels_df = vowels_df.dropna(subset=["f1", "f2"])
print(f"Vowel tokens after dropping missing F1/F2: {len(vowels_df)}")

# ── Lobanov normalisation ──────────────────────────────────────────
# compute per-speaker mean and std for F1 and F2
speaker_stats = vowels_df.groupby("speaker_id")[["f1", "f2"]].agg(
    ["mean", "std"]
)
speaker_stats.columns = ["f1_mean", "f1_std", "f2_mean", "f2_std"]

print("\nPer-speaker F1 mean and std (before normalisation):")
print(speaker_stats.round(1).to_string())

# merge stats back into vowels_df
vowels_norm = vowels_df.merge(
    speaker_stats, on="speaker_id", how="left"
)

# apply normalisation
vowels_norm["f1_norm"] = (
    (vowels_norm["f1"] - vowels_norm["f1_mean"]) / vowels_norm["f1_std"]
)
vowels_norm["f2_norm"] = (
    (vowels_norm["f2"] - vowels_norm["f2_mean"]) / vowels_norm["f2_std"]
)

# also normalise F3 where available
vowels_norm["f3_norm"] = np.where(
    vowels_norm["f3"].notna() & (vowels_norm["f3"] > 0),
    (vowels_norm["f3"] - vowels_norm["f1_mean"]) / vowels_norm["f1_std"],
    np.nan
)

# drop helper columns
vowels_norm = vowels_norm.drop(
    columns=["f1_mean", "f1_std", "f2_mean", "f2_std"]
)

# ── sanity check ───────────────────────────────────────────────────
# after normalisation, per-speaker mean should be ~0 and std ~1
check = vowels_norm.groupby("speaker_id")[["f1_norm", "f2_norm"]].agg(
    ["mean", "std"]
)
print("\nSanity check — per-speaker F1_norm mean and std (should be ~0 and ~1):")
print(check.round(3).to_string())

# ── save output ────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
vowels_norm.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. {len(vowels_norm)} normalised vowel tokens saved to {OUTPUT_CSV}")

# ── summary: mean F1_norm per phoneme ─────────────────────────────
print("\nMean normalised F1 per phoneme (sorted high→low, "
      "should reflect vowel height):")
summary = (
    vowels_norm.groupby("phoneme")["f1_norm"]
    .agg(["mean", "count"])
    .sort_values("mean")
    .round(3)
)
print(summary[summary["count"] >= 10].to_string())