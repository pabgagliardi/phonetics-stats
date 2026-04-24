"""
Stage 2 — extract_acoustics.py
For each phoneme token in data/phonemes.csv, extracts:
  - F1, F2, F3 at midpoint (LPC via parselmouth)
  - f0 mean (autocorrelation, voiced segments only)
  - duration (from TextGrid boundaries)
  - spectral centre of gravity (fricatives only)

LPC parameters follow the lab spec:
  - max_formant: 5000 Hz for female, 4500 Hz for male
  - n_formants: 5

Outputs: data/features_acoustic.csv
"""

import os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

MAX_FORMANT_F  = params["acoustic"]["max_formant_female"]
MAX_FORMANT_M  = params["acoustic"]["max_formant_male"]
N_FORMANTS     = params["acoustic"]["n_formants"]
INPUT_CSV      = "data/phonemes.csv"
OUTPUT_CSV     = "data/features_acoustic.csv"

# ── French oral vowels (for which we extract formants) ─────────────
VOWELS = {
    "a", "e", "i", "o", "u", "y",
    "ø", "œ", "ɛ", "ɑ", "ə",
    "ɑ̃", "ɛ̃", "œ̃"
}

# ── fricatives (for spectral centre of gravity) ────────────────────
FRICATIVES = {"f", "s", "z", "ʃ", "ʒ", "v"}

# ── load phoneme tokens ────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Processing {len(df)} phoneme tokens...")

# ── helper: extract formants at a given time ───────────────────────
def get_formants(sound, time, max_formant):
    """
    Returns F1, F2, F3 at a given time point using LPC Burg method.
    Returns NaN if formant tracker fails.
    """
    formant = call(
        sound, "To Formant (burg)", 0.0, N_FORMANTS,
        max_formant, 0.025, 50
    )
    f1 = call(formant, "Get value at time", 1, time, "Hertz", "Linear")
    f2 = call(formant, "Get value at time", 2, time, "Hertz", "Linear")
    f3 = call(formant, "Get value at time", 3, time, "Hertz", "Linear")
    return f1, f2, f3


# ── helper: extract f0 mean ────────────────────────────────────────
def get_f0(sound):
    """
    Returns mean f0 over the segment using autocorrelation.
    Returns NaN for unvoiced segments.
    """
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    return f0 if f0 != 0 else np.nan


# ── helper: spectral centre of gravity ────────────────────────────
def get_scg(sound):
    """Returns spectral centre of gravity in Hz."""
    spectrum = call(sound, "To Spectrum", True)
    scg = call(spectrum, "Get centre of gravity", 2)
    return scg


# ── main extraction loop ───────────────────────────────────────────
rows = []
missing_wav = 0
errors = 0

# cache loaded sounds to avoid reloading same wav file repeatedly
current_wav = None
current_sound = None

for idx, row in df.iterrows():
    wav_path = row["wav_file"]
    onset    = row["onset"]
    offset   = row["offset"]
    phoneme  = row["phoneme"]
    gender   = row["gender"]

    # select max_formant based on gender
    max_formant = MAX_FORMANT_F if gender == "f" else MAX_FORMANT_M

    # load wav (use cache to avoid reloading same file)
    if wav_path != current_wav:
        if not os.path.exists(wav_path):
            missing_wav += 1
            continue
        try:
            current_sound = parselmouth.Sound(wav_path)
            current_wav   = wav_path
        except Exception as e:
            errors += 1
            continue

    # extract segment
    try:
        segment = current_sound.extract_part(
            from_time=onset,
            to_time=offset,
            preserve_times=True
        )
    except Exception:
        errors += 1
        continue

    # duration (from TextGrid boundaries)
    duration_ms = row["duration_ms"]

    # midpoint time
    midpoint = onset + (offset - onset) / 2

    # initialise all features as NaN
    f1 = f2 = f3 = f0 = scg = np.nan

    # formants — vowels only
    base_phoneme = phoneme.rstrip("ːˑ̥̰̃")  # strip diacritics for matching
    is_vowel = phoneme in VOWELS or base_phoneme in VOWELS

    if is_vowel and duration_ms > 20:  # skip very short segments
        try:
            f1, f2, f3 = get_formants(segment, midpoint, max_formant)
        except Exception:
            pass  # leave as NaN

    # f0 — all voiced segments
    try:
        f0 = get_f0(segment)
    except Exception:
        pass

    # spectral centre of gravity — fricatives only
    is_fricative = phoneme in FRICATIVES or base_phoneme in FRICATIVES
    if is_fricative:
        try:
            scg = get_scg(segment)
        except Exception:
            pass

    rows.append({
        "speaker_id":  row["speaker_id"],
        "l1_status":   row["l1_status"],
        "gender":      row["gender"],
        "frcorp_num":  row["frcorp_num"],
        "word":        row["word"],
        "repetition":  row["repetition"],
        "phoneme":     phoneme,
        "onset":       onset,
        "offset":      offset,
        "duration_ms": duration_ms,
        "is_vowel":    is_vowel,
        "f1":          f1,
        "f2":          f2,
        "f3":          f3,
        "f0":          f0,
        "scg":         scg,
    })

    # progress every 500 tokens
    if (idx + 1) % 500 == 0:
        print(f"  {idx+1}/{len(df)} tokens processed...")

# ── save output ────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
out_df.to_csv(OUTPUT_CSV, index=False)

# ── report ─────────────────────────────────────────────────────────
print(f"\nDone. {len(out_df)} tokens saved to {OUTPUT_CSV}")
print(f"Missing WAV files: {missing_wav}")
print(f"Errors: {errors}")

vowels_df = out_df[out_df["is_vowel"]]
print(f"\nVowel tokens: {len(vowels_df)}")
print(f"F1 missing: {vowels_df['f1'].isna().sum()} "
      f"({vowels_df['f1'].isna().mean()*100:.1f}%)")
print(f"F2 missing: {vowels_df['f2'].isna().sum()} "
      f"({vowels_df['f2'].isna().mean()*100:.1f}%)")
print(f"\nF1 stats (vowels):")
print(vowels_df.groupby("phoneme")["f1"].describe().round(1))