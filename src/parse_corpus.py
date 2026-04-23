"""
Stage 1 — parse_corpus.py
Reads all TextGrid files, metadata_RUFR.csv and RUFRcorr.csv.
Outputs data/phonemes.csv with one row per phoneme token.

Difference from numerical stability version:
- uses the 'phones' tier (not 'words') → phoneme-level tokens
- keeps sentence_id and repetition from RUFRcorr mapping
"""

import os
import re
import pandas as pd
import yaml

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

CORPUS_PATH    = params["corpus"]["path"]
METADATA_PATH  = os.path.join(CORPUS_PATH, "metadata_RUFR.csv")
RUFRCORR_PATH  = os.path.join(CORPUS_PATH, "RUFRcorr.csv")
TEXTGRIDS_PATH = os.path.join(
    CORPUS_PATH, "wav_et_textgrids", "FRcorp_textgrids_only"
)
OUTPUT_PATH = "data/phonemes.csv"

# ── load speaker metadata ──────────────────────────────────────────
metadata = pd.read_csv(METADATA_PATH, sep=";")
metadata.columns = metadata.columns.str.strip()
metadata["spk"] = metadata["spk"].str.strip().str.upper()
spk_info = metadata.set_index("spk").to_dict(orient="index")

# ── load word/sentence mapping ─────────────────────────────────────
rufrcorr = pd.read_csv(RUFRCORR_PATH, sep="\t")
rufrcorr = rufrcorr.dropna(subset=["Word"])

# build: frcorp_number -> {word, repetition}
frcorp_to_word = {}
for _, row in rufrcorr.iterrows():
    word = row["Word"]
    for rep_idx, col in enumerate(["occ.1","occ.2","occ.3",
                                    "occ.4","occ.5","occ.6"], start=1):
        frcorp_num = int(row[col])
        frcorp_to_word[frcorp_num] = {
            "word":       word,
            "repetition": rep_idx
        }

# ── parse phones tier from a TextGrid ─────────────────────────────
def parse_phones_tier(path):
    """
    Returns a list of dicts, one per non-empty phoneme interval.
    Each dict has: phoneme, onset, offset, duration_ms
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # extract the phones tier block
    phones_match = re.search(
        r'name = "phones".*?intervals: size = \d+(.*?)(?=item \[\d+\]:|$)',
        content,
        re.DOTALL,
    )
    if not phones_match:
        return []

    intervals = re.findall(
        r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"',
        phones_match.group(1),
    )

    tokens = []
    for xmin, xmax, text in intervals:
        text = text.strip()
        if text == "":
            continue
        xmin, xmax = float(xmin), float(xmax)
        tokens.append({
            "phoneme":     text,
            "onset":       xmin,
            "offset":      xmax,
            "duration_ms": round((xmax - xmin) * 1000, 2),
        })
    return tokens


# ── walk all speaker folders ───────────────────────────────────────
rows = []
missing_metadata = []

for spk_folder in sorted(os.listdir(TEXTGRIDS_PATH)):
    spk_id   = spk_folder.upper()
    spk_path = os.path.join(TEXTGRIDS_PATH, spk_folder)
    if not os.path.isdir(spk_path):
        continue

    info = spk_info.get(spk_id, None)
    if info is None:
        missing_metadata.append(spk_id)
        continue

    l1_status = "L1" if info["L1"] == "fr" else "L2"
    gender    = info["Gender"].strip()

    for filename in sorted(os.listdir(spk_path)):
        if not filename.endswith(".TextGrid"):
            continue

        match = re.match(
            r"\w+_(fr|rus|fra)_list\d+_FRcorp(\d+)\.TextGrid",
            filename,
            re.IGNORECASE,
        )
        if not match:
            print(f"  WARNING: unexpected filename: {filename}")
            continue

        frcorp_num = int(match.group(2))

        # get word/repetition info if this is a target sentence
        word_info = frcorp_to_word.get(frcorp_num, None)

        tg_path = os.path.join(spk_path, filename)
        phoneme_tokens = parse_phones_tier(tg_path)

        wav_file = os.path.join(
            TEXTGRIDS_PATH, spk_folder,
            filename.replace(".TextGrid", ".wav")
        )

        for tok in phoneme_tokens:
            rows.append({
                "speaker_id":  spk_id,
                "l1_status":   l1_status,
                "gender":      gender,
                "frcorp_num":  frcorp_num,
                "word":        word_info["word"] if word_info else None,
                "repetition":  word_info["repetition"] if word_info else None,
                "phoneme":     tok["phoneme"],
                "onset":       tok["onset"],
                "offset":      tok["offset"],
                "duration_ms": tok["duration_ms"],
                "wav_file":    wav_file,
            })



# ── save output ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)

# ── clean annotation artefacts ─────────────────────────────────────
# remove tokens with 'ding' in the label (annotation errors)
before = len(df)
df = df[~df["phoneme"].str.contains("ding", na=False)]
print(f"Removed {before - len(df)} artefact tokens (ding...)")


df.to_csv(OUTPUT_PATH, index=False)

print(f"Done. {len(df)} phoneme tokens saved to {OUTPUT_PATH}")
print(f"Speakers:  {df['speaker_id'].nunique()}")
print(f"Phonemes:  {df['phoneme'].nunique()} unique — {sorted(df['phoneme'].unique())}")
print(f"Sentences: {df['frcorp_num'].nunique()} unique FRcorp numbers")
print()
print(df.groupby(['l1_status','gender'])['speaker_id'].nunique())
