"""
Stage 6 — extract_neural_xlsr.py (optimised with checkpointing)
Caches XLS-R encoder outputs per WAV file so each file
is only processed ONCE, then slices frames per phoneme.

Optimisations:
- processes one WAV file at a time (encoder runs once per file)
- saves checkpoints every 100 WAV files (safe to interrupt)

Experiments with three layers:
- lower third (layer 4)
- middle third (layer 10)
- upper third (layer 20)
"""

import os
import numpy as np
import pandas as pd
import torch
import yaml
import time
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.io import wavfile

# ── load parameters ────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_NAME      = params["xlsr"]["model"]
LAYERS          = params["xlsr"]["layers"]
INPUT_CSV       = "data/phonemes.csv"
OUTPUT_DIR      = "data"
CHECKPOINT_PATH = "data/xlsr_checkpoint.npz"
XLSR_SR         = 16000
XLSR_HOP_SEC    = 0.02  # wav2vec2 frame hop ~20ms

# ── load model ─────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2Model.from_pretrained(
    MODEL_NAME, output_hidden_states=True
)
model.eval()
print("Model loaded.\n")

# ── load phoneme tokens ────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Processing {len(df)} phoneme tokens across "
      f"{df['wav_file'].nunique()} unique WAV files...")
print(f"Extracting layers: {LAYERS}\n")

# ── helper: load full wav ──────────────────────────────────────────
def load_wav(wav_path):
    sample_rate, audio = wavfile.read(wav_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != XLSR_SR:
        target_len = int(len(audio) * XLSR_SR / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio), target_len),
            np.arange(len(audio)),
            audio
        )
    return audio.astype(np.float32)


# ── helper: overlapping frames ─────────────────────────────────────
def get_overlapping_steps(onset, offset, n_frames):
    frame_times = np.arange(n_frames) * XLSR_HOP_SEC
    mask = (frame_times >= onset) & (frame_times < offset)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        mid = (onset + offset) / 2
        idx = np.argmin(np.abs(frame_times - mid))
        indices = np.array([idx])
    return indices


# ── load checkpoint if exists ──────────────────────────────────────
processed_wavs     = set()
features_per_layer = {layer: {} for layer in LAYERS}

if os.path.exists(CHECKPOINT_PATH):
    print("Found checkpoint, resuming...")
    ckpt = np.load(CHECKPOINT_PATH, allow_pickle=True)
    processed_wavs = set(ckpt["processed_wavs"].tolist())
    for layer in LAYERS:
        key = f"layer_{layer}"
        if key in ckpt:
            features_per_layer[layer] = ckpt[key].item()
    print(f"  Resuming from {len(processed_wavs)} WAV files already done\n")


# ── main loop: one WAV file at a time ─────────────────────────────
wav_files  = df["wav_file"].unique()
n_wav      = len(wav_files)
start_time = time.time()

for wav_idx, wav_path in enumerate(wav_files):

    # skip already processed
    if wav_path in processed_wavs:
        continue

    if not os.path.exists(wav_path):
        processed_wavs.add(wav_path)
        continue

    tokens_in_wav = df[df["wav_file"] == wav_path]

    try:
        # load and encode ONCE per WAV file
        audio  = load_wav(wav_path)
        inputs = processor(
            audio, sampling_rate=XLSR_SR,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True
            )

        n_frames = outputs.hidden_states[0].shape[1]

        # slice per phoneme token
        for idx, row in tokens_in_wav.iterrows():
            indices = get_overlapping_steps(
                row["onset"], row["offset"], n_frames
            )
            for layer in LAYERS:
                hidden = outputs.hidden_states[layer]  # (1, T, D)
                vector = hidden[0, indices, :].mean(dim=0).numpy()
                features_per_layer[layer][idx] = vector

        processed_wavs.add(wav_path)

    except Exception as e:
        print(f"  WARNING: failed on {wav_path}: {e}")
        continue

    # save checkpoint every 100 WAV files
    if (wav_idx + 1) % 100 == 0:
        np.savez(
            CHECKPOINT_PATH,
            processed_wavs=np.array(list(processed_wavs)),
            **{f"layer_{l}": np.array(features_per_layer[l], dtype=object)
               for l in LAYERS}
        )
        elapsed   = time.time() - start_time
        done      = len(processed_wavs)
        remaining = elapsed / done * (n_wav - done) if done > 0 else 0
        print(f"  {wav_idx+1}/{n_wav} WAV files — "
              f"elapsed: {elapsed:.0f}s — "
              f"remaining: ~{remaining:.0f}s — "
              f"checkpoint saved")


# ── save final outputs per layer ───────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
token_ids = sorted(features_per_layer[LAYERS[0]].keys())

for layer in LAYERS:
    feats    = [features_per_layer[layer][i] for i in token_ids]
    X        = np.stack(feats).astype(np.float32)
    out_path = f"{OUTPUT_DIR}/features_xlsr_layer{layer}.npz"
    np.savez(out_path, features=X, token_ids=np.array(token_ids))
    print(f"Saved layer {layer}: shape={X.shape} → {out_path}")

# clean up checkpoint
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint removed.")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.0f}s")