"""
Stage 10 — rsm_mantel.py
Section 5.3: Representational Similarity Matrix (RSM) analysis.

For each representation type (acoustic, Whisper, XLS-R):
  - Compute a pairwise similarity matrix on a subsample of tokens
  - Compare RSMs using the Mantel test
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

OUTPUT_DIR    = "results"
SAMPLE_SIZE   = 300   # keep small for speed
FRENCH_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────
df_ac = pd.read_csv("data/features_acoustic_norm.csv")
df_ac = df_ac[df_ac["phoneme"].isin(FRENCH_VOWELS)].dropna(
    subset=["f1_norm", "f2_norm"]
).reset_index(drop=True)

# subsample
rng   = np.random.RandomState(42)
idx   = rng.choice(len(df_ac), min(SAMPLE_SIZE, len(df_ac)), replace=False)
idx   = np.sort(idx)
df_ac = df_ac.iloc[idx].reset_index(drop=True)
print(f"Subsample: {len(df_ac)} tokens")
print(f"Phonemes:  {sorted(df_ac['phoneme'].unique())}")

# ── acoustic RSM ───────────────────────────────────────────────────
X_ac  = df_ac[["f1_norm", "f2_norm"]].values
N     = len(X_ac)

RSM_ac = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist = np.linalg.norm(X_ac[i] - X_ac[j])
        RSM_ac[i, j] = -dist   # negative Euclidean distance
print("Acoustic RSM computed.")

# ── neural RSMs ────────────────────────────────────────────────────
# load full neural features and select same token indices
df_meta = pd.read_csv("data/phonemes.csv")
df_meta = df_meta[df_meta["phoneme"].isin(FRENCH_VOWELS)].reset_index(drop=True)

# match acoustic subsample to phonemes.csv rows
# we use phoneme + speaker + onset to match
def get_neural_vectors(npz_path, df_sub, df_meta):
    """
    For each row in df_sub (acoustic subsample),
    find the matching row in df_meta and return its neural vector.
    """
    data      = np.load(npz_path)
    X_full    = data["features"]
    token_ids = data["token_ids"]

    # build lookup: token_id -> vector
    vec_lookup = {tid: X_full[i] for i, tid in enumerate(token_ids)}

    vectors = []
    for _, row in df_sub.iterrows():
        # match on speaker + phoneme + onset
        match = df_meta[
            (df_meta["speaker_id"] == row["speaker_id"]) &
            (df_meta["phoneme"]    == row["phoneme"]) &
            (np.abs(df_meta["onset"] - row["onset"]) < 0.001)
        ]
        if len(match) == 0:
            vectors.append(np.zeros(X_full.shape[1]))
        else:
            tid = match.index[0]
            vectors.append(vec_lookup.get(tid, np.zeros(X_full.shape[1])))

    return np.array(vectors)


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)


def compute_neural_rsm(X):
    N = len(X)
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S[i, j] = cosine_sim(X[i], X[j])
    return S


print("Computing Whisper RSM...")
X_wh   = get_neural_vectors(
    "data/features_whisper_layer20_pca50.npz", df_ac, df_meta
)
RSM_wh = compute_neural_rsm(X_wh)
print("Whisper RSM computed.")

print("Computing XLS-R RSM...")
X_xl   = get_neural_vectors(
    "data/features_xlsr_layer20_pca50.npz", df_ac, df_meta
)
RSM_xl = compute_neural_rsm(X_xl)
print("XLS-R RSM computed.")

# ── Mantel test ────────────────────────────────────────────────────
def mantel_test(D1, D2, n_perm=999):
    idx    = np.triu_indices(D1.shape[0], k=1)
    v1, v2 = D1[idx], D2[idx]
    r_obs, _ = stats.spearmanr(v1, v2)
    count = 0
    for _ in range(n_perm):
        perm    = np.random.permutation(D1.shape[0])
        D1_perm = D1[np.ix_(perm, perm)]
        r_perm, _ = stats.spearmanr(D1_perm[idx], v2)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


print("\n── Mantel test results ───────────────────────────────────────")
pairs = [
    ("Acoustic", RSM_ac, "Whisper L20", RSM_wh),
    ("Acoustic", RSM_ac, "XLS-R L20",  RSM_xl),
    ("Whisper L20", RSM_wh, "XLS-R L20", RSM_xl),
]

mantel_rows = []
for n1, M1, n2, M2 in pairs:
    r, p = mantel_test(M1, M2)
    print(f"  {n1} vs {n2}: r={r:.3f}  p={p:.3f}")
    mantel_rows.append({"RSM1": n1, "RSM2": n2,
                        "mantel_r": round(r, 3),
                        "p_value":  round(p, 3)})

pd.DataFrame(mantel_rows).to_csv(
    f"{OUTPUT_DIR}/rsm_mantel_results.csv", index=False
)

# ── plot RSMs ──────────────────────────────────────────────────────
# sort tokens by phoneme for cleaner visualisation
sort_idx = df_ac["phoneme"].argsort().values
labels   = df_ac["phoneme"].values[sort_idx]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, RSM, title in zip(
    axes,
    [RSM_ac[np.ix_(sort_idx, sort_idx)],
     RSM_wh[np.ix_(sort_idx, sort_idx)],
     RSM_xl[np.ix_(sort_idx, sort_idx)]],
    ["Acoustic RSM", "Whisper L20 RSM", "XLS-R L20 RSM"]
):
    im = ax.imshow(RSM, cmap="RdBu_r", aspect="auto")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Token index (sorted by phoneme)")
    ax.set_ylabel("Token index (sorted by phoneme)")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("Representational Similarity Matrices",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_rsm.png", dpi=150)
plt.close()
print("\nSaved plot_rsm.png")
print("Done.")