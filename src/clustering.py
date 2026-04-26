"""
Stage 14 — clustering.py
Section 9: Hierarchical Clustering

9.1 Clustering of French oral vowels
9.2 Consonants vs vowels (with SCG for obstruents)
9.3 Clustering of speakers
9.4 Determining number of clusters (silhouette + dendrogram height)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

OUTPUT_DIR    = "results"
FRENCH_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── vowel linguistic categories ────────────────────────────────────
FRONT_BACK = {
    "i": "front", "e": "front", "ɛ": "front",
    "y": "front", "ø": "front", "a": "front",
    "u": "back",  "o": "back",
    "ɑ": "back",  "ɑ̃": "back",
    "ə": "mid",
}

HEIGHT = {
    "i": "high", "u": "high", "y": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "ə": "mid",
    "a": "low",  "ɑ": "low",  "ɑ̃": "low",
}

# ── consonants ─────────────────────────────────────────────────────
CONSONANTS = ["p", "t", "k", "s", "ʃ", "n"]
ALL_PHONES  = FRENCH_VOWELS + CONSONANTS

# ── load data ──────────────────────────────────────────────────────
df_ac = pd.read_csv("data/features_acoustic_norm.csv")
df_ac = df_ac[df_ac["phoneme"].isin(FRENCH_VOWELS)].dropna(
    subset=["f1_norm", "f2_norm"]
).copy()

meta = pd.read_csv("data/phonemes.csv")
meta = meta[meta["phoneme"].isin(FRENCH_VOWELS)].copy()
meta = meta.reset_index(drop=True)

NEURAL_MODELS = {
    "Whisper L20": "data/features_whisper_layer20_pca50.npz",
    "XLS-R L20":   "data/features_xlsr_layer20_pca50.npz",
}

speakers    = sorted(df_ac["speaker_id"].unique())
spk_meta    = df_ac.groupby("speaker_id")[
    ["l1_status", "gender"]
].first().reindex(speakers)
true_l1     = spk_meta["l1_status"].map({"L1": 0, "L2": 1}).values
true_gender = spk_meta["gender"].map({"f": 0, "m": 1}).values


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════
def compute_ari(labels_pred, labels_true):
    return adjusted_rand_score(labels_true, labels_pred)


def plot_dendrogram(Z, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=labels, ax=ax,
               leaf_rotation=45, leaf_font_size=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def neural_cosine_linkage(X_full, token_ids, meta_df, ph_list):
    """Compute Ward linkage on cosine distances for a phoneme list."""
    ph_means = {}
    for ph in ph_list:
        mask = np.array([
            i < len(meta_df) and meta_df.iloc[i]["phoneme"] == ph
            for i in range(len(token_ids))
        ])
        if mask.sum() > 0:
            ph_means[ph] = X_full[mask].mean(axis=0)

    ph_list_out = [p for p in ph_list if p in ph_means]
    X_ph        = np.array([ph_means[p] for p in ph_list_out])

    norms    = np.linalg.norm(X_ph, axis=1, keepdims=True)
    X_norm   = X_ph / (norms + 1e-10)
    cos_dist = 1 - X_norm @ X_norm.T
    cos_dist = np.clip(cos_dist, 0, None)
    np.fill_diagonal(cos_dist, 0)  # force exact zeros on diagonal

    Z = linkage(squareform(cos_dist), method="ward")
    return Z, ph_list_out


# ══════════════════════════════════════════════════════════════════
# 9.1 Clustering of French oral vowels
# ══════════════════════════════════════════════════════════════════
print("\n── 9.1 Clustering of French oral vowels ─────────────────────")

ari_rows = []

# ── acoustic ──────────────────────────────────────────────────────
ac_means = df_ac.groupby("phoneme")[["f1_norm", "f2_norm"]].mean()
ac_means = ac_means.reindex(
    [p for p in FRENCH_VOWELS if p in ac_means.index]
)
ph_list_ac = ac_means.index.tolist()

Z_ac = linkage(ac_means.values, method="ward", metric="euclidean")

plot_dendrogram(
    Z_ac, ph_list_ac,
    "Acoustic clustering — French vowels (Ward, Euclidean)",
    f"{OUTPUT_DIR}/plot_dendro_acoustic_vowels.png"
)

for k, partition_name, ground_truth_fn in [
    (2, "front_back", lambda ph: FRONT_BACK.get(ph, "mid")),
    (3, "height",     lambda ph: HEIGHT.get(ph, "mid")),
]:
    labels_pred = fcluster(Z_ac, k, criterion="maxclust")
    labels_true = [ground_truth_fn(ph) for ph in ph_list_ac]
    ari = compute_ari(labels_pred, labels_true)
    print(f"  Acoustic k={k} ({partition_name}): ARI={ari:.3f}")
    ari_rows.append({
        "representation": "Acoustic",
        "partition":      partition_name,
        "k":              k,
        "ARI":            round(ari, 3),
    })

# ── neural ─────────────────────────────────────────────────────────
for model_name, path in NEURAL_MODELS.items():
    data      = np.load(path)
    X_full    = data["features"]
    token_ids = data["token_ids"]

    Z_n, ph_list_n = neural_cosine_linkage(
        X_full, token_ids, meta, FRENCH_VOWELS
    )

    safe = model_name.replace(" ", "_").lower()
    plot_dendrogram(
        Z_n, ph_list_n,
        f"{model_name} clustering — French vowels (Ward, cosine)",
        f"{OUTPUT_DIR}/plot_dendro_{safe}_vowels.png"
    )

    for k, partition_name, ground_truth_fn in [
        (2, "front_back", lambda ph: FRONT_BACK.get(ph, "mid")),
        (3, "height",     lambda ph: HEIGHT.get(ph, "mid")),
    ]:
        labels_pred = fcluster(Z_n, k, criterion="maxclust")
        labels_true = [ground_truth_fn(ph) for ph in ph_list_n]
        ari = compute_ari(labels_pred, labels_true)
        print(f"  {model_name} k={k} ({partition_name}): ARI={ari:.3f}")
        ari_rows.append({
            "representation": model_name,
            "partition":      partition_name,
            "k":              k,
            "ARI":            round(ari, 3),
        })


# ══════════════════════════════════════════════════════════════════
# 9.2 Consonants vs vowels
# ══════════════════════════════════════════════════════════════════
print("\n── 9.2 Consonants vs vowels ─────────────────────────────────")

# acoustic: F1, F2, duration, SCG
df_all_raw  = pd.read_csv("data/features_acoustic.csv")
df_all_raw  = df_all_raw[df_all_raw["phoneme"].isin(ALL_PHONES)].copy()
df_ac_norm  = pd.read_csv("data/features_acoustic_norm.csv")

# merge normalised F1/F2 for vowels, raw for consonants
df_vowels_n = df_ac_norm[
    df_ac_norm["phoneme"].isin(FRENCH_VOWELS)
][["phoneme", "speaker_id", "f1_norm", "f2_norm"]].copy()

df_cons_raw = df_all_raw[
    df_all_raw["phoneme"].isin(CONSONANTS)
].copy()
df_cons_raw = df_cons_raw.rename(
    columns={"f1": "f1_norm", "f2": "f2_norm"}
)

# per-phoneme means
phone_means_vowels = df_vowels_n.groupby("phoneme")[
    ["f1_norm", "f2_norm"]
].mean()

phone_means_cons = df_all_raw[
    df_all_raw["phoneme"].isin(CONSONANTS)
].groupby("phoneme")[["duration_ms"]].mean()

# SCG for obstruents
scg_means = df_all_raw.groupby("phoneme")["scg"].mean()

# combine
phone_means_ac = df_all_raw.groupby("phoneme")[
    ["duration_ms"]
].mean()
phone_means_ac["scg"] = scg_means.reindex(
    phone_means_ac.index
).fillna(0)

# add normalised F1/F2
f1f2 = pd.concat([
    phone_means_vowels,
    df_all_raw[df_all_raw["phoneme"].isin(CONSONANTS)].groupby(
        "phoneme"
    )[["f1", "f2"]].mean().rename(
        columns={"f1": "f1_norm", "f2": "f2_norm"}
    )
])
phone_means_ac["f1_norm"] = f1f2["f1_norm"]
phone_means_ac["f2_norm"] = f1f2["f2_norm"]

phone_means_ac = phone_means_ac.reindex(
    [p for p in ALL_PHONES if p in phone_means_ac.index]
).dropna()
ph_list_all = phone_means_ac.index.tolist()

scaler   = StandardScaler()
X_all_ac = scaler.fit_transform(phone_means_ac.values)

Z_all_ac = linkage(X_all_ac, method="ward", metric="euclidean")
plot_dendrogram(
    Z_all_ac, ph_list_all,
    "Acoustic clustering — vowels + consonants\n"
    "(F1, F2, duration, SCG)",
    f"{OUTPUT_DIR}/plot_dendro_acoustic_all.png"
)

labels_pred_cv = fcluster(Z_all_ac, 2, criterion="maxclust")
labels_true_cv = [
    "vowel" if p in FRENCH_VOWELS else "consonant"
    for p in ph_list_all
]
ari_cv_ac = compute_ari(labels_pred_cv, labels_true_cv)
print(f"  Acoustic consonant/vowel ARI: {ari_cv_ac:.3f}")
ari_rows.append({
    "representation": "Acoustic",
    "partition":      "consonant/vowel",
    "k":              2,
    "ARI":            round(ari_cv_ac, 3),
})

# neural consonant/vowel
meta_all = pd.read_csv("data/phonemes.csv")
meta_all = meta_all[
    meta_all["phoneme"].isin(ALL_PHONES)
].copy().reset_index(drop=True)

for model_name, path in NEURAL_MODELS.items():
    data      = np.load(path)
    X_full    = data["features"]
    token_ids = data["token_ids"]

    Z_n2, ph_list_n2 = neural_cosine_linkage(
        X_full, token_ids, meta_all, ALL_PHONES
    )

    safe2 = model_name.replace(" ", "_").lower()
    plot_dendrogram(
        Z_n2, ph_list_n2,
        f"{model_name} — vowels + consonants",
        f"{OUTPUT_DIR}/plot_dendro_{safe2}_all.png"
    )

    labels_pred_cv2 = fcluster(Z_n2, 2, criterion="maxclust")
    labels_true_cv2 = [
        "vowel" if p in FRENCH_VOWELS else "consonant"
        for p in ph_list_n2
    ]
    ari_cv_n = compute_ari(labels_pred_cv2, labels_true_cv2)
    print(f"  {model_name} consonant/vowel ARI: {ari_cv_n:.3f}")
    ari_rows.append({
        "representation": model_name,
        "partition":      "consonant/vowel",
        "k":              2,
        "ARI":            round(ari_cv_n, 3),
    })


# ══════════════════════════════════════════════════════════════════
# 9.3 Clustering of speakers
# ══════════════════════════════════════════════════════════════════
print("\n── 9.3 Clustering of speakers ───────────────────────────────")

# acoustic speaker vectors
spk_ac_vecs = []
for spk in speakers:
    row = []
    for ph in FRENCH_VOWELS:
        sub = df_ac[
            (df_ac["speaker_id"] == spk) &
            (df_ac["phoneme"] == ph)
        ][["f1_norm", "f2_norm"]].mean()
        row.extend(
            sub.values if not sub.isna().any() else [0.0, 0.0]
        )
    spk_ac_vecs.append(row)

X_spk_ac  = np.array(spk_ac_vecs)
scaler2   = StandardScaler()
X_spk_ac  = scaler2.fit_transform(X_spk_ac)
Z_spk_ac  = linkage(X_spk_ac, method="ward", metric="euclidean")

plot_dendrogram(
    Z_spk_ac, speakers,
    "Acoustic speaker clustering",
    f"{OUTPUT_DIR}/plot_dendro_speakers_acoustic.png"
)

for k, name, true_labels in [
    (2, "L1/L2",  true_l1),
    (2, "gender", true_gender),
]:
    pred = fcluster(Z_spk_ac, k, criterion="maxclust") - 1
    ari  = compute_ari(pred, true_labels)
    print(f"  Acoustic speakers — {name}: ARI={ari:.3f}")
    ari_rows.append({
        "representation": "Acoustic (speakers)",
        "partition":      name,
        "k":              k,
        "ARI":            round(ari, 3),
    })

# neural speaker vectors
for model_name, path in NEURAL_MODELS.items():
    data      = np.load(path)
    X_full    = data["features"]
    token_ids = data["token_ids"]

    spk_neural_vecs = []
    for spk in speakers:
        row = []
        for ph in FRENCH_VOWELS:
            mask = np.array([
                i < len(meta) and
                meta.iloc[i]["phoneme"] == ph and
                meta.iloc[i]["speaker_id"] == spk
                for i in range(len(token_ids))
            ])
            if mask.sum() > 0:
                row.extend(X_full[mask].mean(axis=0))
            else:
                row.extend(np.zeros(X_full.shape[1]))
        spk_neural_vecs.append(row)

    X_spk_n = np.array(spk_neural_vecs)
    scaler3  = StandardScaler()
    X_spk_n  = scaler3.fit_transform(X_spk_n)

    pca_spk  = PCA(
        n_components=min(10, X_spk_n.shape[0] - 1),
        random_state=42
    )
    X_spk_n  = pca_spk.fit_transform(X_spk_n)
    Z_spk_n  = linkage(X_spk_n, method="ward", metric="euclidean")

    safe3 = model_name.replace(" ", "_").lower()
    plot_dendrogram(
        Z_spk_n, speakers,
        f"{model_name} speaker clustering",
        f"{OUTPUT_DIR}/plot_dendro_speakers_{safe3}.png"
    )

    for k, name, true_labels in [
        (2, "L1/L2",  true_l1),
        (2, "gender", true_gender),
    ]:
        pred = fcluster(Z_spk_n, k, criterion="maxclust") - 1
        ari  = compute_ari(pred, true_labels)
        print(f"  {model_name} speakers — {name}: ARI={ari:.3f}")
        ari_rows.append({
            "representation": f"{model_name} (speakers)",
            "partition":      name,
            "k":              k,
            "ARI":            round(ari, 3),
        })


# ══════════════════════════════════════════════════════════════════
# 9.4 Determining number of clusters
# ══════════════════════════════════════════════════════════════════
print("\n── 9.4 Number of clusters ───────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── silhouette ─────────────────────────────────────────────────────
ax = axes[0]
sil_scores = []
k_range    = range(2, min(8, len(ph_list_ac)))

for k in k_range:
    labels = fcluster(Z_ac, k, criterion="maxclust")
    if len(np.unique(labels)) < 2:
        continue
    sil = silhouette_score(ac_means.values, labels)
    sil_scores.append(sil)

ax.plot(list(k_range)[:len(sil_scores)], sil_scores,
        marker="o", color="#1565C0", linewidth=2)
best_k = list(k_range)[np.argmax(sil_scores)]
ax.axvline(best_k, color="red", linestyle="--",
           label=f"Best k={best_k}")
ax.set_xlabel("Number of clusters k", fontsize=11)
ax.set_ylabel("Silhouette coefficient", fontsize=11)
ax.set_title("Silhouette analysis\n(acoustic vowel clustering)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
print(f"  Best k by silhouette (acoustic): {best_k}")

# ── dendrogram height ──────────────────────────────────────────────
ax = axes[1]
heights = Z_ac[:, 2]
ax.plot(
    range(1, len(heights) + 1),
    sorted(heights, reverse=True),
    marker="o", color="#C62828", linewidth=2
)
ax.set_xlabel("Number of merges", fontsize=11)
ax.set_ylabel("Merge height (distance)", fontsize=11)
ax.set_title("Dendrogram height — acoustic vowels\n"
             "(large jump = natural number of clusters)",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# find largest jump
diffs   = np.diff(sorted(heights, reverse=True))
jump_idx = np.argmax(np.abs(diffs))
best_k_height = jump_idx + 1
ax.axvline(best_k_height, color="blue", linestyle="--",
           label=f"Largest jump at k={best_k_height}")
ax.legend(fontsize=9)
print(f"  Best k by dendrogram height (acoustic): {best_k_height}")

plt.suptitle("Determining number of clusters — acoustic vowels",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_cluster_selection.png", dpi=150)
plt.close()
print("Saved plot_cluster_selection.png")


# ── save ARI summary ───────────────────────────────────────────────
ari_df = pd.DataFrame(ari_rows)
ari_df.to_csv(f"{OUTPUT_DIR}/ari_results.csv", index=False)

print("\n── ARI summary ──────────────────────────────────────────────")
print(ari_df.pivot_table(
    index="representation",
    columns="partition",
    values="ARI"
).round(3).to_string())

print("\nAll clustering done!")