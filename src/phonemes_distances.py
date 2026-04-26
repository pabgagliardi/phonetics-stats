"""
Stage 11 — phoneme_distances.py
Section 6.2 (extended):

1. Neural distance matrices (Whisper + XLS-R)
2. Mantel test comparing Dac, DWh, DXL
3. Bootstrap CIs on selected phoneme pairs
4. Nearest-centroid classifier (leave-one-speaker-out)
5. McNemar test comparing representation types
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

OUTPUT_DIR    = "results"
FRENCH_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load acoustic features ─────────────────────────────────────────
df_ac = pd.read_csv("data/features_acoustic_norm.csv")
df_ac = df_ac[df_ac["phoneme"].isin(FRENCH_VOWELS)].dropna(
    subset=["f1_norm", "f2_norm"]
).copy()

# ── load metadata ──────────────────────────────────────────────────
meta = pd.read_csv("data/phonemes.csv")
meta = meta[meta["phoneme"].isin(FRENCH_VOWELS)].copy()
meta = meta.reset_index(drop=True)

phoneme_list = sorted(df_ac["phoneme"].unique())
n_ph         = len(phoneme_list)
print(f"Phonemes: {phoneme_list}")

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
# 1. Distance matrices
# ══════════════════════════════════════════════════════════════════
print("\n── 1. Distance matrices ─────────────────────────────────────")

ac_centroids = np.array([
    df_ac[df_ac["phoneme"] == p][["f1_norm", "f2_norm"]].mean().values
    for p in phoneme_list
])

# Euclidean
D_eucl = np.zeros((n_ph, n_ph))
for i in range(n_ph):
    for j in range(n_ph):
        D_eucl[i, j] = np.linalg.norm(
            ac_centroids[i] - ac_centroids[j]
        )

# Mahalanobis
pooled_cov = np.cov(df_ac[["f1_norm", "f2_norm"]].T)
try:
    inv_cov = np.linalg.inv(pooled_cov)
    D_maha  = np.zeros((n_ph, n_ph))
    for i in range(n_ph):
        for j in range(n_ph):
            diff         = ac_centroids[i] - ac_centroids[j]
            D_maha[i, j] = np.sqrt(diff @ inv_cov @ diff)
except np.linalg.LinAlgError:
    D_maha = D_eucl.copy()
    print("  WARNING: Mahalanobis failed, using Euclidean")

print("Acoustic distance matrices computed.")

# neural distance matrices
NEURAL_MODELS = {
    "Whisper L20": "data/features_whisper_layer20_pca50.npz",
    "XLS-R L20":   "data/features_xlsr_layer20_pca50.npz",
}

D_neural = {}
for model_name, path in NEURAL_MODELS.items():
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
    print(f"{model_name} distance matrix computed.")

# ── plot all distance matrices ─────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, D, title in zip(
    axes,
    [D_eucl, D_maha,
     D_neural["Whisper L20"], D_neural["XLS-R L20"]],
    ["Acoustic (Euclidean)", "Acoustic (Mahalanobis)",
     "Whisper L20 (cosine)", "XLS-R L20 (cosine)"]
):
    im = ax.imshow(D, cmap="YlOrRd")
    ax.set_xticks(range(n_ph))
    ax.set_yticks(range(n_ph))
    ax.set_xticklabels(phoneme_list, fontsize=8, rotation=45)
    ax.set_yticklabels(phoneme_list, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("Inter-phoneme distance matrices",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_all_distance_matrices.png", dpi=150)
plt.close()
print("Saved plot_all_distance_matrices.png")


# ══════════════════════════════════════════════════════════════════
# 2. Mantel test
# ══════════════════════════════════════════════════════════════════
print("\n── 2. Mantel test ───────────────────────────────────────────")

def mantel_test(D1, D2, n_perm=999):
    idx      = np.triu_indices(D1.shape[0], k=1)
    v1, v2   = D1[idx], D2[idx]
    r_obs, _ = stats.spearmanr(v1, v2)
    count    = 0
    for _ in range(n_perm):
        perm      = np.random.permutation(D1.shape[0])
        D1_perm   = D1[np.ix_(perm, perm)]
        r_perm, _ = stats.spearmanr(D1_perm[idx], v2)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)

mantel_rows = []
pairs = [
    ("Acoustic (Eucl)", D_eucl,
     "Whisper L20",     D_neural["Whisper L20"]),
    ("Acoustic (Eucl)", D_eucl,
     "XLS-R L20",       D_neural["XLS-R L20"]),
    ("Acoustic (Maha)", D_maha,
     "Whisper L20",     D_neural["Whisper L20"]),
    ("Acoustic (Maha)", D_maha,
     "XLS-R L20",       D_neural["XLS-R L20"]),
    ("Whisper L20",     D_neural["Whisper L20"],
     "XLS-R L20",       D_neural["XLS-R L20"]),
]

for n1, D1, n2, D2 in pairs:
    r, p = mantel_test(D1, D2)
    print(f"  {n1} vs {n2}: r={r:.3f}  p={p:.3f}")
    mantel_rows.append({
        "D1": n1, "D2": n2,
        "mantel_r": round(r, 3),
        "p_value":  round(p, 3),
    })

pd.DataFrame(mantel_rows).to_csv(
    f"{OUTPUT_DIR}/mantel_distance_results.csv", index=False
)


# ══════════════════════════════════════════════════════════════════
# 3. Bootstrap CIs on selected phoneme pairs
# ══════════════════════════════════════════════════════════════════
print("\n── 3. Bootstrap CIs on phoneme pairs ───────────────────────")

PAIRS = [
    ("e", "ɛ"),
    ("o", "u"),
    ("y", "u"),
    ("a", "ɑ"),
]

N_BOOT   = 2000
speakers = sorted(df_ac["speaker_id"].unique())

# precompute per-speaker per-phoneme neural means
print("  Precomputing per-speaker neural means...")
neural_spk_means = {}
for model_name, path in NEURAL_MODELS.items():
    data      = np.load(path)
    X         = data["features"]
    token_ids = data["token_ids"]
    neural_spk_means[model_name] = {}
    for spk in speakers:
        for ph in phoneme_list:
            mask = np.array([
                i < len(meta) and
                meta.iloc[i]["phoneme"] == ph and
                meta.iloc[i]["speaker_id"] == spk
                for i in range(len(token_ids))
            ])
            if mask.sum() > 0:
                neural_spk_means[model_name][(spk, ph)] = \
                    X[mask].mean(axis=0)

boot_rows = []

for p1, p2 in PAIRS:
    if p1 not in phoneme_list or p2 not in phoneme_list:
        continue

    print(f"  /{p1}/–/{p2}/...")

    # acoustic bootstrap
    boot_eucl = []
    for _ in range(N_BOOT):
        boot_spks = np.random.choice(
            speakers, len(speakers), replace=True
        )
        sub1 = pd.concat([
            df_ac[(df_ac["phoneme"] == p1) &
                  (df_ac["speaker_id"] == s)]
            for s in boot_spks
        ])
        sub2 = pd.concat([
            df_ac[(df_ac["phoneme"] == p2) &
                  (df_ac["speaker_id"] == s)]
            for s in boot_spks
        ])
        if len(sub1) == 0 or len(sub2) == 0:
            continue
        c1 = sub1[["f1_norm", "f2_norm"]].mean().values
        c2 = sub2[["f1_norm", "f2_norm"]].mean().values
        boot_eucl.append(np.linalg.norm(c1 - c2))

    # neural bootstrap
    boot_neural = {m: [] for m in NEURAL_MODELS}
    for _ in range(N_BOOT):
        boot_spks = np.random.choice(
            speakers, len(speakers), replace=True
        )
        for model_name in NEURAL_MODELS:
            vecs1 = [
                neural_spk_means[model_name][(s, p1)]
                for s in boot_spks
                if (s, p1) in neural_spk_means[model_name]
            ]
            vecs2 = [
                neural_spk_means[model_name][(s, p2)]
                for s in boot_spks
                if (s, p2) in neural_spk_means[model_name]
            ]
            if len(vecs1) == 0 or len(vecs2) == 0:
                continue
            c1 = np.mean(vecs1, axis=0)
            c2 = np.mean(vecs2, axis=0)
            boot_neural[model_name].append(cosine(c1, c2))

    # observed distances
    obs_eucl = np.linalg.norm(
        ac_centroids[phoneme_list.index(p1)] -
        ac_centroids[phoneme_list.index(p2)]
    )
    obs_wh = D_neural["Whisper L20"][
        phoneme_list.index(p1), phoneme_list.index(p2)
    ]
    obs_xl = D_neural["XLS-R L20"][
        phoneme_list.index(p1), phoneme_list.index(p2)
    ]

    ci_eucl = np.percentile(boot_eucl, [2.5, 97.5])
    ci_wh   = np.percentile(boot_neural["Whisper L20"], [2.5, 97.5])
    ci_xl   = np.percentile(boot_neural["XLS-R L20"],   [2.5, 97.5])

    boot_rows.append({
        "pair":          f"/{p1}/–/{p2}/",
        "obs_euclidean": round(obs_eucl, 4),
        "ci_eucl_lo":    round(ci_eucl[0], 4),
        "ci_eucl_hi":    round(ci_eucl[1], 4),
        "obs_whisper":   round(obs_wh, 4),
        "ci_wh_lo":      round(ci_wh[0], 4),
        "ci_wh_hi":      round(ci_wh[1], 4),
        "obs_xlsr":      round(obs_xl, 4),
        "ci_xl_lo":      round(ci_xl[0], 4),
        "ci_xl_hi":      round(ci_xl[1], 4),
    })

    print(f"    Euclidean: {obs_eucl:.4f} "
          f"[{ci_eucl[0]:.4f}, {ci_eucl[1]:.4f}]")
    print(f"    Whisper:   {obs_wh:.4f} "
          f"[{ci_wh[0]:.4f}, {ci_wh[1]:.4f}]")
    print(f"    XLS-R:     {obs_xl:.4f} "
          f"[{ci_xl[0]:.4f}, {ci_xl[1]:.4f}]")

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(
    f"{OUTPUT_DIR}/bootstrap_ci_distances.csv", index=False
)
print("Saved bootstrap_ci_distances.csv")


# ══════════════════════════════════════════════════════════════════
# 4. Nearest-centroid classifier (leave-one-speaker-out)
# ══════════════════════════════════════════════════════════════════
print("\n── 4. Nearest-centroid classifier (LOSO) ────────────────────")

def nearest_centroid_loso(X, phoneme_labels, speaker_labels):
    """
    Leave-one-speaker-out nearest centroid classifier.
    Vectorised for speed.
    """
    speakers  = np.unique(speaker_labels)
    all_preds = np.empty(len(X), dtype=object)

    for spk in speakers:
        test_mask  = speaker_labels == spk
        train_mask = ~test_mask

        train_phonemes  = np.unique(phoneme_labels[train_mask])
        centroid_matrix = np.array([
            X[train_mask & (phoneme_labels == p)].mean(axis=0)
            for p in train_phonemes
        ])

        X_test = X[test_mask]

        if X.shape[1] == 2:
            # Euclidean distance
            diffs = (X_test[:, np.newaxis, :] -
                     centroid_matrix[np.newaxis, :, :])
            dists = np.linalg.norm(diffs, axis=2)
        else:
            # cosine distance
            X_norm = X_test / (
                np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10
            )
            C_norm = centroid_matrix / (
                np.linalg.norm(
                    centroid_matrix, axis=1, keepdims=True
                ) + 1e-10
            )
            dists = 1 - X_norm @ C_norm.T

        pred_idx = np.argmin(dists, axis=1)
        all_preds[test_mask] = train_phonemes[pred_idx]

    return all_preds


CLASSIFIERS = {
    "Acoustic (F1,F2)": None,
    "Whisper L20":      "data/features_whisper_layer20_pca50.npz",
    "XLS-R L20":        "data/features_xlsr_layer20_pca50.npz",
}

clf_results = {}
all_preds   = {}
true_labels = None

for clf_name, path in CLASSIFIERS.items():
    print(f"\n  {clf_name}...")

    if path is None:
        X_clf      = df_ac[["f1_norm", "f2_norm"]].values
        ph_labels  = df_ac["phoneme"].values
        spk_labels = df_ac["speaker_id"].values
    else:
        data      = np.load(path)
        X_full    = data["features"]
        token_ids = data["token_ids"]

        ph_mask = np.array([
            i < len(meta) and
            meta.iloc[i]["phoneme"] in FRENCH_VOWELS
            for i in range(len(token_ids))
        ])
        X_clf      = X_full[ph_mask]
        ph_labels  = np.array([
            meta.iloc[i]["phoneme"]
            for i in range(len(token_ids))
            if i < len(meta) and
            meta.iloc[i]["phoneme"] in FRENCH_VOWELS
        ])
        spk_labels = np.array([
            meta.iloc[i]["speaker_id"]
            for i in range(len(token_ids))
            if i < len(meta) and
            meta.iloc[i]["phoneme"] in FRENCH_VOWELS
        ])

    preds = nearest_centroid_loso(X_clf, ph_labels, spk_labels)

    if true_labels is None:
        true_labels = ph_labels

    acc    = np.mean(preds == ph_labels)
    f1_mac = f1_score(
        ph_labels, preds, average="macro", zero_division=0
    )

    print(f"    Accuracy: {acc:.3f}  Macro F1: {f1_mac:.3f}")

    clf_results[clf_name] = {
        "accuracy": round(acc, 3),
        "f1_macro": round(f1_mac, 3),
    }
    all_preds[clf_name] = preds

    # confusion matrix
    labels_sorted = sorted(np.unique(ph_labels))
    cm      = confusion_matrix(
        ph_labels, preds, labels=labels_sorted
    )
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        cmap="Blues", ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(
        f"Confusion matrix — {clf_name}\n"
        f"Accuracy={acc:.3f}  F1={f1_mac:.3f}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    safe = clf_name.replace(
        " ", "_").replace("(", "").replace(")", "").replace(",", ""
    )
    plt.savefig(
        f"{OUTPUT_DIR}/plot_confusion_{safe}.png", dpi=150
    )
    plt.close()

acc_df = pd.DataFrame(clf_results).T
acc_df.to_csv(f"{OUTPUT_DIR}/classifier_accuracy.csv")
print(f"\nClassifier accuracy summary:")
print(acc_df.to_string())


# ══════════════════════════════════════════════════════════════════
# 5. McNemar test comparing classifiers
# ══════════════════════════════════════════════════════════════════
print("\n── 5. McNemar test ──────────────────────────────────────────")

def mcnemar_test(preds1, preds2, true):
    correct1 = preds1 == true
    correct2 = preds2 == true
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    if b + c == 0:
        return 1.0
    stat  = (abs(b - c) - 1)**2 / (b + c)
    p_val = 1 - stats.chi2.cdf(stat, df=1)
    return p_val

mcnemar_rows = []
clf_names    = list(all_preds.keys())

for i in range(len(clf_names)):
    for j in range(i + 1, len(clf_names)):
        n1, n2  = clf_names[i], clf_names[j]
        p1_arr  = all_preds[n1]
        p2_arr  = all_preds[n2]
        min_len = min(len(p1_arr), len(p2_arr), len(true_labels))
        p_val   = mcnemar_test(
            p1_arr[:min_len],
            p2_arr[:min_len],
            true_labels[:min_len]
        )
        print(f"  {n1} vs {n2}: p={p_val:.4f}")
        mcnemar_rows.append({
            "clf1":        n1,
            "clf2":        n2,
            "mcnemar_p":   round(p_val, 4),
            "significant": p_val < 0.05,
        })

pd.DataFrame(mcnemar_rows).to_csv(
    f"{OUTPUT_DIR}/mcnemar_results.csv", index=False
)

print("\nAll done!")