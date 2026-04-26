"""
Stage 12 — mixed_effects.py
Section 7: Linear Mixed-Effects Models

7.1 Acoustic features (F1 and F2)
7.2 Neural representations (first 5 PCs)
7.3 Model building strategy:
    - Null model (ICC)
    - Main effects model
    - Full model (L1 x Gender interaction)
    - Extended model (vowel height)
    - Random slope model
7.4 Comparing representation types (marginal/conditional R2)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import yaml

warnings.filterwarnings("ignore")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

OUTPUT_DIR    = "results"
FRENCH_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "ɑ", "ə", "ɛ", "ɑ̃"}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── vowel height classification ────────────────────────────────────
VOWEL_HEIGHT = {
    "i": "high", "u": "high", "y": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "ə": "mid",
    "a": "low",  "ɑ": "low",  "ɑ̃": "low",
}

# ── load acoustic features ─────────────────────────────────────────
df = pd.read_csv("data/features_acoustic_norm.csv")
df = df[df["phoneme"].isin(FRENCH_VOWELS)].copy()
df = df.dropna(subset=["f1_norm", "f2_norm"])

# create dummy variables
df["is_L2"]  = (df["l1_status"] == "L2").astype(int)
df["is_male"] = (df["gender"] == "m").astype(int)
df["height"]  = df["phoneme"].map(VOWEL_HEIGHT)

print(f"Loaded {len(df)} vowel tokens")
phonemes = sorted(df["phoneme"].unique())


# ══════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════

def compute_icc(model):
    """Compute ICC from null model variance components."""
    var_u     = float(model.cov_re.iloc[0, 0])
    var_resid = float(model.scale)
    icc       = var_u / (var_u + var_resid)
    return icc, var_u, var_resid


def compute_r2(model_null, model_fixed, data, response):
    """
    Compute marginal and conditional R2 (Nakagawa & Schielzeth method).
    marginal R2  = variance explained by fixed effects only
    conditional R2 = variance explained by fixed + random effects
    """
    # variance of fixed effects
    fitted_fixed = model_fixed.fittedvalues
    var_fixed    = np.var(fitted_fixed)

    # random effect variance
    var_u    = float(model_fixed.cov_re.iloc[0, 0])
    var_resid = float(model_fixed.scale)

    total = var_fixed + var_u + var_resid

    r2_marginal    = var_fixed / total
    r2_conditional = (var_fixed + var_u) / total

    return r2_marginal, r2_conditional


def likelihood_ratio_test(model1, model2):
    """LRT between two nested models (fitted with ML)."""
    lr_stat = 2 * (model2.llf - model1.llf)
    df_diff = model2.df_modelwc - model1.df_modelwc
    if df_diff <= 0:
        df_diff = 1
    p_val   = 1 - __import__("scipy").stats.chi2.cdf(lr_stat, df_diff)
    return lr_stat, df_diff, p_val


def fit_acoustic_models(data, response, phoneme_label):
    """
    Fit the 5-model sequence for a given response variable
    and return a summary dict.
    """
    results = {"phoneme": phoneme_label, "response": response}

    groups = data["speaker_id"]

    # ── 1. Null model ──────────────────────────────────────────────
    try:
        null = MixedLM(
            data[response], np.ones(len(data)),
            groups=groups
        ).fit(reml=False, method="lbfgs")

        icc, var_u, var_resid = compute_icc(null)
        results["ICC"]      = round(icc, 4)
        results["var_u"]    = round(var_u, 4)
        results["var_resid"] = round(var_resid, 4)
        results["AIC_null"] = round(null.aic, 2)
    except Exception as e:
        print(f"    Null model failed: {e}")
        return results

    # ── 2. Main effects model ──────────────────────────────────────
    try:
        main = smf.mixedlm(
            f"{response} ~ is_L2 + is_male",
            data, groups=data["speaker_id"]
        ).fit(reml=False, method="lbfgs")

        results["beta_L2_main"]   = round(main.fe_params["is_L2"], 4)
        results["beta_male_main"] = round(main.fe_params["is_male"], 4)
        results["AIC_main"]       = round(main.aic, 2)

        lr, df, p = likelihood_ratio_test(null, main)
        results["LRT_main_p"] = round(p, 4)
    except Exception as e:
        print(f"    Main effects model failed: {e}")
        return results

    # ── 3. Full model (L1 x Gender interaction) ────────────────────
    try:
        full = smf.mixedlm(
            f"{response} ~ is_L2 * is_male",
            data, groups=data["speaker_id"]
        ).fit(reml=False, method="lbfgs")

        results["beta_interaction"] = round(
            full.fe_params.get("is_L2:is_male", np.nan), 4
        )
        results["AIC_full"] = round(full.aic, 2)

        lr, df, p = likelihood_ratio_test(main, full)
        results["LRT_interaction_p"] = round(p, 4)
        results["interaction_sig"]   = p < 0.05
    except Exception as e:
        print(f"    Full model failed: {e}")
        full = main

    # ── 4. Extended model (vowel height) ───────────────────────────
    try:
        ext = smf.mixedlm(
            f"{response} ~ is_L2 + is_male + C(height)",
            data, groups=data["speaker_id"]
        ).fit(reml=False, method="lbfgs")

        results["AIC_extended"] = round(ext.aic, 2)
        lr, df, p = likelihood_ratio_test(main, ext)
        results["LRT_height_p"] = round(p, 4)
        results["height_sig"]   = p < 0.05
    except Exception as e:
        print(f"    Extended model failed: {e}")

    # ── 5. Random slope model ──────────────────────────────────────
    try:
        rand_slope = smf.mixedlm(
            f"{response} ~ is_L2 + is_male",
            data,
            groups=data["speaker_id"],
            re_formula="~is_L2"
        ).fit(reml=False, method="lbfgs")

        results["AIC_randslope"] = round(rand_slope.aic, 2)
        lr, df, p = likelihood_ratio_test(main, rand_slope)
        results["LRT_randslope_p"] = round(p, 4)
        results["randslope_sig"]   = p < 0.05
    except Exception as e:
        print(f"    Random slope model failed: {e}")

    # ── R2 ─────────────────────────────────────────────────────────
    try:
        r2_m, r2_c = compute_r2(null, main, data, response)
        results["R2_marginal"]    = round(r2_m, 4)
        results["R2_conditional"] = round(r2_c, 4)
    except Exception as e:
        print(f"    R2 failed: {e}")

    return results


# ══════════════════════════════════════════════════════════════════
# 7.1 Acoustic models — F1 and F2
# ══════════════════════════════════════════════════════════════════
print("\n── 7.1 Acoustic mixed-effects models ────────────────────────")

ac_rows = []
for phoneme in phonemes:
    sub = df[df["phoneme"] == phoneme].copy()
    if len(sub) < 20:
        continue
    print(f"\n  /{phoneme}/  (n={len(sub)})")

    for response in ["f1_norm", "f2_norm"]:
        res = fit_acoustic_models(sub, response, phoneme)
        res["response"] = response
        ac_rows.append(res)
        print(f"    {response}: ICC={res.get('ICC','?')}  "
              f"R2_m={res.get('R2_marginal','?')}  "
              f"R2_c={res.get('R2_conditional','?')}  "
              f"beta_L2={res.get('beta_L2_main','?')}")

ac_df = pd.DataFrame(ac_rows)
ac_df.to_csv(f"{OUTPUT_DIR}/lme_acoustic_results.csv", index=False)
print(f"\nSaved lme_acoustic_results.csv")


# ══════════════════════════════════════════════════════════════════
# 7.2 Neural models — first 5 PCs
# ══════════════════════════════════════════════════════════════════
print("\n── 7.2 Neural mixed-effects models (PCA-5) ──────────────────")

NEURAL_MODELS = {
    "Whisper L20": "data/features_whisper_layer20.npz",
    "XLS-R L20":   "data/features_xlsr_layer20.npz",
}

meta = pd.read_csv("data/phonemes.csv")
meta = meta[meta["phoneme"].isin(FRENCH_VOWELS)].copy()
meta = meta.reset_index(drop=True)
meta["is_L2"]   = (meta["l1_status"] == "L2").astype(int)
meta["is_male"]  = (meta["gender"] == "m").astype(int)
meta["height"]   = meta["phoneme"].map(VOWEL_HEIGHT)

neural_rows = []

for model_name, path in NEURAL_MODELS.items():
    print(f"\n  {model_name}...")

    data_npz  = np.load(path)
    X_full    = data_npz["features"]
    token_ids = data_npz["token_ids"]

    # filter to French vowels
    ph_mask = np.array([
        i < len(meta) and meta.iloc[i]["phoneme"] in FRENCH_VOWELS
        for i in range(len(token_ids))
    ])
    X    = X_full[ph_mask]
    tids = token_ids[ph_mask]
    m    = meta.iloc[
        [i for i in range(len(token_ids))
         if i < len(meta) and
         meta.iloc[i]["phoneme"] in FRENCH_VOWELS]
    ].reset_index(drop=True)

    for phoneme in phonemes:
        ph_mask2 = m["phoneme"].values == phoneme
        if ph_mask2.sum() < 20:
            continue

        X_ph = X[ph_mask2]
        m_ph = m[ph_mask2].copy().reset_index(drop=True)

        # fit PCA-5 on this phoneme's data
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_ph)
        pca     = PCA(n_components=min(5, X_sc.shape[1]),
                      random_state=42)
        X_pca   = pca.fit_transform(X_sc)

        for pc_idx in range(X_pca.shape[1]):
            m_ph[f"pc{pc_idx+1}"] = X_pca[:, pc_idx]
            response               = f"pc{pc_idx+1}"

            res = fit_acoustic_models(m_ph, response, phoneme)
            res["model"]    = model_name
            res["pc"]       = pc_idx + 1
            res["var_expl"] = round(
                pca.explained_variance_ratio_[pc_idx], 4
            )
            neural_rows.append(res)

        # report PC1 only for brevity
        print(f"    /{phoneme}/  PC1: "
              f"ICC={neural_rows[-5].get('ICC','?')}  "
              f"R2_m={neural_rows[-5].get('R2_marginal','?')}  "
              f"beta_L2={neural_rows[-5].get('beta_L2_main','?')}")

neural_df = pd.DataFrame(neural_rows)
neural_df.to_csv(
    f"{OUTPUT_DIR}/lme_neural_results.csv", index=False
)
print(f"\nSaved lme_neural_results.csv")


# ══════════════════════════════════════════════════════════════════
# 7.4 Comparing representation types — R2 summary
# ══════════════════════════════════════════════════════════════════
print("\n── 7.4 R2 comparison across representation types ────────────")

# acoustic R2 per phoneme (F1 only)
ac_f1 = ac_df[ac_df["response"] == "f1_norm"][
    ["phoneme", "ICC", "R2_marginal", "R2_conditional",
     "beta_L2_main", "interaction_sig"]
].copy()
ac_f1["representation"] = "Acoustic F1"

# neural R2 per phoneme — PC1 only, averaged across models
neural_pc1 = neural_df[neural_df["pc"] == 1].groupby(
    ["phoneme", "model"]
)[["ICC", "R2_marginal", "R2_conditional",
   "beta_L2_main"]].mean().reset_index()
neural_pc1.columns = ["phoneme", "representation",
                      "ICC", "R2_marginal",
                      "R2_conditional", "beta_L2_main"]
neural_pc1["interaction_sig"] = False

r2_summary = pd.concat([ac_f1, neural_pc1], ignore_index=True)
r2_summary.to_csv(f"{OUTPUT_DIR}/r2_comparison.csv", index=False)

print("\nR2 comparison (marginal) across phonemes:")
pivot = r2_summary.pivot(
    index="phoneme",
    columns="representation",
    values="R2_marginal"
).round(4)
print(pivot.to_string())

# ── forest plot: beta_L2 per phoneme per representation ───────────
fig, ax = plt.subplots(figsize=(10, 7))

reps    = r2_summary["representation"].unique()
colors  = {"Acoustic F1": "#1565C0",
           "Whisper L20": "#C62828",
           "XLS-R L20":   "#2E7D32"}
y_pos   = np.arange(len(phonemes))
width   = 0.25

for i, rep in enumerate(reps):
    sub    = r2_summary[r2_summary["representation"] == rep]
    sub    = sub.set_index("phoneme").reindex(phonemes)
    betas  = sub["R2_marginal"].values
    offset = (i - 1) * width
    color  = colors.get(rep, "grey")
    ax.barh(y_pos + offset, betas, width,
            label=rep, color=color, alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(phonemes, fontsize=10)
ax.set_xlabel("Marginal R² (L1/L2 fixed effect)", fontsize=11)
ax.set_title("Marginal R² per phoneme across representation types",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
ax.axvline(0, color="black", linewidth=1)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_r2_comparison.png", dpi=150)
plt.close()
print("\nSaved plot_r2_comparison.png")
print("\nAll mixed-effects models done!")

# ── export neural PCA-5 to CSV for R ──────────────────────────────
print("\nExporting neural PCA-5 data for R...")

export_rows = []
for model_name, path in NEURAL_MODELS.items():
    data_npz  = np.load(path)
    X_full    = data_npz["features"]
    token_ids = data_npz["token_ids"]

    ph_mask = np.array([
        i < len(meta) and meta.iloc[i]["phoneme"] in FRENCH_VOWELS
        for i in range(len(token_ids))
    ])
    X    = X_full[ph_mask]
    m    = meta.iloc[
        [i for i in range(len(token_ids))
         if i < len(meta) and
         meta.iloc[i]["phoneme"] in FRENCH_VOWELS]
    ].reset_index(drop=True)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    pca     = PCA(n_components=5, random_state=42)
    X_pca   = pca.fit_transform(X_sc)

    safe_name = model_name.replace(" ", "_").replace("-", "_")
    for idx in range(len(m)):
        row = {
            "model":      safe_name,
            "speaker_id": m.iloc[idx]["speaker_id"],
            "phoneme":    m.iloc[idx]["phoneme"],
            "l1_status":  m.iloc[idx]["l1_status"],
            "gender":     m.iloc[idx]["gender"],
            "is_L2":      1 if m.iloc[idx]["l1_status"] == "L2" else 0,
            "is_male":    1 if m.iloc[idx]["gender"] == "m" else 0,
            "pc1":        X_pca[idx, 0],
            "pc2":        X_pca[idx, 1],
            "pc3":        X_pca[idx, 2],
            "pc4":        X_pca[idx, 3],
            "pc5":        X_pca[idx, 4],
        }
        export_rows.append(row)

export_df = pd.DataFrame(export_rows)
export_df.to_csv("data/neural_pca5_for_r.csv", index=False)
print(f"Exported {len(export_df)} rows to data/neural_pca5_for_r.csv")