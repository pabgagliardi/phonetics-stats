# ── Mixed-Effects Models for Phonetics Lab ────────────────────────
# Section 7: Linear Mixed-Effects Models using lme4

library(lme4)
library(lmerTest)  # adds p-values to lme4
library(MuMIn)     # for R2 (r.squaredGLMM)

# ── load data ──────────────────────────────────────────────────────
df <- read.csv("data/features_acoustic_norm.csv")
df <- df[df$phoneme %in% c("a","e","i","o","u","y",
                             "ø","ɑ","ə","ɛ","ɑ̃"), ]
df <- df[!is.na(df$f1_norm) & !is.na(df$f2_norm), ]

# dummy variables
df$is_L2   <- as.integer(df$l1_status == "L2")
df$is_male <- as.integer(df$gender == "m")
df$height  <- ifelse(df$phoneme %in% c("i","u","y"), "high",
               ifelse(df$phoneme %in% c("e","ø","o","ɛ","ə"), "mid",
               "low"))

cat("Loaded", nrow(df), "vowel tokens\n")
cat("Phonemes:", paste(sort(unique(df$phoneme)), collapse=", "), "\n\n")

# ── helper: fit model sequence for one phoneme ─────────────────────
fit_models <- function(sub, response, phoneme_label) {
  cat("\n══ /", phoneme_label, "/ ══ response:", response, "\n")
  
  f <- as.formula(paste(response, "~ 1 + (1|speaker_id)"))
  
  # 1. Null model
  null <- tryCatch(
    lmer(as.formula(paste(response, "~ 1 + (1|speaker_id)")),
         data=sub, REML=FALSE),
    error=function(e) { cat("  Null failed:", e$message, "\n"); NULL }
  )
  if (is.null(null)) return(NULL)
  
  vc   <- as.data.frame(VarCorr(null))
  var_u     <- vc[vc$grp == "speaker_id", "vcov"]
  var_resid <- vc[vc$grp == "Residual",   "vcov"]
  icc       <- var_u / (var_u + var_resid)
  cat("  Null — ICC:", round(icc, 4),
      " var_u:", round(var_u, 4),
      " var_resid:", round(var_resid, 4), "\n")
  cat("  Null — AIC:", round(AIC(null), 2), "\n")
  
  # 2. Main effects model
  main <- tryCatch(
    lmer(as.formula(paste(response,
         "~ is_L2 + is_male + (1|speaker_id)")),
         data=sub, REML=FALSE),
    error=function(e) { cat("  Main failed:", e$message, "\n"); NULL }
  )
  if (is.null(main)) return(NULL)
  
  cat("  Main — AIC:", round(AIC(main), 2), "\n")
  cat("  Main — beta_L2:", round(fixef(main)["is_L2"], 4),
      " beta_male:", round(fixef(main)["is_male"], 4), "\n")
  cat("  Main — LRT vs null: p =",
      round(anova(null, main)$`Pr(>Chisq)`[2], 4), "\n")
  
  r2 <- tryCatch(r.squaredGLMM(main),
                 error=function(e) matrix(c(NA,NA), nrow=1))
  cat("  Main — R2_marginal:", round(r2[1,1], 4),
      " R2_conditional:", round(r2[1,2], 4), "\n")
  
  # 3. Full model (L1 x Gender interaction)
  full <- tryCatch(
    lmer(as.formula(paste(response,
         "~ is_L2 * is_male + (1|speaker_id)")),
         data=sub, REML=FALSE),
    error=function(e) { cat("  Full failed:", e$message, "\n"); NULL }
  )
  if (!is.null(full)) {
    cat("  Full — AIC:", round(AIC(full), 2), "\n")
    cat("  Full — beta_interaction:",
        round(fixef(full)["is_L2:is_male"], 4), "\n")
    cat("  Full — LRT vs main: p =",
        round(anova(main, full)$`Pr(>Chisq)`[2], 4), "\n")
  }
  
  # 4. Extended model (vowel height)
  ext <- tryCatch(
    lmer(as.formula(paste(response,
         "~ is_L2 + is_male + height + (1|speaker_id)")),
         data=sub, REML=FALSE),
    error=function(e) { cat("  Extended failed:", e$message,"\n"); NULL }
  )
  if (!is.null(ext)) {
    cat("  Extended — AIC:", round(AIC(ext), 2), "\n")
    cat("  Extended — LRT vs main: p =",
        round(anova(main, ext)$`Pr(>Chisq)`[2], 4), "\n")
  }
  
  # 5. Random slope model
  rand <- tryCatch(
    lmer(as.formula(paste(response,
         "~ is_L2 + is_male + (1 + is_L2|speaker_id)")),
         data=sub, REML=FALSE),
    error=function(e) { cat("  RandSlope failed:", e$message,"\n"); NULL }
  )
  if (!is.null(rand)) {
    cat("  RandSlope — AIC:", round(AIC(rand), 2), "\n")
    cat("  RandSlope — LRT vs main: p =",
        round(anova(main, rand)$`Pr(>Chisq)`[2], 4), "\n")
  }
  
  return(list(icc=icc, main=main, full=full))
}

# ── run for all phonemes ───────────────────────────────────────────
phonemes <- sort(unique(df$phoneme))
results  <- list()

sink("results/lme_acoustic_output.txt")  # save output to file
cat("═══════════════════════════════════════════\n")
cat("ACOUSTIC MIXED-EFFECTS MODELS\n")
cat("═══════════════════════════════════════════\n")

for (ph in phonemes) {
  sub <- df[df$phoneme == ph, ]
  if (nrow(sub) < 20) next
  
  for (resp in c("f1_norm", "f2_norm")) {
    res <- fit_models(sub, resp, ph)
    results[[paste(ph, resp, sep="_")]] <- res
  }
}
sink()  # stop saving

cat("\nOutput saved to results/lme_acoustic_output.txt\n")


# ══════════════════════════════════════════════════════════════════
# Neural representations — PC1 of Whisper and XLS-R for each phoneme
# ══════════════════════════════════════════════════════════════════

# We load the PCA-5 data exported from Python
# First check if the file exists
if (file.exists("data/neural_pca5_for_r.csv")) {
  cat("\n═══════════════════════════════════════════\n")
  cat("NEURAL MIXED-EFFECTS MODELS\n")
  cat("═══════════════════════════════════════════\n")
  
  neural_df <- read.csv("data/neural_pca5_for_r.csv")
  
  sink("results/lme_neural_output.txt")
  
  for (model_name in c("Whisper_L20", "XLS_R_L20")) {
    cat("\n── Model:", model_name, "──\n")
    sub_model <- neural_df[neural_df$model == model_name, ]
    
    for (ph in phonemes) {
      sub <- sub_model[sub_model$phoneme == ph, ]
      if (nrow(sub) < 20) next
      fit_models(sub, "pc1", ph)
    }
  }
  
  sink()
  cat("Neural output saved to results/lme_neural_output.txt\n")
}