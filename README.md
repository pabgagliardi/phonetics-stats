# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

M1 Computational Linguistics — Statistics for Textual Data  
Université Paris — 2025–2026  
Pablo Gagliardi

## Project

This project applies a full statistical analysis pipeline to the Russian–French 
Interference Corpus, comparing signal-derived acoustic representations (F1, F2) 
and neural representations (Whisper, XLS-R) along several dimensions: 
variability, inter-phoneme distances, speaker clustering, and hierarchical 
phonological structure.

## Corpus

Russian–French Interference Corpus, ORTOLANG:  
https://www.ortolang.fr/market/corpora/ru-fr_interference

19 speakers (9 L1 French, 10 L2 Russian), 78 sentences, 22,596 phoneme tokens.

## Pipeline (DVC)

| Stage | Script | Output |
|-------|--------|--------|
| 1. Parse corpus | `src/parse_corpus.py` | `data/phonemes.csv` |
| 2. Extract acoustics | `src/extract_acoustics.py` | `data/features_acoustic.csv` |
| 3. Normalise | `src/normalise.py` | `data/features_acoustic