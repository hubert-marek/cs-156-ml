# Complete Experimental Results — Raw Data
## CS156 Assignment 2: Email Classification with Multilingual Transformers

---

## Dataset

- **Source:** 2 Gmail Takeout exports (Polish + English)
- **Total:** 32,783 emails → 32,783 after parsing
- **Labels:** needs_reply (4.7%), no_reply (74.7%), promotional (20.6%)
- **Split:** train 22,977 / val 3,228 / test 6,578 (thread-aware GroupShuffleSplit, SEED=42)
- **Loss:** Weighted cross-entropy (inverse class frequency)
- **Text input format:** `"subject: {subject}\nbody: {body}"`
- **Tokenizer max lengths tested:** 256, 384, 512

---

## Professor Watson's Feedback on Assignment 1 (3/4)

> This was fairly ok, but needs a little more comparison between models. For the next version of this, here's some things you might try:
> 1. Try training an auto-encoder/vectorizer on this data and or try fine tuning an embedding like BERT.
> 2. You might also try a generative model like an RNN/LSTM these are models we'll cover in unit 3, so you can look ahead to get some code bits.
> I'm also a little confused as to what you're trying to predict here. Topic modeling (like Latent Dirichlet Analysis) might be a better option for this.

**How A2 addresses this:**
- Fine-tuned BERT (DistilBERT, mBERT, XLM-R) — directly addresses suggestion #1
- Extensive model comparison (15+ configs, 4 runs) — addresses "needs more comparison"
- Clear problem statement needed: supervised email triage, not topic modeling

---

## A. Assignment 1 Baselines (TF-IDF features, same test set)

| Model | C/alpha | CV Macro F1 | Test Acc | Test Macro F1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 |
|-------|---------|-------------|----------|---------------|-------|------|------|--------|--------|
| Logistic Regression | C=10 | 0.724±0.017 | 0.85 | 0.719 | 0.43 | 0.32 | 0.67 | 0.89 | 0.83 |
| Naive Bayes | α=0.01 | 0.667±0.011 | 0.76 | 0.650 | 0.39 | 0.27 | 0.72 | 0.82 | 0.74 |
| MLP (256→128) | α=1e-5 | 0.702±0.008 | 0.89 | 0.656 | 0.20 | 0.48 | 0.13 | 0.93 | 0.84 |

---

## B. Run 1 — Original batch sizes (BS 16/8, grad_accum 2/4, max_length 256)

| Model | LR | Ep | BS | ML | WD | Test Acc | Test MacF1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 |
|-------|-----|-----|-----|------|------|----------|------------|-------|------|------|--------|--------|
| DistilBERT | 2e-5 | 3 | 16 | 256 | 0.01 | 0.8629 | 0.7551 | 0.5305 | 0.405 | 0.768 | 0.9041 | 0.8307 |
| DistilBERT | 5e-5 | 4 | 16 | 256 | 0.01 | 0.8794 | 0.7587 | 0.5111 | 0.427 | 0.637 | 0.9178 | 0.8471 |
| DistilBERT | 3e-5 | 5 | 16 | 256 | 0.01 | 0.8790 | 0.7603 | 0.5220 | 0.440 | 0.641 | 0.9175 | 0.8415 |
| XLM-R | 2e-5 | 2 | 8 | 256 | 0.01 | 0.8430 | 0.7376 | 0.5051 | 0.366 | 0.814 | 0.8868 | 0.8209 |
| XLM-R | 1e-5 | 4 | 8 | 256 | 0.01 | 0.8553 | 0.7485 | 0.5107 | 0.371 | 0.820 | 0.8974 | 0.8374 |
| XLM-R | 3e-5 | 5 | 8 | 256 | 0.10 | 0.8749 | 0.7587 | 0.5146 | 0.411 | 0.690 | 0.9138 | 0.8476 |
| mBERT | 3e-5 | 3 | 8 | 256 | 0.01 | 0.8676 | 0.7510 | 0.4988 | 0.387 | 0.703 | 0.9076 | 0.8465 |

---

## C. Run 2 — Larger batch sizes (BS 64/32, grad_accum 1, max_length 256)

Only 2 configs ran (old MODEL_SPECS not updated on Colab).

| Model | LR | Ep | BS | ML | WD | Val MacF1 | Test Acc | Test MacF1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 |
|-------|-----|-----|-----|------|------|-----------|----------|------------|-------|------|------|--------|--------|
| DistilBERT | 2e-5 | 5* | 64 | 256 | 0.01 | 0.7309 | 0.8723 | 0.7657 | 0.5452 | 0.423 | 0.768 | 0.9115 | 0.8404 |
| XLM-R | 2e-5 | 5* | 32 | 256 | 0.01 | 0.7461 | 0.8749 | 0.7635 | 0.5303 | 0.421 | 0.716 | 0.9133 | 0.8469 |

*trained 5 epochs, loaded best checkpoint; early stopping was disabled (eval_macro_f1 not found)

---

## D. Run 3 — 8 configs, original batch sizes, max_length 256/384 (overnight run)

| Model | LR | Ep | BS | ML | WD | Val MacF1 | Test Acc | Test MacF1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 |
|-------|-----|-----|-----|------|------|-----------|----------|------------|-------|------|------|--------|--------|
| DistilBERT (BS32) | 3e-5 | 5 | 16 | 256 | 0.01 | 0.7419 | 0.8753 | **0.7642** | 0.5348 | 0.427 | 0.716 | 0.9139 | 0.8437 |
| DistilBERT (ML384) | 5e-5 | 4 | 16 | 384 | 0.01 | 0.7432 | 0.8734 | 0.7616 | 0.5254 | 0.412 | 0.726 | 0.9121 | 0.8473 |
| XLM-R (WD0.1) | 3e-5 | 5 | 16 | 256 | 0.10 | 0.7442 | 0.8740 | 0.7542 | 0.5006 | 0.402 | 0.663 | 0.9132 | 0.8488 |
| DistilBERT | 5e-5 | 4 | 16 | 256 | 0.01 | 0.7462 | 0.8679 | 0.7525 | 0.5034 | 0.387 | 0.719 | 0.9081 | 0.8458 |
| mBERT | 3e-5 | 3 | 16 | 256 | 0.01 | 0.7379 | 0.8658 | 0.7484 | 0.4947 | 0.385 | 0.693 | 0.9062 | 0.8443 |
| XLM-R | 1e-5 | 4 | 16 | 256 | 0.01 | 0.7287 | 0.8548 | 0.7476 | 0.5082 | 0.370 | 0.810 | 0.8968 | 0.8377 |
| DistilBERT | 2e-5 | 3 | 16 | 256 | 0.01 | 0.7274 | 0.8465 | 0.7403 | 0.5020 | 0.359 | 0.833 | 0.8902 | 0.8288 |
| XLM-R | 2e-5 | 2 | 16 | 256 | 0.01 | 0.7137 | 0.8355 | 0.7324 | 0.4962 | 0.348 | 0.863 | 0.8804 | 0.8205 |

---

## E. Run 4 — More epochs (8-12), larger batches (BS 128/64), max_length 512

Early stopping still disabled (eval_macro_f1 bug).

| Model | LR | Ep | BS | ML | WD | Stopped | Val MacF1 | Test Acc | Test MacF1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 |
|-------|-----|-----|------|------|------|---------|-----------|----------|------------|-------|------|------|--------|--------|
| DistilBERT | 2e-5 | 8 | 128 | 512 | 0.01 | ep 11 | 0.7454 | 0.8731 | 0.7632 | 0.5413 | 0.427 | 0.739 | 0.9128 | 0.8356 |
| DistilBERT | 3e-5 | 8 | 128 | 512 | 0.01 | ep 9 | 0.7394 | 0.8696 | 0.7565 | 0.5217 | 0.401 | 0.745 | 0.9105 | 0.8373 |
| XLM-R | 1e-5 | 8 | 64 | 512 | 0.01 | ep 12 | — | 0.8604 | 0.7519 | 0.5172 | 0.392 | 0.761 | 0.9012 | 0.8372 |
| DistilBERT | 5e-5 | 8 | 128 | 512 | 0.01 | ep 12 | 0.7528 | 0.8810 | 0.7483 | 0.4788 | 0.422 | 0.552 | 0.9197 | 0.8463 |
| XLM-R | 3e-5 | 8 | 64 | 512 | 0.10 | ep 7 | 0.7324 | 0.8484 | 0.7417 | 0.4889 | 0.341 | 0.863 | 0.8919 | 0.8442 |
| XLM-R | 2e-5 | 8 | 64 | 512 | 0.01 | ep 7 | 0.7251 | 0.8462 | 0.7389 | 0.4883 | 0.337 | 0.889 | 0.8916 | 0.8369 |
| DistilBERT | 2e-5 | 3 | 128 | 512 | 0.01 | ep 3 | 0.7274 | 0.8498 | 0.7441 | 0.5075 | 0.365 | 0.833 | 0.8928 | 0.8321 |

---

## F. Validation Loss vs Macro F1 Trajectories (Run 4, per-epoch)

### DistilBERT LR 2e-5, BS 128, ML 512

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R | Val NoR_F1 |
|----|-----------|----------|-----------|-----------|----------|------------|
| 1 | 1.082 | 0.579 | 0.531 | 0.282 | 0.949 | 0.711 |
| 2 | 0.444 | 0.392 | 0.671 | 0.461 | 0.897 | 0.827 |
| 3 | 0.328 | 0.377 | 0.701 | 0.449 | 0.934 | 0.868 |
| 4 | 0.279 | 0.363 | 0.713 | 0.486 | 0.882 | 0.871 |
| 5 | 0.231 | 0.400 | 0.736 | 0.488 | 0.897 | 0.898 |
| 6 | 0.196 | 0.391 | 0.728 | 0.479 | 0.853 | 0.889 |
| 7 | 0.170 | 0.456 | 0.737 | 0.484 | 0.816 | 0.901 |
| 8 | 0.146 | 0.529 | 0.741 | 0.485 | 0.713 | 0.910 |
| 9 | 0.128 | 0.587 | 0.745 | 0.500 | 0.706 | 0.913 |
| 10 | 0.123 | 0.653 | 0.739 | 0.488 | 0.676 | 0.912 |
| 11 | 0.110 | 0.664 | 0.742 | 0.489 | 0.647 | 0.915 |

Pattern: Val loss rises after ep 3, but macro F1 keeps climbing until ep 9. NR recall drops steadily (0.95→0.65) while NR precision rises — model gets conservative.

### DistilBERT LR 5e-5, BS 128, ML 512

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R | Val NoR_F1 |
|----|-----------|----------|-----------|-----------|----------|------------|
| 1 | 0.990 | 0.423 | 0.644 | 0.423 | 0.875 | 0.806 |
| 2 | 0.375 | 0.394 | 0.675 | 0.468 | 0.860 | 0.831 |
| 3 | 0.282 | 0.358 | 0.716 | 0.464 | 0.882 | 0.879 |
| 4 | 0.215 | 0.399 | 0.709 | 0.468 | 0.882 | 0.871 |
| 5 | 0.179 | 0.454 | 0.725 | 0.485 | 0.831 | 0.888 |
| 6 | 0.154 | 0.545 | 0.732 | 0.484 | 0.809 | 0.899 |
| 7 | 0.119 | 0.808 | 0.738 | 0.495 | 0.676 | 0.912 |
| 8 | 0.102 | 0.829 | 0.747 | 0.501 | 0.654 | 0.917 |
| 9 | 0.082 | 0.811 | 0.742 | 0.491 | 0.669 | 0.912 |
| 10 | 0.077 | 0.990 | 0.753 | 0.518 | 0.640 | 0.920 |
| 11 | 0.062 | 1.126 | 0.741 | 0.479 | 0.507 | 0.922 |
| 12 | 0.049 | 1.178 | 0.740 | 0.483 | 0.529 | 0.922 |

Pattern: Val loss explodes (0.36→1.18) but macro F1 peaks at ep 10 then drops. NR recall collapses from 0.88→0.51 — severe overfitting on majority class.

### DistilBERT LR 3e-5, BS 128, ML 512

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R | Val NoR_F1 |
|----|-----------|----------|-----------|-----------|----------|------------|
| 1 | 1.048 | 0.498 | 0.617 | 0.363 | 0.934 | 0.797 |
| 2 | 0.415 | 0.360 | 0.667 | 0.432 | 0.926 | 0.830 |
| 3 | 0.296 | 0.350 | 0.702 | 0.443 | 0.941 | 0.864 |
| 4 | 0.236 | 0.381 | 0.716 | 0.481 | 0.846 | 0.878 |
| 5 | 0.189 | 0.402 | 0.725 | 0.473 | 0.831 | 0.890 |
| 6 | 0.164 | 0.521 | 0.738 | 0.501 | 0.757 | 0.900 |
| 7 | 0.137 | 0.637 | 0.739 | 0.506 | 0.728 | 0.909 |
| 8 | 0.117 | 0.729 | 0.739 | 0.500 | 0.713 | 0.910 |
| 9 | 0.099 | 0.761 | 0.730 | 0.475 | 0.632 | 0.906 |

Pattern: Peaks at ep 6-7 then declines. Early stopping at patience=2 would have stopped at ep 9.

### XLM-R LR 2e-5, BS 64, ML 512

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R | Val NoR_F1 |
|----|-----------|----------|-----------|-----------|----------|------------|
| 1 | 0.597 | 0.453 | 0.626 | 0.361 | 0.963 | 0.790 |
| 2 | 0.397 | 0.404 | 0.687 | 0.450 | 0.941 | 0.850 |
| 3 | 0.345 | 0.360 | 0.691 | 0.472 | 0.904 | 0.846 |
| 4 | 0.284 | 0.374 | 0.708 | 0.462 | 0.934 | 0.868 |
| 5 | 0.240 | 0.412 | 0.725 | 0.470 | 0.926 | 0.890 |
| 6 | 0.191 | 0.450 | 0.725 | 0.462 | 0.882 | 0.889 |
| 7 | 0.168 | 0.420 | 0.719 | 0.474 | 0.890 | 0.879 |

Pattern: Slow improvement, peaks at ep 5-6. Very aggressive on needs_reply throughout (recall >88%).

### XLM-R LR 3e-5, BS 64, ML 512, WD 0.1

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R | Val NoR_F1 |
|----|-----------|----------|-----------|-----------|----------|------------|
| 1 | 0.507 | 0.439 | 0.624 | 0.353 | 0.971 | 0.783 |
| 2 | 0.364 | 0.391 | 0.696 | 0.451 | 0.919 | 0.859 |
| 3 | 0.313 | 0.343 | 0.708 | 0.460 | 0.941 | 0.866 |
| 4 | 0.270 | 0.395 | 0.711 | 0.453 | 0.941 | 0.873 |
| 5 | 0.230 | 0.359 | 0.732 | 0.470 | 0.912 | 0.892 |
| 6 | 0.184 | 0.452 | 0.732 | 0.472 | 0.846 | 0.895 |
| 7 | 0.166 | 0.501 | 0.730 | 0.482 | 0.757 | 0.896 |

Pattern: WD 0.1 helps XLM-R converge. Peaks at ep 5-6. NR recall drops from 97%→76%.

---

## G. Final Retrain (train+val combined, DistilBERT LR 5e-5)

Training set: 26,205 (22,977 train + 3,228 val merged)
Test set: 6,578 (unchanged)
No validation → no early stopping

### Attempt 1: 9 epochs

| Metric | Value |
|--------|-------|
| Test Acc | 0.8913 |
| Test Macro F1 | 0.7359 |
| NR F1 | 0.4325 (P: 0.498, R: 0.382) |
| NoR F1 | 0.9281 |
| Pro F1 | 0.8472 |

OVERFIT — trained too long without validation checkpoint selection. NR recall collapsed to 0.38.

### Attempt 2: 4 epochs

| Metric | Value |
|--------|-------|
| Test Acc | 0.8632 |
| Test Macro F1 | 0.7545 |
| NR F1 | 0.5176 (P: 0.384, R: 0.794) |
| NoR F1 | 0.9042 |
| Pro F1 | 0.8418 |

Better, but still worse than sweep best (0.764).

**Conclusion:** Final retrain without validation hurts. Sweep best (0.764) remains the best result.

---

## H. Confusion Matrices (test set, selected models)

Test set: 306 needs_reply, 4917 no_reply, 1355 promotional

### Best overall: DistilBERT LR 3e-5 Ep 5 (Macro F1 = 0.764)

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **219** | 87 | 0 |
| true NoR | 294 | **4354** | 269 |
| true Pro | 0 | 170 | **1185** |

### DistilBERT LR 5e-5 Ep 4, ML 384

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **222** | 84 | 0 |
| true NoR | 317 | **4324** | 276 |
| true Pro | 0 | 156 | **1199** |

### mBERT LR 3e-5 Ep 3

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **212** | 94 | 0 |
| true NoR | 339 | **4263** | 315 |
| true Pro | 0 | 135 | **1220** |

### XLM-R LR 1e-5 Ep 4

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **248** | 58 | 0 |
| true NoR | 422 | **4149** | 346 |
| true Pro | 0 | 129 | **1226** |

### Best NR recall: XLM-R LR 2e-5 Ep 8 (NR recall = 0.889)

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **272** | 34 | 0 |
| true NoR | 536 | **4160** | 221 |
| true Pro | 0 | 221 | **1134** |

### XLM-R LR 3e-5 Ep 8 WD 0.1 (Run 4)

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **264** | 42 | 0 |
| true NoR | 510 | **4111** | 296 |
| true Pro | 0 | 149 | **1206** |

### Most conservative: DistilBERT LR 5e-5 Ep 8 (NR recall = 0.552)

|  | pred NR | pred NoR | pred Pro |
|---|---------|----------|----------|
| true NR | **169** | 137 | 0 |
| true NoR | 231 | **4480** | 206 |
| true Pro | 0 | 192 | **1163** |

---

## I. Error Analysis (best model: DistilBERT LR 3e-5 Ep 5)

Total: 6578 | Correct: 5758 (87.5%) | Wrong: 820 (12.5%)

| Error type | Count | % errors | % of true class |
|-----------|-------|----------|-----------------|
| no_reply → needs_reply | 294 | 35.9% | 6.0% of no_reply |
| no_reply → promotional | 269 | 32.8% | 5.5% of no_reply |
| promotional → no_reply | 170 | 20.7% | 12.5% of promotional |
| needs_reply → no_reply | 87 | 10.6% | 28.4% of needs_reply |

No model ever predicts needs_reply when true=promotional (always 0).

### Sample misclassifications

**needs_reply → no_reply (missed — model failed to flag):**
- Re: Econverse x Freenow
- Re: Symfonia elektroniczny obieg dokumentów
- Re: Econverse x ALK
- Fwd: Faktura numer FV/43/07/2023 - aftermovie PCN
- Webexpenses & Econverse : Pricing Proposal
- Your Minoxidil Express order receipt

**no_reply → needs_reply (false alarm — unnecessary flag):**
- RE: Econverse Cup - zaproszenie do współpracy
- RE: Econverse Cup - zaproszenie do współpracy
- Fwd: Faktury za Econverse Poznań
- RE: Zdjęcia z elimicjacji Econnverse Cup - warszawa
- Post o członkostwie Artura w Jury
- Re: Econverse - FV za wynajem sali

**promotional → no_reply (promo slipped through as personal):**
- Early access to ClickUp AI is now available!
- Alex Terrana has invited you to work with them in Slack
- Welcome to MURAL
- Welcome to Fireflies
- Invite your teammates
- Project Management team Weekly Digest

**no_reply → promotional (personal misread as promo):**
- Zapisz się na webinary o podatkach 2023 i DMS
- Startuje sprzedaż na wszystko.pl!
- Automation filters and Dashboard Enhancements
- Quick Productivity Wins for 2023: Adding Favorites in ClickUp
- Hubert, zostały tylko 2 dni – otrzymaj do470 zł zwrotu!
- Please verify your email address with Citywire

---

## J. Architecture Comparison (best config per architecture)

| Arch | Params | Best Config | Macro F1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 | Acc |
|------|--------|-------------|----------|-------|------|------|--------|--------|-----|
| DistilBERT | 66M | LR=3e-5, Ep=5 | **0.764** | **0.535** | 0.427 | 0.716 | 0.914 | 0.844 | 0.875 |
| XLM-R | 278M | LR=3e-5, Ep=5, WD0.1 | 0.754 | 0.501 | 0.402 | 0.663 | 0.913 | 0.849 | 0.874 |
| mBERT | 178M | LR=3e-5, Ep=3 | 0.748 | 0.495 | 0.385 | 0.693 | 0.906 | 0.844 | 0.866 |

DistilBERT (66M) > XLM-R (278M) > mBERT (178M). Smaller model wins.

---

## K. Hyperparameter Analysis

### Learning rate effect (DistilBERT, all configs sorted by macro F1)

| LR | Ep | Extra | Macro F1 | NR F1 | NR P | NR R |
|-----|-----|-------|----------|-------|------|------|
| 3e-5 | 5 | BS32 | 0.7642 | 0.5348 | 0.427 | 0.716 |
| 2e-5 | 8 | — | 0.7632 | 0.5413 | 0.427 | 0.739 |
| 5e-5 | 4 | ML384 | 0.7616 | 0.5254 | 0.412 | 0.726 |
| 3e-5 | 8 | — | 0.7565 | 0.5217 | 0.401 | 0.745 |
| 5e-5 | 4 | — | 0.7525 | 0.5034 | 0.387 | 0.719 |
| 5e-5 | 8 | — | 0.7483 | 0.4788 | 0.422 | 0.552 |
| 2e-5 | 3 | — | 0.7403 | 0.5020 | 0.359 | 0.833 |

Higher LR → lower recall, higher precision (conservative). Lower LR → higher recall, lower precision (aggressive).

### Epoch effect (same arch + LR, different epochs)

| Config | Epochs | Best Ep | Best MacF1 | Overfit? |
|--------|--------|---------|------------|---------|
| DistilBERT 2e-5 | 3 → 8 | 8 | 0.763 | no |
| DistilBERT 3e-5 | 5 → 8 | 5 | 0.764 | YES |
| DistilBERT 5e-5 | 4 → 4 → 8 | 4 | 0.762 | YES |
| XLM-R 1e-5 | 4 → 8 | 8 | 0.752 | no |
| XLM-R 2e-5 | 2 → 8 | 8 | 0.739 | no |

### max_length effect (same model, different ML)

| ML | Config | Macro F1 |
|-----|--------|----------|
| 256 | DistilBERT 3e-5 Ep5 | **0.764** |
| 384 | DistilBERT 5e-5 Ep4 | 0.762 |
| 512 | DistilBERT 2e-5 Ep8 | 0.763 |

No meaningful improvement from longer input. Classification signal is in subject + first sentences.

### Weight decay effect (XLM-R, LR 3e-5)

| WD | Run | Ep | Macro F1 | NR Recall |
|------|------|-----|----------|-----------|
| 0.01 | (no config tested) | — | — | — |
| 0.10 | Run 1 | 5 | 0.759 | 0.690 |
| 0.10 | Run 3 | 5 | 0.754 | 0.663 |
| 0.10 | Run 4 | 8 | 0.742 | 0.863 |

WD 0.1 helps XLM-R regularize but doesn't close the gap to DistilBERT.

---

## L. Precision-Recall Spectrum on needs_reply (all models)

| Model | NR P | NR R | NR F1 | Profile |
|-------|------|------|-------|---------|
| XLM-R 2e-5 ep8 | 0.337 | 0.889 | 0.488 | AGGRESSIVE |
| XLM-R 2e-5 ep2 | 0.348 | 0.863 | 0.496 | AGGRESSIVE |
| DistilBERT 2e-5 ep3 | 0.359 | 0.833 | 0.502 | AGGRESSIVE |
| XLM-R 1e-5 ep4 | 0.370 | 0.810 | 0.508 | AGGRESSIVE |
| XLM-R 1e-5 ep8 | 0.392 | 0.761 | 0.517 | BALANCED |
| DistilBERT 3e-5 ep8 | 0.401 | 0.745 | 0.522 | BALANCED |
| DistilBERT 2e-5 ep8 | 0.427 | 0.739 | 0.541 | BALANCED |
| DistilBERT 5e-5 ep4 ML384 | 0.412 | 0.726 | 0.525 | BALANCED |
| DistilBERT 5e-5 ep4 | 0.387 | 0.719 | 0.503 | BALANCED |
| DistilBERT 3e-5 ep5 | 0.427 | 0.716 | 0.535 | BALANCED |
| mBERT 3e-5 ep3 | 0.385 | 0.693 | 0.495 | BALANCED |
| XLM-R 3e-5 ep5 WD0.1 | 0.402 | 0.663 | 0.501 | BALANCED |
| DistilBERT 5e-5 ep8 | 0.422 | 0.552 | 0.479 | CONSERVATIVE |

Best F1 comes from balanced models (65-75% recall, 38-43% precision).

---

## M. Technical Notes

1. **Early stopping bug:** `metric_for_best_model="eval_macro_f1"` was not found by EarlyStoppingCallback in any run. All models trained to max epochs. `load_best_model_at_end=True` loaded the best checkpoint by val loss (not macro F1).

2. **LayerNorm warnings:** "missing LayerNorm.weight/bias, unexpected LayerNorm.beta/gamma" — cosmetic, old vs new HuggingFace naming convention. No effect on training.

3. **warmup_ratio deprecation:** Harmless warning, warmup still applied correctly.

4. **Val loss vs macro F1 divergence:** Val loss increases while macro F1 improves. Cross-entropy penalizes confidence; F1 only cares about argmax decisions. Model becomes more decisive but less calibrated — normal for fine-tuned transformers.

5. **Header object bug:** `email.header.Header` objects in from/to/cc fields caused ArrowTypeError on Parquet export. Fixed by wrapping with `str()`.

6. **Transformers API breaking changes (v5.x):**
   - `evaluation_strategy` → `eval_strategy`
   - `overwrite_output_dir` removed
   - `Trainer(tokenizer=)` → `Trainer(processing_class=)`
   - `DataCollatorWithPadding` still uses `tokenizer=`
   - `NotebookProgressCallback` crashes on post-training `evaluate()`
   - `-U` flag on pip install breaks torch/numpy compatibility on Colab

---

## N-1. Run 5 — Additional configs (from report.md paste)

Includes new mBERT configs, XLM-R 1e-5 full 12-epoch trajectory, and mDeBERTa OOM attempts.
Run on Colab A100 22GB, BS 64/128, ML 512. NotebookProgressCallback bug caused test eval failures on some models.

### New test results

| Model | LR | Ep | BS | ML | WD | Stopped | Val MacF1 | Test Acc | Test MacF1 | NR F1 | NR P | NR R | NoR F1 | Pro F1 |
|-------|-----|-----|------|------|------|---------|-----------|----------|------------|-------|------|------|--------|--------|
| mBERT | 3e-5 | 8 | 64 | 512 | 0.01 | ep 8 | 0.7486 | 0.8697 | **0.7572** | 0.5073 | 0.385 | 0.742 | 0.9086 | 0.8558 |
| XLM-R | 1e-5 | 8 | 64 | 512 | 0.01 | ep 12 | 0.7453 | 0.8604 | 0.7519 | 0.5172 | 0.392 | 0.761 | 0.9012 | 0.8372 |

### Per-epoch trajectories (new models)

**mBERT LR 3e-5, Ep 8, BS 64, ML 512:**

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R |
|----|-----------|----------|-----------|-----------|----------|
| 1 | 0.470 | 0.431 | 0.634 | 0.377 | 0.956 |
| 2 | 0.365 | 0.408 | 0.700 | 0.457 | 0.926 |
| 3 | 0.287 | 0.374 | 0.719 | 0.474 | 0.846 |
| 4 | 0.239 | 0.421 | 0.725 | 0.473 | 0.890 |
| 5 | 0.194 | 0.428 | 0.743 | 0.506 | 0.831 |
| 6 | 0.165 | 0.536 | 0.749 | 0.523 | 0.779 |
| 7 | 0.123 | 0.721 | 0.743 | 0.492 | 0.662 |
| 8 | 0.117 | 0.881 | 0.742 | 0.497 | 0.625 |

Peak at epoch 6 (val macro F1 = 0.749). Same conservative drift pattern as DistilBERT.

**XLM-R LR 1e-5, Ep 8, BS 64, ML 512 (full 12 epochs):**

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R |
|----|-----------|----------|-----------|-----------|----------|
| 1 | 0.756 | 0.460 | 0.605 | 0.340 | 0.963 |
| 2 | 0.411 | 0.397 | 0.663 | 0.439 | 0.941 |
| 3 | 0.348 | 0.366 | 0.687 | 0.460 | 0.941 |
| 4 | 0.312 | 0.368 | 0.675 | 0.450 | 0.949 |
| 5 | 0.272 | 0.361 | 0.695 | 0.434 | 0.963 |
| 6 | 0.236 | 0.375 | 0.704 | 0.443 | 0.934 |
| 7 | 0.215 | 0.383 | 0.700 | 0.454 | 0.934 |
| 8 | 0.214 | 0.418 | 0.719 | 0.471 | 0.882 |
| 9 | 0.179 | 0.433 | 0.724 | 0.469 | 0.904 |
| 10 | 0.184 | 0.446 | 0.737 | 0.496 | 0.860 |
| 11 | 0.167 | 0.498 | 0.746 | 0.517 | 0.816 |
| 12 | 0.166 | 0.488 | 0.740 | 0.501 | 0.831 |

Slowest converger — still improving at epoch 11 (val macro F1 = 0.746). Peaked at ep 11 then slight decline.

**mBERT LR 2e-5, Ep 5, BS 32 (new config):**

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_R |
|----|-----------|----------|-----------|-----------|----------|
| 1 | 0.417 | 0.476 | 0.692 | 0.430 | 0.926 |
| 2 | 0.290 | 0.365 | 0.712 | 0.476 | 0.919 |
| 3 | 0.230 | 0.372 | 0.722 | 0.486 | 0.875 |
| 4 | 0.199 | 0.435 | 0.725 | 0.485 | 0.838 |
| 5 | 0.147 | 0.499 | 0.740 | 0.498 | 0.787 |

Test eval crashed (NotebookProgressCallback bug). Val macro F1 = 0.740 at epoch 5, still rising.

### Models that failed to evaluate (training succeeded, per-epoch val data available)

Due to NotebookProgressCallback bug in Run 3, these models completed training but crashed on `trainer.evaluate()`:
- DistilBERT (LR 2e-5, Ep 3)
- DistilBERT (LR 5e-5, Ep 4)
- DistilBERT (LR 3e-5, Ep 5, BS 32)
- XLM-R (LR 2e-5, Ep 2)
- XLM-R (LR 1e-5, Ep 4)
- XLM-R (LR 3e-5, Ep 5, WD 0.1)
- mBERT (LR 3e-5, Ep 3)

Per-epoch validation metrics are in section F trajectories above. Test metrics for these configs come from the successful Run 3 (section D).

### mDeBERTa V3 base — OOM

| Config | Error |
|--------|-------|
| mDeBERTa V3 (LR 2e-5, Ep 3) | CUDA OOM — 22GB A100 insufficient at BS 64, ML 512 |
| mDeBERTa V3 (LR 1e-5, Ep 5) | CUDA OOM — same |

mDeBERTa has 278M params (same as XLM-R) but its disentangled attention mechanism uses more memory. Would require BS 8-16 or gradient checkpointing.

---

## N. Files and Artifacts

### Local CSVs (export_csvs/)
- Run 3: 8 models × (classification_report.csv + test_predictions.csv) + transformer_summary.csv + baseline_vs_transformers.csv

### Local CSVs (export_csvs_v3/)
- Run 4: 6 models × (classification_report.csv + test_predictions.csv)

### Colab artifacts (artifacts/assignment2_transformers/)
- Per-model folders with best_model/, metrics.json, classification_report.csv, test_predictions.csv
- transformer_summary.csv, baseline_vs_transformers.csv

### Notebooks
- `pipeline_final.ipynb` — Assignment 1 full pipeline (source of truth for A1 baselines)
- `assignment2_gpu_transformers.ipynb` — Original GPU notebook
- `assignment2_gpu_transformers_V3.ipynb` — Current working version with all fixes

## O. Complete Per-Epoch Trajectories (all models, all runs)

### Overfitting Summary

| Model | Total Ep | Best Ep | Best Val MacF1 | Final Val MacF1 | NR Recall (ep1→final) | Overfit? |
|-------|---------|---------|----------------|-----------------|----------------------|---------|
| DistilBERT (LR 2e-5, Ep 8) | 11 | 9 | 0.7454 | 0.7423 | 0.949 → 0.647 | YES |
| DistilBERT (LR 5e-5, Ep 8) | 12 | 10 | 0.7528 | 0.7397 | 0.875 → 0.529 | YES |
| DistilBERT (LR 3e-5, Ep 8) | 9 | 7 | 0.7394 | 0.7297 | 0.934 → 0.632 | YES |
| XLM-R base (LR 2e-5, Ep 8) | 7 | 5 | 0.7251 | 0.7188 | 0.963 → 0.890 | YES |
| XLM-R base (LR 1e-5, Ep 8) | 12 | 11 | 0.7463 | 0.7398 | 0.963 → 0.831 | YES |
| XLM-R base (LR 3e-5, Ep 8, WD 0.1) | 7 | 5 | 0.7324 | 0.7303 | 0.971 → 0.757 | YES |
| mBERT base (LR 3e-5, Ep 8) | 8 | 6 | 0.7486 | 0.7422 | 0.956 → 0.625 | YES |
| DistilBERT (LR 2e-5, Ep 3) | 3 | 2 | 0.7227 | 0.7195 | 0.919 → 0.743 | YES |
| DistilBERT (LR 5e-5, Ep 4) | 4 | 4 | 0.7491 | 0.7491 | 0.904 → 0.662 | no |
| DistilBERT (LR 3e-5, Ep 5, BS 32) | 5 | 5 | 0.7400 | 0.7400 | 0.919 → 0.721 | no |
| XLM-R base (LR 2e-5, Ep 2) | 2 | 2 | 0.7172 | 0.7172 | 0.912 → 0.890 | no |
| XLM-R base (LR 1e-5, Ep 4) | 4 | 4 | 0.7271 | 0.7271 | 0.926 → 0.787 | no |
| XLM-R base (LR 3e-5, Ep 5, WD 0.1) | 5 | 5 | 0.7503 | 0.7503 | 0.868 → 0.676 | no |
| mBERT base (LR 3e-5, Ep 3) | 3 | 3 | 0.7389 | 0.7389 | 0.919 → 0.706 | no |
| mBERT base (LR 2e-5, Ep 5, BS 32) | 5 | 5 | 0.7402 | 0.7402 | 0.926 → 0.787 | no |

**Universal pattern:** NR recall drops with training (models become conservative). Every model that ran >5 epochs shows overfitting on val macro F1.

### DistilBERT (LR 2e-5, Ep 8)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 1.082 | 0.579 | 0.5310 | 0.2820 | 0.1656 | 0.9485 | 0.7107 | 0.6003 |
| 2 | 0.444 | 0.392 | 0.6708 | 0.4612 | 0.3104 | 0.8971 | 0.8275 | 0.7238 |
| 3 | 0.328 | 0.377 | 0.7008 | 0.4488 | 0.2953 | 0.9338 | 0.8684 | 0.7853 |
| 4 | 0.279 | 0.363 | 0.7132 | 0.4858 | 0.3352 | 0.8824 | 0.8715 | 0.7824 |
| 5 | 0.231 | 0.400 | 0.7365 | 0.4880 | 0.3352 | 0.8971 | 0.8975 | 0.8239 |
| 6 | 0.196 | 0.391 | 0.7276 | 0.4793 | 0.3333 | 0.8529 | 0.8890 | 0.8144 |
| 7 | 0.170 | 0.456 | 0.7373 | 0.4837 | 0.3437 | 0.8162 | 0.9013 | 0.8270 |
| 8 | 0.146 | 0.529 | 0.7407 | 0.4850 | 0.3674 | 0.7132 | 0.9100 | 0.8270 |
| 9 | 0.128 | 0.587 | 0.7454 | 0.5000 | 0.3871 | 0.7059 | 0.9128 | 0.8234 |
| 10 | 0.123 | 0.653 | 0.7388 | 0.4881 | 0.3817 | 0.6765 | 0.9124 | 0.8160 |
| 11 | 0.110 | 0.664 | 0.7423 | 0.4889 | 0.3929 | 0.6471 | 0.9145 | 0.8236 |

### DistilBERT (LR 5e-5, Ep 8)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.990 | 0.423 | 0.6435 | 0.4235 | 0.2793 | 0.8750 | 0.8060 | 0.7011 |
| 2 | 0.375 | 0.394 | 0.6749 | 0.4680 | 0.3214 | 0.8603 | 0.8311 | 0.7257 |
| 3 | 0.282 | 0.358 | 0.7160 | 0.4642 | 0.3150 | 0.8824 | 0.8789 | 0.8049 |
| 4 | 0.215 | 0.399 | 0.7089 | 0.4678 | 0.3183 | 0.8824 | 0.8706 | 0.7882 |
| 5 | 0.179 | 0.454 | 0.7253 | 0.4850 | 0.3424 | 0.8309 | 0.8883 | 0.8026 |
| 6 | 0.154 | 0.545 | 0.7323 | 0.4835 | 0.3448 | 0.8088 | 0.8995 | 0.8140 |
| 7 | 0.119 | 0.808 | 0.7385 | 0.4946 | 0.3898 | 0.6765 | 0.9116 | 0.8093 |
| 8 | 0.102 | 0.829 | 0.7471 | 0.5014 | 0.4064 | 0.6544 | 0.9175 | 0.8223 |
| 9 | 0.082 | 0.811 | 0.7416 | 0.4906 | 0.3872 | 0.6691 | 0.9115 | 0.8228 |
| 10 | 0.077 | 0.990 | 0.7528 | 0.5179 | 0.4350 | 0.6397 | 0.9202 | 0.8204 |
| 11 | 0.062 | 1.126 | 0.7409 | 0.4792 | 0.4539 | 0.5074 | 0.9218 | 0.8219 |
| 12 | 0.049 | 1.178 | 0.7397 | 0.4832 | 0.4444 | 0.5294 | 0.9220 | 0.8139 |

### DistilBERT (LR 3e-5, Ep 8)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 1.048 | 0.498 | 0.6168 | 0.3634 | 0.2256 | 0.9338 | 0.7969 | 0.6901 |
| 2 | 0.415 | 0.360 | 0.6674 | 0.4315 | 0.2812 | 0.9265 | 0.8299 | 0.7407 |
| 3 | 0.296 | 0.350 | 0.7018 | 0.4429 | 0.2896 | 0.9412 | 0.8645 | 0.7980 |
| 4 | 0.236 | 0.381 | 0.7164 | 0.4812 | 0.3363 | 0.8456 | 0.8778 | 0.7901 |
| 5 | 0.189 | 0.402 | 0.7246 | 0.4728 | 0.3304 | 0.8309 | 0.8898 | 0.8112 |
| 6 | 0.164 | 0.521 | 0.7375 | 0.5012 | 0.3745 | 0.7574 | 0.8996 | 0.8118 |
| 7 | 0.137 | 0.637 | 0.7394 | 0.5064 | 0.3882 | 0.7279 | 0.9085 | 0.8032 |
| 8 | 0.117 | 0.729 | 0.7387 | 0.5000 | 0.3849 | 0.7132 | 0.9098 | 0.8061 |
| 9 | 0.099 | 0.761 | 0.7297 | 0.4751 | 0.3805 | 0.6324 | 0.9056 | 0.8083 |

### XLM-R base (LR 2e-5, Ep 8)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.597 | 0.453 | 0.6263 | 0.3609 | 0.2220 | 0.9632 | 0.7896 | 0.7285 |
| 2 | 0.397 | 0.404 | 0.6871 | 0.4499 | 0.2956 | 0.9412 | 0.8505 | 0.7608 |
| 3 | 0.345 | 0.360 | 0.6911 | 0.4722 | 0.3195 | 0.9044 | 0.8461 | 0.7550 |
| 4 | 0.284 | 0.374 | 0.7079 | 0.4618 | 0.3068 | 0.9338 | 0.8679 | 0.7938 |
| 5 | 0.240 | 0.412 | 0.7251 | 0.4701 | 0.3150 | 0.9265 | 0.8897 | 0.8153 |
| 6 | 0.191 | 0.450 | 0.7248 | 0.4624 | 0.3133 | 0.8824 | 0.8888 | 0.8230 |
| 7 | 0.168 | 0.420 | 0.7188 | 0.4736 | 0.3227 | 0.8897 | 0.8789 | 0.8039 |

### XLM-R base (LR 1e-5, Ep 8)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.756 | 0.460 | 0.6050 | 0.3403 | 0.2066 | 0.9632 | 0.7635 | 0.7113 |
| 2 | 0.411 | 0.397 | 0.6631 | 0.4391 | 0.2864 | 0.9412 | 0.8189 | 0.7314 |
| 3 | 0.348 | 0.366 | 0.6866 | 0.4604 | 0.3048 | 0.9412 | 0.8419 | 0.7574 |
| 4 | 0.312 | 0.368 | 0.6746 | 0.4503 | 0.2952 | 0.9485 | 0.8295 | 0.7441 |
| 5 | 0.272 | 0.361 | 0.6954 | 0.4345 | 0.2805 | 0.9632 | 0.8571 | 0.7947 |
| 6 | 0.236 | 0.375 | 0.7037 | 0.4433 | 0.2906 | 0.9338 | 0.8660 | 0.8017 |
| 7 | 0.215 | 0.383 | 0.7002 | 0.4536 | 0.2995 | 0.9338 | 0.8592 | 0.7879 |
| 8 | 0.214 | 0.418 | 0.7194 | 0.4706 | 0.3209 | 0.8824 | 0.8801 | 0.8076 |
| 9 | 0.179 | 0.433 | 0.7242 | 0.4695 | 0.3170 | 0.9044 | 0.8840 | 0.8191 |
| 10 | 0.184 | 0.446 | 0.7373 | 0.4958 | 0.3482 | 0.8603 | 0.8946 | 0.8214 |
| 11 | 0.167 | 0.498 | 0.7463 | 0.5175 | 0.3788 | 0.8162 | 0.9011 | 0.8203 |
| 12 | 0.166 | 0.488 | 0.7398 | 0.5011 | 0.3587 | 0.8309 | 0.8977 | 0.8206 |

### XLM-R base (LR 3e-5, Ep 8, WD 0.1)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.507 | 0.439 | 0.6245 | 0.3529 | 0.2157 | 0.9706 | 0.7832 | 0.7373 |
| 2 | 0.364 | 0.391 | 0.6957 | 0.4513 | 0.2990 | 0.9191 | 0.8588 | 0.7772 |
| 3 | 0.313 | 0.343 | 0.7084 | 0.4604 | 0.3048 | 0.9412 | 0.8662 | 0.7986 |
| 4 | 0.270 | 0.395 | 0.7111 | 0.4531 | 0.2984 | 0.9412 | 0.8725 | 0.8078 |
| 5 | 0.230 | 0.359 | 0.7324 | 0.4697 | 0.3163 | 0.9118 | 0.8923 | 0.8353 |
| 6 | 0.184 | 0.452 | 0.7322 | 0.4723 | 0.3276 | 0.8456 | 0.8945 | 0.8299 |
| 7 | 0.166 | 0.501 | 0.7303 | 0.4824 | 0.3540 | 0.7574 | 0.8958 | 0.8125 |

### mBERT base (LR 3e-5, Ep 8)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.470 | 0.431 | 0.6336 | 0.3774 | 0.2351 | 0.9559 | 0.7935 | 0.7301 |
| 2 | 0.365 | 0.408 | 0.7003 | 0.4565 | 0.3029 | 0.9265 | 0.8669 | 0.7775 |
| 3 | 0.287 | 0.374 | 0.7186 | 0.4742 | 0.3295 | 0.8456 | 0.8798 | 0.8017 |
| 4 | 0.239 | 0.421 | 0.7254 | 0.4727 | 0.3218 | 0.8897 | 0.8877 | 0.8158 |
| 5 | 0.194 | 0.428 | 0.7435 | 0.5056 | 0.3633 | 0.8309 | 0.9027 | 0.8221 |
| 6 | 0.165 | 0.536 | 0.7486 | 0.5235 | 0.3941 | 0.7794 | 0.9062 | 0.8163 |
| 7 | 0.123 | 0.721 | 0.7427 | 0.4918 | 0.3913 | 0.6618 | 0.9109 | 0.8254 |
| 8 | 0.117 | 0.881 | 0.7422 | 0.4971 | 0.4126 | 0.6250 | 0.9141 | 0.8153 |

### DistilBERT (LR 2e-5, Ep 3)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.410 | 0.390 | 0.6858 | 0.4318 | 0.2822 | 0.9191 | 0.8523 | 0.7734 |
| 2 | 0.295 | 0.411 | 0.7227 | 0.4785 | 0.3314 | 0.8603 | 0.8885 | 0.8012 |
| 3 | 0.247 | 0.453 | 0.7195 | 0.4687 | 0.3424 | 0.7426 | 0.8919 | 0.7979 |

### DistilBERT (LR 5e-5, Ep 4)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.404 | 0.372 | 0.7041 | 0.4556 | 0.3045 | 0.9044 | 0.8685 | 0.7884 |
| 2 | 0.302 | 0.419 | 0.7297 | 0.4874 | 0.3412 | 0.8529 | 0.8966 | 0.8051 |
| 3 | 0.261 | 0.473 | 0.7385 | 0.5035 | 0.3702 | 0.7868 | 0.9042 | 0.8078 |
| 4 | 0.123 | 0.596 | 0.7491 | 0.5187 | 0.4265 | 0.6618 | 0.9151 | 0.8135 |

### DistilBERT (LR 3e-5, Ep 5, BS 32)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.483 | 0.399 | 0.6767 | 0.4371 | 0.2867 | 0.9191 | 0.8433 | 0.7498 |
| 2 | 0.308 | 0.358 | 0.7055 | 0.4573 | 0.3036 | 0.9265 | 0.8668 | 0.7925 |
| 3 | 0.252 | 0.358 | 0.7296 | 0.4926 | 0.3463 | 0.8529 | 0.8873 | 0.8089 |
| 4 | 0.189 | 0.412 | 0.7321 | 0.4937 | 0.3450 | 0.8676 | 0.8916 | 0.8111 |
| 5 | 0.150 | 0.560 | 0.7400 | 0.4987 | 0.3813 | 0.7206 | 0.9064 | 0.8148 |

### XLM-R base (LR 2e-5, Ep 2)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.405 | 0.402 | 0.6926 | 0.4460 | 0.2952 | 0.9118 | 0.8568 | 0.7750 |
| 2 | 0.282 | 0.390 | 0.7172 | 0.4736 | 0.3227 | 0.8897 | 0.8782 | 0.7997 |

### XLM-R base (LR 1e-5, Ep 4)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.466 | 0.419 | 0.6893 | 0.4599 | 0.3058 | 0.9265 | 0.8511 | 0.7570 |
| 2 | 0.337 | 0.415 | 0.7144 | 0.4686 | 0.3162 | 0.9044 | 0.8767 | 0.7980 |
| 3 | 0.319 | 0.400 | 0.7205 | 0.4720 | 0.3202 | 0.8971 | 0.8821 | 0.8076 |
| 4 | 0.235 | 0.438 | 0.7271 | 0.4798 | 0.3452 | 0.7868 | 0.8929 | 0.8087 |

### XLM-R base (LR 3e-5, Ep 5, WD 0.1)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.475 | 0.405 | 0.7050 | 0.4646 | 0.3172 | 0.8676 | 0.8686 | 0.7819 |
| 2 | 0.312 | 0.415 | 0.7264 | 0.4772 | 0.3385 | 0.8088 | 0.8921 | 0.8100 |
| 3 | 0.307 | 0.397 | 0.7135 | 0.4885 | 0.3411 | 0.8603 | 0.8710 | 0.7809 |
| 4 | 0.229 | 0.514 | 0.7470 | 0.5086 | 0.3829 | 0.7574 | 0.9068 | 0.8255 |
| 5 | 0.186 | 0.615 | 0.7503 | 0.5169 | 0.4182 | 0.6765 | 0.9135 | 0.8206 |

### mBERT base (LR 3e-5, Ep 3)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.413 | 0.402 | 0.7028 | 0.4596 | 0.3064 | 0.9191 | 0.8669 | 0.7821 |
| 2 | 0.305 | 0.422 | 0.7386 | 0.4989 | 0.3621 | 0.8015 | 0.9014 | 0.8154 |
| 3 | 0.265 | 0.467 | 0.7389 | 0.4948 | 0.3810 | 0.7059 | 0.9060 | 0.8157 |

### mBERT base (LR 2e-5, Ep 5, BS 32)

| Ep | Train Loss | Val Loss | Val MacF1 | Val NR_F1 | Val NR_P | Val NR_R | Val NoR_F1 | Val Pro_F1 |
|----|-----------|----------|-----------|-----------|----------|----------|------------|------------|
| 1 | 0.417 | 0.476 | 0.6921 | 0.4300 | 0.2800 | 0.9265 | 0.8690 | 0.7773 |
| 2 | 0.290 | 0.365 | 0.7117 | 0.4762 | 0.3213 | 0.9191 | 0.8718 | 0.7871 |
| 3 | 0.230 | 0.372 | 0.7222 | 0.4857 | 0.3362 | 0.8750 | 0.8802 | 0.8008 |
| 4 | 0.199 | 0.435 | 0.7249 | 0.4851 | 0.3413 | 0.8382 | 0.8854 | 0.8042 |
| 5 | 0.147 | 0.499 | 0.7402 | 0.4977 | 0.3639 | 0.7868 | 0.9028 | 0.8201 |
