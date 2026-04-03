# CS156 Machine Learning — Email Classification Pipeline

Multilingual email classification using personal Gmail archives (Polish + English).  
Classifies incoming emails into: `needs_reply`, `no_reply`, `promotional`.

## Repository Structure

```
├── assignment1/
│   └── pipeline_final.ipynb          # A1: TF-IDF + LogReg/NB/MLP baselines
│
├── assignment2/
│   ├── assignment2_report.ipynb      # A2: Final report (export as PDF)
│   ├── experiment_results.md         # Complete raw experimental data
│   ├── generate_charts.py            # Script to regenerate all visualizations
│   ├── charts/                       # Pre-generated charts (10 PNGs)
│   └── gpu_notebook/
│       └── transformer_training.ipynb  # GPU training notebook (run on Colab)
```

## Results Summary

| Model | Macro F1 | needs_reply F1 |
|-------|----------|----------------|
| Logistic Regression (A1 best) | 0.719 | 0.43 |
| **DistilBERT fine-tuned (A2 best)** | **0.764** | **0.54** |

## Dataset

- 32,783 emails from 2 Gmail accounts (Google Takeout)
- 3-class classification with severe imbalance (needs_reply: 4.7%)
- Thread-aware train/val/test splits

## Key Findings

- Fine-tuned multilingual transformers outperform TF-IDF baselines by +6.3% macro F1
- DistilBERT (134M params) slightly outperforms larger XLM-R (278M) and mBERT (177M)
- Hyperparameter tuning (especially learning rate) matters more than architecture choice
- Class imbalance remains the bottleneck — needs_reply F1 of 0.54 is the ceiling

**Author:** Hubert Pysklo  
**Course:** CS156 — Machine Learning, Spring 2026
