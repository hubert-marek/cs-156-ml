"""
Generate all charts for CS156 Assignment 2 final report.
Run: python generate_charts.py
Output: charts/ folder with PNGs
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUT = "charts"

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

# Class distribution
class_dist = {
    "train": {"needs_reply": 1080, "no_reply": 17165, "promotional": 4732},
    "val":   {"needs_reply": 136,  "no_reply": 2468,  "promotional": 624},
    "test":  {"needs_reply": 306,  "no_reply": 4917,  "promotional": 1355},
}

# A1 baselines
a1 = {
    "Logistic Regression": {"acc": 0.85, "macro_f1": 0.719, "nr_f1": 0.43, "nr_p": 0.32, "nr_r": 0.67, "nor_f1": 0.89, "pro_f1": 0.83},
    "Naive Bayes":         {"acc": 0.76, "macro_f1": 0.650, "nr_f1": 0.39, "nr_p": 0.27, "nr_r": 0.72, "nor_f1": 0.82, "pro_f1": 0.74},
    "MLP (256→128)":       {"acc": 0.89, "macro_f1": 0.656, "nr_f1": 0.20, "nr_p": 0.48, "nr_r": 0.13, "nor_f1": 0.93, "pro_f1": 0.84},
}

# Best transformer per architecture (from Run 3 / Run 4)
best_transformers = {
    "DistilBERT\n(66M, LR 3e-5, Ep 5)":  {"acc": 0.875, "macro_f1": 0.764, "nr_f1": 0.535, "nr_p": 0.427, "nr_r": 0.716, "nor_f1": 0.914, "pro_f1": 0.844},
    "XLM-R\n(278M, LR 3e-5, Ep 5)":       {"acc": 0.874, "macro_f1": 0.754, "nr_f1": 0.501, "nr_p": 0.402, "nr_r": 0.663, "nor_f1": 0.913, "pro_f1": 0.849},
    "mBERT\n(178M, LR 3e-5, Ep 8)":       {"acc": 0.870, "macro_f1": 0.757, "nr_f1": 0.507, "nr_p": 0.385, "nr_r": 0.742, "nor_f1": 0.909, "pro_f1": 0.856},
}

# All DistilBERT configs for LR analysis
distilbert_configs = [
    {"label": "LR 2e-5, Ep 3",       "lr": 2e-5, "ep": 3,  "macro_f1": 0.740, "nr_f1": 0.502, "nr_p": 0.359, "nr_r": 0.833},
    {"label": "LR 2e-5, Ep 8",       "lr": 2e-5, "ep": 8,  "macro_f1": 0.763, "nr_f1": 0.541, "nr_p": 0.427, "nr_r": 0.739},
    {"label": "LR 3e-5, Ep 5",       "lr": 3e-5, "ep": 5,  "macro_f1": 0.764, "nr_f1": 0.535, "nr_p": 0.427, "nr_r": 0.716},
    {"label": "LR 3e-5, Ep 8",       "lr": 3e-5, "ep": 8,  "macro_f1": 0.757, "nr_f1": 0.522, "nr_p": 0.401, "nr_r": 0.745},
    {"label": "LR 5e-5, Ep 4",       "lr": 5e-5, "ep": 4,  "macro_f1": 0.753, "nr_f1": 0.503, "nr_p": 0.387, "nr_r": 0.719},
    {"label": "LR 5e-5, Ep 4, ML384","lr": 5e-5, "ep": 4,  "macro_f1": 0.762, "nr_f1": 0.525, "nr_p": 0.412, "nr_r": 0.726},
    {"label": "LR 5e-5, Ep 8",       "lr": 5e-5, "ep": 8,  "macro_f1": 0.748, "nr_f1": 0.479, "nr_p": 0.422, "nr_r": 0.552},
]

# Training trajectories (Run 4 + Run 3 models)
trajectories = {
    "DistilBERT 2e-5": {
        "epochs": [1,2,3,4,5,6,7,8,9,10,11],
        "train_loss": [1.082,0.444,0.328,0.279,0.231,0.196,0.170,0.146,0.128,0.123,0.110],
        "val_loss":   [0.579,0.392,0.377,0.363,0.400,0.391,0.456,0.529,0.587,0.653,0.664],
        "macro_f1":   [0.531,0.671,0.701,0.713,0.736,0.728,0.737,0.741,0.745,0.739,0.742],
        "nr_recall":  [0.949,0.897,0.934,0.882,0.897,0.853,0.816,0.713,0.706,0.676,0.647],
    },
    "DistilBERT 5e-5": {
        "epochs": [1,2,3,4,5,6,7,8,9,10,11,12],
        "train_loss": [0.990,0.375,0.282,0.215,0.179,0.154,0.119,0.102,0.082,0.077,0.062,0.049],
        "val_loss":   [0.423,0.394,0.358,0.399,0.454,0.545,0.808,0.829,0.811,0.990,1.126,1.178],
        "macro_f1":   [0.644,0.675,0.716,0.709,0.725,0.732,0.738,0.747,0.742,0.753,0.741,0.740],
        "nr_recall":  [0.875,0.860,0.882,0.882,0.831,0.809,0.676,0.654,0.669,0.640,0.507,0.529],
    },
    "DistilBERT 3e-5": {
        "epochs": [1,2,3,4,5,6,7,8,9],
        "train_loss": [1.048,0.415,0.296,0.236,0.189,0.164,0.137,0.117,0.099],
        "val_loss":   [0.498,0.360,0.350,0.381,0.402,0.521,0.637,0.729,0.761],
        "macro_f1":   [0.617,0.667,0.702,0.716,0.725,0.738,0.739,0.739,0.730],
        "nr_recall":  [0.934,0.926,0.941,0.846,0.831,0.757,0.728,0.713,0.632],
    },
    "XLM-R 2e-5": {
        "epochs": [1,2,3,4,5,6,7],
        "train_loss": [0.597,0.397,0.345,0.284,0.240,0.191,0.168],
        "val_loss":   [0.453,0.404,0.360,0.374,0.412,0.450,0.420],
        "macro_f1":   [0.626,0.687,0.691,0.708,0.725,0.725,0.719],
        "nr_recall":  [0.963,0.941,0.904,0.934,0.926,0.882,0.890],
    },
    "XLM-R 1e-5": {
        "epochs": [1,2,3,4,5,6,7,8,9,10,11,12],
        "train_loss": [0.756,0.411,0.348,0.312,0.272,0.236,0.215,0.214,0.179,0.184,0.167,0.166],
        "val_loss":   [0.460,0.397,0.366,0.368,0.361,0.375,0.383,0.418,0.433,0.446,0.498,0.488],
        "macro_f1":   [0.605,0.663,0.687,0.675,0.695,0.704,0.700,0.719,0.724,0.737,0.746,0.740],
        "nr_recall":  [0.963,0.941,0.941,0.949,0.963,0.934,0.934,0.882,0.904,0.860,0.816,0.831],
    },
    "mBERT 3e-5": {
        "epochs": [1,2,3,4,5,6,7,8],
        "train_loss": [0.470,0.365,0.287,0.239,0.194,0.165,0.123,0.117],
        "val_loss":   [0.431,0.408,0.374,0.421,0.428,0.536,0.721,0.881],
        "macro_f1":   [0.634,0.700,0.719,0.725,0.743,0.749,0.743,0.742],
        "nr_recall":  [0.956,0.926,0.846,0.890,0.831,0.779,0.662,0.625],
    },
}

# Confusion matrices
confusion_matrices = {
    "DistilBERT (best)\nMacro F1 = 0.764": np.array([[219, 87, 0], [294, 4354, 269], [0, 170, 1185]]),
    "XLM-R 2e-5\nMacro F1 = 0.739": np.array([[272, 34, 0], [536, 4160, 221], [0, 221, 1134]]),
    "DistilBERT 5e-5 Ep8\nMacro F1 = 0.748": np.array([[169, 137, 0], [231, 4480, 206], [0, 192, 1163]]),
}

# Error analysis
errors = {
    "no_reply → needs_reply\n(false alarm)": 294,
    "no_reply → promotional\n(personal as promo)": 269,
    "promotional → no_reply\n(promo as personal)": 170,
    "needs_reply → no_reply\n(missed reply)": 87,
}

# All models for precision-recall scatter
all_models_pr = [
    {"name": "XLM-R 2e-5 ep8",        "p": 0.337, "r": 0.889, "f1": 0.488, "arch": "XLM-R"},
    {"name": "XLM-R 2e-5 ep2",        "p": 0.348, "r": 0.863, "f1": 0.496, "arch": "XLM-R"},
    {"name": "DistilBERT 2e-5 ep3",   "p": 0.359, "r": 0.833, "f1": 0.502, "arch": "DistilBERT"},
    {"name": "XLM-R 1e-5 ep4",        "p": 0.370, "r": 0.810, "f1": 0.508, "arch": "XLM-R"},
    {"name": "XLM-R 1e-5 ep8",        "p": 0.392, "r": 0.761, "f1": 0.517, "arch": "XLM-R"},
    {"name": "XLM-R 3e-5 WD0.1",      "p": 0.402, "r": 0.663, "f1": 0.501, "arch": "XLM-R"},
    {"name": "DistilBERT 3e-5 ep8",   "p": 0.401, "r": 0.745, "f1": 0.522, "arch": "DistilBERT"},
    {"name": "DistilBERT 2e-5 ep8",   "p": 0.427, "r": 0.739, "f1": 0.541, "arch": "DistilBERT"},
    {"name": "DistilBERT 5e-5 ep4",   "p": 0.387, "r": 0.719, "f1": 0.503, "arch": "DistilBERT"},
    {"name": "DistilBERT 5e-5 ML384", "p": 0.412, "r": 0.726, "f1": 0.525, "arch": "DistilBERT"},
    {"name": "DistilBERT 3e-5 ep5 ★", "p": 0.427, "r": 0.716, "f1": 0.535, "arch": "DistilBERT"},
    {"name": "mBERT 3e-5 ep3",        "p": 0.385, "r": 0.693, "f1": 0.495, "arch": "mBERT"},
    {"name": "mBERT 3e-5 ep8",        "p": 0.385, "r": 0.742, "f1": 0.507, "arch": "mBERT"},
    {"name": "DistilBERT 5e-5 ep8",   "p": 0.422, "r": 0.552, "f1": 0.479, "arch": "DistilBERT"},
]

LABELS = ["needs_reply", "no_reply", "promotional"]

# ═══════════════════════════════════════════════════════════════
# CHART 1: Class distribution
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Bar chart
splits = ["train", "val", "test"]
x = np.arange(len(LABELS))
w = 0.25
colors_split = ["#2ecc71", "#3498db", "#e74c3c"]
for i, split in enumerate(splits):
    counts = [class_dist[split][l] for l in LABELS]
    bars = axes[0].bar(x + i*w, counts, w, label=split, color=colors_split[i], alpha=0.85)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f"{count:,}", ha='center', fontsize=8)
axes[0].set_xticks(x + w)
axes[0].set_xticklabels(LABELS)
axes[0].set_ylabel("Count")
axes[0].set_title("Class Distribution by Split")
axes[0].legend()

# Pie chart (total)
total = {l: sum(class_dist[s][l] for s in splits) for l in LABELS}
colors_class = ["#e74c3c", "#3498db", "#f39c12"]
wedges, texts, autotexts = axes[1].pie(
    total.values(), labels=total.keys(), autopct='%1.1f%%',
    colors=colors_class, startangle=90
)
axes[1].set_title("Overall Class Distribution (n=32,783)")

plt.tight_layout()
plt.savefig(f"{OUT}/01_class_distribution.png")
plt.close()
print("1/10 Class distribution")

# ═══════════════════════════════════════════════════════════════
# CHART 2: A1 vs A2 macro F1 comparison
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

all_models = list(a1.keys()) + list(best_transformers.keys())
all_f1 = [a1[m]["macro_f1"] for m in a1] + [best_transformers[m]["macro_f1"] for m in best_transformers]
colors = ["#95a5a6"]*len(a1) + ["#2ecc71"]*len(best_transformers)

bars = ax.bar(range(len(all_models)), all_f1, color=colors, edgecolor='white', linewidth=0.5)
for i, (bar, f1) in enumerate(zip(bars, all_f1)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{f1:.3f}", ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(all_models)))
ax.set_xticklabels(all_models, rotation=20, ha='right', fontsize=9)
ax.set_ylabel("Test Macro F1")
ax.set_ylim(0.5, 0.85)
ax.set_title("Assignment 1 Baselines vs Assignment 2 Transformers")
ax.axhline(y=0.719, color='gray', linestyle='--', alpha=0.5, label='Best A1 (LogReg)')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_a1_vs_a2_macro_f1.png")
plt.close()
print("2/10 A1 vs A2 comparison")

# ═══════════════════════════════════════════════════════════════
# CHART 3: Per-class F1 comparison (all A1 + best A2)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))

classes = ["needs_reply", "no_reply", "promotional", "macro avg"]
logreg_f1 =     [0.43,  0.89,  0.83,  0.719]
nb_f1 =         [0.39,  0.82,  0.74,  0.650]
mlp_f1 =        [0.20,  0.93,  0.84,  0.656]
distilbert_f1 = [0.535, 0.914, 0.844, 0.764]

x = np.arange(len(classes))
w = 0.2
b1 = ax.bar(x - 1.5*w, nb_f1, w, label="Naive Bayes (A1)", color="#bdc3c7")
b2 = ax.bar(x - 0.5*w, mlp_f1, w, label="MLP (A1)", color="#95a5a6")
b3 = ax.bar(x + 0.5*w, logreg_f1, w, label="Logistic Regression (A1)", color="#7f8c8d")
b4 = ax.bar(x + 1.5*w, distilbert_f1, w, label="DistilBERT (A2 best)", color="#2ecc71")

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha='center', fontsize=7, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.1)
ax.set_title("Per-Class F1: All A1 Baselines vs Best A2 Transformer")
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/03_per_class_f1_comparison.png")
plt.close()
print("3/10 Per-class F1")

# ═══════════════════════════════════════════════════════════════
# CHART 4: Architecture comparison
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

archs = list(best_transformers.keys())
arch_colors = ["#2ecc71", "#3498db", "#e67e22"]

# Macro F1
vals = [best_transformers[a]["macro_f1"] for a in archs]
bars = axes[0].bar(archs, vals, color=arch_colors)
for bar, v in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
axes[0].set_ylim(0.7, 0.8)
axes[0].set_ylabel("Test Macro F1")
axes[0].set_title("Macro F1 by Architecture")

# needs_reply F1
vals = [best_transformers[a]["nr_f1"] for a in archs]
bars = axes[1].bar(archs, vals, color=arch_colors)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
axes[1].set_ylim(0.4, 0.6)
axes[1].set_ylabel("needs_reply F1")
axes[1].set_title("Minority Class F1 by Architecture")

# Params vs performance
params = [66, 278, 178]
macro = [best_transformers[a]["macro_f1"] for a in archs]
axes[2].scatter(params, macro, s=200, c=arch_colors, zorder=5, edgecolors='black')
for p, m, name in zip(params, macro, ["DistilBERT", "XLM-R", "mBERT"]):
    axes[2].annotate(name, (p, m), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)
axes[2].set_xlabel("Parameters (millions)")
axes[2].set_ylabel("Test Macro F1")
axes[2].set_title("Model Size vs Performance")
axes[2].set_ylim(0.74, 0.77)

plt.tight_layout()
plt.savefig(f"{OUT}/04_architecture_comparison.png")
plt.close()
print("4/10 Architecture comparison")

# ═══════════════════════════════════════════════════════════════
# CHART 5: Training curves (val loss + macro F1 vs epoch)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors_traj = {"DistilBERT 2e-5": "#2ecc71", "DistilBERT 5e-5": "#27ae60",
               "DistilBERT 3e-5": "#1abc9c", "XLM-R 2e-5": "#3498db",
               "XLM-R 1e-5": "#2980b9", "mBERT 3e-5": "#e67e22"}

for idx, (name, data) in enumerate(trajectories.items()):
    row, col = idx // 3, idx % 3
    ax1 = axes[row, col]
    ax2 = ax1.twinx()

    ep = data["epochs"]
    ax1.plot(ep, data["val_loss"], 'o-', color='#e74c3c', markersize=4, label='Val Loss')
    ax2.plot(ep, data["macro_f1"], 's-', color='#2ecc71', markersize=4, label='Val Macro F1')

    # Mark best epoch
    best_idx = np.argmax(data["macro_f1"])
    ax2.axvline(x=ep[best_idx], color='green', linestyle=':', alpha=0.5)
    ax2.annotate(f"best: ep {ep[best_idx]}\nF1={data['macro_f1'][best_idx]:.3f}",
                 xy=(ep[best_idx], data["macro_f1"][best_idx]),
                 fontsize=7, color='green', ha='center',
                 xytext=(0, 15), textcoords='offset points')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss", color='#e74c3c')
    ax2.set_ylabel("Val Macro F1", color='#2ecc71')
    ax1.set_title(name, fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')

plt.suptitle("Training Dynamics: Validation Loss vs Macro F1", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/05_training_curves.png")
plt.close()
print("5/10 Training curves")

# ═══════════════════════════════════════════════════════════════
# CHART 6: NR recall collapse over epochs
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for name, data in trajectories.items():
    ax.plot(data["epochs"], data["nr_recall"], 'o-', label=name, markersize=4)

ax.set_xlabel("Epoch")
ax.set_ylabel("needs_reply Recall (validation)")
ax.set_title("Conservative Drift: needs_reply Recall Drops with Training")
ax.set_ylim(0.4, 1.0)
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3)
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/06_nr_recall_collapse.png")
plt.close()
print("6/10 NR recall collapse")

# ═══════════════════════════════════════════════════════════════
# CHART 7: Confusion matrices
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (title, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=8, rotation=30, ha='right')
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)

    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, f"{cm[i,j]:,}", ha='center', va='center', color=color, fontsize=11)

plt.suptitle("Confusion Matrices: Best, Most Aggressive, Most Conservative", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/07_confusion_matrices.png")
plt.close()
print("7/10 Confusion matrices")

# ═══════════════════════════════════════════════════════════════
# CHART 8: Precision-recall scatter for needs_reply
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

arch_colors_map = {"DistilBERT": "#2ecc71", "XLM-R": "#3498db", "mBERT": "#e67e22", "A1 Baseline": "#95a5a6"}
arch_markers = {"DistilBERT": "o", "XLM-R": "s", "mBERT": "^", "A1 Baseline": "D"}

# Add A1 baselines to the scatter
all_models_pr.extend([
    {"name": "LogReg (A1)", "p": 0.32, "r": 0.67, "f1": 0.43, "arch": "A1 Baseline"},
    {"name": "Naive Bayes (A1)", "p": 0.27, "r": 0.72, "f1": 0.39, "arch": "A1 Baseline"},
    {"name": "MLP (A1)", "p": 0.48, "r": 0.13, "f1": 0.20, "arch": "A1 Baseline"},
])

for m in all_models_pr:
    c = arch_colors_map[m["arch"]]
    mk = arch_markers[m["arch"]]
    size = 150 if "★" in m["name"] else 80
    edge = 'red' if "★" in m["name"] else 'black'
    lw = 2 if "★" in m["name"] else 0.5
    ax.scatter(m["r"], m["p"], s=size, c=c, marker=mk, edgecolors=edge, linewidths=lw, zorder=5)
    ax.annotate(m["name"], (m["r"], m["p"]), fontsize=7, xytext=(5, 5),
                textcoords='offset points', alpha=0.8)

# F1 isocurves
for f1_val in [0.45, 0.50, 0.55]:
    r_range = np.linspace(0.01, 1.0, 100)
    p_range = f1_val * r_range / (2 * r_range - f1_val)
    valid = (p_range > 0) & (p_range <= 1)
    ax.plot(r_range[valid], p_range[valid], '--', color='gray', alpha=0.3, linewidth=1)
    # Label
    idx_label = np.argmin(np.abs(r_range - 0.95))
    if valid[idx_label]:
        ax.text(0.95, p_range[idx_label], f"F1={f1_val}", fontsize=7, color='gray', alpha=0.5)

# Legend
for arch in arch_colors_map:
    ax.scatter([], [], c=arch_colors_map[arch], marker=arch_markers[arch], s=80, label=arch, edgecolors='black')
ax.legend(loc='upper right', fontsize=10)

ax.set_xlabel("needs_reply Recall")
ax.set_ylabel("needs_reply Precision")
ax.set_title("Precision-Recall Tradeoff on needs_reply (all models)")
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0.20, 0.55)
ax.grid(alpha=0.2)

# Annotate regions
ax.annotate("AGGRESSIVE\n(high recall,\nlow precision)", xy=(0.87, 0.23), fontsize=9,
            color='#e74c3c', ha='center', alpha=0.7)
ax.annotate("BALANCED\n(best F1)", xy=(0.72, 0.47), fontsize=9,
            color='#2ecc71', ha='center', alpha=0.7)
ax.annotate("A1 BASELINES", xy=(0.35, 0.22), fontsize=9,
            color='#95a5a6', ha='center', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{OUT}/08_precision_recall_scatter.png")
plt.close()
print("8/10 Precision-recall scatter")

# ═══════════════════════════════════════════════════════════════
# CHART 9: Error analysis
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Error type bar chart
err_labels = list(errors.keys())
err_counts = list(errors.values())
err_colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
bars = axes[0].barh(err_labels, err_counts, color=err_colors, edgecolor='white')
for bar, count in zip(bars, err_counts):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f"{count} ({100*count/820:.0f}%)", va='center', fontsize=10)
axes[0].set_xlabel("Count")
axes[0].set_title(f"Error Breakdown (820 errors / 6,578 total = 12.5%)")
axes[0].invert_yaxis()

# Error by true class
true_classes = ["needs_reply\n(306)", "no_reply\n(4917)", "promotional\n(1355)"]
error_rates = [87/306*100, (294+269)/4917*100, 170/1355*100]
correct_rates = [100 - e for e in error_rates]
bars_correct = axes[1].bar(true_classes, correct_rates, color="#2ecc71", label="Correct")
bars_error = axes[1].bar(true_classes, error_rates, bottom=correct_rates, color="#e74c3c", label="Error")
for i, (c, e) in enumerate(zip(correct_rates, error_rates)):
    axes[1].text(i, c/2, f"{c:.1f}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    axes[1].text(i, c + e/2, f"{e:.1f}%", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
axes[1].set_ylabel("Percentage")
axes[1].set_title("Error Rate by True Class")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT}/09_error_analysis.png")
plt.close()
print("9/10 Error analysis")

# ═══════════════════════════════════════════════════════════════
# CHART 10: DistilBERT hyperparameter heatmap
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Sort by macro F1
configs_sorted = sorted(distilbert_configs, key=lambda x: -x["macro_f1"])
labels = [c["label"] for c in configs_sorted]
macro_vals = [c["macro_f1"] for c in configs_sorted]
nr_vals = [c["nr_f1"] for c in configs_sorted]

x = np.arange(len(labels))
w = 0.35
b1 = axes[0].bar(x - w/2, macro_vals, w, label="Macro F1", color="#2ecc71")
b2 = axes[0].bar(x + w/2, nr_vals, w, label="needs_reply F1", color="#e74c3c")
for bars in [b1, b2]:
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{bar.get_height():.3f}", ha='center', fontsize=7, rotation=45)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
axes[0].set_ylim(0.4, 0.85)
axes[0].set_title("DistilBERT: All Hyperparameter Configs")
axes[0].legend(fontsize=9)

# NR precision vs recall by LR
lr_colors = {"2e-5": "#3498db", "3e-5": "#2ecc71", "5e-5": "#e74c3c"}
for c in distilbert_configs:
    lr_str = f"{c['lr']:.0e}".replace("+0", "").replace("-0", "-")
    if lr_str == "2e-5": lr_label = "2e-5"
    elif lr_str == "3e-5": lr_label = "3e-5"
    else: lr_label = "5e-5"
    color = lr_colors.get(lr_label, "gray")
    axes[1].scatter(c["nr_r"], c["nr_p"], s=100, c=color, edgecolors='black', linewidths=0.5, zorder=5)
    axes[1].annotate(f"ep{c['ep']}", (c["nr_r"], c["nr_p"]), fontsize=7,
                     xytext=(4, 4), textcoords='offset points')

for lr, color in lr_colors.items():
    axes[1].scatter([], [], c=color, s=80, label=f"LR {lr}", edgecolors='black')
axes[1].legend(fontsize=9)
axes[1].set_xlabel("needs_reply Recall")
axes[1].set_ylabel("needs_reply Precision")
axes[1].set_title("DistilBERT: LR Effect on Precision-Recall")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUT}/10_distilbert_hyperparams.png")
plt.close()
print("10/10 Hyperparameter analysis")

print(f"\nAll charts saved to {OUT}/")
