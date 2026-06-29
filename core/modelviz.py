"""Model-page visualisations: feature charts, importance, PCA, confusion matrix.

Reads the cached training features (features_cache.csv) and the trained pipeline
(trained_model.pkl), and renders matplotlib PNGs in the app's palette.
"""

import io
import os
import csv
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import features
from .charts import GREEN, AMBER, GRID, AXIS, LABEL, TICK, TITLE, BG, _style_ax
from .paths import CACHE_PATH, MODEL_PATH, META_PATH

KEYS = features.MODEL_FEATURE_KEYS
GREY = "#9aa3a7"


# ----- data loading -----

def load_meta():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)


def _load_cache():
    """Return (X, labels) from the feature cache; labels in {clean,dirty,nolabel}."""
    X, labels = [], []
    if not os.path.exists(CACHE_PATH):
        return np.empty((0, len(KEYS))), []
    with open(CACHE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                X.append([float(row[k]) for k in KEYS])
                labels.append(row["label"])
            except (KeyError, ValueError):
                continue
    return np.array(X), labels


def _load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _finish(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _empty(msg="No data"):
    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=LABEL, fontsize=12)
    return _finish(fig)


# ----- charts -----

def cv_scores_png(meta) -> bytes:
    """Bar chart of cross-validated accuracy per candidate classifier."""
    cv = (meta or {}).get("cv_results") or {}
    if not cv:
        return _empty("Train the model to see CV scores")
    names = list(cv.keys())
    means = [cv[n]["mean"] for n in names]
    stds = [cv[n]["std"] for n in names]
    best = max(range(len(names)), key=lambda i: means[i])

    fig, ax = plt.subplots(figsize=(4.8, 2.6), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    colors = [GREEN if i == best else "#c7d2cb" for i in range(len(names))]
    ax.bar(names, means, yerr=stds, color=colors, zorder=3,
           capsize=4, ecolor=GREY, width=0.6)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle="--", zorder=2)
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.02, f"{m:.2f}", ha="center", fontsize=10,
                fontweight="600", color=TITLE)
    ax.tick_params(axis="x", colors=LABEL, labelsize=9)
    fig.tight_layout(pad=1.0)
    return _finish(fig)


def confusion_matrix_png(meta) -> bytes:
    """2x2 confusion matrix from honest out-of-fold predictions."""
    cm = (meta or {}).get("confusion_matrix")
    if not cm:
        return _empty("Train the model to see the confusion matrix")
    cm = np.array(cm, dtype=float)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.imshow(cm, cmap="Greens", aspect="auto")
    labels = ["Empty", "Full"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", color=LABEL, fontsize=10)
    ax.set_ylabel("Actual", color=LABEL, fontsize=10)
    ax.tick_params(colors=LABEL)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=15, fontweight="700",
                    color="white" if cm[i, j] > thresh else TITLE)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=1.0)
    return _finish(fig)


def feature_importance_png() -> bytes:
    """Feature ranking from the trained model (|coef| or impurity importance)."""
    bundle = _load_pipeline()
    if not bundle:
        return _empty("Train the model to see feature importance")
    clf = bundle["pipeline"].named_steps["clf"]
    keys = bundle.get("feature_keys", KEYS)

    if hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
        title = "|standardised coefficient|"
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        title = "impurity importance"
    else:
        return _empty("Model exposes no importances")

    order = np.argsort(importance)
    names = [keys[i] for i in order]
    vals = importance[order]

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.barh(names, vals, color=GREEN, zorder=3, height=0.66)
    ax.set_xlabel(title, color=LABEL, fontsize=10)
    ax.tick_params(axis="y", colors=TITLE, labelsize=9)
    fig.tight_layout(pad=1.0)
    return _finish(fig)


def feature_distributions_png() -> bytes:
    """Clean vs dirty distributions for the most discriminative features."""
    X, labels = _load_cache()
    if len(X) == 0:
        return _empty("No cached features")
    labels = np.array(labels)
    clean_mask = labels == "clean"
    dirty_mask = labels == "dirty"
    if not clean_mask.any() or not dirty_mask.any():
        return _empty("No labeled features")

    # Rank features by separation between class means (standardised).
    Xs = StandardScaler().fit_transform(X)
    sep = np.abs(Xs[clean_mask].mean(0) - Xs[dirty_mask].mean(0))
    top = np.argsort(sep)[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(9.8, 4.6), dpi=100)
    fig.patch.set_facecolor(BG)
    for ax, idx in zip(axes.ravel(), top):
        _style_ax(ax)
        col = X[:, idx]
        rng = (float(col.min()), float(col.max()))
        ax.hist(col[clean_mask], bins=12, range=rng, color=GREEN, alpha=0.55,
                label="empty", zorder=3)
        ax.hist(col[dirty_mask], bins=12, range=rng, color=AMBER, alpha=0.55,
                label="full", zorder=3)
        ax.set_title(KEYS[idx], fontsize=10, color=TITLE)
        ax.tick_params(labelsize=8, colors=TICK)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(pad=1.0)
    return _finish(fig)


def pca_scatter_png() -> bytes:
    """2D PCA of all feature vectors, coloured by label."""
    X, labels = _load_cache()
    if len(X) < 3:
        return _empty("Not enough data for PCA")
    labels = np.array(labels)
    Xs = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    groups = [("nolabel", GREY, 10, 0.5, "no_label"),
              ("clean", GREEN, 26, 0.9, "empty"),
              ("dirty", AMBER, 26, 0.9, "full")]
    for key, color, size, alpha, lab in groups:
        m = labels == key
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=size, c=color,
                       alpha=alpha, edgecolors="none", zorder=3, label=lab)
    ax.set_xlabel("PC 1", color=LABEL, fontsize=10)
    ax.set_ylabel("PC 2", color=LABEL, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(pad=1.0)
    return _finish(fig)
