"""Train a classifier on the Data/ folder and save to trained_model.pkl.

Strategy:
  - extract a rich feature vector per image (features.model_vector)
  - compare several classifiers by 5-fold cross-validation (honest estimate)
  - keep the best, refit on all labeled data, pickle it

Dataset layout:
  Data/train/with_label/clean/*.jpg   — labeled clean
  Data/train/with_label/dirty/*.jpg   — labeled dirty
  Data/train/no_label/*.jpg           — unlabeled (unused for now)
  Data/test/*.jpg                     — held-out test images (no labels)
"""

import os
import glob
import csv
import json
import pickle
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from core import features

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "features_cache.csv")

FEATURE_KEYS = features.MODEL_FEATURE_KEYS


def _load_cache():
    """Return {path: (mtime, label, [feature values])} from the CSV cache."""
    cache = {}
    if not os.path.exists(CACHE_PATH):
        return cache
    with open(CACHE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                vals = [float(row[k]) for k in FEATURE_KEYS]
                cache[row["path"]] = (float(row["mtime"]), row["label"], vals)
            except (KeyError, ValueError):
                continue  # stale/incompatible row — ignore, will re-extract
    return cache


def _save_cache(cache):
    """Write the feature cache back to disk (one row per image)."""
    with open(CACHE_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "mtime", "label"] + FEATURE_KEYS)
        for path, (mtime, label, vals) in sorted(cache.items()):
            w.writerow([path, mtime, label] + vals)


def _vectorize(paths, label, cache, fresh, counters):
    """Feature vectors for a list of images, reusing the cache when unchanged
    (matched on path + modification time). Updates `fresh` and `counters`."""
    out = []
    for p in paths:
        try:
            mtime = os.path.getmtime(p)
            cached = cache.get(p)
            if cached and abs(cached[0] - mtime) < 1e-6:
                vals = cached[2]
                counters[0] += 1  # hits
            else:
                feats = features.model_vector(p)
                vals = [feats[k] for k in FEATURE_KEYS]
                counters[1] += 1  # misses
            fresh[p] = (mtime, label, vals)
            out.append(vals)
        except Exception as e:
            print(f"  SKIP {os.path.basename(p)}: {e}")
    return out


def _glob_jpgs(folder):
    return glob.glob(os.path.join(folder, "*.jpg")) + \
           glob.glob(os.path.join(folder, "*.jpeg"))


def _collect():
    """Collect labeled and unlabeled feature vectors, sharing one feature cache.

    Returns (X, y, X_unlabeled): X/y are the labeled set, X_unlabeled holds the
    no_label images used for semi-supervised self-training.
    """
    cache = _load_cache()
    fresh = {}
    counters = [0, 0]  # [hits, misses]
    X, y = [], []

    for label in ("clean", "dirty"):
        folder = os.path.join(DATA_DIR, "train", "with_label", label)
        paths = _glob_jpgs(folder)
        print(f"  {label}: {len(paths)} images")
        vecs = _vectorize(paths, label, cache, fresh, counters)
        X.extend(vecs)
        y.extend(0 if label == "clean" else 1 for _ in vecs)

    nolabel_paths = _glob_jpgs(os.path.join(DATA_DIR, "train", "no_label"))
    print(f"  no_label: {len(nolabel_paths)} images")
    X_unlabeled = _vectorize(nolabel_paths, "nolabel", cache, fresh, counters)

    _save_cache(fresh)
    print(f"  cache: {counters[0]} reused, {counters[1]} extracted "
          f"-> {os.path.basename(CACHE_PATH)}")
    return np.array(X), np.array(y), np.array(X_unlabeled)


def _candidates():
    """Classifiers to compare. Each is wrapped with feature scaling."""
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "grad_boost": GradientBoostingClassifier(random_state=42),
    }


# Self-training: only pseudo-label no_label images the model is very sure about.
PSEUDO_CONFIDENCE = 0.90


def train():
    print("Collecting training data...")
    X, y, X_unlabeled = _collect()
    print(f"Labeled samples: {len(y)}  (clean={int(sum(y==0))}, dirty={int(sum(y==1))})")
    print(f"Unlabeled (no_label) samples: {len(X_unlabeled)}")
    print(f"Features per image: {len(FEATURE_KEYS)}")

    if len(y) < 8:
        print("ERROR: not enough labeled images to train.")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model selection is done on the LABELED data only, so the accuracy estimate
    # stays honest (pseudo-labels never leak into validation folds).
    print("\n— Cross-validated accuracy on labeled data (honest estimate) —")
    cv_results = {}
    best_name, best_pipe, best_score, best_std = None, None, -1.0, 0.0
    for name, clf in _candidates().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": round(float(scores.mean()), 4),
                            "std": round(float(scores.std()), 4)}
        print(f"  {name:>14}: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_score:
            best_name, best_pipe = name, pipe
            best_score, best_std = scores.mean(), scores.std()
    print(f"\nBest model: {best_name}  (CV accuracy {best_score:.3f})")

    # Honest per-class metrics + confusion matrix from out-of-fold predictions.
    cv_pred = cross_val_predict(best_pipe, X, y, cv=cv)
    cm = confusion_matrix(y, cv_pred).tolist()
    report = classification_report(y, cv_pred, target_names=["clean", "dirty"],
                                   output_dict=True, zero_division=0)

    # Semi-supervised step: fit on labeled data, pseudo-label the confident
    # no_label images, then refit on the combined set.
    X_train, y_train = X, y
    n_pseudo = n_pseudo_clean = n_pseudo_dirty = 0
    if len(X_unlabeled):
        best_pipe.fit(X, y)
        proba = best_pipe.predict_proba(X_unlabeled)
        keep = proba.max(axis=1) >= PSEUDO_CONFIDENCE
        pseudo_y = proba.argmax(axis=1)[keep]
        n_pseudo = int(keep.sum())
        n_pseudo_clean = int((pseudo_y == 0).sum())
        n_pseudo_dirty = int((pseudo_y == 1).sum())
        print(f"\n— Self-training on no_label —")
        print(f"  confident pseudo-labels (>= {PSEUDO_CONFIDENCE:.0%}): "
              f"{n_pseudo}/{len(X_unlabeled)}  "
              f"(clean={n_pseudo_clean}, dirty={n_pseudo_dirty})")
        if n_pseudo:
            X_train = np.vstack([X, X_unlabeled[keep]])
            y_train = np.concatenate([y, pseudo_y])

    best_pipe.fit(X_train, y_train)
    print("\n— Training-set report on labeled data (optimistic, for reference) —")
    print(classification_report(y, best_pipe.predict(X), target_names=["clean", "dirty"]))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {"pipeline": best_pipe, "feature_keys": FEATURE_KEYS, "model_name": best_name},
            f,
        )
    print(f"Model saved to {MODEL_PATH}  (trained on {len(y_train)} samples)")

    # Persist training metadata for the Model page.
    meta = {
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_keys": FEATURE_KEYS,
        "n_features": len(FEATURE_KEYS),
        "n_labeled": int(len(y)),
        "n_clean": int((y == 0).sum()),
        "n_dirty": int((y == 1).sum()),
        "n_unlabeled": int(len(X_unlabeled)),
        "n_pseudo": n_pseudo,
        "n_pseudo_clean": n_pseudo_clean,
        "n_pseudo_dirty": n_pseudo_dirty,
        "n_train_total": int(len(y_train)),
        "pseudo_confidence": PSEUDO_CONFIDENCE,
        "cv_results": cv_results,
        "cv_accuracy": round(float(best_score), 4),
        "cv_std": round(float(best_std), 4),
        "confusion_matrix": cm,          # rows = true [clean, dirty], cols = pred
        "report": report,                # honest out-of-fold precision/recall/f1
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    train()
