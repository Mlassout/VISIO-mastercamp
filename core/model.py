"""Classifier — loads trained_model.pkl if available, else falls back to a stub."""

import os
import pickle

from .paths import MODEL_PATH

_model = None


def load_model():
    global _model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        return _model
    return None


def predict(features: dict) -> tuple:
    """Return (label, confidence)."""
    global _model
    if _model is None:
        load_model()

    if _model is not None:
        keys = _model["feature_keys"]
        X = [[features.get(k, 0) for k in keys]]
        pipe = _model["pipeline"]
        proba = pipe.predict_proba(X)[0]
        label_idx = int(proba.argmax())
        label = "clean" if label_idx == 0 else "dirty"
        confidence = round(float(proba[label_idx]), 2)
        return label, confidence

    # Stub fallback when no trained model exists
    ed = features.get("edge_density", 0)
    if ed > 0.15:
        return "dirty", round(0.5 + ed, 2)
    return "clean", round(0.8 - ed, 2)
