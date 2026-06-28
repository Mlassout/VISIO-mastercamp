"""Rule-based classifier (no machine learning).

Simulates an automatic decision from the extracted features using simple,
configurable conditional rules — as required by the project brief. The default
example mirrors the brief: a dark average colour AND a large file lean "dirty"
(full/overflowing); we also factor in edge density, since clutter from rubbish
raises the contour count.
"""

# Default thresholds. Each can be overridden from the /rules page (stored in DB).
DEFAULT_RULES = {
    "dark_threshold": 110.0,     # mean brightness below this looks full/cluttered
    "edge_threshold": 0.14,      # Canny edge density above this = lots of clutter
    "size_threshold_kb": 250.0,  # bigger files carry more visual detail
}


def classify(features: dict, config: dict = None):
    """Return (label, confidence, reason) from conditional rules.

    label: 'clean' (empty) or 'dirty' (full/overflowing)
    confidence: derived from how many rules fired
    reason: short human-readable explanation of the decision
    """
    cfg = {**DEFAULT_RULES, **(config or {})}

    brightness = (features["mean_r"] + features["mean_g"] + features["mean_b"]) / 3.0
    edge = features.get("edge_density", 0.0)
    size = features.get("file_size_kb", 0.0)

    fired = []
    if brightness < cfg["dark_threshold"]:
        fired.append(f"dark image (brightness {brightness:.0f} < {cfg['dark_threshold']:.0f})")
    if edge > cfg["edge_threshold"]:
        fired.append(f"high edge density ({edge:.2f} > {cfg['edge_threshold']:.2f})")
    if size > cfg["size_threshold_kb"]:
        fired.append(f"large file ({size:.0f} KB > {cfg['size_threshold_kb']:.0f} KB)")

    score = len(fired)
    if score >= 2:
        label = "dirty"
        confidence = round(0.5 + 0.16 * score, 2)  # 2->0.82, 3->0.98
        reason = "Dirty: " + ", ".join(fired)
    else:
        label = "clean"
        confidence = round(0.6 + 0.2 * (1 - score), 2)  # 0->0.80, 1->0.60
        reason = "Clean: " + (", ".join(fired) if fired else "no rules triggered")

    return label, min(confidence, 0.99), reason
