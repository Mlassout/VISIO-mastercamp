"""Image feature extraction using Pillow and OpenCV."""

import os
import io
import json
import numpy as np
import cv2
from PIL import Image

HIST_BINS = 16


def normalized_png(image_path: str) -> bytes:
    """Grayscale, contrast-stretched view of an image — what the model 'sees'."""
    arr = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    span = (mx - mn) or 1
    stretched = ((arr - mn) / span * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(stretched, mode="L").save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _histograms(rgb: np.ndarray, gray: np.ndarray) -> str:
    """16-bin colour + luminance histograms as a compact JSON string.

    Values are fractions of pixels per bin (sum to 1) so they're comparable
    across images of different sizes.
    """
    def binned(channel):
        counts, _ = np.histogram(channel, bins=HIST_BINS, range=(0, 255))
        total = counts.sum() or 1
        return [round(float(c) / total, 4) for c in counts]

    return json.dumps({
        "gray": binned(gray),
        "r": binned(rgb[:, :, 0]),
        "g": binned(rgb[:, :, 1]),
        "b": binned(rgb[:, :, 2]),
    })


def extract(image_path: str) -> dict:
    """Extract display features (stored in the DB) from an image file."""
    img_pil = Image.open(image_path).convert("RGB")
    width, height = img_pil.size
    file_size_kb = round(os.path.getsize(image_path) / 1024, 1)

    rgb = np.array(img_pil)
    arr = rgb.astype(np.float64)
    mean_r = round(float(arr[:, :, 0].mean()), 1)
    mean_g = round(float(arr[:, :, 1].mean()), 1)
    mean_b = round(float(arr[:, :, 2].mean()), 1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contrast = round(float(gray.astype(np.float64).std() / 255), 2)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = round(float(edges.mean() / 255), 2)

    return {
        "width": width,
        "height": height,
        "file_size_kb": file_size_kb,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "contrast": contrast,
        "edge_density": edge_density,
        "hist": _histograms(rgb, gray),
    }


# Order of the model's feature vector. Kept here so train.py and model.py agree.
MODEL_FEATURE_KEYS = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v", "std_s",
    "contrast", "edge_density", "edge_top", "edge_bottom",
    "laplacian_var", "colorfulness",
]


def model_vector(image_path: str) -> dict:
    """Richer feature set used only by the classifier (not stored in the DB).

    Adds colour spread, HSV stats, regional edge density (overflowing bins have
    clutter near the top), texture (Laplacian variance) and a colourfulness
    metric on top of the basic display features.
    """
    img_pil = Image.open(image_path).convert("RGB")
    rgb = np.array(img_pil)
    h = rgb.shape[0]

    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_top = float(edges[: h // 2, :].mean() / 255)
    edge_bottom = float(edges[h // 2:, :].mean() / 255)

    # Hasler-Süsstrunk colourfulness.
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "std_r": float(r.std()), "std_g": float(g.std()), "std_b": float(b.std()),
        "mean_h": float(hsv[:, :, 0].mean()), "mean_s": float(hsv[:, :, 1].mean()),
        "mean_v": float(hsv[:, :, 2].mean()), "std_s": float(hsv[:, :, 1].std()),
        "contrast": float(gray.astype(np.float64).std() / 255),
        "edge_density": float(edges.mean() / 255),
        "edge_top": edge_top, "edge_bottom": edge_bottom,
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "colorfulness": colorfulness,
    }
