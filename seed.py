"""Seed the database from the real Data/ dataset so every row has a real image.

Copies a bounded subset of dataset images into static/uploads/, extracts real
features, labels them from the folder structure, and assigns demo coordinates to
a subset so the map renders real, file-backed points (never phantom rows).
"""

import os
import glob
import shutil
import random
from datetime import datetime, timedelta

from core import db
from core import features
from core import rules

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "Data")
UPLOAD_DIR = os.path.join(ROOT, "static", "uploads")

# How many images to pull from each part of the dataset.
N_CLEAN = 20      # Data/train/with_label/clean
N_DIRTY = 20      # Data/train/with_label/dirty
N_NOLABEL = 20    # Data/train/no_label  -> annotation queue

# Demo coordinates: scatter geolocated bins within ~1 km of the EFREI Paris
# campus (30-32 Av. de la République, 94800 Villejuif). The dataset has no geo.
GEO_CENTER = (48.78937, 2.36266)
GEO_SPREAD = (0.008, 0.012)  # ~0.9 km lat, ~0.9 km lng at this latitude
N_GEOLOCATED = 16

random.seed(42)


def _pick(folder, n):
    """First n jpg/jpeg files in a folder (sorted for determinism)."""
    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
    )
    return paths[:n]


def _sources():
    """Yield (source_path, dest_filename, manual_label) for each seed image."""
    groups = [
        (os.path.join(DATA_DIR, "train", "with_label", "clean"), N_CLEAN, "clean"),
        (os.path.join(DATA_DIR, "train", "with_label", "dirty"), N_DIRTY, "dirty"),
        (os.path.join(DATA_DIR, "train", "no_label"), N_NOLABEL, None),
    ]
    for folder, n, label in groups:
        prefix = label if label else "nolabel"
        for p in _pick(folder, n):
            dest = f"{prefix}_{os.path.basename(p)}"
            yield p, dest, label


def seed():
    # Start clean: fresh DB + empty uploads dir.
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    if os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    db.init_db()

    items = list(_sources())
    base = datetime.now() - timedelta(days=70)
    geo_idx = set(random.sample(range(len(items)), min(N_GEOLOCATED, len(items))))

    inserted = 0
    for i, (src, dest, manual_label) in enumerate(items):
        dest_path = os.path.join(UPLOAD_DIR, dest)
        try:
            shutil.copy(src, dest_path)
            feats = features.extract(dest_path)
            auto_label, confidence, reason = rules.classify(feats)
        except Exception as e:
            print(f"  SKIP {dest}: {e}")
            continue

        # Spread uploads across the last ~70 days for a realistic trend chart.
        dt = base + timedelta(
            days=int(70 * i / max(1, len(items))),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        lat = lon = None
        if i in geo_idx:
            lat = round(GEO_CENTER[0] + random.uniform(-GEO_SPREAD[0], GEO_SPREAD[0]), 5)
            lon = round(GEO_CENTER[1] + random.uniform(-GEO_SPREAD[1], GEO_SPREAD[1]), 5)

        db.insert({
            "filename": dest,
            "upload_date": dt.isoformat(),
            **feats,
            "manual_label": manual_label,
            "auto_label": auto_label,
            "confidence": confidence,
            "rule_reason": reason,
            "latitude": lat,
            "longitude": lon,
        })
        inserted += 1

    print(f"Seeded {inserted} real images into {db.DB_PATH}")
    print(f"  copied to {UPLOAD_DIR}")


if __name__ == "__main__":
    seed()
