"""Centralised filesystem paths.

All generated artifacts and data live at the project root (the parent of this
package), so the paths are stable no matter which module imports them.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)  # project root

SCHEMA_PATH = os.path.join(PACKAGE_DIR, "schema.sql")
DB_PATH = os.path.join(BASE_DIR, "binsight.db")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")
CACHE_PATH = os.path.join(BASE_DIR, "features_cache.csv")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATA_DIR = os.path.join(BASE_DIR, "Data")
