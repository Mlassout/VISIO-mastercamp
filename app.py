

import os
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from datetime import datetime

from core import db
from core import features
from core import model
from core import charts
from core import modelviz
from core import rules

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg"}



def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.before_request
def _ensure_db():
    if not os.path.exists(db.DB_PATH):
        db.init_db()
        import seed
        seed.seed()


@app.route("/")
def dashboard():
    stats = db.get_stats()
    recent = db.get_recent(6)
    return render_template("dashboard.html", active="dashboard", stats=stats, recent=recent)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not _allowed(f.filename):
            return redirect(url_for("upload"))

        fname = secure_filename(f.filename)
        ts = datetime.now().strftime("%Y%m%d%H%M%S_")
        fname = ts + fname
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)

        feats = features.extract(path)              # display features for the DB

        # Primary auto-classification: configurable conditional rules (no ML).
        cfg = db.get_setting("rules", rules.DEFAULT_RULES)
        auto_label, confidence, reason = rules.classify(feats, cfg)

        record = {
            "filename": fname,
            "upload_date": datetime.now().isoformat(),
            **feats,
            "manual_label": None,
            "auto_label": auto_label,
            "confidence": confidence,
            "rule_reason": reason,
            "latitude": None,
            "longitude": None,
        }
        new_id = db.insert(record)
        return redirect(url_for("annotate", id=new_id))

    # Upload now lives on the fused annotate page.
    return redirect(url_for("annotate"))


@app.route("/annotate")
def annotate():
    image_id = request.args.get("id", type=int)
    if image_id:
        image = db.get_by_id(image_id)
    else:
        image = db.get_next_unannotated()

    _, total = db.count_annotated()
    current_pos = 0
    ml_label = ml_conf = None

    if image:
        current_pos, total = db.get_position(image["id"])
        # Live ML prediction (not stored) — shown alongside the rule label.
        img_path = os.path.join(UPLOAD_DIR, image["filename"])
        if os.path.exists(img_path):
            try:
                ml_label, ml_conf = model.predict(features.model_vector(img_path))
            except Exception:
                ml_label = ml_conf = None

    return render_template(
        "annotate.html",
        active="annotate",
        image=image,
        images=db.list_images(),
        current_pos=current_pos,
        total_count=total,
        ml_label=ml_label,
        ml_conf=ml_conf,
    )


@app.route("/annotate/<int:id>", methods=["POST"])
def annotate_submit(id):
    label = request.form.get("label")
    if label in ("clean", "dirty"):
        db.set_label(id, label)
    # Stay on the same image so the saved label is visible; navigation is via
    # the image picker, not prev/next.
    return redirect(url_for("annotate", id=id))


@app.route("/image/<int:id>/normalized.png")
def image_normalized(id):
    row = db.get_by_id(id)
    if not row:
        return Response(status=404)
    path = os.path.join(UPLOAD_DIR, row["filename"])
    if not os.path.exists(path):
        return Response(status=404)
    return Response(features.normalized_png(path), mimetype="image/png")


# EFREI Paris main campus — 30-32 Av. de la République, 94800 Villejuif.
EFREI = {"lat": 48.78937, "lng": 2.36266}


@app.route("/map")
def map_page():
    locations = db.get_located()
    markers = [
        {
            "lat": l["latitude"],
            "lng": l["longitude"],
            "label": l["manual_label"] or l["auto_label"] or "unknown",
            "filename": l["filename"],
            "id": l["id"],
        }
        for l in locations
    ]
    return render_template(
        "map.html",
        active="map",
        locations=locations,
        markers=markers,
        efrei=EFREI,
    )


@app.route("/rules", methods=["GET", "POST"])
def rules_page():
    cfg = db.get_setting("rules", rules.DEFAULT_RULES)
    applied = False

    if request.method == "POST":
        # Read and save user-defined thresholds, then re-classify every image.
        new_cfg = {}
        for key in rules.DEFAULT_RULES:
            try:
                new_cfg[key] = float(request.form.get(key, cfg[key]))
            except (TypeError, ValueError):
                new_cfg[key] = cfg[key]
        db.set_setting("rules", new_cfg)
        cfg = new_cfg

        for row in db.get_all_for_charts():
            label, conf, reason = rules.classify(row, cfg)
            db.update_auto_label(row["id"], label, conf, reason)
        applied = True

    return render_template("rules.html", active="rules", cfg=cfg, applied=applied)


@app.route("/model")
def model_page():
    meta = modelviz.load_meta()
    return render_template("model.html", active="model", meta=meta)


# Model-page charts dispatched by name.
_MODEL_CHARTS = {
    "cv_scores": lambda: modelviz.cv_scores_png(modelviz.load_meta()),
    "confusion_matrix": lambda: modelviz.confusion_matrix_png(modelviz.load_meta()),
    "feature_importance": modelviz.feature_importance_png,
    "feature_distributions": modelviz.feature_distributions_png,
    "pca": modelviz.pca_scatter_png,
}


@app.route("/charts/model/<chart>.png")
def chart_model(chart):
    fn = _MODEL_CHARTS.get(chart)
    if fn is None:
        return Response(status=404)
    return Response(fn(), mimetype="image/png")


@app.route("/charts/class_distribution.png")
def chart_class_distribution():
    stats = db.get_stats()
    png = charts.class_distribution_png(stats["clean"], stats["dirty"])
    return Response(png, mimetype="image/png")


@app.route("/charts/uploads_over_time.png")
def chart_uploads_over_time():
    rows = db.get_all_for_charts()
    png = charts.uploads_over_time_png(rows)
    return Response(png, mimetype="image/png")


@app.route("/charts/file_size_distribution.png")
def chart_file_size_distribution():
    rows = db.get_all_for_charts()
    png = charts.file_size_distribution_png(rows)
    return Response(png, mimetype="image/png")


@app.route("/charts/histogram/<int:id>.png")
def chart_histogram(id):
    row = db.get_by_id(id)
    hist = row["hist"] if row else None
    return Response(charts.histogram_png(hist), mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
