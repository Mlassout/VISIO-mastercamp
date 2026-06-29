"""Server-rendered matplotlib chart PNGs matching the mockup palette."""

import io
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

FONT_FAMILY = "sans-serif"
GREEN = "#2f6b4f"
AMBER = "#cf9a4a"
GRID = "#eef1f1"
AXIS = "#cdd3d4"
LABEL = "#7b878d"
TICK = "#a7afb3"
TITLE = "#37444b"
BG = "#ffffff"


def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(colors=TICK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def class_distribution_png(clean_count: int, dirty_count: int) -> bytes:
    fig, ax = plt.subplots(figsize=(4.8, 2.5), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    bars = ax.bar(
        ["Empty", "Full"],
        [clean_count, dirty_count],
        color=[GREEN, AMBER],
        width=0.45,
        zorder=3,
    )
    for bar, val in zip(bars, [clean_count, dirty_count]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(clean_count, dirty_count) * 0.02,
            str(val),
            ha="center", va="bottom",
            fontsize=11, fontweight="600", color=TITLE,
        )

    ax.set_ylabel("")
    ax.tick_params(axis="x", colors=LABEL)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def uploads_over_time_png(rows: list) -> bytes:
    by_date = defaultdict(int)
    for r in rows:
        d = r["upload_date"][:10]
        by_date[d] += 1

    dates = sorted(by_date.keys())
    counts = [by_date[d] for d in dates]
    x = [datetime.fromisoformat(d) for d in dates]

    fig, ax = plt.subplots(figsize=(4.8, 2.5), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    if x:
        ax.fill_between(x, counts, alpha=0.08, color=GREEN, zorder=2)
        ax.plot(x, counts, color=GREEN, linewidth=2, zorder=3)
        ax.scatter(x[-2:] if len(x) >= 2 else x, counts[-2:] if len(counts) >= 2 else counts,
                   color=GREEN, s=18, zorder=4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate(rotation=0, ha="center")

    ax.tick_params(axis="x", colors=LABEL)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def file_size_distribution_png(rows: list) -> bytes:
    """Histogram of uploaded image file sizes (KB)."""
    sizes = [r["file_size_kb"] for r in rows if r.get("file_size_kb")]

    fig, ax = plt.subplots(figsize=(9.8, 2.6), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    if sizes:
        ax.hist(sizes, bins=12, color=GREEN, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("File size (KB)", color=LABEL, fontsize=10)
    ax.tick_params(axis="x", colors=LABEL)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def histogram_png(hist_json: str) -> bytes:
    """Colour + luminance histograms for a single image, from stored JSON."""
    try:
        h = json.loads(hist_json) if hist_json else {}
    except (ValueError, TypeError):
        h = {}

    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=100)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    n = len(h.get("gray", [])) or 16
    x = list(range(n))
    for key, colour in (("r", "#c0533b"), ("g", "#2f6b4f"), ("b", "#3b6fc0")):
        if h.get(key):
            ax.plot(x, h[key], color=colour, linewidth=1.5, zorder=3)
    if h.get("gray"):
        ax.fill_between(x, h["gray"], color="#7b878d", alpha=0.18, zorder=2)

    ax.set_xlim(0, n - 1)
    ax.set_xticks([0, n // 2, n - 1])
    ax.set_xticklabels(["0", "128", "255"])
    ax.set_xlabel("Pixel intensity", color=LABEL, fontsize=10)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
