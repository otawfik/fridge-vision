"""Render the README demo figure: annotated photo + inventory + recipes.

Usage: python scripts/make_demo.py [--photo assets/fridge.jpg] [--out screenshots/demo.svg]
Requires the trained freshness model (run scripts/train_freshness.py first).
The output is SVG (plain text) so it can live in the repo without binary blobs.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.pipeline import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="assets/fridge.jpg")
    ap.add_argument("--out", default="screenshots/demo.svg")
    args = ap.parse_args()

    result = analyze(Image.open(args.photo).convert("RGB"))
    annotated = result["annotated"]
    inventory = result["inventory"]
    recipes = result["recipes"]

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(annotated)
    ax0.axis("off")
    ax0.set_title(f"Fridge Vision: {len(result['detections'])} items detected",
                  fontsize=13, fontweight="bold", loc="left")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    inv_lines = ["INVENTORY (use-soon-first)", ""]
    for row in inventory:
        mark = "READY" if row["ready_to_eat"] else f"{row['days_left']}d left"
        inv_lines.append(
            f"{row['item']} x{row['count']}  |  fresh {row['freshness']:.0%}  |  {mark}"
        )
    if not inv_lines[2:]:
        inv_lines.append("(no items detected)")
    ax1.text(0, 1, "\n".join(inv_lines), va="top", ha="left", fontsize=10,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f4f8f4", edgecolor="#2ecc71"))

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    rec_lines = ["RECIPES FOR WHAT YOU HAVE", ""]
    for recipe, score in recipes[:4]:
        rec_lines.append(f"* {recipe['title']} ({recipe['time_min']} min)")
        rec_lines.append(f"  needs: {', '.join(recipe['ingredients'][:5])}")
        rec_lines.append("")
    if not recipes:
        rec_lines.append("(no matching recipes)")
    ax2.text(0, 1, "\n".join(rec_lines).rstrip(), va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#fef9ef", edgecolor="#f0ad4e"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print(f"saved {args.out}")
    print(f"detections: {[(d['label'], round(d['confidence'], 2)) for d in result['detections']]}")


if __name__ == "__main__":
    main()
