"""Fridge inventory: turn detections + freshness scores into a grocery-style list.

Each detected item gets an estimated days-left computed from a shelf-life
table (fridge storage, USDA FoodKeeper-style guidance) scaled by the ML
freshness score: a fresher item keeps closer to its full shelf life, a
borderline one is flagged to use soon. Estimates are heuristic, the table is
the honest part.
"""
import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shelf_life.csv")

FALLBACK = {"fridge_days": 3, "ready_to_eat": 0, "tip": "When in doubt, use within 3 days."}


def load_shelf_life(path=None):
    table = {}
    with open(path or DATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            table[row["item"].strip().lower()] = {
                "fridge_days": int(row["fridge_days"]),
                "ready_to_eat": row["ready_to_eat"].strip() == "1",
                "tip": row["tip"].strip(),
            }
    return table


def estimate_days_left(item, freshness_score, table=None):
    """Heuristic days-left: shelf life scaled by freshness. Returns an int >= 0."""
    table = table or load_shelf_life()
    entry = table.get(item.lower(), FALLBACK)
    days = entry["fridge_days"] * (0.35 + 0.65 * max(0.0, min(1.0, freshness_score)))
    return max(0, int(round(days)))


def build_inventory(detections, freshness_scores, table=None):
    """Merge detections into one row per item.

    detections: list of {"label", "confidence", ...}
    freshness_scores: dict label -> P(fresh) in [0, 1]
    Returns rows sorted by days left (use-soon-first).
    """
    table = table or load_shelf_life()
    grouped = {}
    for det in detections:
        label = det["label"]
        grouped.setdefault(label, {"count": 0, "confidences": []})
        grouped[label]["count"] += 1
        grouped[label]["confidences"].append(det["confidence"])

    rows = []
    for label, info in grouped.items():
        score = float(freshness_scores.get(label, 0.75))
        entry = table.get(label.lower(), FALLBACK)
        rows.append({
            "item": label,
            "count": info["count"],
            "confidence": sum(info["confidences"]) / len(info["confidences"]),
            "freshness": score,
            "freshness_label": "fresh" if score >= 0.5 else "spoiled",
            "days_left": estimate_days_left(label, score, table),
            "ready_to_eat": bool(entry["ready_to_eat"]),
            "tip": entry["tip"],
        })
    rows.sort(key=lambda r: (r["days_left"], -r["freshness"]))
    return rows
