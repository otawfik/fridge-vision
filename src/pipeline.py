"""End-to-end pipeline: fridge photo -> detections -> freshness -> inventory -> recipes."""
from PIL import Image

from .detect import annotate, crop_item, detect_items
from .freshness import get_model
from .inventory import build_inventory
from .recipes import RecipeRecommender


def analyze(image, conf=0.35, top_recipes=5):
    """Run the full Fridge Vision pipeline on a PIL image or image path.

    Returns a dict with:
      detections: [{"label", "confidence", "bbox", "freshness"}]
      inventory:  rows from build_inventory(), use-soon-first
      recipes:    [(recipe, score)] from the recommender
      annotated:  PIL image with detection boxes drawn
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    detections = detect_items(image, conf=conf)
    model = get_model()
    for det in detections:
        try:
            # center-crop the box: score the item, not the background
            det["freshness"] = model.score(crop_item(image, det["bbox"], center_frac=0.6))
        except Exception:
            det["freshness"] = 0.75

    freshness_by_label = {}
    for det in detections:
        freshness_by_label.setdefault(det["label"], []).append(det["freshness"])
    freshness_by_label = {k: sum(v) / len(v) for k, v in freshness_by_label.items()}

    inventory = build_inventory(detections, freshness_by_label)
    labels = [d["label"] for d in detections]
    recipes = RecipeRecommender().recommend(labels, top_k=top_recipes)

    return {
        "detections": detections,
        "inventory": inventory,
        "recipes": recipes,
        "annotated": annotate(image, detections),
    }
