"""Fridge item detection with a pretrained YOLOv8 detector.

We use YOLOv8n pretrained on COCO and keep only the classes that show up in
fridges: produce, ready-to-eat food, and containers. No training needed here,
the model is used as a frozen feature: image in, labeled boxes out.
"""
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# COCO class id -> friendly label, fridge-relevant subset only.
FRIDGE_CLASSES = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
}

_model = None


def load_detector(weights="yolov8n.pt"):
    """Load (and cache) the YOLOv8 detector. Downloads weights on first use."""
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _dedup(detections, iou_thresh=0.5):
    """Drop lower-confidence boxes that heavily overlap a same-label box."""
    kept = []
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        if any(d["label"] == det["label"] and _iou(d["bbox"], det["bbox"]) > iou_thresh
               for d in kept):
            continue
        kept.append(det)
    return kept


def detect_items(image, conf=0.35):
    """Detect fridge items in a PIL image or image path.

    Returns a list of dicts: {"label", "confidence", "bbox": (x1, y1, x2, y2)}.
    """
    model = load_detector()
    results = model.predict(image, conf=conf, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in FRIDGE_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append({
                "label": FRIDGE_CLASSES[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
            })
    return _dedup(detections)


def crop_item(image, bbox, center_frac=1.0):
    """Crop a detection box out of a PIL image, clamped to the image bounds.

    center_frac < 1.0 keeps only the central fraction (per side), which focuses
    freshness scoring on the item instead of background inside the box.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    w, h = image.size
    x1, y1, x2, y2 = bbox
    box = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
    crop = image.crop(box)
    if center_frac < 1.0:
        cw, ch = crop.size
        m = (1.0 - center_frac) / 2.0
        crop = crop.crop((int(cw * m), int(ch * m), int(cw * (1 - m)), int(ch * (1 - m))))
    return crop


def annotate(image, detections):
    """Draw labeled boxes on a copy of the image. Returns a new PIL image."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(46, 204, 113), width=3)
        text = f'{det["label"]} {det["confidence"]:.0%}'
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tx1 - 2, ty1 - 2, tx2 + 2, ty2 + 2], fill=(46, 204, 113))
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    return out
