from src.inventory import build_inventory, estimate_days_left, load_shelf_life


def test_shelf_life_table_loads():
    table = load_shelf_life()
    assert table["apple"]["fridge_days"] == 21
    assert table["sandwich"]["ready_to_eat"] is True


def test_fresher_item_gets_more_days():
    assert estimate_days_left("apple", 1.0) >= estimate_days_left("apple", 0.0)


def test_days_left_never_negative():
    assert estimate_days_left("apple", 0.0) >= 0


def test_unknown_item_uses_fallback():
    assert estimate_days_left("dragonfruit", 0.9) > 0


def test_build_inventory_merges_duplicates():
    dets = [
        {"label": "apple", "confidence": 0.9, "bbox": (0, 0, 10, 10)},
        {"label": "apple", "confidence": 0.7, "bbox": (20, 20, 30, 30)},
        {"label": "banana", "confidence": 0.8, "bbox": (0, 0, 10, 10)},
    ]
    rows = build_inventory(dets, {"apple": 0.9, "banana": 0.4})
    by_item = {r["item"]: r for r in rows}
    assert by_item["apple"]["count"] == 2
    assert by_item["apple"]["confidence"] == 0.8
    assert by_item["banana"]["freshness_label"] == "spoiled"


def test_inventory_sorted_use_soon_first():
    dets = [
        {"label": "banana", "confidence": 0.9, "bbox": (0, 0, 10, 10)},
        {"label": "apple", "confidence": 0.9, "bbox": (0, 0, 10, 10)},
    ]
    rows = build_inventory(dets, {"banana": 0.9, "apple": 0.9})
    assert rows[0]["item"] == "banana"  # 5 days vs 21 days shelf life
