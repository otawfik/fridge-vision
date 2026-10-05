"""Stream a stratified sample of fresh/rotten produce photos from Hugging Face.

Source: Project-AgML/fresh_rotten_fruit_classification (CC-BY-4.0), labels are
fresh (0) vs rotten (1) with a fruit_type field. We stream the augmented split
so we never download the full ~900MB, and save a small balanced sample under
data/freshness_sample/{fresh,rotten}/ for training.

Usage: python scripts/download_data.py [--per-class 700] [--out data/freshness_sample]
"""
import argparse
import os
import sys

# Sanitize proxy bypass lists: httpx (used by huggingface_hub) chokes on the
# bracketed IPv6 entries in no_proxy/NO_PROXY ("Invalid port: ':1]'").
# Hugging Face is not a bypass host, so dropping the bypass list is safe.
for _var in ("no_proxy", "NO_PROXY"):
    os.environ.pop(_var, None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET = "Project-AgML/fresh_rotten_fruit_classification"
CONFIG = "augmented"


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=600,
                    help="total images per label (fresh/rotten)")
    ap.add_argument("--per-fruit", type=int, default=150,
                    help="max images per (fruit, label)")
    ap.add_argument("--fruits", default="apple,banana,grape,guava",
                    help="comma-separated fruit types to sample")
    ap.add_argument("--out", default="data/freshness_sample")
    args = ap.parse_args()
    fruits = [f.strip() for f in args.fruits.split(",") if f.strip()]

    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    counts = {"fresh": 0, "rotten": 0}
    per_fruit = {}
    for row in ds:
        label = "fresh" if int(row["label"]) == 0 else "rotten"
        fruit = str(row.get("fruit_type", "fruit")).lower()
        if fruit not in fruits:
            continue
        key = (fruit, label)
        if counts[label] >= args.per_class or per_fruit.get(key, 0) >= args.per_fruit:
            if all(c >= args.per_class for c in counts.values()):
                break
            continue
        dest_dir = os.path.join(args.out, label)
        os.makedirs(dest_dir, exist_ok=True)
        fname = f"{fruit}_{per_fruit.get(key, 0):04d}.jpg"
        row["image"].convert("RGB").resize((256, 256)).save(
            os.path.join(dest_dir, fname), quality=88
        )
        counts[label] += 1
        per_fruit[key] = per_fruit.get(key, 0) + 1
        if sum(counts.values()) % 200 == 0:
            print(f"saved {counts} {dict(per_fruit)}", flush=True)
    print(f"done: {counts} {dict(per_fruit)}")


if __name__ == "__main__":
    main()
