"""Recipe recommendations from detected fridge items.

A TF-IDF retriever over a curated recipe set: each recipe is a document made
of its ingredient tokens, the user's detected items are the query. Cosine
similarity ranks recipes by ingredient overlap, weighted so rare ingredients
count more than pantry staples. Simple, explainable, no API calls.
"""
import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes.csv")

# Detection label -> ingredient tokens it implies. Non-food detections map to [].
INGREDIENT_MAP = {
    "apple": ["apple"],
    "banana": ["banana"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "sandwich": ["bread"],
    "hot dog": ["hot dog", "bread"],
    "pizza": ["pizza"],
    "donut": ["donut"],
    "cake": ["cake"],
    "bottle": ["milk"],
    "cup": [],
    "bowl": [],
    "wine glass": [],
}


def load_recipes(path=None):
    recipes = []
    with open(path or DATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            recipes.append({
                "title": row["title"].strip(),
                "ingredients": [i.strip() for i in row["ingredients"].split(",") if i.strip()],
                "time_min": int(row["time_min"]),
                "steps": [s.strip() for s in row["steps"].split("|") if s.strip()],
            })
    return recipes


class RecipeRecommender:
    def __init__(self, recipes=None):
        self.recipes = recipes or load_recipes()
        docs = [" ".join(r["ingredients"]).lower() for r in self.recipes]
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z][a-zA-Z ]*?[a-zA-Z]")
        self.doc_matrix = self.vectorizer.fit_transform(docs)

    def pantry_tokens(self, labels):
        tokens = []
        for label in labels:
            tokens.extend(INGREDIENT_MAP.get(label.lower(), []))
        # de-dup, keep order
        return list(dict.fromkeys(tokens))

    def recommend(self, labels, top_k=5):
        """Rank recipes for the detected labels. Returns list of (recipe, score)."""
        tokens = self.pantry_tokens(labels)
        if not tokens:
            return []
        query = self.vectorizer.transform([" ".join(tokens).lower()])
        scores = cosine_similarity(query, self.doc_matrix)[0]
        ranked = sorted(zip(self.recipes, scores), key=lambda rs: -rs[1])
        return [(r, float(s)) for r, s in ranked[:top_k] if s > 0]
