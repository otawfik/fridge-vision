from src.recipes import RecipeRecommender


def test_recommend_returns_ranked_results():
    rec = RecipeRecommender()
    results = rec.recommend(["banana", "apple"], top_k=3)
    assert 1 <= len(results) <= 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_banana_query_surfaces_banana_recipes():
    rec = RecipeRecommender()
    results = rec.recommend(["banana"], top_k=5)
    titles = " ".join(r["title"].lower() for r, _ in results)
    assert "banana" in titles


def test_top_k_respected():
    rec = RecipeRecommender()
    assert len(rec.recommend(["apple", "carrot", "broccoli"], top_k=2)) <= 2


def test_empty_pantry_returns_nothing():
    rec = RecipeRecommender()
    assert rec.recommend([]) == []
    assert rec.recommend(["wine glass", "cup"]) == []  # non-food detections


def test_pantry_tokens_dedup_and_map():
    rec = RecipeRecommender()
    assert rec.pantry_tokens(["apple", "apple", "bottle"]) == ["apple", "milk"]
