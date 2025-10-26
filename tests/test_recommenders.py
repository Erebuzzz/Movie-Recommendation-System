from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ modules are on path when running tests directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_preparation import create_user_item_matrix, load_movielens_data, merge_datasets
from collaborative_filtering import get_collab_recommendations, init_collaborative_model
from content_based import get_content_recommendations, init_content_model, score_external_movie
from hybrid import get_hybrid_recommendations, init_hybrid_model, suggest_titles
from advanced_enhancements import build_svd_model, hybrid_with_svd_blend


def test_data_loading_shapes() -> None:
    ratings, movies = load_movielens_data()
    merged = merge_datasets(ratings, movies)
    assert {"userId", "movieId", "rating"}.issubset(merged.columns)
    assert len(ratings) == len(merged)


def test_user_item_matrix_contains_users() -> None:
    ratings, movies = load_movielens_data()
    merged = merge_datasets(ratings, movies)
    matrix = create_user_item_matrix(merged)
    assert matrix.index.size > 0
    assert matrix.columns.size > 0


def test_collaborative_recommendations_have_titles() -> None:
    model = init_collaborative_model()
    recs = get_collab_recommendations(model, "Toy Story (1995)", top_n=3)
    assert len(recs) == 3
    assert recs["title"].notna().all()


def test_content_recommendations_similarity() -> None:
    model = init_content_model()
    recs = get_content_recommendations(model, "Toy Story (1995)", top_n=5)
    assert len(recs) == 5
    assert recs["score"].iloc[0] >= recs["score"].iloc[-1]


def test_hybrid_recommendations_columns() -> None:
    model = init_hybrid_model(min_support=5)
    recs = get_hybrid_recommendations(model, "Toy Story (1995)", top_n=5)
    expected_cols = {"movieId", "title", "final_score", "collab_score", "content_score"}
    assert expected_cols.issubset(recs.columns)
    assert len(recs) == 5


def test_svd_blended_scores_sorted() -> None:
    _, _, item_factors = build_svd_model(n_factors=20, min_support=15, verbosity=False)
    blended = hybrid_with_svd_blend(
        title="Toy Story (1995)",
        svd_weight=0.3,
        item_factors=item_factors,
        top_n=5,
    )
    assert "svd_score" in blended.columns
    scores = blended["final_score"].to_numpy()
    assert np.all(scores[:-1] >= scores[1:])


def test_suggest_titles_returns_match() -> None:
    hybrid_model = init_hybrid_model()
    suggestions = suggest_titles(hybrid_model, "Jurrasic Park", limit=3)
    assert suggestions
    assert any("Jurassic Park" in item["title"] for item in suggestions)


def test_score_external_movie_works_with_metadata() -> None:
    content_model = init_content_model()
    recs = score_external_movie(
        content_model,
        title="Toy Story 4",
        genres="Adventure Animation",
        plot="Woody and Buzz embark on a road trip with new friend Forky.",
        top_n=3,
    )
    assert not recs.empty
    assert recs["score"].iloc[0] >= recs["score"].iloc[-1]
