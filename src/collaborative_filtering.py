from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from data_preparation import (
    PROCESSED_DATA_DIR,
    create_user_item_matrix,
    load_movielens_data,
    merge_datasets,
)


@dataclass
class CollaborativeModel:
    merged: pd.DataFrame
    user_item: pd.DataFrame
    movie_index: Dict[int, int]
    movies: List[int]
    item_user_matrix: sparse.csr_matrix
    movie_lookup: pd.Series
    rating_counts: pd.Series


def _load_phase1_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_path = PROCESSED_DATA_DIR / "ratings_with_movies.csv"
    matrix_path = PROCESSED_DATA_DIR / "user_item_matrix.pkl"

    if merged_path.exists() and matrix_path.exists():
        merged = pd.read_csv(merged_path, parse_dates=["timestamp"])
        user_item = pd.read_pickle(matrix_path)
        return merged, user_item

    ratings, movies = load_movielens_data()
    merged = merge_datasets(ratings, movies)
    user_item = create_user_item_matrix(merged)
    return merged, user_item


def _build_sparse_item_user(user_item: pd.DataFrame) -> sparse.csr_matrix:
    filled = user_item.fillna(0.0)
    return sparse.csr_matrix(filled.values, dtype=np.float32)


def init_collaborative_model() -> CollaborativeModel:
    merged, user_item = _load_phase1_outputs()
    item_user = user_item.transpose()
    sparse_item_user = _build_sparse_item_user(item_user)

    movie_ids = item_user.index.to_list()
    index_lookup = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    movie_titles = merged.drop_duplicates("movieId").set_index("movieId")["title"]
    rating_counts = merged.groupby("movieId")["rating"].count()

    return CollaborativeModel(
        merged=merged,
        user_item=user_item,
        movie_index=index_lookup,
        movies=movie_ids,
        item_user_matrix=sparse_item_user,
        movie_lookup=movie_titles,
        rating_counts=rating_counts,
    )


def _resolve_movie_id(model: CollaborativeModel, movie_name: str) -> int:
    matches = model.movie_lookup[model.movie_lookup.str.lower() == movie_name.lower()]
    if matches.empty:
        partial_matches = model.movie_lookup[model.movie_lookup.str.contains(movie_name, case=False)]
        if partial_matches.empty:
            raise ValueError(f"Movie '{movie_name}' was not found in the catalog.")
        return partial_matches.index[0]
    return matches.index[0]


def get_collab_recommendations(
    model: CollaborativeModel, movie_name: str, top_n: int = 5
) -> pd.DataFrame:
    movie_id = _resolve_movie_id(model, movie_name)
    row_idx = model.movie_index.get(movie_id)
    if row_idx is None:
        raise ValueError(f"Movie id {movie_id} was not present in the similarity matrix.")

    target_vector = model.item_user_matrix[row_idx]
    similarity_scores = cosine_similarity(target_vector, model.item_user_matrix).flatten()

    similarity_scores[row_idx] = -1.0
    top_indices = np.argpartition(similarity_scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]

    recommended_ids = [model.movies[idx] for idx in top_indices]

    recommendations: List[Dict[str, object]] = []
    for mid in recommended_ids:
        title = model.movie_lookup.get(mid, "Unknown")
        score = float(similarity_scores[model.movie_index[mid]])
        support = int(model.rating_counts.get(mid, 0))
        recommendations.append(
            {"movieId": mid, "title": title, "score": round(score, 4), "ratings": support}
        )

    return pd.DataFrame(recommendations)


def main() -> None:
    model = init_collaborative_model()
    movie_title = "Toy Story (1995)"
    print(f"Sample collaborative recommendations for '{movie_title}':")
    sample = get_collab_recommendations(model, movie_title, top_n=5)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
