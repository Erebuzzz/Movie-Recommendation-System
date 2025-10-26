from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preparation import RAW_DATA_DIR


@dataclass
class ContentModel:
    movies: pd.DataFrame
    tfidf_matrix: np.ndarray
    title_index: Dict[str, int]
    movie_ids: List[int]
    vectorizer: TfidfVectorizer


def _load_metadata() -> pd.DataFrame:
    movies_path = RAW_DATA_DIR / "movies.csv"
    tags_path = RAW_DATA_DIR / "tags.csv"

    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)

    tag_strings = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str))).rename("tags")
    movies = movies.join(tag_strings, on="movieId")

    movies["genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["tags"] = movies["tags"].fillna("")
    movies["content"] = (movies["title"].fillna("") + " " + movies["genres"] + " " + movies["tags"]).str.strip()
    movies["content"] = movies["content"].replace("", np.nan).fillna(movies["title"])

    return movies


def init_content_model() -> ContentModel:
    movies = _load_metadata()
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(movies["content"])

    title_index = {title.lower(): idx for idx, title in enumerate(movies["title"]) if isinstance(title, str)}
    movie_ids = movies["movieId"].tolist()

    return ContentModel(
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        title_index=title_index,
        movie_ids=movie_ids,
        vectorizer=vectorizer,
    )


def _resolve_title(model: ContentModel, movie_name: str) -> int:
    lower = movie_name.lower()
    if lower in model.title_index:
        return model.title_index[lower]

    matches = model.movies[model.movies["title"].str.contains(movie_name, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Movie '{movie_name}' not found for content-based lookup.")
    return matches.index[0]


def get_content_recommendations(model: ContentModel, movie_name: str, top_n: int = 5) -> pd.DataFrame:
    idx = _resolve_title(model, movie_name)
    cosine_scores = cosine_similarity(model.tfidf_matrix[idx], model.tfidf_matrix).flatten()
    cosine_scores[idx] = -1

    top_indices = np.argpartition(cosine_scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(cosine_scores[top_indices])[::-1]]

    recs = []
    for rec_idx in top_indices:
        row = model.movies.iloc[rec_idx]
        recs.append(
            {
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "score": round(float(cosine_scores[rec_idx]), 4),
            }
        )
    return pd.DataFrame(recs)


def score_external_movie(
    model: ContentModel,
    title: str,
    genres: str = "",
    plot: str = "",
    tags: str = "",
    top_n: int = 5,
) -> pd.DataFrame:
    """Score an external movie description against the catalog."""

    document = " ".join(filter(None, [title, genres, plot, tags]))
    if not document.strip():
        raise ValueError("At least one metadata field (title/genres/plot/tags) is required.")

    tfidf_vector = model.vectorizer.transform([document])
    similarity = cosine_similarity(tfidf_vector, model.tfidf_matrix).flatten()

    top_indices = np.argpartition(similarity, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarity[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        row = model.movies.iloc[idx]
        results.append(
            {
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "score": round(float(similarity[idx]), 4),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    model = init_content_model()
    sample_movie = "Toy Story (1995)"
    print(f"Sample content-based neighbours for '{sample_movie}':")
    recs = get_content_recommendations(model, sample_movie, top_n=5)
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
