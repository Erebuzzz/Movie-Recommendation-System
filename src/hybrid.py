from __future__ import annotations

from dataclasses import dataclass
from typing import List

import difflib
import numpy as np
import pandas as pd

from collaborative_filtering import (
    CollaborativeModel,
    get_collab_recommendations,
    init_collaborative_model,
)
from content_based import (
    ContentModel,
    get_content_recommendations,
    init_content_model,
)


@dataclass
class HybridModel:
    collab: CollaborativeModel
    content: ContentModel
    popularity: pd.DataFrame
    mean_ratings: pd.Series
    min_support: int = 20


def _build_popularity_table(collab_model: CollaborativeModel) -> pd.DataFrame:
    merged = collab_model.merged

    stats = (
        merged.groupby("movieId")["rating"]
        .agg([("mean_rating", "mean"), ("rating_count", "count")])
        .reset_index()
    )

    titles = (
        merged.drop_duplicates("movieId")[["movieId", "title", "genres"]]
        .set_index("movieId")
        .reset_index()
    )

    popularity = stats.merge(titles, on="movieId", how="left")
    popularity = popularity.sort_values(
        ["rating_count", "mean_rating"], ascending=[False, False]
    ).reset_index(drop=True)
    return popularity


def _normalize_scores(scores: pd.Series) -> pd.Series:
    clean = scores.astype(float).fillna(0.0)
    if clean.empty:
        return clean

    max_val = clean.max()
    min_val = clean.min()

    if np.isclose(max_val, min_val):
        return (clean > 0).astype(float)

    normalized = (clean - min_val) / (max_val - min_val)
    return normalized.clip(0.0, 1.0)


def _parse_genre_string(genres: str | float | None) -> set[str]:
    if not isinstance(genres, str):
        return set()
    tokens = genres.replace(",", "|").split("|")
    cleaned = {token.strip().lower() for token in tokens if token and token.strip()}
    cleaned.discard("(no genres listed)")
    return cleaned


def _compute_genre_similarity(
    seed_genres: set[str], candidate_genres: str | float | None
) -> float:
    if not seed_genres:
        return 0.0
    candidate = _parse_genre_string(candidate_genres)
    if not candidate:
        return 0.0
    overlap = seed_genres & candidate
    if not overlap:
        return 0.0
    union = seed_genres | candidate
    if not union:
        return 0.0
    return len(overlap) / len(union)


def _find_movie_id(hybrid: HybridModel, movie_name: str) -> int | None:
    lookup = hybrid.collab.movie_lookup.dropna()
    lower_name = movie_name.lower()
    exact = lookup[lookup.str.lower() == lower_name]
    if not exact.empty:
        return int(exact.index[0])
    partial = lookup[lookup.str.contains(movie_name, case=False, regex=False)]
    if not partial.empty:
        return int(partial.index[0])
    return None


def suggest_titles(
    hybrid: HybridModel, query: str, limit: int = 5
) -> list[dict[str, str | int]]:
    if not query:
        return []

    lookup = hybrid.collab.movie_lookup.dropna()
    suggestions: list[dict[str, str | int]] = []
    seen_titles: set[str] = set()

    partial = lookup[lookup.str.contains(query, case=False, regex=False)]
    for movie_id, title in partial.head(limit).items():
        suggestions.append({"movieId": int(movie_id), "title": title})
        seen_titles.add(title)
        if len(suggestions) >= limit:
            return suggestions

    close_matches = difflib.get_close_matches(query, lookup.tolist(), n=limit, cutoff=0.4)
    for title in close_matches:
        if title in seen_titles:
            continue
        match_series = lookup[lookup == title]
        if match_series.empty:
            continue
        suggestions.append({"movieId": int(match_series.index[0]), "title": title})
        seen_titles.add(title)
        if len(suggestions) >= limit:
            break

    return suggestions


def init_hybrid_model(min_support: int = 20) -> HybridModel:
    collab_model = init_collaborative_model()
    content_model = init_content_model()
    popularity = _build_popularity_table(collab_model)
    mean_ratings = collab_model.merged.groupby("movieId")["rating"].mean()
    return HybridModel(
        collab=collab_model,
        content=content_model,
        popularity=popularity,
        mean_ratings=mean_ratings,
        min_support=min_support,
    )


def _hydrate_recommendations(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["movieId", source])
    rename_map = {"score": f"{source}_score"}
    return df.rename(columns=rename_map)[["movieId", f"{source}_score"]]


def _lookup_titles(hybrid: HybridModel, movie_ids: List[int]) -> pd.DataFrame:
    result = pd.DataFrame({"movieId": movie_ids})

    collab_titles = hybrid.collab.movie_lookup.dropna()
    content_meta = hybrid.content.movies.set_index("movieId")

    result["title"] = result["movieId"].map(collab_titles)
    result["genres"] = result["movieId"].map(content_meta["genres"])

    missing_title_mask = result["title"].isna()
    if missing_title_mask.any():
        result.loc[missing_title_mask, "title"] = result.loc[missing_title_mask, "movieId"].map(
            content_meta["title"]
        )

    result["title"] = result["title"].fillna("Unknown")
    result["genres"] = result["genres"].fillna("")
    result["rating_count"] = (
        result["movieId"].map(hybrid.collab.rating_counts).fillna(0).astype(int)
    )
    result["mean_rating"] = result["movieId"].map(hybrid.mean_ratings).round(2)
    return result


def _is_cold_start(hybrid: HybridModel, movie_name: str) -> bool:
    movie_id = _find_movie_id(hybrid, movie_name)
    if movie_id is None:
        return True

    support = int(hybrid.collab.rating_counts.get(movie_id, 0))
    return support < hybrid.min_support


def _get_popular_fallback(
    hybrid: HybridModel, exclude: List[int], top_n: int
) -> pd.DataFrame:
    available = hybrid.popularity[~hybrid.popularity["movieId"].isin(exclude)]
    fallback = available.head(top_n).copy()
    fallback["collab_score"] = 0.0
    fallback["content_score"] = 0.0
    fallback["final_score"] = (fallback["mean_rating"].fillna(0.0) / 5.0).round(3)
    fallback["genre_similarity"] = 0.0
    fallback["reason"] = "popular"
    columns = [
        "reason",
        "movieId",
        "title",
        "genres",
        "final_score",
        "collab_score",
        "content_score",
        "genre_similarity",
        "rating_count",
        "mean_rating",
    ]
    return fallback[columns]


def get_hybrid_recommendations(
    hybrid: HybridModel, movie_name: str, top_n: int = 5
) -> pd.DataFrame:
    seed_id = _find_movie_id(hybrid, movie_name)
    seed_genres: set[str] = set()
    if seed_id is not None:
        seed_row = hybrid.content.movies.loc[hybrid.content.movies["movieId"] == seed_id]
        if not seed_row.empty:
            seed_genres = _parse_genre_string(seed_row.iloc[0].get("genres"))

    if _is_cold_start(hybrid, movie_name):
        exclude = [seed_id] if seed_id is not None else []
        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)

    try:
        collab_df = get_collab_recommendations(
            hybrid.collab, movie_name, top_n=top_n * 3
        )
    except ValueError:
        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])

    try:
        content_df = get_content_recommendations(
            hybrid.content, movie_name, top_n=top_n * 3
        )
    except ValueError:
        content_df = pd.DataFrame(columns=["movieId", "score", "genres"])

    collab_scores = _hydrate_recommendations(collab_df, "collab")
    content_scores = _hydrate_recommendations(content_df, "content")

    combined = pd.merge(collab_scores, content_scores, on="movieId", how="outer").fillna(0.0)

    if seed_id is not None:
        combined = combined[combined["movieId"] != seed_id]

    if combined.empty:
        exclude = [seed_id] if seed_id is not None else []
        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)

    combined["collab_score"] = combined.get("collab_score", 0.0).astype(float)
    combined["content_score"] = combined.get("content_score", 0.0).astype(float)

    collab_norm = _normalize_scores(combined["collab_score"])
    content_norm = _normalize_scores(combined["content_score"])

    combined["collab_score"] = collab_norm
    combined["content_score"] = content_norm

    combined["hybrid_score"] = 0.65 * collab_norm + 0.35 * content_norm
    overlap_mask = (collab_norm > 0) & (content_norm > 0)
    combined.loc[overlap_mask, "hybrid_score"] += 0.1
    combined["hybrid_score"] = combined["hybrid_score"].clip(0.0, 1.0)

    combined = combined.sort_values("hybrid_score", ascending=False)
    movie_ids = combined["movieId"].astype(int).tolist()

    meta = _lookup_titles(hybrid, movie_ids)
    results = combined.merge(meta, on="movieId", how="left")
    results["reason"] = "hybrid"
    if seed_genres:
        results["genre_similarity"] = results["genres"].apply(
            lambda genres: _compute_genre_similarity(seed_genres, genres)
        )
    else:
        results["genre_similarity"] = 0.0

    results["final_score"] = (
        0.55 * results["collab_score"]
        + 0.3 * results["content_score"]
        + 0.15 * results["genre_similarity"]
    ).clip(0.0, 1.0)

    results = results[
        [
            "reason",
            "movieId",
            "title",
            "genres",
            "final_score",
            "collab_score",
            "content_score",
            "genre_similarity",
            "rating_count",
            "mean_rating",
        ]
    ]

    results = results.drop_duplicates("movieId")
    results = results.sort_values(
        ["final_score", "genre_similarity", "rating_count", "mean_rating"],
        ascending=[False, False, False, False],
    ).head(top_n)

    if len(results) < top_n:
        exclude_ids = results["movieId"].tolist()
        if seed_id is not None:
            exclude_ids.append(seed_id)
        extra = _get_popular_fallback(
            hybrid, exclude=exclude_ids, top_n=top_n - len(results)
        )
        results = pd.concat([results, extra], ignore_index=True)
        results = results.sort_values(
            ["final_score", "genre_similarity", "rating_count", "mean_rating"],
            ascending=[False, False, False, False],
        ).head(top_n)

    results["final_score"] = results["final_score"].round(4)
    results["collab_score"] = results["collab_score"].round(4)
    results["content_score"] = results["content_score"].round(4)
    results["genre_similarity"] = results["genre_similarity"].round(4)
    return results


def _print_recommendations(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no recommendations)")
        return

    for _, row in df.iterrows():
        title = row["title"][:42]
        genres = row["genres"][:45]
        avg_rating = "n/a" if pd.isna(row["mean_rating"]) else f"{row['mean_rating']:.2f}"
        print(
            f"  {row['reason']:<7} | {title:<42} | {genres:<45} | final={row['final_score']:.4f} "
            f"(collab={row['collab_score']:.4f}, content={row['content_score']:.4f}, "
            f"genre={row.get('genre_similarity', 0.0):.4f}) | "
            f"ratings={row['rating_count']:>3} | avg={avg_rating}"
        )


def main() -> None:
    hybrid_model = init_hybrid_model(min_support=20)
    samples = ["Toy Story (1995)", "Jurassic Park (1993)"]
    for movie in samples:
        print(f"\nHybrid recommendations for '{movie}':")
        recs = get_hybrid_recommendations(hybrid_model, movie, top_n=5)
        _print_recommendations(recs)


if __name__ == "__main__":
    main()
