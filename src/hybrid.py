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

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 60)


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


def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:
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


def suggest_titles(hybrid: HybridModel, query: str, limit: int = 5) -> list[dict[str, str | int]]:
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


def _get_popular_fallback(hybrid: HybridModel, exclude: List[int], top_n: int) -> pd.DataFrame:
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
        collab_df = get_collab_recommendations(hybrid.collab, movie_name, top_n=top_n * 3)
    except ValueError:
        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])

    try:
        content_df = get_content_recommendations(hybrid.content, movie_name, top_n=top_n * 3)
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
        extra = _get_popular_fallback(hybrid, exclude=exclude_ids, top_n=top_n - len(results))
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
from __future__ import annotationsfrom __future__ import annotations



from dataclasses import dataclassfrom dataclasses import dataclass

from typing import Listfrom typing import List



import difflibimport difflib

import numpy as npimport numpy as np

import pandas as pdimport pandas as pd



from collaborative_filtering import (from collaborative_filtering import (

    CollaborativeModel,    CollaborativeModel,

    get_collab_recommendations,    get_collab_recommendations,

    init_collaborative_model,    init_collaborative_model,

))

from content_based import (from content_based import (

    ContentModel,    ContentModel,

    get_content_recommendations,    get_content_recommendations,

    init_content_model,    init_content_model,

))



pd.set_option("display.width", 140)pd.set_option("display.width", 140)

pd.set_option("display.max_colwidth", 60)pd.set_option("display.max_colwidth", 60)





@dataclass@dataclass

class HybridModel:class HybridModel:

    collab: CollaborativeModel    collab: CollaborativeModel

    content: ContentModel    content: ContentModel

    popularity: pd.DataFrame    popularity: pd.DataFrame

    mean_ratings: pd.Series    mean_ratings: pd.Series

    min_support: int = 20    min_support: int = 20





def _build_popularity_table(collab_model: CollaborativeModel) -> pd.DataFrame:def _build_popularity_table(collab_model: CollaborativeModel) -> pd.DataFrame:

    merged = collab_model.merged    merged = collab_model.merged



    stats = (    stats = (

        merged.groupby("movieId")["rating"]        merged.groupby("movieId")["rating"]

        .agg([("mean_rating", "mean"), ("rating_count", "count")])        .agg([("mean_rating", "mean"), ("rating_count", "count")])

        .reset_index()        .reset_index()

    )    )



    titles = (    titles = (

        merged.drop_duplicates("movieId")["movieId"]        merged.drop_duplicates("movieId")[['movieId', 'title', 'genres']]

        .to_frame()        .set_index("movieId")

        .join(        .reset_index()

            merged.drop_duplicates("movieId")[["movieId", "title", "genres"]]    )

            .set_index("movieId"),

            on="movieId",    popularity = stats.merge(titles, on="movieId", how="left")

        )    popularity = popularity.sort_values(

        .reset_index(drop=True)        ["rating_count", "mean_rating"], ascending=[False, False]

    )    ).reset_index(drop=True)

    return popularity

    popularity = stats.merge(titles, on="movieId", how="left")

    popularity = popularity.sort_values(

        ["rating_count", "mean_rating"], ascending=[False, False]def _normalize_scores(scores: pd.Series) -> pd.Series:

    ).reset_index(drop=True)    clean = scores.astype(float).fillna(0.0)

    return popularity    if clean.empty:

        return clean



def _normalize_scores(scores: pd.Series) -> pd.Series:    max_val = clean.max()

    clean = scores.astype(float).fillna(0.0)    min_val = clean.min()

    if clean.empty:

        return clean    if np.isclose(max_val, min_val):

        return (clean > 0).astype(float)

    max_val = clean.max()

    min_val = clean.min()    normalized = (clean - min_val) / (max_val - min_val)

    return normalized.clip(0.0, 1.0)

    if np.isclose(max_val, min_val):

        return (clean > 0).astype(float)

def _parse_genre_string(genres: str | float | None) -> set[str]:

    normalized = (clean - min_val) / (max_val - min_val)    if not isinstance(genres, str):

    return normalized.clip(0.0, 1.0)        return set()

    tokens = genres.replace(",", "|").split("|")

    cleaned = {token.strip().lower() for token in tokens if token and token.strip()}

def _parse_genre_string(genres: str | float | None) -> set[str]:    cleaned.discard("(no genres listed)")

    if not isinstance(genres, str):    return cleaned

        return set()

    tokens = genres.replace(",", "|").split("|")

    cleaned = {token.strip().lower() for token in tokens if token and token.strip()}def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:

    cleaned.discard("(no genres listed)")    if not seed_genres:

    return cleaned        return 0.0

    candidate = _parse_genre_string(candidate_genres)

    if not candidate:

def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:        return 0.0

    if not seed_genres:    overlap = seed_genres & candidate

        return 0.0    if not overlap:

    candidate = _parse_genre_string(candidate_genres)        return 0.0

    if not candidate:    union = seed_genres | candidate

        return 0.0    if not union:

    overlap = seed_genres & candidate        return 0.0

    if not overlap:    return len(overlap) / len(union)

        return 0.0

    union = seed_genres | candidate

    if not union:def _find_movie_id(hybrid: HybridModel, movie_name: str) -> int | None:

        return 0.0    lookup = hybrid.collab.movie_lookup.dropna()

    return len(overlap) / len(union)    lower_name = movie_name.lower()

    exact = lookup[lookup.str.lower() == lower_name]

    if not exact.empty:

def _find_movie_id(hybrid: HybridModel, movie_name: str) -> int | None:        return int(exact.index[0])

    lookup = hybrid.collab.movie_lookup.dropna()    partial = lookup[lookup.str.contains(movie_name, case=False, regex=False)]

    lower_name = movie_name.lower()    if not partial.empty:

    exact = lookup[lookup.str.lower() == lower_name]        return int(partial.index[0])

    if not exact.empty:    return Nonefrom __future__ import annotations

        return int(exact.index[0])

    partial = lookup[lookup.str.contains(movie_name, case=False, regex=False)]from dataclasses import dataclass

    if not partial.empty:from typing import List

        return int(partial.index[0])

    return Noneimport difflib

import numpy as np

import pandas as pd

def suggest_titles(hybrid: HybridModel, query: str, limit: int = 5) -> list[dict[str, str | int]]:

    if not query:from collaborative_filtering import (

        return []    CollaborativeModel,

    get_collab_recommendations,

    lookup = hybrid.collab.movie_lookup.dropna()    init_collaborative_model,

    suggestions: list[dict[str, str | int]] = [])

    seen_titles: set[str] = set()from content_based import (

    ContentModel,

    partial = lookup[lookup.str.contains(query, case=False, regex=False)]    get_content_recommendations,

    for movie_id, title in partial.head(limit).items():    init_content_model,

        suggestions.append({"movieId": int(movie_id), "title": title}))

        seen_titles.add(title)

        if len(suggestions) >= limit:

            return suggestionspd.set_option("display.width", 140)

pd.set_option("display.max_colwidth", 60)

    close_matches = difflib.get_close_matches(query, lookup.tolist(), n=limit, cutoff=0.4)

    for title in close_matches:

        if title in seen_titles:@dataclass

            continueclass HybridModel:

        match_series = lookup[lookup == title]    collab: CollaborativeModel

        if match_series.empty:    content: ContentModel

            continue    popularity: pd.DataFrame

        suggestions.append({"movieId": int(match_series.index[0]), "title": title})    mean_ratings: pd.Series

        seen_titles.add(title)    min_support: int = 20

        if len(suggestions) >= limit:

            break

def _build_popularity_table(collab_model: CollaborativeModel) -> pd.DataFrame:

    return suggestions    merged = collab_model.merged



    stats = (

def init_hybrid_model(min_support: int = 20) -> HybridModel:        merged.groupby("movieId")["rating"]

    collab_model = init_collaborative_model()        .agg([("mean_rating", "mean"), ("rating_count", "count")])

    content_model = init_content_model()        .reset_index()

    popularity = _build_popularity_table(collab_model)    )

    mean_ratings = collab_model.merged.groupby("movieId")["rating"].mean()

    return HybridModel(    titles = (

        collab=collab_model,        merged.drop_duplicates("movieId")["movieId"]

        content=content_model,        .to_frame()

        popularity=popularity,        .join(

        mean_ratings=mean_ratings,            merged.drop_duplicates("movieId")[["movieId", "title", "genres"]]

        min_support=min_support,            .set_index("movieId"),

    )            on="movieId",

        )

        .reset_index(drop=True)

def _hydrate_recommendations(df: pd.DataFrame, source: str) -> pd.DataFrame:    )

    if df.empty:

        return pd.DataFrame(columns=["movieId", source])    popularity = stats.merge(titles, on="movieId", how="left")

    rename_map = {"score": f"{source}_score"}    popularity = popularity.sort_values(

    return df.rename(columns=rename_map)[["movieId", f"{source}_score"]]        ["rating_count", "mean_rating"], ascending=[False, False]

    ).reset_index(drop=True)

    return popularity

def _lookup_titles(hybrid: HybridModel, movie_ids: List[int]) -> pd.DataFrame:

    result = pd.DataFrame({"movieId": movie_ids})

def _normalize_scores(scores: pd.Series) -> pd.Series:

    collab_titles = hybrid.collab.movie_lookup.dropna()    clean = scores.astype(float).fillna(0.0)

    content_meta = hybrid.content.movies.set_index("movieId")    if clean.empty:

        return clean

    result["title"] = result["movieId"].map(collab_titles)

    result["genres"] = result["movieId"].map(content_meta["genres"])    max_val = clean.max()

    min_val = clean.min()

    missing_title_mask = result["title"].isna()

    if missing_title_mask.any():    if np.isclose(max_val, min_val):

        result.loc[missing_title_mask, "title"] = result.loc[missing_title_mask, "movieId"].map(        return (clean > 0).astype(float)

            content_meta["title"]

        )    normalized = (clean - min_val) / (max_val - min_val)

    return normalized.clip(0.0, 1.0)

    result["title"] = result["title"].fillna("Unknown")

    result["genres"] = result["genres"].fillna("")

    result["rating_count"] = (def _parse_genre_string(genres: str | float | None) -> set[str]:

        result["movieId"].map(hybrid.collab.rating_counts).fillna(0).astype(int)    if not isinstance(genres, str):

    )        return set()

    result["mean_rating"] = result["movieId"].map(hybrid.mean_ratings).round(2)    tokens = genres.replace(",", "|").split("|")

    return result    cleaned = {token.strip().lower() for token in tokens if token and token.strip()}

    cleaned.discard("(no genres listed)")

    return cleaned

def _is_cold_start(hybrid: HybridModel, movie_name: str) -> bool:

    movie_id = _find_movie_id(hybrid, movie_name)

    if movie_id is None:def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:

        return True    if not seed_genres:

        return 0.0

    support = int(hybrid.collab.rating_counts.get(movie_id, 0))    candidate = _parse_genre_string(candidate_genres)

    return support < hybrid.min_support    if not candidate:

        return 0.0

    overlap = seed_genres & candidate

def _get_popular_fallback(hybrid: HybridModel, exclude: List[int], top_n: int) -> pd.DataFrame:    if not overlap:

    available = hybrid.popularity[~hybrid.popularity["movieId"].isin(exclude)]        return 0.0

    fallback = available.head(top_n).copy()    union = seed_genres | candidate

    fallback["collab_score"] = 0.0    if not union:

    fallback["content_score"] = 0.0        return 0.0

    fallback["final_score"] = (fallback["mean_rating"].fillna(0.0) / 5.0).round(3)    return len(overlap) / len(union)

    fallback["genre_similarity"] = 0.0

    fallback["reason"] = "popular"

    columns = [def _find_movie_id(hybrid: HybridModel, movie_name: str) -> int | None:

        "reason",    lookup = hybrid.collab.movie_lookup.dropna()

        "movieId",    lower_name = movie_name.lower()

        "title",    exact = lookup[lookup.str.lower() == lower_name]

        "genres",    if not exact.empty:

        "final_score",        return int(exact.index[0])

        "collab_score",    partial = lookup[lookup.str.contains(movie_name, case=False, regex=False)]

        "content_score",    if not partial.empty:

        "genre_similarity",        return int(partial.index[0])

        "rating_count",    return None

        "mean_rating",

    ]

    return fallback[columns]def suggest_titles(hybrid: HybridModel, query: str, limit: int = 5) -> list[dict[str, str | int]]:

    if not query:

        return []

def get_hybrid_recommendations(

    hybrid: HybridModel, movie_name: str, top_n: int = 5    lookup = hybrid.collab.movie_lookup.dropna()

) -> pd.DataFrame:    suggestions: list[dict[str, str | int]] = []

    seed_id = _find_movie_id(hybrid, movie_name)    seen_titles: set[str] = set()

    seed_genres: set[str] = set()

    if seed_id is not None:    partial = lookup[lookup.str.contains(query, case=False, regex=False)]

        seed_row = hybrid.content.movies.loc[hybrid.content.movies["movieId"] == seed_id]    for movie_id, title in partial.head(limit).items():

        if not seed_row.empty:        suggestions.append({"movieId": int(movie_id), "title": title})

            seed_genres = _parse_genre_string(seed_row.iloc[0].get("genres"))        seen_titles.add(title)

        if len(suggestions) >= limit:

    if _is_cold_start(hybrid, movie_name):            return suggestions

        exclude = [seed_id] if seed_id is not None else []

        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)    close_matches = difflib.get_close_matches(query, lookup.tolist(), n=limit, cutoff=0.4)

    for title in close_matches:

    try:        if title in seen_titles:

        collab_df = get_collab_recommendations(hybrid.collab, movie_name, top_n=top_n * 3)            continue

    except ValueError:        match_series = lookup[lookup == title]

        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])        if match_series.empty:

            continue

    try:        suggestions.append({"movieId": int(match_series.index[0]), "title": title})

        content_df = get_content_recommendations(hybrid.content, movie_name, top_n=top_n * 3)        seen_titles.add(title)

    except ValueError:        if len(suggestions) >= limit:

        content_df = pd.DataFrame(columns=["movieId", "score", "genres"])            break



    collab_scores = _hydrate_recommendations(collab_df, "collab")    return suggestions

    content_scores = _hydrate_recommendations(content_df, "content")



    combined = pd.merge(collab_scores, content_scores, on="movieId", how="outer").fillna(0.0)def init_hybrid_model(min_support: int = 20) -> HybridModel:

    collab_model = init_collaborative_model()

    if seed_id is not None:    content_model = init_content_model()

        combined = combined[combined["movieId"] != seed_id]    popularity = _build_popularity_table(collab_model)

    mean_ratings = collab_model.merged.groupby("movieId")["rating"].mean()

    if combined.empty:    return HybridModel(

        exclude = [seed_id] if seed_id is not None else []        collab=collab_model,

        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)        content=content_model,

        popularity=popularity,

    combined["collab_score"] = combined.get("collab_score", 0.0).astype(float)        mean_ratings=mean_ratings,

    combined["content_score"] = combined.get("content_score", 0.0).astype(float)        min_support=min_support,

    )

    collab_norm = _normalize_scores(combined["collab_score"])

    content_norm = _normalize_scores(combined["content_score"])

def _hydrate_recommendations(df: pd.DataFrame, source: str) -> pd.DataFrame:

    combined["collab_score"] = collab_norm    if df.empty:

    combined["content_score"] = content_norm        return pd.DataFrame(columns=["movieId", source])

    rename_map = {"score": f"{source}_score"}

    combined["hybrid_score"] = 0.65 * collab_norm + 0.35 * content_norm    return df.rename(columns=rename_map)[["movieId", f"{source}_score"]]

    overlap_mask = (collab_norm > 0) & (content_norm > 0)

    combined.loc[overlap_mask, "hybrid_score"] += 0.1

    combined["hybrid_score"] = combined["hybrid_score"].clip(0.0, 1.0)def _lookup_titles(hybrid: HybridModel, movie_ids: List[int]) -> pd.DataFrame:

    result = pd.DataFrame({"movieId": movie_ids})

    combined = combined.sort_values("hybrid_score", ascending=False)

    movie_ids = combined["movieId"].astype(int).tolist()    collab_titles = hybrid.collab.movie_lookup.dropna()

    content_meta = hybrid.content.movies.set_index("movieId")

    meta = _lookup_titles(hybrid, movie_ids)

    results = combined.merge(meta, on="movieId", how="left")    result["title"] = result["movieId"].map(collab_titles)

    results["reason"] = "hybrid"    result["genres"] = result["movieId"].map(content_meta["genres"])

    if seed_genres:

        results["genre_similarity"] = results["genres"].apply(    missing_title_mask = result["title"].isna()

            lambda genres: _compute_genre_similarity(seed_genres, genres)    if missing_title_mask.any():

        )        result.loc[missing_title_mask, "title"] = result.loc[missing_title_mask, "movieId"].map(

    else:            content_meta["title"]

        results["genre_similarity"] = 0.0        )



    results["final_score"] = (    result["title"] = result["title"].fillna("Unknown")

        0.55 * results["collab_score"]    result["genres"] = result["genres"].fillna("")

        + 0.3 * results["content_score"]    result["rating_count"] = (

        + 0.15 * results["genre_similarity"]        result["movieId"].map(hybrid.collab.rating_counts).fillna(0).astype(int)

    ).clip(0.0, 1.0)    )

    result["mean_rating"] = result["movieId"].map(hybrid.mean_ratings).round(2)

    results = results[    return result

        [

            "reason",

            "movieId",def _is_cold_start(hybrid: HybridModel, movie_name: str) -> bool:

            "title",    movie_id = _find_movie_id(hybrid, movie_name)

            "genres",    if movie_id is None:

            "final_score",        return True

            "collab_score",

            "content_score",    support = int(hybrid.collab.rating_counts.get(movie_id, 0))

            "genre_similarity",    return support < hybrid.min_support

            "rating_count",

            "mean_rating",

        ]def _get_popular_fallback(hybrid: HybridModel, exclude: List[int], top_n: int) -> pd.DataFrame:

    ]    available = hybrid.popularity[~hybrid.popularity["movieId"].isin(exclude)]

    fallback = available.head(top_n).copy()

    results = results.drop_duplicates("movieId")    fallback["collab_score"] = 0.0

    results = results.sort_values(    fallback["content_score"] = 0.0

        ["final_score", "genre_similarity", "rating_count", "mean_rating"],    fallback["final_score"] = (fallback["mean_rating"].fillna(0.0) / 5.0).round(3)

        ascending=[False, False, False, False],    fallback["genre_similarity"] = 0.0

    ).head(top_n)    fallback["reason"] = "popular"

    columns = [

    if len(results) < top_n:        "reason",

        exclude_ids = results["movieId"].tolist()        "movieId",

        if seed_id is not None:        "title",

            exclude_ids.append(seed_id)        "genres",

        extra = _get_popular_fallback(hybrid, exclude=exclude_ids, top_n=top_n - len(results))        "final_score",

        results = pd.concat([results, extra], ignore_index=True)        "collab_score",

        results = results.sort_values(        "content_score",

            ["final_score", "genre_similarity", "rating_count", "mean_rating"],        "genre_similarity",

            ascending=[False, False, False, False],        "rating_count",

        ).head(top_n)        "mean_rating",

    ]

    results["final_score"] = results["final_score"].round(4)    return fallback[columns]

    results["collab_score"] = results["collab_score"].round(4)

    results["content_score"] = results["content_score"].round(4)

    results["genre_similarity"] = results["genre_similarity"].round(4)def get_hybrid_recommendations(

    return results    hybrid: HybridModel, movie_name: str, top_n: int = 5

) -> pd.DataFrame:

    seed_id = _find_movie_id(hybrid, movie_name)

def _print_recommendations(df: pd.DataFrame) -> None:    seed_genres: set[str] = set()

    if df.empty:    if seed_id is not None:

        print("  (no recommendations)")        seed_row = hybrid.content.movies.loc[hybrid.content.movies["movieId"] == seed_id]

        return        if not seed_row.empty:

            seed_genres = _parse_genre_string(seed_row.iloc[0].get("genres"))

    for _, row in df.iterrows():

        title = row["title"][:42]    if _is_cold_start(hybrid, movie_name):

        genres = row["genres"][:45]        exclude = [seed_id] if seed_id is not None else []

        avg_rating = "n/a" if pd.isna(row["mean_rating"]) else f"{row['mean_rating']:.2f}"        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)

        print(

            f"  {row['reason']:<7} | {title:<42} | {genres:<45} | final={row['final_score']:.4f} "    try:

            f"(collab={row['collab_score']:.4f}, content={row['content_score']:.4f}, "        collab_df = get_collab_recommendations(hybrid.collab, movie_name, top_n=top_n * 3)

            f"genre={row.get('genre_similarity', 0.0):.4f}) | "    except ValueError:

            f"ratings={row['rating_count']:>3} | avg={avg_rating}"        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])

        )

    try:

        content_df = get_content_recommendations(hybrid.content, movie_name, top_n=top_n * 3)

def main() -> None:    except ValueError:

    hybrid_model = init_hybrid_model(min_support=20)        content_df = pd.DataFrame(columns=["movieId", "score", "genres"])

    samples = ["Toy Story (1995)", "Jurassic Park (1993)"]

    for movie in samples:    collab_scores = _hydrate_recommendations(collab_df, "collab")

        print(f"\nHybrid recommendations for '{movie}':")    content_scores = _hydrate_recommendations(content_df, "content")

        recs = get_hybrid_recommendations(hybrid_model, movie, top_n=5)

        _print_recommendations(recs)    combined = pd.merge(collab_scores, content_scores, on="movieId", how="outer").fillna(0.0)



    if seed_id is not None:

if __name__ == "__main__":        combined = combined[combined["movieId"] != seed_id]

    main()

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
        extra = _get_popular_fallback(hybrid, exclude=exclude_ids, top_n=top_n - len(results))
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


pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 60)


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
        merged.drop_duplicates("movieId")["movieId"].to_frame()
        .join(merged.drop_duplicates("movieId")[["movieId", "title", "genres"]].set_index("movieId"), on="movieId")
    ).reset_index(drop=True)

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


def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:
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


def suggest_titles(hybrid: HybridModel, query: str, limit: int = 5) -> list[dict[str, str | int]]:
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


def _get_popular_fallback(hybrid: HybridModel, exclude: List[int], top_n: int) -> pd.DataFrame:
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
        collab_df = get_collab_recommendations(hybrid.collab, movie_name, top_n=top_n * 3)
    except ValueError:
        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])

    try:
        content_df = get_content_recommendations(hybrid.content, movie_name, top_n=top_n * 3)
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
        extra = _get_popular_fallback(hybrid, exclude=exclude_ids, top_n=top_n - len(results))
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
from __future__ import annotationsfrom __future__ import annotations



from dataclasses import dataclassfrom dataclasses import dataclass

from typing import Listfrom typing import List



import difflibimport difflib

import numpy as npimport numpy as np

import pandas as pdimport pandas as pd



from collaborative_filtering import (from collaborative_filtering import (

    CollaborativeModel,    CollaborativeModel,

    get_collab_recommendations,    get_collab_recommendations,

    init_collaborative_model,    init_collaborative_model,

))

from content_based import (from content_based import (

    ContentModel,    ContentModel,

    get_content_recommendations,    get_content_recommendations,

    init_content_model,    init_content_model,

))





pd.set_option("display.width", 140)pd.set_option("display.width", 140)

pd.set_option("display.max_colwidth", 60)pd.set_option("display.max_colwidth", 60)





@dataclass@dataclass

class HybridModel:class HybridModel:

    collab: CollaborativeModel    collab: CollaborativeModel

    content: ContentModel    content: ContentModel

    popularity: pd.DataFrame    popularity: pd.DataFrame

    mean_ratings: pd.Series    mean_ratings: pd.Series

    min_support: int = 20    min_support: int = 20





def _build_popularity_table(collab_model: CollaborativeModel) -> pd.DataFrame:def _build_popularity_table(collab_model: CollaborativeModel) -> pd.DataFrame:

    merged = collab_model.merged    merged = collab_model.merged



    stats = (    stats = (

        merged.groupby("movieId")["rating"]        merged.groupby("movieId")["rating"]

        .agg([("mean_rating", "mean"), ("rating_count", "count")])        .agg([("mean_rating", "mean"), ("rating_count", "count")])

        .reset_index()        .reset_index()

    )    )



    titles = (    titles = (

        merged.drop_duplicates("movieId")[["movieId", "title", "genres"]]        merged.drop_duplicates("movieId")[["movieId", "title", "genres"]]

        .set_index("movieId")        .set_index("movieId")

        .reset_index()    )

    )            "rating_count",



    popularity = stats.merge(titles, on="movieId", how="left")    popularity = stats.merge(titles, on="movieId", how="left")

    popularity = popularity.sort_values(    popularity = popularity.sort_values(

        ["rating_count", "mean_rating"], ascending=[False, False]        ["rating_count", "mean_rating"], ascending=[False, False]

    ).reset_index(drop=True)    ).reset_index(drop=True)

    return popularity    return popularity

        ["final_score", "genre_similarity", "rating_count", "mean_rating"],

        ascending=[False, False, False, False],

def _normalize_scores(scores: pd.Series) -> pd.Series:

    clean = scores.astype(float).fillna(0.0)def _normalize_scores(scores: pd.Series) -> pd.Series:

    if clean.empty:    clean = scores.astype(float).fillna(0.0)

        return clean    if clean.empty:

        return clean

    max_val = clean.max()

    min_val = clean.min()    max_val = clean.max()

    min_val = clean.min()

    if np.isclose(max_val, min_val):

        return (clean > 0).astype(float)        results = pd.concat([results, extra], ignore_index=True)



    normalized = (clean - min_val) / (max_val - min_val)    normalized = (clean - min_val) / (max_val - min_val)

    return normalized.clip(0.0, 1.0)    return normalized.clip(0.0, 1.0)





def _parse_genre_string(genres: str | float | None) -> set[str]:    results["final_score"] = results["final_score"].round(4)

    if not isinstance(genres, str):    results["collab_score"] = results["collab_score"].round(4)

        return set()    results["content_score"] = results["content_score"].round(4)

    tokens = genres.replace(",", "|").split("|")    results["genre_similarity"] = results["genre_similarity"].round(4)

    cleaned = {token.strip().lower() for token in tokens if token and token.strip()}    return results

    cleaned.discard("(no genres listed)")    cleaned = {token.strip().lower() for token in tokens if token and token.strip()}

    return cleaned    cleaned.discard("(no genres listed)")

    return cleaned



def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:

    if not seed_genres:def _compute_genre_similarity(seed_genres: set[str], candidate_genres: str | float | None) -> float:

        return 0.0    if not seed_genres:

    candidate = _parse_genre_string(candidate_genres)        return 0.0

    if not candidate:    candidate = _parse_genre_string(candidate_genres)

        return 0.0    if not candidate:

    overlap = seed_genres & candidate        return 0.0

    if not overlap:    overlap = seed_genres & candidate

        return 0.0    if not overlap:

    union = seed_genres | candidate        return 0.0

    if not union:    union = seed_genres | candidate

        return 0.0    if not union:

    return len(overlap) / len(union)        return 0.0

    return len(overlap) / len(union)



def _find_movie_id(hybrid: HybridModel, movie_name: str) -> int | None:

    lookup = hybrid.collab.movie_lookup.dropna()def _find_movie_id(hybrid: HybridModel, movie_name: str) -> int | None:

    lower_name = movie_name.lower()    lookup = hybrid.collab.movie_lookup.dropna()

    exact = lookup[lookup.str.lower() == lower_name]    lower_name = movie_name.lower()

    if not exact.empty:    exact = lookup[lookup.str.lower() == lower_name]

        return int(exact.index[0])    if not exact.empty:

    partial = lookup[lookup.str.contains(movie_name, case=False, regex=False)]        return int(exact.index[0])

    if not partial.empty:    partial = lookup[lookup.str.contains(movie_name, case=False, regex=False)]

        return int(partial.index[0])    if not partial.empty:

    return None        return int(partial.index[0])

    return None



def suggest_titles(hybrid: HybridModel, query: str, limit: int = 5) -> list[dict[str, str | int]]:

    if not query:def suggest_titles(hybrid: HybridModel, query: str, limit: int = 5) -> list[dict[str, str | int]]:

        return []    if not query:

        return []

    lookup = hybrid.collab.movie_lookup.dropna()

    suggestions: list[dict[str, str | int]] = []    lookup = hybrid.collab.movie_lookup.dropna()

    seen_titles: set[str] = set()    suggestions: list[dict[str, str | int]] = []

    seen_titles: set[str] = set()

    partial = lookup[lookup.str.contains(query, case=False, regex=False)]

    for movie_id, title in partial.head(limit).items():    partial = lookup[lookup.str.contains(query, case=False, regex=False)]

        suggestions.append({"movieId": int(movie_id), "title": title})    for movie_id, title in partial.head(limit).items():

        seen_titles.add(title)        suggestions.append({"movieId": int(movie_id), "title": title})

        if len(suggestions) >= limit:        seen_titles.add(title)

            return suggestions        if len(suggestions) >= limit:

            return suggestions

    close_matches = difflib.get_close_matches(query, lookup.tolist(), n=limit, cutoff=0.4)

    for title in close_matches:    close_matches = difflib.get_close_matches(query, lookup.tolist(), n=limit, cutoff=0.4)

        if title in seen_titles:    for title in close_matches:

            continue        if title in seen_titles:

        match_series = lookup[lookup == title]            continue

        if match_series.empty:        match_series = lookup[lookup == title]

            continue        if match_series.empty:

        suggestions.append({"movieId": int(match_series.index[0]), "title": title})            continue

        seen_titles.add(title)        suggestions.append({"movieId": int(match_series.index[0]), "title": title})

        if len(suggestions) >= limit:        seen_titles.add(title)

            break        if len(suggestions) >= limit:

            break

    return suggestions

    return suggestions



def init_hybrid_model(min_support: int = 20) -> HybridModel:

    collab_model = init_collaborative_model()def init_hybrid_model(min_support: int = 20) -> HybridModel:

    content_model = init_content_model()    collab_model = init_collaborative_model()

    popularity = _build_popularity_table(collab_model)    content_model = init_content_model()

    mean_ratings = collab_model.merged.groupby("movieId")["rating"].mean()    popularity = _build_popularity_table(collab_model)

    return HybridModel(    mean_ratings = collab_model.merged.groupby("movieId")["rating"].mean()

        collab=collab_model,    return HybridModel(

        content=content_model,        collab=collab_model,

        popularity=popularity,        content=content_model,

        mean_ratings=mean_ratings,        popularity=popularity,

        min_support=min_support,        mean_ratings=mean_ratings,

    )        min_support=min_support,

    )



def _hydrate_recommendations(df: pd.DataFrame, source: str) -> pd.DataFrame:

    if df.empty:def _hydrate_recommendations(df: pd.DataFrame, source: str) -> pd.DataFrame:

        return pd.DataFrame(columns=["movieId", source])    if df.empty:

    rename_map = {"score": f"{source}_score"}        return pd.DataFrame(columns=["movieId", source])

    return df.rename(columns=rename_map)[["movieId", f"{source}_score"]]    rename_map = {"score": f"{source}_score"}

    return df.rename(columns=rename_map)[["movieId", f"{source}_score"]]



def _lookup_titles(hybrid: HybridModel, movie_ids: List[int]) -> pd.DataFrame:

    result = pd.DataFrame({"movieId": movie_ids})def _lookup_titles(hybrid: HybridModel, movie_ids: List[int]) -> pd.DataFrame:

    result = pd.DataFrame({"movieId": movie_ids})

    collab_titles = hybrid.collab.movie_lookup.dropna()

    content_meta = hybrid.content.movies.set_index("movieId")    collab_titles = hybrid.collab.movie_lookup.dropna()

    content_meta = hybrid.content.movies.set_index("movieId")

    result["title"] = result["movieId"].map(collab_titles)

    result["genres"] = result["movieId"].map(content_meta["genres"])    result["title"] = result["movieId"].map(collab_titles)

    result["genres"] = result["movieId"].map(content_meta["genres"])

    missing_title_mask = result["title"].isna()

    if missing_title_mask.any():    missing_title_mask = result["title"].isna()

        result.loc[missing_title_mask, "title"] = result.loc[missing_title_mask, "movieId"].map(    if missing_title_mask.any():

            content_meta["title"]        result.loc[missing_title_mask, "title"] = result.loc[missing_title_mask, "movieId"].map(

        )            content_meta["title"]

        )

    result["title"] = result["title"].fillna("Unknown")

    result["genres"] = result["genres"].fillna("")    result["title"] = result["title"].fillna("Unknown")

    result["rating_count"] = (    result["genres"] = result["genres"].fillna("")

        result["movieId"].map(hybrid.collab.rating_counts).fillna(0).astype(int)    result["rating_count"] = (

    )        result["movieId"].map(hybrid.collab.rating_counts).fillna(0).astype(int)

    result["mean_rating"] = result["movieId"].map(hybrid.mean_ratings).round(2)    )

    return result    result["mean_rating"] = result["movieId"].map(hybrid.mean_ratings).round(2)

    return result



def _is_cold_start(hybrid: HybridModel, movie_name: str) -> bool:

    movie_id = _find_movie_id(hybrid, movie_name)def _is_cold_start(hybrid: HybridModel, movie_name: str) -> bool:

    if movie_id is None:    movie_id = _find_movie_id(hybrid, movie_name)

        return True    if movie_id is None:

        return True

    support = int(hybrid.collab.rating_counts.get(movie_id, 0))

    return support < hybrid.min_support    support = int(hybrid.collab.rating_counts.get(movie_id, 0))

    return support < hybrid.min_support



def _get_popular_fallback(hybrid: HybridModel, exclude: List[int], top_n: int) -> pd.DataFrame:

    available = hybrid.popularity[~hybrid.popularity["movieId"].isin(exclude)]def _get_popular_fallback(hybrid: HybridModel, exclude: List[int], top_n: int) -> pd.DataFrame:

    fallback = available.head(top_n).copy()    available = hybrid.popularity[~hybrid.popularity["movieId"].isin(exclude)]

    fallback["collab_score"] = 0.0    fallback = available.head(top_n).copy()

    fallback["content_score"] = 0.0    fallback["collab_score"] = 0.0

    fallback["final_score"] = (fallback["mean_rating"].fillna(0.0) / 5.0).round(3)    fallback["content_score"] = 0.0

    fallback["genre_similarity"] = 0.0    fallback["final_score"] = (fallback["mean_rating"].fillna(0.0) / 5.0).round(3)

    fallback["reason"] = "popular"    fallback["genre_similarity"] = 0.0

    columns = [    fallback["reason"] = "popular"

        "reason",    columns = [

        "movieId",        "reason",

        "title",        "movieId",

        "genres",        "title",

        "final_score",        "genres",

        "collab_score",        "final_score",

        "content_score",        "collab_score",

        "genre_similarity",        "content_score",

        "rating_count",        "genre_similarity",

        "mean_rating",        "rating_count",

    ]        "mean_rating",

    return fallback[columns]    ]

    return fallback[columns]



def get_hybrid_recommendations(

    hybrid: HybridModel, movie_name: str, top_n: int = 5def get_hybrid_recommendations(

) -> pd.DataFrame:    hybrid: HybridModel, movie_name: str, top_n: int = 5

    seed_id = _find_movie_id(hybrid, movie_name)) -> pd.DataFrame:

    seed_genres: set[str] = set()    seed_id = _find_movie_id(hybrid, movie_name)

    if seed_id is not None:    seed_genres: set[str] = set()

        seed_row = hybrid.content.movies.loc[hybrid.content.movies["movieId"] == seed_id]    if seed_id is not None:

        if not seed_row.empty:        seed_row = hybrid.content.movies.loc[hybrid.content.movies["movieId"] == seed_id]

            seed_genres = _parse_genre_string(seed_row.iloc[0].get("genres"))        if not seed_row.empty:

            seed_genres = _parse_genre_string(seed_row.iloc[0].get("genres"))

    if _is_cold_start(hybrid, movie_name):

        exclude = [seed_id] if seed_id is not None else []    if _is_cold_start(hybrid, movie_name):

        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)        exclude = [seed_id] if seed_id is not None else []

        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)

    try:

        collab_df = get_collab_recommendations(hybrid.collab, movie_name, top_n=top_n * 3)    try:

    except ValueError:        collab_df = get_collab_recommendations(hybrid.collab, movie_name, top_n=top_n * 3)

        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])    except ValueError:

        collab_df = pd.DataFrame(columns=["movieId", "score", "ratings"])

    try:

        content_df = get_content_recommendations(hybrid.content, movie_name, top_n=top_n * 3)    try:

    except ValueError:        content_df = get_content_recommendations(hybrid.content, movie_name, top_n=top_n * 3)

        content_df = pd.DataFrame(columns=["movieId", "score", "genres"])    except ValueError:

        content_df = pd.DataFrame(columns=["movieId", "score", "genres"])

    collab_scores = _hydrate_recommendations(collab_df, "collab")

    content_scores = _hydrate_recommendations(content_df, "content")    collab_scores = _hydrate_recommendations(collab_df, "collab")

    content_scores = _hydrate_recommendations(content_df, "content")

    combined = pd.merge(collab_scores, content_scores, on="movieId", how="outer").fillna(0.0)

    combined = pd.merge(collab_scores, content_scores, on="movieId", how="outer").fillna(0.0)

    if seed_id is not None:

        combined = combined[combined["movieId"] != seed_id]    if seed_id is not None:

        combined = combined[combined["movieId"] != seed_id]

    if combined.empty:

        exclude = [seed_id] if seed_id is not None else []    if combined.empty:

        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)        exclude = [seed_id] if seed_id is not None else []

        return _get_popular_fallback(hybrid, exclude=exclude, top_n=top_n)

    combined["collab_score"] = combined.get("collab_score", 0.0).astype(float)

    combined["content_score"] = combined.get("content_score", 0.0).astype(float)    combined["collab_score"] = combined.get("collab_score", 0.0).astype(float)

    combined["content_score"] = combined.get("content_score", 0.0).astype(float)

    collab_norm = _normalize_scores(combined["collab_score"])

    content_norm = _normalize_scores(combined["content_score"])    collab_norm = _normalize_scores(combined["collab_score"])

    content_norm = _normalize_scores(combined["content_score"])

    combined["collab_score"] = collab_norm

    combined["content_score"] = content_norm    combined["collab_score"] = collab_norm

    combined["content_score"] = content_norm

    combined["hybrid_score"] = 0.65 * collab_norm + 0.35 * content_norm

    overlap_mask = (collab_norm > 0) & (content_norm > 0)    combined["hybrid_score"] = 0.65 * collab_norm + 0.35 * content_norm

    combined.loc[overlap_mask, "hybrid_score"] += 0.1    overlap_mask = (collab_norm > 0) & (content_norm > 0)

    combined["hybrid_score"] = combined["hybrid_score"].clip(0.0, 1.0)    combined.loc[overlap_mask, "hybrid_score"] += 0.1

    combined["hybrid_score"] = combined["hybrid_score"].clip(0.0, 1.0)

    combined = combined.sort_values("hybrid_score", ascending=False)

    movie_ids = combined["movieId"].astype(int).tolist()    combined = combined.sort_values("hybrid_score", ascending=False)

    movie_ids = combined["movieId"].astype(int).tolist()

    meta = _lookup_titles(hybrid, movie_ids)

    results = combined.merge(meta, on="movieId", how="left")    meta = _lookup_titles(hybrid, movie_ids)

    results["reason"] = "hybrid"    results = combined.merge(meta, on="movieId", how="left")

    if seed_genres:    results["reason"] = "hybrid"

        results["genre_similarity"] = results["genres"].apply(    if seed_genres:

            lambda genres: _compute_genre_similarity(seed_genres, genres)        results["genre_similarity"] = results["genres"].apply(

        )            lambda genres: _compute_genre_similarity(seed_genres, genres)

    else:        )

        results["genre_similarity"] = 0.0    else:

        results["genre_similarity"] = 0.0

    results["final_score"] = (

        0.55 * results["collab_score"]    results["final_score"] = (

        + 0.3 * results["content_score"]        0.55 * results["collab_score"]

        + 0.15 * results["genre_similarity"]        + 0.3 * results["content_score"]

    ).clip(0.0, 1.0)        + 0.15 * results["genre_similarity"]

    ).clip(0.0, 1.0)

    results = results[    results = results[

        [        [

            "reason",            "reason",

            "movieId",            "movieId",

            "title",            "title",

            "genres",            "genres",

            "final_score",            "final_score",

            "collab_score",            "collab_score",

            "content_score",            "content_score",

            "genre_similarity",            "genre_similarity",

            "rating_count",            "rating_count",

            "mean_rating",            "mean_rating",

        ]        ]

    ]    ]



    results = results.drop_duplicates("movieId")    results = results.drop_duplicates("movieId")

    results = results.sort_values(    results = results.sort_values(

        ["final_score", "genre_similarity", "rating_count", "mean_rating"],        ["final_score", "genre_similarity", "rating_count", "mean_rating"],

        ascending=[False, False, False, False],        ascending=[False, False, False, False],

    ).head(top_n)    ).head(top_n)



    if len(results) < top_n:    if len(results) < top_n:

        exclude_ids = results["movieId"].tolist()        exclude_ids = results["movieId"].tolist()

        if seed_id is not None:        if seed_id is not None:

            exclude_ids.append(seed_id)            exclude_ids.append(seed_id)

        extra = _get_popular_fallback(hybrid, exclude=exclude_ids, top_n=top_n - len(results))        extra = _get_popular_fallback(

        results = pd.concat([results, extra], ignore_index=True)            hybrid, exclude=exclude_ids, top_n=top_n - len(results)

        results = results.sort_values(        )

            ["final_score", "genre_similarity", "rating_count", "mean_rating"],        results = pd.concat([results, extra], ignore_index=True)

            ascending=[False, False, False, False],        results = results.sort_values(

        ).head(top_n)            ["final_score", "genre_similarity", "rating_count", "mean_rating"],

            ascending=[False, False, False, False],

    results["final_score"] = results["final_score"].round(4)        ).head(top_n)

    results["collab_score"] = results["collab_score"].round(4)

    results["content_score"] = results["content_score"].round(4)    results["final_score"] = results["final_score"].round(4)

    results["genre_similarity"] = results["genre_similarity"].round(4)    results["collab_score"] = results["collab_score"].round(4)

    return results    results["content_score"] = results["content_score"].round(4)

    results["genre_similarity"] = results["genre_similarity"].round(4)

    return results

def _print_recommendations(df: pd.DataFrame) -> None:

    if df.empty:

        print("  (no recommendations)")def _print_recommendations(df: pd.DataFrame) -> None:

        return    if df.empty:

        print("  (no recommendations)")

    for _, row in df.iterrows():        return

        title = row["title"][:42]

        genres = row["genres"][:45]    for _, row in df.iterrows():

        avg_rating = "n/a" if pd.isna(row["mean_rating"]) else f"{row['mean_rating']:.2f}"        title = row["title"][:42]

        print(        genres = row["genres"][:45]

            f"  {row['reason']:<7} | {title:<42} | {genres:<45} | final={row['final_score']:.4f} "        avg_rating = "n/a" if pd.isna(row["mean_rating"]) else f"{row['mean_rating']:.2f}"

            f"(collab={row['collab_score']:.4f}, content={row['content_score']:.4f}, "        print(

            f"genre={row.get('genre_similarity', 0.0):.4f}) | "            f"  {row['reason']:<7} | {title:<42} | {genres:<45} | final={row['final_score']:.4f} "

            f"ratings={row['rating_count']:>3} | avg={avg_rating}"            f"(collab={row['collab_score']:.4f}, content={row['content_score']:.4f}) | "

        )            f"ratings={row['rating_count']:>3} | avg={avg_rating}"

        )



def main() -> None:

    hybrid_model = init_hybrid_model(min_support=20)def main() -> None:

    samples = ["Toy Story (1995)", "Jurassic Park (1993)"]    hybrid_model = init_hybrid_model(min_support=20)

    for movie in samples:    samples = ["Toy Story (1995)", "Jurassic Park (1993)"]

        print(f"\nHybrid recommendations for '{movie}':")    for movie in samples:

        recs = get_hybrid_recommendations(hybrid_model, movie, top_n=5)        print(f"\nHybrid recommendations for '{movie}':")

        _print_recommendations(recs)        recs = get_hybrid_recommendations(hybrid_model, movie, top_n=5)

        _print_recommendations(recs)



if __name__ == "__main__":

    main()if __name__ == "__main__":

    main()
