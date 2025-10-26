from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from textwrap import shorten
from typing import Optional

import numpy as np
import pandas as pd

from content_based import init_content_model, score_external_movie
from hybrid import (
    _find_movie_id,
    get_hybrid_recommendations,
    init_hybrid_model,
    suggest_titles,
)
from metadata_enrichment import OmdbClient, enrich_recommendations
from advanced_enhancements import build_svd_model
from sklearn.metrics.pairwise import cosine_similarity
from data_preparation import PROCESSED_DATA_DIR

SVD_FACTORS_FILE = PROCESSED_DATA_DIR / "svd_item_factors.pkl"
SVD_META_FILE = PROCESSED_DATA_DIR / "svd_item_factors_meta.json"


@dataclass
class SessionConfig:
    top_n: int
    min_support: int
    enrich: bool
    svd_weight: float
    svd_factors: int
    svd_min_support: int


def parse_args() -> SessionConfig:
    parser = argparse.ArgumentParser(description="Interactive Hybrid Movie Recommender")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to display")
    parser.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="Minimum rating count required before using collaborative scores",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Fetch posters, plots, and cast details from OMDb (requires API key)",
    )
    parser.add_argument(
        "--svd-weight",
        type=float,
        default=0.0,
        help="Optional weight (0-1) to blend TruncatedSVD latent similarity into scores",
    )
    parser.add_argument(
        "--svd-factors",
        type=int,
        default=40,
        help="Number of latent factors to learn if SVD blending is enabled",
    )
    parser.add_argument(
        "--svd-min-support",
        type=int,
        default=10,
        help="Minimum rating count per movie before it is included in SVD training",
    )
    args = parser.parse_args()
    return SessionConfig(
        top_n=max(1, args.top),
        min_support=max(1, args.min_support),
        enrich=args.enrich,
        svd_weight=max(0.0, min(1.0, args.svd_weight)),
        svd_factors=max(2, args.svd_factors),
        svd_min_support=max(1, args.svd_min_support),
    )


def extract_year(title: str) -> Optional[int]:
    if not isinstance(title, str):
        return None
    if title.endswith(")") and "(" in title:
        segment = title[title.rfind("(") + 1 : title.rfind(")")]
        if segment.isdigit():
            return int(segment)
    return None


def apply_filters(
    df: pd.DataFrame,
    genre_keyword: Optional[str],
    min_year: Optional[int],
    min_mean_rating: Optional[float],
) -> pd.DataFrame:
    filtered = df.copy()

    if genre_keyword:
        mask = filtered["genres"].str.contains(genre_keyword, case=False, na=False)
        filtered = filtered[mask]

    if min_year:
        filtered["_year"] = filtered["title"].apply(extract_year)
        filtered = filtered[filtered["_year"].fillna(0) >= min_year]
        filtered = filtered.drop(columns=["_year"], errors="ignore")

    if min_mean_rating:
        filtered = filtered[filtered["mean_rating"].fillna(0) >= min_mean_rating]

    return filtered


def prompt_float(prompt: str) -> Optional[float]:
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        print("  Could not parse a number. Ignoring filter.")
        return None
    return value


def prompt_int(prompt: str) -> Optional[int]:
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print("  Could not parse an integer. Ignoring value.")
        return None
    return value


def render_rows(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no recommendations matched the current filters)")
        return

    for _, row in df.iterrows():
        title = shorten(str(row.get("title", "")), width=46, placeholder="...")
        genres = shorten(str(row.get("genres", "")), width=40, placeholder="...")
        final_score = float(row.get("final_score", 0.0) or 0.0)
        collab_score = float(row.get("collab_score", 0.0) or 0.0)
        content_score = float(row.get("content_score", 0.0) or 0.0)
        genre_score = float(row.get("genre_similarity", 0.0) or 0.0)
        svd_score_val = row.get("svd_score")
        svd_text = ""
        if svd_score_val is not None and pd.notna(svd_score_val):
            svd_text = f", svd={float(svd_score_val):.4f}"
        rating_count_val = row.get("rating_count")
        rating_count = int(rating_count_val) if pd.notna(rating_count_val) else "n/a"
        mean_rating_val = row.get("mean_rating")
        mean_rating = f"{float(mean_rating_val):.2f}" if pd.notna(mean_rating_val) else "n/a"
        imdb = row.get("imdb_rating", "") or "n/a"
        poster = row.get("poster", "")
        plot = row.get("plot", "")
        actors = row.get("actors", "")

        print(
            f"  {title:<46} | genres={genres:<40} | final={final_score:.4f} "
            f"(collab={collab_score:.4f}, content={content_score:.4f}, genre={genre_score:.4f}{svd_text}) | "
            f"ratings={rating_count} | avg={mean_rating}"
        )
        if imdb != "n/a" or poster:
            print(f"      imdb={imdb:<8} poster={shorten(poster, width=70, placeholder='...')}")
        if plot:
            print(f"      plot: {shorten(plot, width=90, placeholder='...')}")
        if actors:
            print(f"      cast: {shorten(actors, width=70, placeholder='...')}")


def collect_feedback(df: pd.DataFrame) -> None:
    if df.empty:
        return
    response = input("  Rate any of the movies above (format movieId:rating, comma separated) or press Enter to skip: ").strip()
    if not response:
        return
    pairs = [segment.strip() for segment in response.split(",") if segment.strip()]
    acknowledged = []
    for pair in pairs:
        if ":" not in pair:
            continue
        movie_id_str, rating_str = pair.split(":", maxsplit=1)
        try:
            movie_id = int(movie_id_str)
            rating = float(rating_str)
        except ValueError:
            continue
        acknowledged.append((movie_id, rating))
    if acknowledged:
        print("  Thanks for the feedback:")
        for movie_id, rating in acknowledged:
            title = df.loc[df["movieId"] == movie_id, "title"].head(1).tolist()
            title_display = title[0] if title else "unknown"
            print(f"    - {title_display} -> {rating:.1f}")
    else:
        print("  Did not recognize any ratings input. Skipping.")


def blend_with_svd(
    df: pd.DataFrame,
    item_factors: Optional[pd.DataFrame],
    seed_id: Optional[int],
    weight: float,
) -> pd.DataFrame:
    if weight <= 0 or item_factors is None or seed_id is None:
        return df

    if seed_id not in item_factors.index:
        print("  SVD blend skipped: seed movie missing from latent factors.")
        return df

    working = df.copy()
    candidate_ids = [int(mid) for mid in working["movieId"].tolist() if mid in item_factors.index]
    if not candidate_ids:
        print("  SVD blend skipped: no candidates with latent factors after filters.")
        return df

    seed_vector = item_factors.loc[seed_id].values.reshape(1, -1)
    candidate_vectors = item_factors.loc[candidate_ids]
    similarities = cosine_similarity(seed_vector, candidate_vectors).flatten()
    similarities = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)

    svd_map = {mid: float(score) for mid, score in zip(candidate_ids, similarities)}
    working["svd_score"] = working["movieId"].map(svd_map).fillna(0.0)
    working["final_score"] = (1 - weight) * working["final_score"].astype(float) + weight * working["svd_score"]
    return working.sort_values("final_score", ascending=False)


def main() -> None:
    config = parse_args()
    hybrid_model = init_hybrid_model(min_support=config.min_support)
    client = OmdbClient()
    content_model = init_content_model()
    svd_item_factors: Optional[pd.DataFrame] = None

    def load_cached_item_factors(n_factors: int, min_support: int) -> Optional[pd.DataFrame]:
        if not SVD_FACTORS_FILE.exists():
            return None
        if not SVD_META_FILE.exists():
            return pd.read_pickle(SVD_FACTORS_FILE)

        try:
            meta = json.loads(SVD_META_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return pd.read_pickle(SVD_FACTORS_FILE)

        if meta.get("n_factors") == n_factors and meta.get("min_support") == min_support:
            return pd.read_pickle(SVD_FACTORS_FILE)
        return None

    def ensure_svd_item_factors(n_factors: int, min_support: int) -> Optional[pd.DataFrame]:
        nonlocal svd_item_factors
        if svd_item_factors is not None:
            return svd_item_factors

        svd_item_factors = load_cached_item_factors(n_factors, min_support)
        if svd_item_factors is not None:
            print(
                f"  Loaded precomputed SVD factors ({svd_item_factors.shape[1]} dims, min_support={min_support})."
            )
            return svd_item_factors

        print("  Training TruncatedSVD latent factors (one-time step)...")
        _, _, svd_item_factors = build_svd_model(
            n_factors=n_factors,
            min_support=min_support,
            verbosity=False,
        )

        try:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            svd_item_factors.to_pickle(SVD_FACTORS_FILE)
            metadata = {
                "n_factors": n_factors,
                "min_support": min_support,
                "rows": svd_item_factors.shape[0],
                "cols": svd_item_factors.shape[1],
            }
            SVD_META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"  Cached SVD factors at {SVD_FACTORS_FILE}.")
        except OSError:
            pass

        return svd_item_factors

    def handle_external_choice(
        option: dict,
        genre_filter_val: Optional[str],
        min_year_val: Optional[int],
        min_rating_val: Optional[float],
    ) -> None:
        if not client.is_enabled():
            print("  External lookups require an OMDb API key. Skipping.")
            return

        title = option.get("Title", "")
        year = option.get("Year")
        detail = client.fetch(title, year)

        print(f"\nTop matches for new release '{detail.get('Title', title)}' ({detail.get('Year', year)})")
        poster = detail.get("Poster", "")
        if poster:
            print(f"  Poster: {poster}")
        if detail.get("Plot"):
            print(f"  Plot: {shorten(detail['Plot'], width=100, placeholder='...')}")
        if detail.get("Actors"):
            print(f"  Cast: {shorten(detail['Actors'], width=80, placeholder='...')}")

        try:
            recs = score_external_movie(
                content_model,
                title=detail.get("Title", title),
                genres=detail.get("Genre", ""),
                plot=detail.get("Plot", ""),
                tags=detail.get("Actors", ""),
                top_n=config.top_n * 2,
            )
        except ValueError:
            print("  Not enough metadata to score this title against the catalog.")
            return

        if recs.empty:
            print("  Unable to find similar catalog titles yet.")
            return

        recs = recs.rename(columns={"score": "content_score"})
        recs["final_score"] = recs["content_score"]
        recs["collab_score"] = 0.0
        recs["rating_count"] = (
            recs["movieId"].map(hybrid_model.collab.rating_counts).fillna(0).astype(int)
        )
        recs["mean_rating"] = recs["movieId"].map(hybrid_model.mean_ratings)

        recs = apply_filters(recs, genre_filter_val or None, min_year_val, min_rating_val)
        recs = recs.head(config.top_n)

        if recs.empty:
            print("  No catalog matches met your filter criteria yet. Try relaxing them.")
            return

        render_rows(recs)
        collect_feedback(recs)

    if config.enrich and not client.is_enabled():
        print("OMDb enrichment requested but no API key detected. Set OMDB_API_KEY to enable.")

    print("\nWelcome to the Hybrid Movie Recommendation CLI.")
    print("Type a movie title you like. Enter 'q' to quit.")

    while True:
        seed = input("\nEnter a movie title (q to quit): ").strip()
        if not seed:
            print("  Please enter a title or 'q' to exit.")
            continue
        if seed.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        top_n_override = prompt_int("  How many recommendations do you want? (press Enter for default): ")
        top_n = top_n_override or config.top_n

        genre_filter = input("  Filter by genre keyword? (optional): ").strip()
        min_year = prompt_int("  Minimum release year? (optional): ")
        min_avg_rating = prompt_float("  Minimum average rating (0-5)? (optional): ")

        seed_id = _find_movie_id(hybrid_model, seed)
        if seed_id is None:
            print("  Could not locate that exact title; falling back to popularity/similarity defaults.")
            suggestions = []
            catalog_suggestions = suggest_titles(hybrid_model, seed, limit=5)
            if catalog_suggestions:
                print("  Did you mean one of these?")
                for idx, item in enumerate(catalog_suggestions, start=1):
                    print(f"    {idx}. {item['title']}")
                    suggestions.append(("catalog", item))

            external_suggestions = []
            if client.is_enabled():
                external_suggestions = client.search(seed, limit=5, year_from=2018)
                if external_suggestions:
                    print("  Newer releases that match your search:")
                    offset = len(suggestions)
                    for idx, item in enumerate(external_suggestions, start=1):
                        title = item.get("Title", "?")
                        year = item.get("Year", "?")
                        print(f"    {offset + idx}. {title} ({year})")
                        suggestions.append(("external", item))

            if suggestions:
                choice = input("  Select a suggestion number or press Enter to continue with fallback: ").strip()
                if choice.isdigit():
                    selected = int(choice) - 1
                    if 0 <= selected < len(suggestions):
                        kind, payload = suggestions[selected]
                        if kind == "catalog":
                            seed = payload["title"]
                            seed_id = payload["movieId"]
                        else:
                            handle_external_choice(payload, genre_filter, min_year, min_avg_rating)
                            continue

        try:
            base = get_hybrid_recommendations(hybrid_model, seed, top_n=top_n * 3)
        except ValueError as exc:
            print(f"  Could not generate recommendations: {exc}")
            continue

        base = apply_filters(base, genre_filter or None, min_year, min_avg_rating)
        if base.empty:
            print("  No recommendations after applying filters. Try relaxing them.")
            continue

        weight_override = prompt_float("  SVD blend weight (0-1)? (press Enter to keep default): ")
        svd_weight = config.svd_weight if weight_override is None else max(0.0, min(1.0, weight_override))

        if svd_weight > 0:
            item_factors = ensure_svd_item_factors(config.svd_factors, config.svd_min_support)
            base = blend_with_svd(base, item_factors, seed_id, svd_weight)

        base = base.head(top_n)

        if config.enrich and client.is_enabled():
            base = enrich_recommendations(base, client)

        render_rows(base)
        collect_feedback(base)


if __name__ == "__main__":
    main()
