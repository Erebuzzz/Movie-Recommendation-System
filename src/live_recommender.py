from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tmdb_client import TMDbClient


@dataclass
class LiveRecommendation:
    movie_id: int
    title: str
    release_year: Optional[int]
    genres: str
    final_score: float
    tmdb_score: float
    content_score: float
    popularity_score: float
    vote_average: float
    vote_count: int
    popularity: float
    poster: Optional[str]
    plot: Optional[str]


class LiveRecommender:
    """Recommendation helper backed by live TMDb data instead of MovieLens."""

    def __init__(self, client: TMDbClient, catalog_pages: int = 5) -> None:
        self.client = client
        self.catalog_pages = catalog_pages
        self._catalog: Optional[pd.DataFrame] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._genre_lookup: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prime_catalog(self) -> None:
        """Trigger catalog download/vectorisation ahead of serving requests."""

        self._ensure_catalog()

    def suggest_titles(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        if not query:
            return []
        results = self.client.search_movie(query, page=1)
        suggestions: List[Dict[str, str]] = []
        for item in results[:limit]:
            title = item.get("title") or item.get("name")
            if not title:
                continue
            release_date = item.get("release_date") or ""
            year = release_date.split("-")[0] if release_date else ""
            suggestions.append(
                {
                    "id": int(item.get("id")),
                    "title": title,
                    "year": year,
                }
            )
        return suggestions

    def get_recommendations(self, movie_name: str, top_n: int = 10) -> pd.DataFrame:
        self._ensure_catalog()

        search_results = self.client.search_movie(movie_name)
        if not search_results:
            raise ValueError(f"TMDb did not return results for '{movie_name}'.")

        seed_raw = search_results[0]
        seed_id = int(seed_raw.get("id"))
        seed_details = self.client.get_movie(seed_id)
        seed_document = self._build_document(seed_details)
        seed_vector = self._vectorizer.transform([seed_document]) if self._vectorizer else None

        candidate_rows: Dict[int, Dict[str, Any]] = {}

        # Gather candidates from TMDb recommendations and similar movies.
        tmdb_rank = 0
        for item in self._iter_unique(self.client.get_recommendations(seed_id), self.client.get_similar(seed_id)):
            movie_id = int(item.get("id"))
            if movie_id == seed_id:
                continue
            tmdb_rank += 1
            candidate_rows[movie_id] = self._hydrate_movie(item, tmdb_rank)

        # Add top content-similarity matches from the prepared catalog.
        if seed_vector is not None and self._matrix is not None:
            similarities = cosine_similarity(seed_vector, self._matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][: top_n * 3]
            for idx in top_indices:
                row = self._catalog.iloc[idx]
                movie_id = int(row["movie_id"])
                if movie_id == seed_id:
                    continue
                entry = candidate_rows.setdefault(movie_id, row.to_dict())
                entry["content_score"] = float(similarities[idx])
                entry.setdefault("tmdb_score", 0.0)

        if not candidate_rows:
            raise ValueError("No live recommendations were returned for this title yet.")

        records = []
        seed_popularity = seed_details.get("popularity") or 0
        seed_vote_count = seed_details.get("vote_count") or 0
        max_vote_count = max(seed_vote_count, max((row.get("vote_count", 0) for row in candidate_rows.values()), default=1)) or 1
        max_popularity = max(seed_popularity, max((row.get("popularity", 0.0) for row in candidate_rows.values()), default=1.0)) or 1.0

        for movie_id, data in candidate_rows.items():
            details = data if "title" in data and data.get("populated") else self._hydrate_movie_by_id(movie_id)
            if not details:
                continue
            content_score = float(details.get("content_score", 0.0))
            if seed_vector is not None and self._vectorizer is not None:
                doc = " ".join(
                    filter(
                        None,
                        [
                            details.get("title"),
                            details.get("plot"),
                            details.get("genres"),
                        ],
                    )
                )
                if doc.strip():
                    candidate_vector = self._vectorizer.transform([doc])
                    similarity = float(cosine_similarity(seed_vector, candidate_vector)[0][0])
                    if similarity > content_score:
                        content_score = similarity
                        details["content_score"] = content_score
            tmdb_score = float(details.get("tmdb_score", 0.0))
            popularity_score = self._compute_popularity_score(details, max_vote_count, max_popularity)

            final_score = (
                0.45 * content_score
                + 0.35 * tmdb_score
                + 0.20 * popularity_score
            )

            records.append(
                LiveRecommendation(
                    movie_id=movie_id,
                    title=details.get("title", "Unknown"),
                    release_year=details.get("release_year"),
                    genres=details.get("genres", ""),
                    final_score=float(final_score),
                    tmdb_score=float(tmdb_score),
                    content_score=float(content_score),
                    popularity_score=float(popularity_score),
                    vote_average=float(details.get("vote_average", 0.0)),
                    vote_count=int(details.get("vote_count", 0)),
                    popularity=float(details.get("popularity", 0.0)),
                    poster=details.get("poster"),
                    plot=details.get("plot"),
                ).__dict__
            )

        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("No live recommendations were produced after scoring.")

        df["movieId"] = df["movie_id"]
        df["reason"] = "tmdb_live"
        df["mean_rating"] = df["vote_average"].round(2)
        df["rating_count"] = df["vote_count"]
        df["final_score"] = df["final_score"].clip(0, 1).round(4)
        df["tmdb_score"] = df["tmdb_score"].clip(0, 1).round(4)
        df["content_score"] = df["content_score"].clip(0, 1).round(4)
        df["popularity_score"] = df["popularity_score"].clip(0, 1).round(4)
        df = df.sort_values(["final_score", "content_score", "popularity_score"], ascending=False).head(top_n)
        df["title"] = df.apply(self._format_title_with_year, axis=1)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_catalog(self) -> None:
        if self._catalog is not None:
            return

        self._genre_lookup = self.client.get_genres()

        records: List[Dict[str, Any]] = []
        seen_ids = set()
        for endpoint in (self.client.get_trending, self.client.get_popular):
            for page in range(1, self.catalog_pages + 1):
                try:
                    movies = endpoint(page=page)
                except TypeError:
                    # trending has signature (time_window, page)
                    movies = endpoint("week", page=page)
                for item in movies:
                    movie_id = item.get("id")
                    if movie_id in seen_ids:
                        continue
                    seen_ids.add(movie_id)
                    records.append(self._summarize_catalog_entry(item))

        if not records:
            raise RuntimeError("TMDb catalog download returned no records.")

        df = pd.DataFrame(records).dropna(subset=["title", "content"])
        df = df.drop_duplicates("movie_id")
        if df.empty:
            raise RuntimeError("Live catalog is empty after cleaning.")

        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)
        matrix = vectorizer.fit_transform(df["content"])

        self._catalog = df
        self._vectorizer = vectorizer
        self._matrix = matrix

    def _summarize_catalog_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        movie_id = int(item.get("id"))
        title = item.get("title") or item.get("name") or "Unknown"
        release_date = item.get("release_date")
        year = int(release_date.split("-")[0]) if release_date else None
        genres = self._resolve_genres(item.get("genre_ids", []))
        overview = item.get("overview") or ""
        poster = self.client.build_poster_url(item.get("poster_path"))
        content = " ".join(filter(None, [title, overview, genres]))
        return {
            "movie_id": movie_id,
            "title": title,
            "release_year": year,
            "genres": genres,
            "overview": overview,
            "poster": poster,
            "vote_average": float(item.get("vote_average", 0.0)),
            "vote_count": int(item.get("vote_count", 0)),
            "popularity": float(item.get("popularity", 0.0)),
            "content": content,
            "plot": overview,
            "populated": True,
        }

    def _hydrate_movie(self, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        details = self._summarize_catalog_entry(item)
        details["tmdb_score"] = 1.0 - 0.02 * (rank - 1)
        details["content_score"] = 0.0
        details["populated"] = True
        return details

    def _hydrate_movie_by_id(self, movie_id: int) -> Dict[str, Any]:
        raw = self.client.get_movie(movie_id)
        if not raw:
            return {}
        details = self._summarize_catalog_entry(raw)
        details["tmdb_score"] = details.get("tmdb_score", 0.0)
        details["content_score"] = details.get("content_score", 0.0)
        return details

    def _build_document(self, details: Dict[str, Any]) -> str:
        title = details.get("title") or details.get("name") or ""
        overview = details.get("overview") or ""
        genres = ", ".join(g.get("name") for g in details.get("genres", []) if g.get("name"))
        keywords = ""
        if "keywords" in details:
            keywords = " ".join(k.get("name") for k in details["keywords"].get("keywords", []))
        return " ".join(filter(None, [title, overview, genres, keywords]))

    def _compute_popularity_score(self, details: Dict[str, Any], max_vote_count: int, max_popularity: float) -> float:
        vote_count = min(int(details.get("vote_count", 0)), max_vote_count)
        popularity = min(float(details.get("popularity", 0.0)), max_popularity)
        vote_component = math.sqrt(vote_count / max_vote_count) if max_vote_count else 0.0
        popularity_component = popularity / max_popularity if max_popularity else 0.0
        return float(0.6 * vote_component + 0.4 * popularity_component)

    def _resolve_genres(self, genre_ids: Iterable[int]) -> str:
        names = [self._genre_lookup.get(int(gid), "") for gid in genre_ids]
        return ", ".join(sorted({name for name in names if name}))

    @staticmethod
    def _iter_unique(*iterables: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        seen = set()
        for iterable in iterables:
            for item in iterable or []:
                movie_id = item.get("id")
                if movie_id in seen:
                    continue
                seen.add(movie_id)
                yield item

    @staticmethod
    def _format_title_with_year(row: pd.Series) -> str:
        title = row.get("title", "Unknown")
        year = row.get("release_year")
        if year and str(year) not in title:
            return f"{title} ({year})"
        return title
