from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class TMDbClient:
    """Lightweight client for TMDb v3 API with simple caching."""

    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

    def __init__(
        self,
        api_key: Optional[str] = None,
        read_token: Optional[str] = None,
        cache_ttl: int = 3600,
    ) -> None:
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.read_token = read_token or os.getenv("TMDB_READ_TOKEN")
        self.cache_ttl = cache_ttl
        self._session = requests.Session()
        self._cache: Dict[str, Tuple[float, Any]] = {}

        if self.read_token:
            self._session.headers.update({"Authorization": f"Bearer {self.read_token}"})
        if self.api_key:
            self._session.params.update({"api_key": self.api_key})

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return bool(self.api_key or self.read_token)

    def build_poster_url(self, path: Optional[str], size: str = "w500") -> Optional[str]:
        if not path:
            return None
        return f"{self.IMAGE_BASE_URL}/{size}{path}"

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------
    def search_movie(self, query: str, page: int = 1, include_adult: bool = False) -> List[Dict[str, Any]]:
        payload = self._request("/search/movie", params={"query": query, "page": page, "include_adult": include_adult})
        return payload.get("results", []) if payload else []

    def get_movie(self, movie_id: int, append: Optional[str] = None) -> Dict[str, Any]:
        params = {"append_to_response": append} if append else None
        payload = self._request(f"/movie/{movie_id}", params=params)
        return payload or {}

    def get_recommendations(self, movie_id: int, page: int = 1) -> List[Dict[str, Any]]:
        payload = self._request(f"/movie/{movie_id}/recommendations", params={"page": page})
        return payload.get("results", []) if payload else []

    def get_similar(self, movie_id: int, page: int = 1) -> List[Dict[str, Any]]:
        payload = self._request(f"/movie/{movie_id}/similar", params={"page": page})
        return payload.get("results", []) if payload else []

    def get_trending(self, time_window: str = "week", page: int = 1) -> List[Dict[str, Any]]:
        payload = self._request(f"/trending/movie/{time_window}", params={"page": page})
        return payload.get("results", []) if payload else []

    def get_popular(self, page: int = 1) -> List[Dict[str, Any]]:
        payload = self._request("/movie/popular", params={"page": page})
        return payload.get("results", []) if payload else []

    def get_genres(self) -> Dict[int, str]:
        payload = self._request("/genre/movie/list")
        genres = payload.get("genres", []) if payload else []
        return {genre["id"]: genre["name"] for genre in genres}

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------
    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.is_enabled():
            raise RuntimeError("TMDb client requested but no API credentials were provided.")

        cache_key = self._cache_key(path, params)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]

        url = f"{self.BASE_URL}{path}"
        response = self._session.get(url, params=params, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(
                f"TMDb request failed: {response.status_code} {response.text}"
            )
        payload = response.json()
        self._cache[cache_key] = (now, payload)
        return payload

    @staticmethod
    def _cache_key(path: str, params: Optional[Dict[str, Any]]) -> str:
        if not params:
            return path
        sorted_items = sorted(params.items())
        query = "&".join(f"{key}={value}" for key, value in sorted_items)
        return f"{path}?{query}"


def get_tmdb_client() -> TMDbClient:
    """Factory that ensures credentials are loaded from the environment."""

    client = TMDbClient()
    if not client.is_enabled():
        raise RuntimeError(
            "TMDB_API_KEY or TMDB_READ_TOKEN must be set to use TMDb live recommendations."
        )
    return client
