from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import requests
import pandas as pd

from data_preparation import PROCESSED_DATA_DIR

OMDB_URL = "http://www.omdbapi.com/"


@dataclass
class OmdbClient:
    api_key: Optional[str] = None
    cache_path: Path = PROCESSED_DATA_DIR / "omdb_cache.json"
    timeout: int = 8
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("OMDB_API_KEY", "trilogy")
        self._cache: Dict[str, Dict[str, str]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r", encoding="utf-8") as handle:
                    self._cache = json.load(handle)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _write_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle, indent=2)
        except OSError:
            pass

    def _cache_key(self, title: str, year: Optional[str]) -> str:
        safe_title = title.lower().strip()
        safe_year = year.strip() if year else ""
        return f"{safe_title}::{safe_year}"

    def _request_enabled(self) -> bool:
        return bool(self.api_key)

    def is_enabled(self) -> bool:
        return self._request_enabled()

    def fetch(self, title: str, year: Optional[str] = None) -> Dict[str, str]:
        key = self._cache_key(title, year)
        if key in self._cache:
            return self._cache[key]

        if not self._request_enabled():
            payload = {"status": "skipped", "reason": "missing_api_key"}
            self._cache[key] = payload
            self._write_cache()
            return payload

        params = {"t": title, "apikey": self.api_key, "plot": "short"}
        if year:
            params["y"] = year

        try:
            response = self.session.get(OMDB_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            payload = {"status": "error", "reason": str(exc)}
            self._cache[key] = payload
            self._write_cache()
            return payload

        if data.get("Response") != "True":
            payload = {"status": "error", "reason": data.get("Error", "unknown")}
            self._cache[key] = payload
            self._write_cache()
            return payload

        fields = {
            "status": "ok",
            "Title": data.get("Title", title),
            "Year": data.get("Year", ""),
            "Genre": data.get("Genre", ""),
            "Poster": data.get("Poster", ""),
            "Plot": data.get("Plot", ""),
            "Actors": data.get("Actors", ""),
            "imdbRating": data.get("imdbRating", ""),
        }

        for field in ("Poster", "Plot", "Actors", "imdbRating"):
            value = fields.get(field)
            if isinstance(value, str) and value.strip().upper() == "N/A":
                fields[field] = ""

        self._cache[key] = fields
        self._write_cache()
        return fields


    def search(
        self,
        query: str,
        limit: int = 6,
        year_from: Optional[int] = None,
    ) -> list[Dict[str, str]]:
        """Search OMDb for movies matching a query."""

        key = self._cache_key(f"search::{query}", str(year_from) if year_from else None)
        if key in self._cache:
            return self._cache[key]

        if not self._request_enabled():
            self._cache[key] = []
            self._write_cache()
            return []

        results: list[Dict[str, str]] = []
        page = 1

        def parse_year(value: str) -> Optional[int]:
            if not value:
                return None
            digits = "".join(ch for ch in value if ch.isdigit())
            if len(digits) >= 4:
                try:
                    return int(digits[:4])
                except ValueError:
                    return None
            return None

        while len(results) < limit and page <= 5:
            params = {"s": query, "type": "movie", "apikey": self.api_key, "page": page}
            try:
                response = self.session.get(OMDB_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError):
                break

            if data.get("Response") != "True":
                break

            for entry in data.get("Search", []):
                year_val = entry.get("Year", "")
                year_int = parse_year(year_val)
                if year_from and (year_int is None or year_int < year_from):
                    continue

                results.append(
                    {
                        "Title": entry.get("Title", ""),
                        "Year": year_val,
                        "imdbID": entry.get("imdbID", ""),
                        "Poster": entry.get("Poster", ""),
                    }
                )
                if len(results) >= limit:
                    break

            total = int(data.get("totalResults", "0")) if str(data.get("totalResults", "0")).isdigit() else 0
            if len(results) >= limit or total <= page * 10:
                break
            page += 1

        self._cache[key] = results
        self._write_cache()
        return results


def enrich_recommendations(df, client: OmdbClient) -> pd.DataFrame:
    """Attach OMDb metadata to recommendation rows."""
    records = []
    for row in df.to_dict(orient="records"):
        original_title = row.get("title", "")
        clean_title = original_title
        year = None

        if clean_title.endswith(")") and "(" in clean_title:
            possible = clean_title[clean_title.rfind("(") + 1 : clean_title.rfind(")")]
            if possible.isdigit():
                year = possible
                clean_title = clean_title[: clean_title.rfind("(")].strip()

        lower = clean_title.lower()
        if "(a.k.a" in lower:
            clean_title = clean_title[: lower.index("(a.k.a")].strip()

        suffix_map = {", the": "The", ", a": "A", ", an": "An"}
        for suffix, prefix in suffix_map.items():
            if clean_title.lower().endswith(suffix):
                clean_title = f"{prefix} {clean_title[:-len(suffix)].strip()}"
                break

        meta = client.fetch(clean_title, year)

        enriched = row.copy()
        enriched["poster"] = meta.get("Poster", "")
        enriched["plot"] = meta.get("Plot", "")[:320]
        enriched["actors"] = meta.get("Actors", "")
        enriched["imdb_rating"] = meta.get("imdbRating", "")
        enriched["metadata_status"] = meta.get("status", "error")
        enriched["metadata_reason"] = meta.get("reason", "")
        records.append(enriched)

    return pd.DataFrame(records)
