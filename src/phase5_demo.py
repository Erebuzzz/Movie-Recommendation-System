from __future__ import annotations

from textwrap import shorten

from hybrid import get_hybrid_recommendations, init_hybrid_model
from metadata_enrichment import OmdbClient, enrich_recommendations


def _print_header(message: str) -> None:
    print("\n" + message)
    print("-" * len(message))


def _print_row(row: dict) -> None:
    title = shorten(row.get("title", ""), width=40, placeholder="...")
    poster = shorten(row.get("poster", ""), width=60, placeholder="...")
    plot = shorten(row.get("plot", ""), width=70, placeholder="...")
    actors = shorten(row.get("actors", ""), width=50, placeholder="...")
    imdb = row.get("imdb_rating", "") or "n/a"
    status = row.get("metadata_status", "unknown")
    reason = row.get("metadata_reason", "")

    print(
        f"  {title:<40} | imdb={imdb:<4} | status={status:<7} {reason:<25} | "
        f"poster={poster:<60}"
    )
    if plot:
        print(f"    plot: {plot}")
    if actors:
        print(f"    cast: {actors}")


def main() -> None:
    hybrid_model = init_hybrid_model(min_support=20)
    client = OmdbClient()

    if not client.is_enabled():
        print(
            "OMDb API key not configured (set OMDB_API_KEY). Using cached/error responses if available."
        )

    seeds = [
        "Toy Story (1995)",
        "Jurassic Park (1993)",
        "Forrest Gump (1994)",
    ]

    for seed in seeds:
        header = f"Enriched recommendations for '{seed}'"
        _print_header(header)

        base_recs = get_hybrid_recommendations(hybrid_model, seed, top_n=3)
        enriched = enrich_recommendations(base_recs, client)

        if enriched.empty:
            print("  (no recommendations produced)")
            continue

        for _, row in enriched.iterrows():
            _print_row(row)


if __name__ == "__main__":
    main()
