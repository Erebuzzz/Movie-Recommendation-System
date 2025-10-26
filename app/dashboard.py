from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from advanced_enhancements import build_svd_model
from content_based import ContentModel, init_content_model, score_external_movie
from hybrid import (
    HybridModel,
    _find_movie_id,
    get_hybrid_recommendations,
    init_hybrid_model,
    suggest_titles,
)
from metadata_enrichment import OmdbClient, enrich_recommendations
from data_preparation import PROCESSED_DATA_DIR

SVD_FACTORS_FILE = PROCESSED_DATA_DIR / "svd_item_factors.pkl"
SVD_META_FILE = PROCESSED_DATA_DIR / "svd_item_factors_meta.json"


@st.cache_resource(show_spinner=True)
def load_hybrid_model(min_support: int) -> HybridModel:
    return init_hybrid_model(min_support=min_support)


@st.cache_resource(show_spinner=True)
def load_content_model() -> ContentModel:
    return init_content_model()


@st.cache_resource(show_spinner=True)
def load_svd_item_factors(n_factors: int, min_support: int) -> pd.DataFrame:
    if SVD_FACTORS_FILE.exists():
        try:
            if SVD_META_FILE.exists():
                meta = json.loads(SVD_META_FILE.read_text(encoding="utf-8"))
                if meta.get("n_factors") == n_factors and meta.get("min_support") == min_support:
                    return pd.read_pickle(SVD_FACTORS_FILE)
            else:
                return pd.read_pickle(SVD_FACTORS_FILE)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    _, _, item_factors = build_svd_model(
        n_factors=n_factors,
        min_support=min_support,
        verbosity=False,
    )

    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        item_factors.to_pickle(SVD_FACTORS_FILE)
        metadata = {
            "n_factors": n_factors,
            "min_support": min_support,
            "rows": item_factors.shape[0],
            "cols": item_factors.shape[1],
        }
        SVD_META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError:
        pass

    return item_factors


@st.cache_resource(show_spinner=False)
def load_omdb_client() -> OmdbClient:
    return OmdbClient(api_key=os.getenv("OMDB_API_KEY"))


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    legacy_rerun = getattr(st, "experimental_rerun", None)
    if callable(legacy_rerun):
        legacy_rerun()


def apply_theme(dark_mode: bool) -> None:
    if dark_mode:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #121212; color: #f5f5f5; }
            .st-emotion-cache-6qob1r, .stSidebar { background-color: #1e1e1e; }
            .stMarkdown a { color: #8ab4f8; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #fafafa; color: #111; }
            .st-emotion-cache-6qob1r, .stSidebar { background-color: #ffffff; }
            .stMarkdown a { color: #3366cc; }
            </style>
            """,
            unsafe_allow_html=True,
        )


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
        years = filtered["title"].str.extract(r"(\d{4})").astype(float)
        filtered = filtered[years[0].fillna(0) >= min_year]

    if min_mean_rating:
        filtered = filtered[filtered["mean_rating"].fillna(0) >= min_mean_rating]

    return filtered


def blend_with_svd(
    df: pd.DataFrame,
    item_factors: Optional[pd.DataFrame],
    seed_id: Optional[int],
    weight: float,
) -> pd.DataFrame:
    if weight <= 0 or item_factors is None or seed_id is None:
        return df

    if seed_id not in item_factors.index:
        st.info("SVD blend skipped: seed movie missing from latent factors.")
        return df

    working = df.copy()
    candidate_ids = [int(mid) for mid in working["movieId"].tolist() if mid in item_factors.index]
    if not candidate_ids:
        st.info("SVD blend skipped: no candidates with latent factors after filters.")
        return df

    seed_vector = item_factors.loc[seed_id].to_numpy()
    candidate_vectors = item_factors.loc[candidate_ids].to_numpy()

    seed_norm = float(np.linalg.norm(seed_vector)) or 1e-9
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    candidate_norms[candidate_norms == 0] = 1e-9

    similarities = candidate_vectors.dot(seed_vector) / (candidate_norms * seed_norm)
    similarities = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
    svd_map = {mid: float(score) for mid, score in zip(candidate_ids, similarities)}
    working["svd_score"] = working["movieId"].map(svd_map).fillna(0.0)
    working["final_score"] = (1 - weight) * working["final_score"].astype(float) + weight * working["svd_score"]
    return working.sort_values("final_score", ascending=False)


def render_recommendations(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    if df.empty:
        st.warning("No recommendations matched the selected filters.")
        return

    for _, row in df.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                poster = row.get("poster")
                if poster:
                    st.image(poster, width="stretch")
            with cols[1]:
                st.markdown(f"### {row.get('title', 'Unknown')}")
                st.write(f"Genres: {row.get('genres', 'n/a')}")
                st.write(
                    f"Final score: {row.get('final_score', 0):.3f} "
                    f"(collab={row.get('collab_score', 0):.3f}, content={row.get('content_score', 0):.3f}, "
                    f"genre={row.get('genre_similarity', 0):.3f}, svd={row.get('svd_score', 0):.3f})"
                )
                st.write(
                    f"Ratings count: {row.get('rating_count', 'n/a')} | "
                    f"Avg rating: {row.get('mean_rating', 'n/a')} | IMDb: {row.get('imdb_rating', 'n/a')}"
                )
                if row.get("plot"):
                    st.write(row["plot"])
                if row.get("actors"):
                    st.caption(f"Cast: {row['actors']}")


def handle_external_movie(
    selected: dict,
    content_model: ContentModel,
    hybrid_model: HybridModel,
    client: OmdbClient,
    genre_filter: Optional[str],
    min_year: Optional[int],
    min_avg_rating: Optional[float],
    top_n: int,
    enrich: bool,
) -> None:
    title = selected.get("Title", "")
    year = selected.get("Year")
    details = client.fetch(title, year)

    st.info(
        f"Showing catalog matches for new release **{details.get('Title', title)} ({details.get('Year', year)})**"
    )

    recs = score_external_movie(
        content_model,
        title=details.get("Title", title),
        genres=details.get("Genre", ""),
        plot=details.get("Plot", ""),
        tags=details.get("Actors", ""),
        top_n=top_n * 4,
    )

    recs = recs.rename(columns={"score": "content_score"})
    recs["final_score"] = recs["content_score"]
    recs["collab_score"] = 0.0
    recs["svd_score"] = recs["content_score"]
    recs["rating_count"] = recs["movieId"].map(hybrid_model.collab.rating_counts).fillna(0).astype(int)
    recs["mean_rating"] = recs["movieId"].map(hybrid_model.mean_ratings)

    recs = apply_filters(recs, genre_filter, min_year, min_avg_rating)
    recs = recs.head(top_n)

    if recs.empty:
        st.warning("No catalog matches met your filter criteria for this new release yet.")
        return

    if enrich and client.is_enabled():
        recs = enrich_recommendations(recs, client)

    render_recommendations(recs, "Similar titles from catalog")


def main() -> None:
    st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

    st.sidebar.title("Controls")

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    dark_mode = st.sidebar.checkbox("Dark mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode
    apply_theme(dark_mode)

    top_n = st.sidebar.slider("Number of recommendations", 3, 15, 6)
    genre_filter = st.sidebar.text_input("Genre keyword filter")
    min_year = st.sidebar.number_input("Minimum release year", min_value=1900, max_value=2030, value=1900, step=1)
    min_year = int(min_year) if min_year else None
    min_avg_rating = st.sidebar.slider("Minimum avg rating", 0.0, 5.0, 0.0, step=0.1)
    min_avg_rating = min_avg_rating or None

    st.sidebar.markdown("---")
    svd_weight = st.sidebar.slider("SVD blend weight", 0.0, 1.0, 0.2, step=0.05)
    svd_factors = st.sidebar.selectbox("Latent factors", [20, 40, 60, 80], index=1)
    svd_min_support = st.sidebar.slider("SVD min ratings", 5, 50, 15, step=5)

    st.sidebar.markdown("---")
    enrich_enabled = st.sidebar.checkbox("Include OMDb posters & plots", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Enter an OMDb API key via OMDB_API_KEY env var for live metadata.")

    movie_query = st.text_input("Movie you like", value=st.session_state.get("movie_query", ""))
    st.session_state.movie_query = movie_query

    if not movie_query:
        st.write("Enter a movie title to begin exploring recommendations.")
        return

    hybrid_model = load_hybrid_model(min_support=10)
    content_model = load_content_model()
    client = load_omdb_client()

    seed_id = _find_movie_id(hybrid_model, movie_query)

    if seed_id is None:
        st.warning("Exact title not found. Here are some close matches.")
        catalog_suggestions = suggest_titles(hybrid_model, movie_query, limit=5)
        external_suggestions = client.search(movie_query, limit=5, year_from=2018) if client.is_enabled() else []

        if catalog_suggestions:
            st.markdown("#### Catalog suggestions")
            for idx, suggestion in enumerate(catalog_suggestions, start=1):
                if st.button(f"Use {suggestion['title']}", key=f"catalog-{idx}"):
                    st.session_state.movie_query = suggestion["title"]
                    rerun_app()

        if external_suggestions:
            st.markdown("#### New releases")
            for idx, suggestion in enumerate(external_suggestions, start=1):
                col = st.columns([3, 1])
                with col[0]:
                    st.write(f"{suggestion.get('Title', '?')} ({suggestion.get('Year', '?')})")
                with col[1]:
                    if st.button("View", key=f"external-{idx}"):
                        handle_external_movie(
                            suggestion,
                            content_model,
                            hybrid_model,
                            client,
                            genre_filter,
                            min_year,
                            min_avg_rating,
                            top_n,
                            enrich_enabled,
                        )
                        return

        if not catalog_suggestions and not external_suggestions:
            st.info("No close matches found yet; showing popular fallback recommendations instead.")
    try:
        recs = get_hybrid_recommendations(hybrid_model, movie_query, top_n=top_n * 3)
    except ValueError as exc:
        st.error(str(exc))
        return

    recs = apply_filters(recs, genre_filter, min_year, min_avg_rating)
    if recs.empty:
        st.warning("No recommendations after applying filters. Try adjusting your filters.")
        return

    if svd_weight > 0:
        svd_item_factors = load_svd_item_factors(svd_factors, svd_min_support)
        recs = blend_with_svd(recs, svd_item_factors, seed_id, svd_weight)

    recs = recs.head(top_n)

    if enrich_enabled and client.is_enabled():
        recs = enrich_recommendations(recs, client)

    render_recommendations(recs, "Hybrid recommendations")


if __name__ == "__main__":
    main()
