# Movie Recommendation System

A hybrid recommender stack built on the MovieLens 100k dataset with collaborative, content, and latent-factor (SVD) models. The project includes:

- Offline preparation scripts and persisted artifacts for ratings + metadata.
- Collaborative, content-based, and hybrid recommenders with fuzzy title matching and a cold-start fallback.
- OMDb enrichment (posters, plots, cast) with caching and external search support for newer releases.
- A console workflow for power users and a deployable Streamlit dashboard with light/dark modes.
- Optional advanced enhancements: SVD blending, diagnostics, and quick visualization exports.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download the MovieLens bundle you want to work with (the prep script now fetches it automatically if it is missing). Examples:

```bash
/workspaces/Movie-Recommendation-System/.venv/bin/python src/data_preparation.py  # defaults to ml-latest-small

# Use the larger ml-latest bundle (newer releases, more ratings)
MOVIELENS_DATASET=ml-latest /workspaces/Movie-Recommendation-System/.venv/bin/python src/data_preparation.py

# Point to an already downloaded directory
MOVIELENS_DATA_DIR=/data/movielens/ml-25m /workspaces/Movie-Recommendation-System/.venv/bin/python src/data_preparation.py
```

Supported dataset names mirror the official archives (`ml-latest-small`, `ml-latest`, `ml-25m`, etc.). The script will download the zip from [grouplens.org](https://grouplens.org/datasets/movielens/) when the target directory is missing.

This creates `data/processed/ratings_with_movies.csv` and a cached user-item matrix for downstream phases.

## Running the recommender scripts

```bash
# Collaborative filtering sanity check
python src/collaborative_filtering.py

# Content-based neighbours
python src/content_based.py

# Weighted hybrid sample
python src/hybrid.py

# Metadata enrichment demo (Phase 5)
OMDB_API_KEY=<your_key> python src/phase5_demo.py

# Console workflow with filters, OMDb enrichment, and SVD blending
OMDB_API_KEY=<your_key> python src/interactive_cli.py --top 5 --enrich --svd-weight 0.3
```

Set `OMDB_API_KEY` to a valid key (a free key from [omdbapi.com](https://www.omdbapi.com/)); the default fallback (`trilogy`) works for light testing but is rate limited.

## Streamlit Dashboard

The deployable app lives at `app/dashboard.py` and bundles the full feature set:

- Fuzzy "Did you mean" title suggestions and OMDb-powered search for post-2018 releases (configure `OMDB_API_KEY` for live metadata).
- Adjustable genre/year/rating filters, top-N slider, and latent-factor (SVD) weight blending.
- Dark mode is the default (with a light-mode toggle) and poster/plot display when an OMDb key is configured.

Launch locally:

```bash
OMDB_API_KEY=<your_key> streamlit run app/dashboard.py
```

The dashboard caches models per session. SVD factors are trained lazily when a non-zero weight is chosen. Deploy via Streamlit Community Cloud, Azure App Service, or any platform that can run `streamlit run` with the repo and environment variables configured.

## Advanced Enhancements

- `src/advanced_enhancements.py` demonstrates latent SVD blending, rating distribution visualisations, and similarity heatmaps. Example:

	```bash
	python src/advanced_enhancements.py "Toy Story (1995)" --svd-weight 0.35 --factors 40 --top 5
	```
- Artifacts are dropped under `figures/` (ignored by git).

## Tests

Lightweight smoke tests cover data prep, recommendation integrity, fuzzy suggestions, and external metadata scoring:

```bash
pytest -q
```

## Folder Layout

```
app/                # Streamlit dashboard
data/               # Raw + processed data (ignored)
src/                # Core recommendation modules and scripts
tests/              # Pytest-based smoke tests
figures/            # Generated plots (ignored)
```

## Notes

- All modules default to ASCII output; comments are intentionally brief.
- OMDb responses are cached locally under `data/processed/omdb_cache.json` to avoid repeated calls.
- The CLI and dashboard share the same underlying models, ensuring behaviour parity between interfaces.