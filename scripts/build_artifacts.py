from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_preparation import (
    PROCESSED_DATA_DIR,
    create_user_item_matrix,
    load_movielens_data,
    merge_datasets,
    persist_outputs,
)
from advanced_enhancements import build_svd_model


ARTIFACT_DIR = PROCESSED_DATA_DIR
SVD_FACTORS_FILE = ARTIFACT_DIR / "svd_item_factors.pkl"
SVD_META_FILE = ARTIFACT_DIR / "svd_item_factors_meta.json"


def run_data_preparation() -> None:
    ratings, movies = load_movielens_data()
    merged = merge_datasets(ratings, movies)
    user_item = create_user_item_matrix(merged)
    persist_outputs(merged, user_item)


def write_svd_item_factors(n_factors: int, min_support: int) -> None:
    _, _, item_factors = build_svd_model(
        n_factors=n_factors,
        min_support=min_support,
        verbosity=True,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    item_factors.to_pickle(SVD_FACTORS_FILE)

    metadata = {
        "n_factors": n_factors,
        "min_support": min_support,
        "rows": item_factors.shape[0],
        "cols": item_factors.shape[1],
    }
    SVD_META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved SVD item factors to {SVD_FACTORS_FILE}")
    print(f"Saved metadata to {SVD_META_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-build data artifacts for deployment")
    parser.add_argument("--no-prep", action="store_true", help="Skip data preparation step")
    parser.add_argument("--svd-factors", type=int, default=60, help="Number of latent factors for SVD")
    parser.add_argument("--svd-min-support", type=int, default=10, help="Minimum rating count before SVD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_prep:
        print("Running Phase 1 data preparation...")
        run_data_preparation()

    print("Building SVD latent factors...")
    write_svd_item_factors(args.svd_factors, args.svd_min_support)

    print("Artifacts ready.")


if __name__ == "__main__":
    main()
