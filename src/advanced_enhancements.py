from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances

from collaborative_filtering import create_user_item_matrix, load_movielens_data, merge_datasets
from hybrid import get_hybrid_recommendations, init_hybrid_model


def build_svd_model(
    n_factors: int = 50,
    random_state: int = 42,
    min_support: int = 5,
    verbosity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings, movies = load_movielens_data()
    merged = merge_datasets(ratings, movies)

    if min_support > 1:
        counts = merged.groupby("movieId")["rating"].count()
        keep = counts[counts >= min_support].index
        merged = merged[merged["movieId"].isin(keep)]

    user_item = create_user_item_matrix(merged)
    filled = user_item.fillna(0.0)

    if verbosity:
        print(f"Running TruncatedSVD with {n_factors} factors on shape {filled.shape}...")

    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    latent = svd.fit_transform(filled)

    user_factors = pd.DataFrame(latent, index=filled.index)
    item_factors = pd.DataFrame(svd.components_.T, index=filled.columns)

    if verbosity:
        explained = svd.explained_variance_ratio_.sum()
        print(f"Explained variance ratio: {explained:.4f}")

    return merged, user_factors, item_factors


def hybrid_with_svd_blend(
    title: str,
    svd_weight: float,
    hybrid_model=None,
    item_factors: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    hybrid_model = hybrid_model or init_hybrid_model()
    base = get_hybrid_recommendations(hybrid_model, title, top_n=top_n * 2)

    if item_factors is None or svd_weight <= 0:
        return base.head(top_n)

    base = base.head(top_n * 2)
    movie_ids = [int(mid) for mid in base["movieId"].tolist() if mid in item_factors.index]

    if not movie_ids:
        return base.head(top_n)

    seed_id = movie_ids[0]
    if seed_id not in item_factors.index:
        return base.head(top_n)

    seed_vector = item_factors.loc[seed_id].values.reshape(1, -1)
    candidate_vectors = item_factors.loc[movie_ids]
    distances = pairwise_distances(seed_vector, candidate_vectors, metric="cosine").flatten()
    svd_scores = 1 - distances
    svd_scores = np.clip((svd_scores + 1.0) / 2.0, 0.0, 1.0)

    for mid, score in zip(movie_ids, svd_scores):
        base.loc[base["movieId"] == mid, "svd_score"] = float(score)

    base["svd_score"] = base["svd_score"].fillna(base["svd_score"].mean(skipna=True) or 0.0)
    base["final_score"] = (1 - svd_weight) * base["final_score"] + svd_weight * base["svd_score"]

    return base.sort_values("final_score", ascending=False).head(top_n)


def plot_rating_distribution(output_dir: Path) -> None:
    ratings, _ = load_movielens_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(ratings["rating"], bins=np.arange(0.5, 5.5, 0.5), edgecolor="black")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png")
    plt.close()


def plot_similarity_heatmap(item_factors: pd.DataFrame, output_dir: Path, sample_size: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if item_factors.empty:
        return

    sample = item_factors.sample(n=min(sample_size, len(item_factors)), random_state=0)
    similarity = 1 - pairwise_distances(sample, metric="cosine")

    plt.figure(figsize=(7, 6))
    plt.imshow(similarity, cmap="viridis")
    plt.colorbar(label="Cosine Similarity")
    plt.title("Item Factor Similarity (sample)")
    plt.xticks(ticks=range(sample.shape[0]), labels=sample.index, rotation=90)
    plt.yticks(ticks=range(sample.shape[0]), labels=sample.index)
    plt.tight_layout()
    plt.savefig(output_dir / "item_similarity_heatmap.png")
    plt.close()


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 advanced enhancements")
    parser.add_argument("title", help="Seed movie title for demonstration")
    parser.add_argument("--svd-weight", type=float, default=0.3, help="Blend weight for SVD latent similarity")
    parser.add_argument("--factors", type=int, default=60, help="Number of latent factors for SVD")
    parser.add_argument("--output", type=Path, default=Path("figures"), help="Where to store diagnostic plots")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to show")
    parser.add_argument("--min-support", type=int, default=10, help="Minimum ratings per movie before SVD")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    merged, _, item_factors = build_svd_model(n_factors=args.factors, min_support=args.min_support)

    output_dir = Path(args.output)
    plot_rating_distribution(output_dir)
    plot_similarity_heatmap(item_factors, output_dir)

    hybrid_model = init_hybrid_model()

    print("\nHybrid + SVD recommendations:")
    blended = hybrid_with_svd_blend(
        title=args.title,
        svd_weight=args.svd_weight,
        hybrid_model=hybrid_model,
        item_factors=item_factors,
        top_n=args.top,
    )
    print(blended[["movieId", "title", "final_score", "svd_score", "collab_score", "content_score"]].to_string(index=False))

    print("\nCreated plots:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file}")


if __name__ == "__main__":
    main()
