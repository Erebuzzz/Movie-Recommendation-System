from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "ml-latest-small"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def load_movielens_data(dataset_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movie metadata from the MovieLens dataset."""
    ratings_path = dataset_dir / "ratings.csv"
    movies_path = dataset_dir / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "Expected ratings.csv and movies.csv inside the MovieLens directory."
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies


def merge_datasets(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Join ratings with movie metadata."""
    merged = ratings.merge(movies, on="movieId", how="left")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="s")
    return merged


def create_user_item_matrix(
    merged_df: pd.DataFrame, min_ratings_per_user: int = 0
) -> pd.DataFrame:
    """Pivot the merged rating table into a user-item matrix."""
    if min_ratings_per_user > 0:
        user_counts = merged_df.groupby("userId")["movieId"].size()
        keep_users = user_counts[user_counts >= min_ratings_per_user].index
        filtered = merged_df[merged_df["userId"].isin(keep_users)]
    else:
        filtered = merged_df

    user_item = filtered.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    )
    return user_item


def report_dataset_health(merged_df: pd.DataFrame) -> None:
    """Print a short health report to the console."""
    print("\nCombined ratings shape:", merged_df.shape)
    print("Rating distribution:")
    print(merged_df["rating"].describe().round(3))

    missing = merged_df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values detected after merge.")
    else:
        print("Columns with missing values:")
        print(missing)


def persist_outputs(merged_df: pd.DataFrame, user_item: pd.DataFrame) -> None:
    """Save cleaned datasets for later phases."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(PROCESSED_DATA_DIR / "ratings_with_movies.csv", index=False)
    user_item.to_pickle(PROCESSED_DATA_DIR / "user_item_matrix.pkl")


def main() -> None:
    ratings, movies = load_movielens_data()
    merged = merge_datasets(ratings, movies)

    print("First five merged rows:")
    preview_cols = ["userId", "movieId", "rating", "timestamp", "title", "genres"]
    head_preview = merged.loc[:, preview_cols].head().copy()
    head_preview["genres"] = head_preview["genres"].str.split("|").str[:3].str.join("|")
    preview_lines = []
    for row in head_preview.to_dict(orient="records"):
        ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        preview_lines.append(
            f"  userId={row['userId']:>3} | movieId={row['movieId']:>5} | rating={row['rating']:.1f} | "
            f"{ts} | {row['title']} | genres={row['genres']}"
        )
    print("\n".join(preview_lines))

    report_dataset_health(merged)

    user_item_matrix = create_user_item_matrix(merged)
    print("\nUser-item matrix shape:", user_item_matrix.shape)
    top_populated = (
        user_item_matrix.notna().sum().sort_values(ascending=False).head(5).index.tolist()
    )
    print("Sample of user-item matrix (densest 5 movies, first 5 users):")
    print(user_item_matrix.loc[:, top_populated].head().to_string())

    persist_outputs(merged, user_item_matrix)


if __name__ == "__main__":
    main()
