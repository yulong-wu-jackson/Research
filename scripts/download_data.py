"""CLI entry point for downloading and preprocessing MS MARCO data.

Usage:
    uv run python scripts/download_data.py [--embedding-samples N] [--reranking-samples N]
"""

from __future__ import annotations

import argparse
from collections import Counter

from unimoe.config import DataConfig
from unimoe.data.msmarco import load_embedding_dataset, load_reranking_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess MS MARCO data")
    parser.add_argument(
        "--embedding-samples",
        type=int,
        default=None,
        help="Max embedding samples (default: full dataset)",
    )
    parser.add_argument(
        "--reranking-samples",
        type=int,
        default=None,
        help="Max reranking samples (default: full dataset)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data",
        help="Cache directory for processed data",
    )
    args = parser.parse_args()

    config = DataConfig(
        embedding_samples=args.embedding_samples,
        reranking_samples=args.reranking_samples,
    )

    print("=" * 60)
    print("MS MARCO Data Pipeline")
    print("=" * 60)

    # Download and preprocess embedding dataset
    print("\n--- Embedding Dataset ---")
    emb_ds = load_embedding_dataset(config, cache_dir=args.cache_dir)
    print(f"  Rows: {len(emb_ds)}")
    print(f"  Columns: {emb_ds.column_names}")
    if len(emb_ds) > 0:
        sample = emb_ds[0]
        print(f"  Negatives per query: {len(sample['negatives'])}")
        print(f"  Sample query: {sample['query'][:80]}...")

    # Download and preprocess reranking dataset
    print("\n--- Reranking Dataset ---")
    rerank_ds = load_reranking_dataset(config, cache_dir=args.cache_dir)
    print(f"  Rows: {len(rerank_ds)}")
    print(f"  Columns: {rerank_ds.column_names}")

    # Label distribution
    label_counts = Counter(rerank_ds["label"])
    total = sum(label_counts.values())
    print("  Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {label}: {count} ({pct:.1f}%)")

    pos_count = label_counts.get("yes", 0)
    neg_count = label_counts.get("no", 0)
    if pos_count > 0:
        print(f"  Pos:Neg ratio: 1:{neg_count / pos_count:.1f}")

    print("\n" + "=" * 60)
    print("Done! Data cached in:", args.cache_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
