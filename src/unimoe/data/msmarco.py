"""MS MARCO data pipeline for embedding and reranking tasks.

Downloads and preprocesses MS MARCO passage data into two formats:
1. Embedding: (query, positive, [hard_neg_1, ..., hard_neg_K]) for InfoNCE
2. Reranking: (query, document, label) with yes/no labels for SFT-style training

Data sources:
- sentence-transformers/msmarco: corpus, queries, triplets, labeled-list
- sentence-transformers/msmarco-hard-negatives: pre-mined BM25 + dense negatives
"""

from __future__ import annotations

import gc
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

from unimoe.config import DataConfig


def _build_corpus_lookup(corpus_ds: Dataset) -> dict[str, str]:
    """Build passage_id -> passage text lookup dict."""
    lookup = {}
    for row in tqdm(corpus_ds, desc="Building corpus lookup"):
        lookup[str(row["passage_id"])] = row["passage"]
    return lookup


def _build_query_lookup(queries_ds: Dataset) -> dict[str, str]:
    """Build query_id -> query text lookup dict."""
    lookup = {}
    for row in tqdm(queries_ds, desc="Building query lookup"):
        lookup[str(row["query_id"])] = row["query"]
    return lookup


def load_embedding_dataset(
    config: DataConfig, cache_dir: str = "data", seed: int = 42
) -> Dataset:
    """Load and preprocess MS MARCO for embedding training.

    Each row contains: query, positive passage, and K hard negative passages.
    Hard negatives are mined from ranks 30-100 of the pre-mined dataset
    to avoid false negatives.

    Args:
        config: Data configuration with num_hard_negatives and sample limits.
        cache_dir: Directory to cache processed data.
        seed: Random seed for hard negative sampling (included in cache key).

    Returns:
        HuggingFace Dataset with columns: query, positive, negatives (list of str).
    """
    cache_path = Path(cache_dir) / f"embedding_dataset_seed_{seed}_n{config.embedding_samples}_neg{config.num_hard_negatives}"
    if cache_path.exists():
        return Dataset.load_from_disk(str(cache_path))

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    corpus_ds = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    queries_ds = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    # Load only the original hard negatives file (with bm25 + dense retriever keys).
    # Cannot use load_dataset("sentence-transformers/msmarco-hard-negatives") directly
    # because the repo contains multiple JSONL files with incompatible schemas.
    hard_neg_ds = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
        split="train",
    )

    corpus_lookup = _build_corpus_lookup(corpus_ds)
    query_lookup = _build_query_lookup(queries_ds)

    # Build hard negatives index: qid -> {retriever_name: [neg_passage_ids]}
    hard_neg_index: dict[int, dict] = {}
    for row in tqdm(hard_neg_ds, desc="Indexing hard negatives"):
        hard_neg_index[row["qid"]] = {
            "pos": row["pos"],
            "neg": row["neg"],
        }

    num_hard_negs = config.num_hard_negatives
    records = []

    # Set seed for reproducible hard negative sampling
    random.seed(seed)

    for qid, info in tqdm(hard_neg_index.items(), desc="Building embedding dataset"):
        qid_str = str(qid)
        if qid_str not in query_lookup:
            continue

        query_text = query_lookup[qid_str]
        pos_ids = info["pos"]
        if not pos_ids:
            continue

        pos_id_str = str(pos_ids[0])
        if pos_id_str not in corpus_lookup:
            continue
        positive_text = corpus_lookup[pos_id_str]

        # Positive IDs to exclude from negatives (avoid false negatives)
        pos_id_set = {str(pid) for pid in pos_ids}

        # Get BM25 hard negatives, sampling from ranks 30-100 to avoid false negatives
        neg_dict = info["neg"]
        bm25_negs = neg_dict.get("bm25", [])

        # Sample from ranks 30-100 (index 29-99) to avoid false negatives
        candidate_negs = [
            nid for nid in (bm25_negs[29:100] if len(bm25_negs) >= 30 else bm25_negs)
            if str(nid) not in pos_id_set
        ]
        if not candidate_negs:
            continue

        # Also use dense retriever negatives as fallback
        for retriever in ["msmarco-distilbert-base-tas-b", "msmarco-distilbert-base-v3"]:
            if retriever in neg_dict:
                dense_negs = neg_dict[retriever]
                if isinstance(dense_negs, list) and dense_negs:
                    filtered = [
                        nid for nid in (dense_negs[29:100] if len(dense_negs) >= 30 else dense_negs)
                        if str(nid) not in pos_id_set
                    ]
                    candidate_negs.extend(filtered)

        # Sample K negatives and resolve to text
        if len(candidate_negs) < num_hard_negs:
            # Fallback: use all available candidates
            sampled_neg_ids = candidate_negs[:num_hard_negs]
        else:
            sampled_neg_ids = random.sample(candidate_negs, num_hard_negs)

        neg_texts = []
        for neg_id in sampled_neg_ids:
            neg_id_str = str(neg_id)
            if neg_id_str in corpus_lookup:
                neg_texts.append(corpus_lookup[neg_id_str])

        if len(neg_texts) < num_hard_negs:
            continue  # Skip if we can't find enough negatives

        records.append({
            "query": query_text,
            "positive": positive_text,
            "negatives": neg_texts,
        })

        if config.embedding_samples and len(records) >= config.embedding_samples:
            break

    # Release lookup dicts and raw datasets to reclaim memory before
    # building the final Dataset.  The corpus_lookup alone holds ~8.8M
    # entries (several GB).  Explicit deletion + gc.collect() is needed
    # because Python's small-object arena allocator may not return
    # memory to the OS on its own.
    del corpus_lookup, query_lookup, hard_neg_index
    del corpus_ds, queries_ds, hard_neg_ds
    gc.collect()

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(cache_path))
    return dataset


def load_reranking_dataset(
    config: DataConfig, cache_dir: str = "data", seed: int = 42
) -> Dataset:
    """Load and preprocess MS MARCO for reranking training.

    Flattens the labeled-list subset into individual (query, document, label) rows.
    Subsamples 5-7 hard negatives per query (highest BM25-ranked but labeled 0).

    Args:
        config: Data configuration with sample limits.
        cache_dir: Directory to cache processed data.
        seed: Random seed (included in cache key for reproducibility).

    Returns:
        HuggingFace Dataset with columns: query, document, label ("yes"/"no").
    """
    cache_path = Path(cache_dir) / f"reranking_dataset_seed_{seed}_n{config.reranking_samples}"
    if cache_path.exists():
        return Dataset.load_from_disk(str(cache_path))

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    corpus_ds = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    queries_ds = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    labeled_ds = load_dataset("sentence-transformers/msmarco", "labeled-list", split="train")

    corpus_lookup = _build_corpus_lookup(corpus_ds)
    query_lookup = _build_query_lookup(queries_ds)

    records = []
    max_neg_per_query = 7
    min_neg_per_query = 5

    # Seed for reproducible negative sampling across seeds
    random.seed(seed)

    for row in tqdm(labeled_ds, desc="Building reranking dataset"):
        qid_str = str(row["query_id"])
        if qid_str not in query_lookup:
            continue
        query_text = query_lookup[qid_str]

        doc_ids = row["doc_ids"]
        labels = row["labels"]

        positives = []
        negatives = []

        for doc_id, label in zip(doc_ids, labels):
            doc_id_str = str(doc_id)
            if doc_id_str not in corpus_lookup:
                continue
            doc_text = corpus_lookup[doc_id_str]

            if label == 1:
                positives.append(doc_text)
            else:
                negatives.append(doc_text)

        # Add all positive examples
        for pos_text in positives:
            records.append({
                "query": query_text,
                "document": pos_text,
                "label": "yes",
            })

        # Subsample negatives (5-7 per query, randomly sampled)
        num_negs = min(max_neg_per_query, len(negatives))
        if num_negs < min_neg_per_query and len(negatives) >= min_neg_per_query:
            num_negs = min_neg_per_query

        sampled_negs = random.sample(negatives, num_negs) if len(negatives) > num_negs else negatives[:num_negs]
        for neg_text in sampled_negs:
            records.append({
                "query": query_text,
                "document": neg_text,
                "label": "no",
            })

        if config.reranking_samples and len(records) >= config.reranking_samples:
            break

    # Release lookup dicts and raw datasets to reclaim memory (see
    # load_embedding_dataset for rationale).
    del corpus_lookup, query_lookup
    del corpus_ds, queries_ds, labeled_ds
    gc.collect()

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(cache_path))
    return dataset
