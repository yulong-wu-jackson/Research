"""Reranking evaluation on BEIR datasets and MS MARCO dev.

Uses yes/no token logit scoring to rerank BM25 top-100 candidates,
then computes nDCG@10 and MRR@10 via pytrec_eval.
"""

from __future__ import annotations

import gc

import numpy as np
import pytrec_eval
import torch
from tqdm import tqdm

from unimoe.config import ExperimentConfig
from unimoe.data.templates import DEFAULT_INSTRUCTION, format_reranking_input
from unimoe.model.lora_model import UnimodelForExp1


def _score_pairs(
    model: UnimodelForExp1,
    tokenizer,
    queries: list[str],
    documents: list[str],
    batch_size: int = 64,
    max_len: int = 512,
    device: str = "cpu",
) -> list[float]:
    """Score query-document pairs using yes/no logit scoring."""
    all_scores = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]
            batch_d = documents[i : i + batch_size]

            formatted = [
                format_reranking_input(DEFAULT_INSTRUCTION, q, d)
                for q, d in zip(batch_q, batch_d)
            ]

            encoded = tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            scores = model.rerank(encoded["input_ids"], encoded["attention_mask"])
            all_scores.extend(scores.cpu().tolist())

    return all_scores


def evaluate_msmarco_dev(
    model: UnimodelForExp1,
    tokenizer,
    config: ExperimentConfig,
) -> dict:
    """Evaluate reranking on MS MARCO dev set.

    Uses BM25 top-100 candidates, reranks with yes/no scoring,
    and computes MRR@10 via pytrec_eval.

    Returns dict with per-query and aggregate scores.
    """
    from datasets import load_dataset

    device = config.model.resolve_device()

    # Load MS MARCO dev queries and qrels
    queries_ds = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    query_lookup = {str(row["query_id"]): row["query"] for row in queries_ds}

    # Load dev qrels from MS MARCO
    # Use the labeled-list subset to get dev queries with relevance labels
    labeled_ds = load_dataset("sentence-transformers/msmarco", "labeled-list", split="train")
    corpus_ds = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    corpus_lookup = {str(row["passage_id"]): row["passage"] for row in corpus_ds}

    # Build qrels and candidate lists
    qrels = {}
    candidates = {}

    for row in labeled_ds:
        qid = str(row["query_id"])
        if qid not in query_lookup:
            continue
        doc_ids = row["doc_ids"]
        labels = row["labels"]

        qrels[qid] = {}
        candidates[qid] = []
        for doc_id, label in zip(doc_ids, labels):
            doc_id_str = str(doc_id)
            qrels[qid][doc_id_str] = int(label)
            if doc_id_str in corpus_lookup:
                candidates[qid].append((doc_id_str, corpus_lookup[doc_id_str]))

        # Limit to top-100 candidates
        candidates[qid] = candidates[qid][:100]

    # Release corpus_lookup (~8.8M entries) and raw datasets; only
    # query_lookup is still needed for the reranking loop below.
    del corpus_lookup, corpus_ds, queries_ds, labeled_ds
    gc.collect()

    # Rerank candidates for each query
    run = {}
    per_query_scores = {}

    for qid in tqdm(list(candidates.keys())[:6980], desc="MS MARCO dev reranking"):
        if not candidates.get(qid):
            continue
        query_text = query_lookup[qid]
        cand_ids = [c[0] for c in candidates[qid]]
        cand_texts = [c[1] for c in candidates[qid]]

        query_list = [query_text] * len(cand_texts)
        scores = _score_pairs(
            model, tokenizer, query_list, cand_texts,
            batch_size=config.eval.eval_batch_size,
            max_len=config.data.reranking_max_len,
            device=device,
        )
        run[qid] = {cid: float(s) for cid, s in zip(cand_ids, scores)}

    # Truncate run to top-10 per query for MRR@10
    # (pytrec_eval's recip_rank has no built-in cutoff)
    for qid in run:
        sorted_docs = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:10]
        run[qid] = dict(sorted_docs)

    # Compute MRR@10 via pytrec_eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    results = evaluator.evaluate(run)

    for qid in results:
        per_query_scores[qid] = results[qid]["recip_rank"]

    mrr_at_10 = np.mean(list(per_query_scores.values())) if per_query_scores else 0.0

    return {
        "dataset": "msmarco_dev",
        "per_query": per_query_scores,
        "aggregate": {"mrr@10": float(mrr_at_10)},
    }


def evaluate_beir(
    model: UnimodelForExp1,
    tokenizer,
    config: ExperimentConfig,
    dataset_name: str,
) -> dict:
    """Evaluate reranking on a BEIR dataset.

    Downloads the dataset, retrieves BM25 top-100 candidates, reranks
    with the model, and computes nDCG@10 via pytrec_eval.

    Returns dict with per-query and aggregate scores.
    """
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader
    from rank_bm25 import BM25Okapi

    device = config.model.resolve_device()

    # Download and load BEIR dataset
    data_path = beir_util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        "data/beir",
    )
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # BM25 retrieval for top-100 candidates
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
        for cid in corpus_ids
    ]
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # For each query, get BM25 top-100 and rerank
    run = {}
    per_query_scores = {}

    for qid, query_text in tqdm(queries.items(), desc=f"Evaluating {dataset_name}"):
        tokenized_query = query_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:100]

        candidate_ids = [corpus_ids[idx] for idx in top_indices]
        candidate_texts = [corpus_texts[idx] for idx in top_indices]

        # Rerank candidates
        query_list = [query_text] * len(candidate_texts)
        scores = _score_pairs(
            model,
            tokenizer,
            query_list,
            candidate_texts,
            batch_size=config.eval.eval_batch_size,
            max_len=config.data.reranking_max_len,
            device=device,
        )

        run[qid] = {cid: float(s) for cid, s in zip(candidate_ids, scores)}

    # Release BM25 index, corpus structures, and query dict.  These can
    # be 500 MB–2 GB for large BEIR datasets, and evaluate_reranking_suite
    # calls this function multiple times sequentially.
    del bm25, tokenized_corpus, corpus_ids, corpus_texts, corpus, queries
    gc.collect()

    # Compute nDCG@10 via pytrec_eval
    qrels_int = {
        qid: {did: int(rel) for did, rel in rels.items()}
        for qid, rels in qrels.items()
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_int, {"ndcg_cut_10"})
    results = evaluator.evaluate(run)

    # Extract per-query nDCG@10
    for qid in results:
        per_query_scores[qid] = results[qid]["ndcg_cut_10"]

    aggregate = {
        "ndcg@10": np.mean(list(per_query_scores.values())),
    }

    return {
        "dataset": dataset_name,
        "per_query": per_query_scores,
        "aggregate": aggregate,
    }


def evaluate_reranking_suite(
    model: UnimodelForExp1,
    tokenizer,
    config: ExperimentConfig,
) -> dict:
    """Run reranking evaluation on all configured BEIR datasets + MS MARCO dev.

    Returns aggregated results dict with per-dataset and average scores.
    """
    all_results = {}

    # BEIR datasets
    for dataset_name in config.eval.beir_datasets:
        result = evaluate_beir(model, tokenizer, config, dataset_name)
        all_results[dataset_name] = result

    # Compute average across datasets
    ndcg_scores = [r["aggregate"]["ndcg@10"] for r in all_results.values()]
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return {
        "per_dataset": all_results,
        "average": {"ndcg@10": float(avg_ndcg)},
    }
