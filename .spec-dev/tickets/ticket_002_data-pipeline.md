# MS MARCO Data Pipeline

**Created:** 2026-02-06
**Updated:** 2026-02-07
**Status:** Draft
**Depends on:** ticket_001 (config system, project structure)
**Blocks:** ticket_003 (model+training needs data), ticket_004 (eval needs data)

## Overview

Download and preprocess MS MARCO passage data into two task-specific formats (embedding with hard negatives and reranking in Qwen3-Reranker chat template format), and implement batch collators that tokenize data for each task using the Qwen3 tokenizer.

## Context

- **Data source:** `sentence-transformers/msmarco` on HuggingFace — subsets: `triplets`, `labeled-list`, `corpus`, `queries`
- **Hard negatives source:** `sentence-transformers/msmarco-hard-negatives` — pre-mined BM25 + dense retriever negatives with 160M cross-encoder scores
- **Embedding format:** `(query, positive_passage, [hard_neg_1, ..., hard_neg_K])` for InfoNCE with hard negatives
- **Reranking format:** Qwen3-Reranker chat template with yes/no labels for SFT-style training
- **Tokenizer:** Qwen3-0.6B tokenizer, uses `<|endoftext|>` as EOS, `<|im_start|>`/`<|im_end|>` for chat template
- **Sequence lengths:** query max 128, passage max 256, reranking pair max 512

## Requirements

- [ ] Implement `src/unimoe/data/msmarco.py`:
  - Download `sentence-transformers/msmarco` subsets (corpus, queries, triplets, labeled-list) via HuggingFace `datasets`
  - Download `sentence-transformers/msmarco-hard-negatives` for pre-mined hard negatives
  - For embedding: resolve triplet IDs to text, use full ~500K queries, mine 7 hard negatives per query from pre-mined dataset (sample from ranks 30-100 to avoid false negatives), cache as Arrow/Parquet in `data/`
  - For reranking: flatten labeled-list, subsample 5-7 hard negatives per query (highest BM25-ranked but labeled 0), resolve IDs, cache ~500K pairs
  - Functions: `load_embedding_dataset(config)` and `load_reranking_dataset(config)` that return HuggingFace Dataset objects
  - Configurable `num_samples` parameter for quick sanity runs (default: full dataset)
- [ ] Implement `src/unimoe/data/collators.py`:
  - `EmbeddingCollator`: tokenizes query (with instruction prefix), positive, and hard negatives separately; appends EOS token; returns dict with `query_input_ids`, `query_attention_mask`, `pos_input_ids`, `pos_attention_mask`, `neg_input_ids` (list), `neg_attention_mask` (list)
  - `RerankingCollator`: formats input using Qwen3-Reranker chat template via **pre-tokenized prefix/suffix concatenation** (NOT full-string tokenization — BPE is context-sensitive at join boundaries). Pre-compute `prefix_token_ids` and `suffix_token_ids` once at init, tokenize only the user content (`<Instruct>...<Document>...`) per sample, then concatenate `prefix + content + suffix` at the token-ID level. Returns `input_ids`, `attention_mask`, `labels` (token ID for "yes" or "no")
  - Both use the Qwen3 tokenizer with appropriate max lengths from config
- [ ] Implement `src/unimoe/data/templates.py`:
  - `format_embedding_query(instruction, query)` -> `"Instruct: {instruction}\nQuery:{query}"` (Qwen3-Embedding format — NOTE: no space after `Query:`, matching official `config_sentence_transformers.json`)
  - `format_reranking_input(instruction, query, document)` -> full Qwen3-Reranker chat template string, including the `<think>\n\n</think>\n\n` suffix that Qwen3-Reranker expects before yes/no prediction
  - Default instruction: `"Given a web search query, retrieve relevant passages that answer the query"`
  - `set_tokenizer_config(tokenizer)` -> set `padding_side='left'` for causal attention / flash_attention_2 compatibility (matches Qwen3-Embedding)
- [ ] Implement `scripts/download_data.py`: CLI entry point that triggers download + preprocessing, reports dataset sizes and label distributions

## Design Decisions

- **Full ~500K queries (not 200K):** 200K risks underfitting, which confounds interference measurements. Top embedding models train on millions; 500K is the minimum for reliable results per NV-Retriever (ICLR 2025), F2LLM, and KaLM-Embedding-V2 findings.
- **Pre-mined hard negatives over random negatives:** Hard negatives are the single most impactful factor for embedding quality (NV-Retriever). The `sentence-transformers/msmarco-hard-negatives` dataset provides high-quality BM25 + dense retriever negatives with cross-encoder scores. Mining from ranks 30-100 avoids false negatives (~70% of top-ranked "hard negatives" in MS MARCO are actually false negatives per NV-Retriever).
- **7 hard negatives per query:** Standard in F2LLM and KaLM-Embedding-V2. Balances training signal with compute cost.
- **Qwen3-Reranker chat template (not raw concatenation):** Aligns with how Qwen3-Reranker is trained, enabling fair comparison. Uses yes/no token prediction instead of custom MLP head scoring.
- **Instruction prefixes:** Qwen3-Embedding reports 1-5% improvement with task-specific instructions. Queries get `"Instruct: ...\nQuery: ..."` prefix; documents get plain text only.
- **Reranking max length 512 (not 384):** Chat template + instruction prefix + query + document requires more tokens. 384 truncates too aggressively with the template overhead. Note: Qwen3-Reranker's official code uses `max_length=8192` (model supports 32K), but 512 is a deliberate compute-saving choice justified by MS MARCO's short passages (avg ~60 words). Make configurable.
- **Pre-tokenized prefix/suffix concatenation (not full-string tokenization):** Matches the official Qwen3-Reranker scoring code. BPE tokenization is context-sensitive at boundary characters — tokenizing the full chat string as one piece produces different token IDs at join points than concatenating pre-tokenized segments. The official code pre-tokenizes prefix and suffix, tokenizes user content separately, then concatenates at the token-ID level.
- **Cache as Arrow files:** Avoids re-downloading and re-processing on subsequent runs. HuggingFace `datasets` handles this natively.
- **Separate collators (not a unified one):** Each task has fundamentally different tokenization logic. Keeping them separate is cleaner.
- **Reranking pos:neg ratio 1:5-7 with pos_weight:** Standard practice for imbalanced reranking data (sentence-transformers blog, F2LLM).

## Scope

**In scope:** MS MARCO download, hard negative preparation, preprocessing for both tasks, chat template formatting, collators, download script

**Out of scope:** BEIR dataset downloading (handled in evaluation ticket), any model or training code, additional training datasets (NQ, NLI — deferred to future work)

## Technical Notes

- The `triplets` subset uses IDs that must be cross-referenced with `corpus` and `queries` subsets to get actual text. Build lookup dicts first.
- The `labeled-list` subset has lists of doc_ids and labels per query. Must be flattened into individual rows.
- The `msmarco-hard-negatives` dataset maps query IDs to negative passage IDs from multiple retrieval systems. Use BM25 negatives as primary, supplemented by dense model negatives.
- EOS token handling: Qwen3-Embedding pools at the `<|endoftext|>` EOS token. The tokenizer must be configured to append EOS. Use `TemplateProcessing` post-processor if needed.
- Embedding queries use left padding (`padding_side='left'`) for compatibility with causal attention and optional flash_attention_2.
- The exact Qwen3-Reranker chat template format is:
  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: {instruction}
  <Query>: {query}
  <Document>: {document}<|im_end|>
  <|im_start|>assistant
  <think>

  </think>

  ```
  The model then predicts "yes" or "no" as the next token. The `<think>\n\n</think>\n\n` is part of Qwen3's thinking format — the model skips actual reasoning and goes directly to yes/no. This must be included for compatibility with Qwen3-Reranker's expected input format.
- The download may take 20-40 minutes depending on network speed (hard negatives dataset is large). Progress bars via `tqdm` are essential.

## Acceptance Criteria

- [ ] `uv run python scripts/download_data.py` completes, creates files in `data/`
- [ ] Embedding dataset has ~500K rows with text columns including 7 hard negatives per query
- [ ] Reranking dataset has ~500K rows with Qwen3-Reranker chat template format and yes/no labels
- [ ] Label distribution is documented (pos:neg ratio ~1:5-7 for reranking)
- [ ] `tests/test_data.py` passes: verifies batch shapes, token ID ranges, attention mask values, label values (yes/no token IDs), sequence length constraints, instruction prefix presence, EOS token appended
- [ ] A DataLoader with each collator produces correctly-shaped batches without errors
- [ ] Chat template output matches Qwen3-Reranker expected format
- [ ] `uv run pytest tests/test_data.py -v` all green
