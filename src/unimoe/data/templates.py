"""Template formatting for Qwen3-Embedding and Qwen3-Reranker.

Query format matches Qwen3-Embedding's config_sentence_transformers.json:
  "Instruct: {instruction}\\nQuery:{query}"  (no space after "Query:")

Reranking format matches Qwen3-Reranker's chat template with yes/no prediction.
"""

RERANKING_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def format_embedding_query(instruction: str, query: str) -> str:
    """Format a query for Qwen3-Embedding.

    Matches the official format from config_sentence_transformers.json:
    "Instruct: {instruction}\\nQuery:{query}" — no space after "Query:".
    """
    return f"Instruct: {instruction}\nQuery:{query}"


def format_reranking_input(instruction: str, query: str, document: str) -> str:
    """Format input for Qwen3-Reranker chat template.

    Returns the full chat string including system message, user content,
    and the assistant's <think>\\n\\n</think>\\n\\n prefix.
    The model then predicts "yes" or "no" as the next token.
    """
    return (
        f"<|im_start|>system\n"
        f"{RERANKING_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<Instruct>: {instruction}\n\n"
        f"<Query>: {query}\n\n"
        f"<Document>: {document}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )


def build_reranking_token_ids(
    tokenizer,
    instruction: str,
    query: str,
    document: str,
    max_len: int = 512,
) -> list[int]:
    """Build reranking input token IDs via pre-tokenized concatenation.

    Tokenizes prefix, user content, and suffix separately, then concatenates
    at the token-ID level.  This matches the official Qwen3-Reranker scoring
    approach and avoids BPE boundary mismatches between training and evaluation.

    Args:
        tokenizer: HuggingFace tokenizer.
        instruction: Task instruction string.
        query: Query text.
        document: Document text.
        max_len: Maximum sequence length (truncates content to fit).

    Returns:
        List of token IDs.
    """
    prefix_text = (
        f"<|im_start|>system\n"
        f"{RERANKING_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
    )
    suffix_text = (
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )
    user_content = (
        f"<Instruct>: {instruction}\n\n"
        f"<Query>: {query}\n\n"
        f"<Document>: {document}"
    )

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    content_ids = tokenizer.encode(user_content, add_special_tokens=False)

    # Truncate content if needed (keep prefix + suffix intact)
    max_content_len = max_len - len(prefix_ids) - len(suffix_ids)
    if len(content_ids) > max_content_len:
        content_ids = content_ids[:max_content_len]

    return prefix_ids + content_ids + suffix_ids


def set_tokenizer_config(tokenizer) -> None:
    """Configure tokenizer for causal attention / flash_attention_2 compatibility.

    Sets padding_side='left' to match Qwen3-Embedding's configuration.
    """
    tokenizer.padding_side = "left"
