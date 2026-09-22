"""
LLM Keyword Generator - Core extraction logic

Public entry point: extract_keywords(text, n, model, restrictions)
"""

from typing import List, Optional

from .config import DEFAULTS
from .llm_client import LLMClient, LLMError
from .prompts import KeywordRestrictions, build_keyword_extraction_prompt

EXTRACTION_MAX_TOKENS = 512


def _normalize_keywords(raw_keywords: List, n: int, restrictions: KeywordRestrictions) -> List[str]:
    """Clean, dedupe, case-normalize and trim the model's keyword list."""
    seen = set()
    normalized: List[str] = []

    for item in raw_keywords:
        keyword = str(item).strip()
        if not keyword:
            continue

        if restrictions.case == "lower":
            keyword = keyword.lower()
        elif restrictions.case == "title":
            keyword = keyword.title()

        dedupe_key = keyword.lower()
        if not restrictions.allow_duplicates:
            if dedupe_key in seen:
                continue

        excluded = {term.lower() for term in restrictions.exclude_terms}
        if dedupe_key in excluded:
            continue

        seen.add(dedupe_key)
        normalized.append(keyword)

        if len(normalized) >= n:
            break

    return normalized


def extract_keywords(
    text: str,
    n: int = None,
    model: Optional[str] = None,
    restrictions: Optional[KeywordRestrictions] = None,
    client: Optional[LLMClient] = None,
) -> List[str]:
    """Extract the top `n` keywords from `text` using an LLM.

    Args:
        text: Source content to analyze.
        n: Desired number of keywords (defaults to DEFAULT_KEYWORD_COUNT).
        model: Registered model name (see config.list_models()). Ignored if
            `client` is provided.
        restrictions: Optional KeywordRestrictions controlling prompt
            behavior (phrase length, case, language, exclusions, etc.).
        client: Optional pre-built LLMClient to reuse across calls (e.g. to
            avoid re-initializing per request in a server context).

    Returns:
        A list of up to `n` keyword strings, ranked most to least relevant.

    Raises:
        ValueError: If `text` is empty or `n` is not positive.
        LLMError: If the LLM call fails or returns an unusable response.
    """
    if not text or not str(text).strip():
        raise ValueError("text must be non-empty")

    n = n if n is not None else DEFAULTS["default_keyword_count"]
    if n <= 0:
        raise ValueError("n must be a positive integer")

    restrictions = restrictions or KeywordRestrictions()
    llm_client = client or LLMClient(model)

    prompt = build_keyword_extraction_prompt(text, n, restrictions)
    response = llm_client.complete_json(
        prompt,
        system_message=restrictions.system_message_text(),
        max_tokens=EXTRACTION_MAX_TOKENS,
    )

    if not isinstance(response, dict) or "keywords" not in response:
        raise LLMError(f"Expected a JSON object with a 'keywords' array, got: {response!r}")

    raw_keywords = response["keywords"]
    if not isinstance(raw_keywords, list):
        raise LLMError(f"Expected 'keywords' to be a list, got {type(raw_keywords)}: {raw_keywords!r}")

    return _normalize_keywords(raw_keywords, n, restrictions)
