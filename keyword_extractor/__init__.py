"""
llm-keyword-generator

A lightweight, standalone keyword extraction module powered by an LLM.
Extracted and simplified from tef-service-graph's keyword extraction
pipeline, with the service-graph-specific data model, database, and
semantic-retrieval logic removed.
"""

from .config import DEFAULTS, MODEL_REGISTRY, get_model_config, list_models
from .extractor import extract_keywords
from .llm_client import LLMClient, LLMError
from .prompts import KeywordRestrictions, build_keyword_extraction_prompt

__all__ = [
    "extract_keywords",
    "LLMClient",
    "LLMError",
    "KeywordRestrictions",
    "build_keyword_extraction_prompt",
    "get_model_config",
    "list_models",
    "MODEL_REGISTRY",
    "DEFAULTS",
]

__version__ = "0.1.0"
