"""
LLM Keyword Generator - Configuration

Model registry and defaults, resolved from environment variables at call time
(not at import time), so the library stays testable and side-effect free.

Connection is intentionally generic and shared across all models: one
API_KEY + ENDPOINT talk to a single OpenAI-compatible chat completions API
(OpenAI itself, Azure OpenAI's OpenAI-compatible surface, or any compatible
gateway/proxy). "Model selection" picks which deployment/model id is sent
in each request and which tuning defaults (max tokens, JSON mode support,
etc.) apply.
"""

import os
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can be set any other way.
    pass


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a string environment variable."""
    return os.getenv(key, default)


def env_int(key: str, default: int = 0) -> int:
    """Get an integer environment variable."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float = 0.0) -> float:
    """Get a float environment variable."""
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# ============================================================================
# Model registry
# ============================================================================
#
# Lightweight presets: a friendly name -> the default deployment/model id to
# send to the API, plus tuning defaults. All entries share the same
# API_KEY/ENDPOINT connection (see get_model_config). A model name that is
# not in this registry is still allowed — the name itself is used as the
# deployment id, so any custom deployment can be selected without editing
# this file.
#
MODEL_REGISTRY: Dict[str, dict] = {
    "gpt-4o-mini": {
        "deployment_default": "gpt-4o-mini",
        "default_max_tokens": 512,
    },
    "gpt-4o": {
        "deployment_default": "gpt-4o",
        "default_max_tokens": 512,
    },
    "gpt-5.4": {
        "deployment_default": "gpt-5.4",
        "default_max_tokens": 1024,
        # This model family rejects `max_tokens` and requires
        # `max_completion_tokens` instead.
        "requires_completion_tokens": True,
    },
    "mistral-large": {
        "deployment_default": "mistral-large-latest",
        "default_max_tokens": 512,
    },
}

DEFAULTS = {
    "default_model": env("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
    "default_keyword_count": env_int("DEFAULT_KEYWORD_COUNT", 10),
    "default_theme": env("DEFAULT_THEME", "AI and Robotics within healthcare"),
    "request_timeout": env_float("LLM_REQUEST_TIMEOUT", 60.0),
    "max_retries": env_int("LLM_MAX_RETRIES", 2),
}


def list_models() -> List[str]:
    """Return the names of the built-in model presets.

    Any other model/deployment name is also accepted by `get_model_config`;
    this list is just a set of convenience suggestions.
    """
    return sorted(MODEL_REGISTRY.keys())


def get_model_config(model_name: Optional[str] = None) -> dict:
    """Resolve a fully-populated model configuration from the shared
    connection env vars + optional registry preset.

    Args:
        model_name: Preset name from `list_models()`, or any arbitrary
            deployment/model id. Defaults to DEFAULT_LLM_MODEL.

    Returns:
        Dict with resolved name, deployment, api_key, endpoint, and tuning
        defaults.

    Raises:
        ValueError: If the required API_KEY environment variable is missing.
    """
    name = model_name or DEFAULTS["default_model"]
    spec = MODEL_REGISTRY.get(name, {})

    api_key = env("API_KEY")
    if not api_key:
        raise ValueError("Environment variable 'API_KEY' is not set")

    endpoint = env("ENDPOINT")

    # DEPLOYMENT_NAME is a global override: if set, it is used regardless of
    # which model preset is selected (typical when a single deployment is
    # configured behind the shared endpoint). Otherwise fall back to the
    # preset's default deployment id, or the model name itself.
    deployment = env("DEPLOYMENT_NAME") or spec.get("deployment_default") or name

    return {
        "name": name,
        "deployment": deployment,
        "api_key": api_key,
        "endpoint": endpoint,
        "request_timeout": env_float("LLM_REQUEST_TIMEOUT", DEFAULTS["request_timeout"]),
        "max_retries": env_int("LLM_MAX_RETRIES", DEFAULTS["max_retries"]),
        "supports_json_mode": spec.get("supports_json_mode", True),
        "supports_system_message": spec.get("supports_system_message", True),
        "default_max_tokens": spec.get("default_max_tokens", 512),
        "requires_completion_tokens": spec.get("requires_completion_tokens", False),
    }
