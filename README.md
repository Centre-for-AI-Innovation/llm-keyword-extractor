# llm-keyword-generator

A lightweight, standalone keyword extraction module. It accepts free-form
text content and a desired number of keywords (`n`), sends a configurable
prompt to an LLM, and returns the extracted keywords.

This is a simplified, independent extraction of the keyword extraction
functionality from `tef-service-graph` (`llm_extractor.py`, `llm_client.py`,
`prompts.py`, `config.py`). All service-graph-specific concerns — the
SQLite database, service-offering/domain enrichment, semantic
neighbor-retrieval, extraction-progress persistence, and needs extraction —
have been removed. What remains is a small, self-contained library:
input text + n → keywords.

## Features

- **Theme-guided extraction** — a configurable theme (default: "AI and
  Robotics within healthcare") steers which keywords are prioritized.
- **Model selection** — pick from a small registry of model presets, or
  pass any custom deployment/model id, all via one shared connection.
- **Configurable prompt & restrictions** — control the theme, phrase length,
  casing, output language, excluded terms, prioritized topics, a focus
  hint, and free-form custom instructions, all without touching code.
- **No external database or service-specific data model** — just text in,
  keywords out.
- **CLI and library usage.**

## Install

```bash
pip install -r requirements.txt
```

Only `openai` and `python-dotenv` are required — the library speaks to any
OpenAI-compatible chat completions API (OpenAI, Azure OpenAI's
OpenAI-compatible surface, or a compatible gateway/proxy).

## Configure

Copy `.env.example` to `.env` and fill in your connection details:

```bash
cp .env.example .env
```

The connection is a single shared `API_KEY` + `ENDPOINT` (+ optional
`DEPLOYMENT_NAME`) used for whichever model you select — there are no
per-model credentials to manage. At minimum, set `API_KEY`.

## CLI usage

```bash
# Extract 5 keywords from inline text
python -m keyword_extractor.cli --text "Your article content here..." --n 5

# From a file
python -m keyword_extractor.cli --file article.txt --n 10 --model gpt-4o

# From stdin
cat article.txt | python -m keyword_extractor.cli --n 8

# List available models
python -m keyword_extractor.cli --list-models

# Apply a custom theme and restrictions
python -m keyword_extractor.cli --file article.txt --n 5 \
    --theme "surgical robotics" \
    --max-words-per-keyword 1 \
    --case lower \
    --exclude "the" --exclude "company" \
    --focus "regulatory compliance" \
    --json
```

## Library usage

```python
from keyword_extractor import extract_keywords, KeywordRestrictions

text = "Your article or document content..."

# Basic usage with defaults
keywords = extract_keywords(text, n=10)

# With a custom theme, model selection, and restrictions
restrictions = KeywordRestrictions(
    theme="AI and Robotics within healthcare",  # this is also the default
    max_words_per_keyword=2,
    case="lower",
    language="English",
    exclude_terms=["example", "company"],
    must_include_terms=["compliance", "certification"],
    focus_hint="medical device regulation",
    custom_instructions="Prefer technical terms over generic business language.",
)

keywords = extract_keywords(text, n=8, model="gpt-4o", restrictions=restrictions)
print(keywords)
```

See `examples/example.py` for a runnable script.

## Configuring models

All models share a single connection, configured once via `.env`:

| Variable          | Purpose                                                        |
|-------------------|-----------------------------------------------------------------|
| `API_KEY`         | Required. API key for the chat completions endpoint.            |
| `ENDPOINT`        | Optional. Base URL, for Azure OpenAI or any OpenAI-compatible gateway. Omit to use the public OpenAI API. |
| `DEPLOYMENT_NAME` | Optional. Overrides the deployment/model id sent to the API, regardless of which `--model` preset is selected — useful when your endpoint only exposes one deployment. |
| `DEFAULT_LLM_MODEL` | Which model preset to use when `--model` / `model=` is not specified. |

"Model selection" (`--model` / `model=`) picks a preset from
`keyword_extractor/config.py`'s `MODEL_REGISTRY`, which sets the deployment
id to call (unless overridden by `DEPLOYMENT_NAME`) and tuning defaults like
max tokens. Built-in presets: `gpt-4o-mini` (default), `gpt-4o`, `gpt-5.4`,
`mistral-large`. Any other string is also accepted as a model name — it is
used directly as the deployment id, so custom/self-hosted deployments work
without editing code.

To add a named preset (e.g. to set custom tuning defaults), add an entry to
`MODEL_REGISTRY`:

```python
MODEL_REGISTRY["my-custom-model"] = {
    "deployment_default": "my-deployment-id",
    "default_max_tokens": 512,
}
```

## Configuring the prompt and restrictions

`KeywordRestrictions` (in `keyword_extractor/prompts.py`) is the single
place to tune extraction behavior:

- `theme` — guides which keywords are prioritized (default: `"AI and
  Robotics within healthcare"`, overridable via the `DEFAULT_THEME` env var
  or per-call). The model is instructed to favor theme-relevant terms when
  present, but to fall back to general extraction for unrelated text rather
  than fabricating theme terms.
- `max_words_per_keyword` — cap phrase length (1 = single words only).
- `case` — `"lower"`, `"title"`, or `"preserve"`.
- `allow_duplicates` — allow overlapping/near-duplicate keywords.
- `language` — force output language regardless of source text language.
- `exclude_terms` — list of terms/phrases that must never appear in results.
- `must_include_terms` — topics to prioritize when relevant to the text.
- `focus_hint` — a domain/context hint to bias extraction (e.g. "medical
  regulatory compliance").
- `custom_instructions` — raw text appended verbatim to the prompt for full
  customization beyond the structured knobs above.
- `system_message` — override the default system message entirely.

The base prompt template itself lives in
`build_keyword_extraction_prompt()` / `KEYWORD_EXTRACTION_PROMPT` in
`keyword_extractor/prompts.py` and can be edited directly for structural
changes (e.g. changing the JSON output schema).

## Project layout

```
keyword_extractor/
  __init__.py       # public API surface
  config.py         # model registry + shared connection resolution
  prompts.py         # prompt template, theme, and KeywordRestrictions
  llm_client.py      # LLM client for any OpenAI-compatible endpoint
  extractor.py       # extract_keywords(text, n, model, restrictions)
  cli.py             # command-line interface
examples/
  example.py         # minimal usage example
tests/
  test_extractor.py  # unit tests using a fake LLM client (no network calls)
```

## Testing

```bash
python -m unittest discover -s tests -v
```

Tests exercise the normalization/dedup/restriction logic in `extractor.py`
using a fake LLM client, so they run without any API keys or network access.
