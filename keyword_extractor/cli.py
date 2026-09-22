"""
LLM Keyword Generator - Command Line Interface

Usage:
    python -m keyword_extractor.cli --text "some content" --n 5
    python -m keyword_extractor.cli --file article.txt --n 10 --model gpt-4o
    cat article.txt | python -m keyword_extractor.cli --n 8
    python -m keyword_extractor.cli --list-models
    python -m keyword_extractor.cli --file article.txt --theme "AI and Robotics within healthcare"
"""

import argparse
import json
import sys
from typing import Optional

from .config import DEFAULTS, list_models
from .extractor import extract_keywords
from .llm_client import LLMError
from .prompts import KeywordRestrictions


def _read_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No input provided. Use --text, --file, or pipe content via stdin.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyword-extractor",
        description="Extract keywords from text using an LLM.",
    )
    parser.add_argument("--text", help="Text content to analyze")
    parser.add_argument("--file", help="Path to a text file to analyze")
    parser.add_argument(
        "--n", type=int, default=DEFAULTS["default_keyword_count"],
        help=f"Number of keywords to extract (default: {DEFAULTS['default_keyword_count']})",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Model to use (default: {DEFAULTS['default_model']}). See --list-models.",
    )
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    parser.add_argument(
        "--theme", default=None,
        help=f"Theme guiding keyword selection (default: {DEFAULTS['default_theme']!r})",
    )

    # Restriction knobs
    parser.add_argument("--max-words-per-keyword", type=int, default=3,
                         help="Max words allowed per keyword/phrase (default: 3)")
    parser.add_argument("--case", choices=["lower", "title", "preserve"], default="lower",
                         help="Casing applied to output keywords (default: lower)")
    parser.add_argument("--allow-duplicates", action="store_true",
                         help="Allow near-duplicate/overlapping keywords")
    parser.add_argument("--language", default=None, help="Force output keyword language, e.g. 'English'")
    parser.add_argument("--exclude", action="append", default=[],
                         help="Term to exclude from results (repeatable)")
    parser.add_argument("--must-include", action="append", default=[],
                         help="Topic to prioritize if relevant (repeatable)")
    parser.add_argument("--focus", default=None, help="Domain/context hint to focus extraction on")
    parser.add_argument("--custom-instructions", default=None,
                         help="Raw extra instruction text appended to the prompt")
    parser.add_argument("--system-message", default=None, help="Override the default system message")

    parser.add_argument("--json", action="store_true", help="Print output as JSON instead of plain lines")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for name in list_models():
            print(name)
        return 0

    try:
        content = _read_input(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    restriction_kwargs = dict(
        max_words_per_keyword=args.max_words_per_keyword,
        case=args.case,
        allow_duplicates=args.allow_duplicates,
        language=args.language,
        exclude_terms=args.exclude,
        must_include_terms=args.must_include,
        focus_hint=args.focus,
        custom_instructions=args.custom_instructions,
        system_message=args.system_message,
    )
    if args.theme is not None:
        restriction_kwargs["theme"] = args.theme
    restrictions = KeywordRestrictions(**restriction_kwargs)

    try:
        keywords = extract_keywords(content, n=args.n, model=args.model, restrictions=restrictions)
    except (ValueError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"keywords": keywords}, indent=2))
    else:
        for keyword in keywords:
            print(keyword)

    return 0


if __name__ == "__main__":
    sys.exit(main())
