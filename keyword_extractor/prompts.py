"""
LLM Keyword Generator - Prompt construction

Builds a model-agnostic keyword extraction prompt from free text, a target
keyword count, and a set of configurable restrictions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .config import DEFAULTS


DEFAULT_SYSTEM_MESSAGE = (
    "You are a precise keyword extraction assistant. You read arbitrary text "
    "and identify the most salient, specific keywords or key phrases. "
    "Always respond with valid JSON only, no explanatory text."
)

MAX_TEXT_CHARS_DEFAULT = 12000


@dataclass
class KeywordRestrictions:
    """Configurable constraints applied to keyword extraction.

    All fields are optional knobs; leave them at their defaults for a
    generic, unrestricted extraction.
    """

    # Theme guiding keyword selection, e.g. "AI and Robotics within healthcare".
    # Keywords related to the theme are prioritized, but extraction still
    # only draws on terms clearly supported by the source text.
    theme: str = field(default_factory=lambda: DEFAULTS["default_theme"])

    # Formatting restrictions
    max_words_per_keyword: int = 3        # e.g. 1 = single words only, 3 = short phrases
    case: str = "lower"                   # "lower" | "title" | "preserve"
    allow_duplicates: bool = False        # allow near-duplicate keywords across the list

    # Content restrictions
    language: Optional[str] = None        # e.g. "English" — instruct output language
    exclude_terms: List[str] = field(default_factory=list)   # terms/phrases to never output
    focus_hint: Optional[str] = None      # extra domain/context hint, e.g. "medical technology"
    must_include_terms: List[str] = field(default_factory=list)  # terms to prioritize if relevant

    # Escape hatch for full customization without touching code
    custom_instructions: Optional[str] = None   # raw text appended verbatim to the prompt
    system_message: Optional[str] = None        # override the default system message

    def system_message_text(self) -> str:
        return self.system_message or DEFAULT_SYSTEM_MESSAGE


def _truncate(text: str, max_chars: int = MAX_TEXT_CHARS_DEFAULT) -> str:
    """Truncate text for prompt inclusion, cutting at a word boundary."""
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _format_restrictions(n: int, restrictions: KeywordRestrictions) -> str:
    """Render restrictions as human-readable instruction lines."""
    lines: List[str] = []

    if restrictions.max_words_per_keyword <= 1:
        lines.append("- Each keyword must be a single word.")
    else:
        lines.append(
            f"- Each keyword must be at most {restrictions.max_words_per_keyword} words."
        )

    case_instruction = {
        "lower": "Use lowercase for all keywords.",
        "title": "Use Title Case for all keywords.",
        "preserve": "Preserve the original casing used in the source text.",
    }.get(restrictions.case, "Use lowercase for all keywords.")
    lines.append(f"- {case_instruction}")

    if not restrictions.allow_duplicates:
        lines.append("- Do not repeat the same concept twice; each keyword must be distinct.")

    if restrictions.language:
        lines.append(f"- Output keywords in {restrictions.language}, even if the source text is in another language.")

    if restrictions.exclude_terms:
        excluded = ", ".join(restrictions.exclude_terms)
        lines.append(f"- Never output any of these terms or close variants: {excluded}.")

    if restrictions.must_include_terms:
        preferred = ", ".join(restrictions.must_include_terms)
        lines.append(
            f"- If clearly relevant to the text, prioritize covering these topics: {preferred}."
        )

    if restrictions.focus_hint:
        lines.append(f"- Focus especially on aspects related to: {restrictions.focus_hint}.")

    lines.append(f"- Return exactly {n} keyword(s), ranked from most to least relevant.")
    lines.append("- Only extract keywords clearly supported by the text; do not invent unrelated terms.")

    if restrictions.custom_instructions:
        lines.append(f"- {restrictions.custom_instructions}")

    return "\n".join(lines)


KEYWORD_EXTRACTION_PROMPT = """Analyze the following text and extract the {n} most relevant keywords or key phrases.

THEME: {theme}
Use this theme to guide which keywords matter most: prioritize terms connected
to the theme when they are present in the text. If the text is unrelated to
the theme, ignore the theme and extract the most relevant general keywords
instead — never fabricate theme-related terms that are not supported by the
text.

TEXT:
\"\"\"
{content}
\"\"\"

INSTRUCTIONS:
{restrictions}

OUTPUT FORMAT:
Return ONLY a JSON object in this exact structure:
{{
  "keywords": ["keyword1", "keyword2", "..."]
}}

Do not include any explanatory text, markdown formatting, or code fences — only the JSON object."""


def build_keyword_extraction_prompt(
    content: str,
    n: int,
    restrictions: Optional[KeywordRestrictions] = None,
    max_text_chars: int = MAX_TEXT_CHARS_DEFAULT,
) -> str:
    """Build the user prompt for keyword extraction.

    Args:
        content: The source text to extract keywords from.
        n: Desired number of keywords.
        restrictions: Optional KeywordRestrictions to shape the output.
        max_text_chars: Truncate very long input text to this many characters.

    Returns:
        A fully-formatted prompt string ready to send to an LLM.
    """
    restrictions = restrictions or KeywordRestrictions()
    return KEYWORD_EXTRACTION_PROMPT.format(
        n=n,
        theme=restrictions.theme,
        content=_truncate(content, max_text_chars),
        restrictions=_format_restrictions(n, restrictions),
    )
