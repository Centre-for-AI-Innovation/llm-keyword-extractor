"""
Minimal usage example for the keyword_extractor library.

Run from the project root:
    python examples/example.py
"""

from keyword_extractor import KeywordRestrictions, extract_keywords

TEXT = """
The clinic offers AI-assisted diagnostic imaging for early-stage lung cancer
detection, combining deep learning models with radiologist review. Services
include CE-marked software validation, GDPR-compliant data handling, and
support for clinical trial recruitment across European hospitals.
"""

if __name__ == "__main__":
    # Basic extraction using the default model, theme
    # ("AI and Robotics within healthcare"), and settings.
    keywords = extract_keywords(TEXT, n=6)
    print("Default extraction:")
    for kw in keywords:
        print(f"  - {kw}")

    # Extraction with custom restrictions: single words only, focus hint,
    # a couple of excluded terms, and an overridden theme.
    restrictions = KeywordRestrictions(
        theme="medical regulatory compliance",
        max_words_per_keyword=1,
        case="lower",
        exclude_terms=["clinic"],
    )
    keywords_restricted = extract_keywords(TEXT, n=5, restrictions=restrictions)
    print("\nRestricted extraction (single words, compliance theme):")
    for kw in keywords_restricted:
        print(f"  - {kw}")
