"""
Unit tests for keyword_extractor.extractor.

These tests use a fake LLMClient so no network calls or API keys are needed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keyword_extractor.extractor import extract_keywords
from keyword_extractor.llm_client import LLMError
from keyword_extractor.prompts import KeywordRestrictions


class FakeLLMClient:
    """Stand-in for LLMClient that returns a canned response."""

    def __init__(self, response):
        self._response = response

    def complete_json(self, prompt, system_message=None, max_tokens=None):
        return self._response


class ExtractKeywordsTests(unittest.TestCase):
    def test_basic_extraction_returns_requested_count(self):
        client = FakeLLMClient({"keywords": ["Alpha", "Beta", "Gamma", "Delta"]})
        result = extract_keywords("some text", n=3, client=client)
        self.assertEqual(result, ["alpha", "beta", "gamma"])

    def test_deduplicates_case_insensitively(self):
        client = FakeLLMClient({"keywords": ["Data", "data", "DATA", "Analytics"]})
        result = extract_keywords("some text", n=5, client=client)
        self.assertEqual(result, ["data", "analytics"])

    def test_allow_duplicates_keeps_repeats(self):
        client = FakeLLMClient({"keywords": ["data", "data", "analytics"]})
        restrictions = KeywordRestrictions(allow_duplicates=True)
        result = extract_keywords("some text", n=3, client=client, restrictions=restrictions)
        self.assertEqual(result, ["data", "data", "analytics"])

    def test_exclude_terms_are_filtered(self):
        client = FakeLLMClient({"keywords": ["data", "analytics", "banned"]})
        restrictions = KeywordRestrictions(exclude_terms=["banned"])
        result = extract_keywords("some text", n=3, client=client, restrictions=restrictions)
        self.assertEqual(result, ["data", "analytics"])

    def test_case_title(self):
        client = FakeLLMClient({"keywords": ["data science"]})
        restrictions = KeywordRestrictions(case="title")
        result = extract_keywords("some text", n=1, client=client, restrictions=restrictions)
        self.assertEqual(result, ["Data Science"])

    def test_missing_keywords_field_raises(self):
        client = FakeLLMClient({"not_keywords": []})
        with self.assertRaises(LLMError):
            extract_keywords("some text", n=3, client=client)

    def test_empty_text_raises(self):
        client = FakeLLMClient({"keywords": []})
        with self.assertRaises(ValueError):
            extract_keywords("   ", n=3, client=client)

    def test_non_positive_n_raises(self):
        client = FakeLLMClient({"keywords": []})
        with self.assertRaises(ValueError):
            extract_keywords("some text", n=0, client=client)


if __name__ == "__main__":
    unittest.main()
