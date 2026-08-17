from __future__ import annotations

import json
import os
from typing import Dict, List

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Gemini client that returns structured grounded claims.
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = (
            model
            or os.getenv(
                "RAG_LLM_MODEL",
                "gemini-3.5-flash",
            )
        )

    @staticmethod
    def build_context(
        results: List[Dict],
    ) -> str:
        blocks = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            blocks.append(
                f"""
SOURCE {index}
Specification: {result['specification']}
Version: {result['version']}
Release: {result['release']}
Section: {result['section']}
Section title: {result['section_title']}
Content type: {result['content_type']}

{result['content']}
""".strip()
            )

        return "\n\n".join(blocks)

    def answer(
        self,
        query: str,
        evidence: List[Dict],
    ) -> Dict:

        context = self.build_context(
            evidence
        )

        prompt = f"""
You are a conservative 3GPP standards assistant.

Use ONLY the supplied 3GPP evidence.

Return a JSON object with exactly this structure:

{{
  "answer": "short answer",
  "claims": [
    {{
      "text": "one factual claim",
      "section": "4.2.2"
    }}
  ]
}}

Rules:

1. Every factual statement must be a separate claim.
2. Each claim must have exactly one section.
3. The section MUST exist in the supplied evidence.
4. Never invent section numbers.
5. Do not use outside knowledge.
6. Do not combine independent facts into one claim.
7. Keep the answer concise.
8. Do not use Markdown.
9. Return ONLY JSON.
10. If evidence is insufficient, return:

{{
  "answer": "I do not have sufficient evidence in the provided 3GPP documentation to answer this question.",
  "claims": []
}}

QUESTION:
{query}

SUPPLIED 3GPP EVIDENCE:
{context}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        text = (
            response.text
            if response.text
            else ""
        ).strip()

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        try:
            data = json.loads(text)

        except json.JSONDecodeError as exc:
            print("\n--- RAW GEMINI RESPONSE ---")
            print(text)
            print("--- END RAW RESPONSE ---\n")

            raise RuntimeError(
                "Gemini returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Gemini response must be a JSON object."
            )

        if "answer" not in data:
            raise RuntimeError(
                "Gemini JSON is missing 'answer'."
            )

        if "claims" not in data:
            raise RuntimeError(
                "Gemini JSON is missing 'claims'."
            )

        if not isinstance(
            data["claims"],
            list,
        ):
            raise RuntimeError(
                "Gemini 'claims' must be a list."
            )

        return data