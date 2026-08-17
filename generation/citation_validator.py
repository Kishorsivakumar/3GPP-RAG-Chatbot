from __future__ import annotations

import re
from typing import Dict, List


CITATION_PATTERN = re.compile(
    r"\[TS\s+23\.501,\s+Section\s+([A-Za-z0-9.]+)\]"
)


def extract_citations(
    answer: str,
) -> List[str]:
    return CITATION_PATTERN.findall(
        answer
    )


def validate_citations(
    answer: str,
    evidence: List[Dict],
) -> Dict:

    citations = extract_citations(
        answer
    )

    available_sections = {
        str(result["section"])
        for result in evidence
    }

    invalid = [
        citation
        for citation in citations
        if citation not in available_sections
    ]

    valid = [
        citation
        for citation in citations
        if citation in available_sections
    ]

    if invalid:
        return {
            "valid": False,
            "reason": "invalid_citation",
            "citations": citations,
            "valid_citations": valid,
            "invalid_citations": invalid,
        }

    if not citations:
        return {
            "valid": False,
            "reason": "no_citations",
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
        }

    return {
        "valid": True,
        "reason": "citations_valid",
        "citations": citations,
        "valid_citations": valid,
        "invalid_citations": [],
    }