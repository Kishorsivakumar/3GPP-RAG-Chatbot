from __future__ import annotations

import re
from typing import Dict, List


LIST_QUERY_PATTERNS = [
    "what network functions",
    "which network functions",
    "what are the network functions",
    "which functions are part",
    "list the network functions",
    "list all network functions",
    "list all functions",
]


def normalize(text: str) -> str:
    text = text.lower()

    # Normalize punctuation / spacing.
    text = re.sub(r"[^a-z0-9()\- ]+", " ", text)

    return " ".join(text.split())


def is_list_query(query: str) -> bool:
    """
    Completeness is required only when the user explicitly
    requests an exhaustive list.
    """

    q = normalize(query)

    exhaustive_patterns = [
        "list all",
        "list every",
        "enumerate all",
        "enumerate every",
        "give me all",
        "provide all",
        "all network functions",
        "all the network functions",
        "complete list",
        "full list",
        "entire list",
    ]

    return any(
        pattern in q
        for pattern in exhaustive_patterns
    )


def extract_source_items(
    evidence_text: str,
) -> List[str]:
    """
    Extract likely 3GPP network-function/entity names
    from section 4.2.2 content.

    Expected source style:
        - Authentication Server Function (AUSF).
        - Access and Mobility Management Function (AMF).
        ...
    """

    items: List[str] = []

    for raw_line in evidence_text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # Remove leading bullet / whitespace.
        line = re.sub(
            r"^[-•]\s*",
            "",
            line,
        )

        # Remove trailing period.
        line = line.rstrip(".")

        normalized = normalize(line)

        if not normalized:
            continue

        # Network functions.
        if (
            " function " in f" {normalized} "
            or normalized.endswith("function")
            or "(ran)" in normalized
            or normalized.startswith(
                "user equipment"
            )
        ):
            items.append(line)

    return items


def item_present(
    item: str,
    answer_text: str,
) -> bool:
    """
    Check whether the important identity of a source item
    appears in the generated answer.
    """

    source = normalize(item)
    answer = normalize(answer_text)

    # Full normalized match.
    if source in answer:
        return True

    # Extract abbreviation, e.g. (AMF).
    abbreviation_match = re.search(
        r"\(([A-Z0-9][A-Z0-9\-]*)\)",
        item,
    )

    if abbreviation_match:
        abbreviation = (
            abbreviation_match.group(1)
            .lower()
        )

        if re.search(
            rf"\b{re.escape(abbreviation)}\b",
            answer,
        ):
            return True

    # Extract the main name before abbreviation.
    name_without_abbreviation = re.sub(
        r"\s*\([^)]+\)",
        "",
        source,
    ).strip()

    if (
        name_without_abbreviation
        and name_without_abbreviation in answer
    ):
        return True

    return False


def validate_completeness(
    query: str,
    response: Dict,
    evidence: List[Dict],
) -> Dict:
    """
    Validate completeness for list-style questions.

    Currently the authoritative source for the network-function
    list is Section 4.2.2.
    """

    if not is_list_query(query):
        return {
            "required": False,
            "valid": True,
            "reason": "not_a_list_query",
            "expected_items": 0,
            "covered_items": 0,
            "coverage": 1.0,
            "missing_items": [],
        }

    authoritative = [
        item
        for item in evidence
        if str(item.get("section")) == "4.2.2"
    ]

    if not authoritative:
        return {
            "required": True,
            "valid": False,
            "reason": "authoritative_section_missing",
            "expected_items": 0,
            "covered_items": 0,
            "coverage": 0.0,
            "missing_items": [],
        }

    source_text = "\n".join(
        item.get("content", "")
        for item in authoritative
    )

    source_items = extract_source_items(
        source_text
    )

    claims = response.get(
        "claims",
        [],
    )

    answer_text = " ".join(
        claim.get("text", "")
        for claim in claims
        if isinstance(claim, dict)
    )

    missing_items = [
        item
        for item in source_items
        if not item_present(
            item,
            answer_text,
        )
    ]

    expected_count = len(
        source_items
    )

    covered_count = (
        expected_count
        - len(missing_items)
    )

    coverage = (
        covered_count / expected_count
        if expected_count
        else 0.0
    )

    # Strict for explicit list questions.
    valid = (
        expected_count > 0
        and coverage >= 0.95
    )

    return {
        "required": True,
        "valid": valid,
        "reason": (
            "complete"
            if valid
            else "missing_items"
        ),
        "expected_items": expected_count,
        "covered_items": covered_count,
        "coverage": round(
            coverage,
            3,
        ),
        "missing_items": missing_items,
    }