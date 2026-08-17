from __future__ import annotations

import re
from typing import Dict


# ============================================================
# TERM DEFINITIONS
# ============================================================

NETWORK_FUNCTIONS = {
    "amf": "Access and Mobility Management Function",
    "smf": "Session Management Function",
    "upf": "User Plane Function",
    "ausf": "Authentication Server Function",
    "nef": "Network Exposure Function",
    "nrf": "Network Repository Function",
    "nssf": "Network Slice Selection Function",
    "pcf": "Policy Control Function",
    "udm": "Unified Data Management",
    "udr": "Unified Data Repository",
    "nwdaf": "Network Data Analytics Function",
    "chf": "Charging Function",
    "scp": "Service Communication Proxy",
    "sepp": "Security Edge Protection Proxy",
}


# Dedicated functional-description sections in TS 23.501.
NETWORK_FUNCTION_SECTIONS = {
    "amf": "6.2.1",
    "smf": "6.2.2",
    "upf": "6.2.3",
    "pcf": "6.2.4",
    "udm": "6.2.5",
    "nrf": "6.2.6",
}


ARCHITECTURE_PATTERNS = [
    "network functions",
    "network function",
    "reference architecture",
    "5g system architecture",
    "5g architecture",
    "architecture",
    "reference point",
    "interface",
    "interaction between",
]


ROLE_PATTERNS = [
    "role",
    "responsibility",
    "responsibilities",
    "what does",
    "what is responsible",
    "purpose",
]


DEFINITION_PATTERNS = [
    "define",
    "definition",
    "meaning",
    "what is",
]


TABLE_PATTERNS = [
    "table",
    "mapping",
    "attributes",
    "which attributes",
    "what attributes",
    "what information",
    "rules",
    "services specified",
    "standardized mapping",
    "standardised mapping",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(text: str) -> str:
    """
    Normalize whitespace and lowercase text.
    """
    return " ".join(
        text.lower().split()
    )


def tokenize(text: str) -> set[str]:
    """
    Tokenize technical text while retaining terms such as:

        AMF
        S-NSSAI
        5QI
        N11
        non-3GPP
    """

    return set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            normalize(text),
        )
    )


# ============================================================
# QUERY INTENT
# ============================================================

def detect_intent(query: str) -> str:
    """
    Detect the primary query intent.

    Priority:

        table
        architecture
        role
        definition
        general
    """

    q = normalize(query)

    # ========================================================
    # TABLE / STRUCTURED-DATA QUESTIONS
    # ========================================================

    table_patterns = [
        "table",
        "mapping",
        "attributes",
        "which attributes",
        "what attributes",
        "what information",
        "what services",
        "services specified",
        "which services",
        "what rules",
        "which rules",
        "atsss rules",
        "what are the atsss rules",
        "standardized mapping",
        "standardised mapping",
    ]

    if any(
        pattern in q
        for pattern in table_patterns
    ):
        return "table"

    # ========================================================
    # ARCHITECTURE
    # ========================================================

    architecture_patterns = [
        "network functions",
        "network function",
        "reference architecture",
        "5g system architecture",
        "5g architecture",
        "architecture",
        "reference point",
        "interface",
        "interaction between",
    ]

    if any(
        pattern in q
        for pattern in architecture_patterns
    ):
        return "architecture"

    # ========================================================
    # ROLE
    # ========================================================

    role_patterns = [
        "role",
        "responsibility",
        "responsibilities",
        "what does",
        "what is responsible",
        "purpose",
    ]

    if any(
        pattern in q
        for pattern in role_patterns
    ):
        return "role"

    # ========================================================
    # DEFINITION
    # ========================================================

    definition_patterns = [
        "define",
        "definition",
        "meaning",
        "what is",
    ]

    if any(
        pattern in q
        for pattern in definition_patterns
    ):
        return "definition"

    return "general"

# ============================================================
# ENTITY DETECTION
# ============================================================

def detect_entity(
    query: str,
) -> str | None:
    """
    Detect a known 3GPP network function in the query.
    """

    q = normalize(query)

    # Full names first.
    for abbreviation, full_name in NETWORK_FUNCTIONS.items():

        if normalize(full_name) in q:
            return abbreviation

    # Then abbreviations.
    for abbreviation in NETWORK_FUNCTIONS:

        if re.search(
            rf"\b{re.escape(abbreviation)}\b",
            q,
        ):
            return abbreviation

    return None


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    query: str,
    chunk: Dict,
) -> float:
    """
    Calculate lexical overlap between query and chunk.
    """

    query_tokens = tokenize(query)

    chunk_text = " ".join(
        [
            chunk["section_title"],
            chunk["content"],
        ]
    )

    chunk_tokens = tokenize(
        chunk_text
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        .intersection(chunk_tokens)
    )

    return len(overlap) / len(
        query_tokens
    )


# ============================================================
# ENTITY SCORE
# ============================================================

def entity_score(
    query: str,
    chunk: Dict,
) -> float:
    """
    Score how strongly the retrieved chunk relates
    to the network function mentioned in the query.
    """

    entity = detect_entity(query)

    if entity is None:
        return 0.0

    abbreviation = entity

    full_name = NETWORK_FUNCTIONS[
        entity
    ]

    section_title = normalize(
        chunk["section_title"]
    )

    content = normalize(
        chunk["content"]
    )

    score = 0.0

    # Exact abbreviation in section title.
    if re.search(
        rf"\b{re.escape(abbreviation)}\b",
        section_title,
    ):
        score += 0.35

    # Full function name in title.
    if normalize(full_name) in section_title:
        score += 0.50

    # Exact abbreviation in content.
    if re.search(
        rf"\b{re.escape(abbreviation)}\b",
        content,
    ):
        score += 0.10

    # Full name in content.
    if normalize(full_name) in content:
        score += 0.15

    return score


# ============================================================
# ROLE EVIDENCE
# ============================================================

def role_evidence_score(
    query: str,
    chunk: Dict,
) -> float:
    """
    Score evidence that actually describes the role/function
    of the queried network function.
    """

    if detect_intent(query) != "role":
        return 0.0

    entity = detect_entity(query)

    if entity is None:
        return 0.0

    full_name = NETWORK_FUNCTIONS[
        entity
    ]

    section_title = normalize(
        chunk["section_title"]
    )

    content = normalize(
        chunk["content"]
    )

    section_number = (
        str(chunk["section"])
        .strip()
        .lower()
    )

    score = 0.0

    # --------------------------------------------------------
    # Dedicated functional-description section.
    # --------------------------------------------------------

    expected_section = (
        NETWORK_FUNCTION_SECTIONS.get(
            entity
        )
    )

    if (
        expected_section is not None
        and section_number == expected_section
    ):
        score += 0.45

    # --------------------------------------------------------
    # Strong section-level signals.
    # --------------------------------------------------------

    if re.search(
        rf"\b{re.escape(entity)}\b",
        section_title,
    ):
        score += 0.25

    if normalize(full_name) in section_title:
        score += 0.50

    # --------------------------------------------------------
    # Functional language.
    # --------------------------------------------------------

    role_phrases = [
        "functional description",
        "functional descriptions",
        "functionality",
        "provides",
        "shall support",
        "supports",
        "responsible for",
        "performs",
        "used to",
    ]

    for phrase in role_phrases:

        if phrase in content:
            score += 0.06

    # --------------------------------------------------------
    # Specialized sections are useful but not ideal for
    # a broad role question.
    # --------------------------------------------------------

    specialized_terms = [
        "overload",
        "emergency",
        "load balancing",
        "charging",
        "mobility event",
        "satellite",
        "selection",
        "discovery",
    ]

    for term in specialized_terms:

        if term in section_title:
            score -= 0.10

    return score


# ============================================================
# DEFINITION EVIDENCE
# ============================================================

def definition_evidence_score(
    query: str,
    chunk: Dict,
) -> float:

    if detect_intent(query) != "definition":
        return 0.0

    query_text = normalize(query)

    content = normalize(
        chunk["content"]
    )

    section_title = normalize(
        chunk["section_title"]
    )

    section_number = (
        chunk["section"]
        .strip()
        .lower()
    )

    score = 0.0

    # --------------------------------------------------------
    # Entity-specific definition handling.
    # --------------------------------------------------------

    entity = detect_entity(query)

    if entity is not None:

        expected_section = (
            NETWORK_FUNCTION_SECTIONS.get(
                entity
            )
        )

        # Strong preference for the dedicated NF section.
        if (
            expected_section is not None
            and section_number == expected_section
        ):
            score += 0.55

        # Entity in title.
        if re.search(
            rf"\b{re.escape(entity)}\b",
            section_title,
        ):
            score += 0.25

        full_name = NETWORK_FUNCTIONS[
            entity
        ]

        if normalize(full_name) in section_title:
            score += 0.30

    # --------------------------------------------------------
    # Extract the concept being asked about.
    # --------------------------------------------------------

    concept = query_text

    concept = concept.replace(
        "what is",
        "",
    ).strip()

    concept = concept.replace(
        "define",
        "",
    ).strip()

    concept = concept.replace(
        "definition of",
        "",
    ).strip()

    # Remove article.
    if concept.startswith("a "):
        concept = concept[2:]

    if concept.startswith("an "):
        concept = concept[3:]

    # --------------------------------------------------------
    # Exact concept in section title.
    # --------------------------------------------------------

    if concept and concept in section_title:
        score += 0.40

    # --------------------------------------------------------
    # Exact concept in passage.
    # --------------------------------------------------------

    if concept and concept in content:
        score += 0.20

    # --------------------------------------------------------
    # Domain-specific definition language.
    # --------------------------------------------------------

    definition_phrases = [
        "is a",
        "is defined",
        "is referred to as",
        "means",
        "refers to",
        "denotes",
        "is identified as",
    ]

    for phrase in definition_phrases:

        if phrase in content:
            score += 0.05

    # --------------------------------------------------------
    # General Definitions section.
    # --------------------------------------------------------

    if section_number.startswith("3.1"):

        if concept and concept in content:
            score += 0.35

    # --------------------------------------------------------
    # PDU Session-specific preference.
    # --------------------------------------------------------

    if concept == "pdu session":

        if section_number == "5.6.1":
            score += 0.50

        elif section_number.startswith("5.6."):
            score += 0.20

        specialized_terms = [
            "multi-homing",
            "emergency",
            "redundant",
        ]

        if any(
            term in section_title
            for term in specialized_terms
        ):
            score -= 0.15

    return score


# ============================================================
# TABLE EVIDENCE
# ============================================================

def table_evidence_score(
    query: str,
    chunk: Dict,
) -> float:
    """
    Score evidence for table-oriented questions.
    """

    if detect_intent(query) != "table":
        return 0.0

    section_title = normalize(
        chunk["section_title"]
    )

    content_type = (
        chunk.get(
            "content_type",
            "text",
        )
        .lower()
    )

    score = 0.0

    # Actual table chunk.
    if content_type == "table":
        score += 0.40

    # Table-oriented section names.
    table_terms = [
        "mapping",
        "rules",
        "services",
        "attributes",
        "overview",
        "general",
    ]

    for term in table_terms:

        if term in section_title:
            score += 0.08

    return score


# ============================================================
# INTENT SCORE
# ============================================================

def intent_bonus(
    query: str,
    chunk: Dict,
) -> float:
    """
    Apply lightweight 3GPP structural relevance.
    """

    intent = detect_intent(query)

    section_number = (
        chunk["section"]
        .strip()
        .lower()
    )

    section_title = normalize(
        chunk["section_title"]
    )

    score = 0.0

    # ========================================================
    # DEFINITION
    # ========================================================

    if intent == "definition":

        if section_number.startswith("3.1"):
            score += 0.40

        if "definition" in section_title:
            score += 0.20

        entity = detect_entity(query)

        if entity is not None:

            expected_section = (
                NETWORK_FUNCTION_SECTIONS.get(
                    entity
                )
            )

            if (
                expected_section is not None
                and section_number
                == expected_section
            ):
                score += 0.35

    # ========================================================
    # ROLE
    # ========================================================

    elif intent == "role":

        # Definitions are usually not the best source
        # for a role/function question.
        if section_number.startswith("3.1"):
            score -= 0.60

        if section_number.startswith("3.2"):
            score -= 0.50

        entity = detect_entity(query)

        if entity is not None:

            expected_section = (
                NETWORK_FUNCTION_SECTIONS.get(
                    entity
                )
            )

            if (
                expected_section is not None
                and section_number
                == expected_section
            ):
                score += 0.40

        if "functionality" in section_title:
            score += 0.30

        if "function" in section_title:
            score += 0.20

        if "service" in section_title:
            score += 0.20

        if "general" in section_title:
            score += 0.05

        if "architecture" in section_title:
            score += 0.10

    # ========================================================
    # TABLE
    # ========================================================

    elif intent == "table":

        # Actual technical table.
        if chunk.get(
            "content_type",
            ""
        ).lower() == "table":
            score += 0.35

        # Exact/strong table-oriented titles.
        if "mapping" in section_title:
            score += 0.30

        if "rules" in section_title:
            score += 0.30

        if "services" in section_title:
            score += 0.25

        if "attributes" in section_title:
            score += 0.25

        # Tables embedded in General / Overview sections.
        if "general" in section_title:
            score += 0.10

        if "overview" in section_title:
            score += 0.10

    # ========================================================
    # ARCHITECTURE
    # ========================================================

    elif intent == "architecture":

        if "network functions" in section_title:
            score += 0.45

        if "architecture" in section_title:
            score += 0.30

        if "reference point" in section_title:
            score += 0.20

        if "interface" in section_title:
            score += 0.15

        if section_number.startswith("4."):
            score += 0.10

    # ========================================================
    # COMMON PENALTIES
    # ========================================================

    if "abbreviations" in section_title:
        score -= 0.40

    if "change history" in section_title:
        score -= 1.0

    return score


# ============================================================
# FINAL RELEVANCE SCORE
# ============================================================

def relevance_score(
    query: str,
    chunk: Dict,
    reranker_score: float,
) -> float:
    """
    Combine:

        - Cross-encoder relevance
        - Lexical overlap
        - Entity relevance
        - Intent relevance
        - Role evidence
        - Definition evidence
        - Table evidence
    """

    lexical = keyword_overlap(
        query,
        chunk,
    )

    entity = entity_score(
        query,
        chunk,
    )

    intent = intent_bonus(
        query,
        chunk,
    )

    role_evidence = role_evidence_score(
        query,
        chunk,
    )

    definition_evidence = (
        definition_evidence_score(
            query,
            chunk,
        )
    )

    table_evidence = (
        table_evidence_score(
            query,
            chunk,
        )
    )

    return (
        reranker_score
        + lexical
        + entity
        + intent
        + role_evidence
        + definition_evidence
        + table_evidence
    )