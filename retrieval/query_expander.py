from __future__ import annotations

from retrieval.relevance import (
    detect_entity,
    detect_intent,
    NETWORK_FUNCTIONS,
)


def expand_query(query: str) -> str:
    """
    Expand a natural-language question with
    3GPP-oriented retrieval terms.
    """

    intent = detect_intent(query)
    entity = detect_entity(query)

    terms = [query]

    # ---------------------------------------------------------
    # Network-function queries
    # ---------------------------------------------------------

    if entity is not None:

        full_name = NETWORK_FUNCTIONS[
            entity
        ]

        terms.extend(
            [
                entity.upper(),
                full_name,
            ]
        )

        if intent == "role":
            terms.extend(
                [
                    f"{entity.upper()} functionality",
                    f"{entity.upper()} services",
                    f"{entity.upper()} responsibilities",
                    f"{entity.upper()} functions",
                ]
            )

        elif intent == "definition":
            terms.extend(
                [
                    f"{entity.upper()} definition",
                    f"{full_name} definition",
                ]
            )

    # ---------------------------------------------------------
    # PDU Session
    # ---------------------------------------------------------

    if "pdu session" in query.lower():

        if intent == "definition":
            terms.extend(
                [
                    "PDU Session definition",
                    "PDU Session management",
                    "Session Management",
                    "PDU Session attributes",
                    "clause 5.6",
                ]
            )

    # ---------------------------------------------------------
    # Architecture
    # ---------------------------------------------------------

    if intent == "architecture":

        terms.extend(
            [
                "5G System architecture",
                "network functions",
                "reference architecture",
            ]
        )

    return " ".join(terms)