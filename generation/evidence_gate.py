from __future__ import annotations

import re
from typing import Dict, List

from retrieval.relevance import (
    detect_entity,
    detect_intent,
)


# ============================================================
# Dedicated functional-description sections
# ============================================================

NETWORK_FUNCTION_SECTIONS = {
    "amf": "6.2.1",
    "smf": "6.2.2",
    "upf": "6.2.3",
    "pcf": "6.2.4",
    "udm": "6.2.5",
    "nrf": "6.2.6",
}


class EvidenceGate:
    """
    Conservative gate for grounded 3GPP answers.

    Intent-specific behavior:

        definition:
            One strong and relevant section can be enough.

        table:
            One strong and relevant section can be enough.

        entity-specific role:
            One dedicated NF section can be enough.

        architecture/general:
            Multiple strong supporting sections are required.
    """

    def __init__(
        self,
        min_results: int = 2,
        min_score: float = 1.5,
        min_entity_matches: int = 1,
        min_evidence_quality: float = 0.55,
    ):
        self.min_results = min_results
        self.min_score = min_score
        self.min_entity_matches = min_entity_matches
        self.min_evidence_quality = (
            min_evidence_quality
        )

    # ========================================================
    # CONTENT CHECK
    # ========================================================

    @staticmethod
    def _has_content(
        result: Dict,
    ) -> bool:
        """
        Return True when a retrieved result has
        non-empty content.
        """

        return bool(
            result.get(
                "content",
                "",
            ).strip()
        )

    # ========================================================
    # ENTITY CHECK
    # ========================================================

    @staticmethod
    def _entity_present(
        query: str,
        result: Dict,
    ) -> bool:
        """
        Check whether the network-function entity
        requested by the query is present in the
        retrieved result.
        """

        entity = detect_entity(query)

        # No explicit entity in the question.
        if entity is None:
            return True

        content = str(
            result.get(
                "content",
                "",
            )
        ).lower()

        section_title = str(
            result.get(
                "section_title",
                "",
            )
        ).lower()

        return (
            re.search(
                rf"\b{re.escape(entity)}\b",
                content,
            )
            is not None
            or re.search(
                rf"\b{re.escape(entity)}\b",
                section_title,
            )
            is not None
        )

    # ========================================================
    # SECTION QUALITY
    # ========================================================

    @staticmethod
    def _section_quality(
        query: str,
        result: Dict,
    ) -> float:
        """
        Estimate how appropriate the retrieved section is
        for the query intent.
        """

        intent = detect_intent(query)

        section = str(
            result.get(
                "section",
                "",
            )
        ).strip().lower()

        title = str(
            result.get(
                "section_title",
                "",
            )
        ).strip().lower()

        content_type = str(
            result.get(
                "content_type",
                "text",
            )
        ).strip().lower()

        # Neutral starting point.
        score = 0.50

        # ====================================================
        # DEFINITION
        # ====================================================

        if intent == "definition":

            entity = detect_entity(query)

            # Formal definitions.
            if section.startswith("3.1"):
                score += 0.30

            if "definition" in title:
                score += 0.20

            # Overview sections.
            if "overview" in title:
                score += 0.10

            # Core PDU Session section.
            if section == "5.6.1":
                score += 0.20

            # Dedicated NF sections.
            if entity is not None:

                expected_section = (
                    NETWORK_FUNCTION_SECTIONS.get(
                        entity
                    )
                )

                if (
                    expected_section is not None
                    and section == expected_section
                ):
                    score += 0.40

                # Entity appears in title.
                if re.search(
                    rf"\b{re.escape(entity)}\b",
                    title,
                ):
                    score += 0.20

        # ====================================================
        # ROLE
        # ====================================================

        elif intent == "role":

            entity = detect_entity(query)

            # General definitions are less useful
            # for entity-role questions.
            if section.startswith("3.1"):
                score -= 0.35

            if section.startswith("3.2"):
                score -= 0.35

            # Dedicated NF section.
            if entity is not None:

                expected_section = (
                    NETWORK_FUNCTION_SECTIONS.get(
                        entity
                    )
                )

                if (
                    expected_section is not None
                    and section == expected_section
                ):
                    score += 0.35

                # Entity in section title.
                if re.search(
                    rf"\b{re.escape(entity)}\b",
                    title,
                ):
                    score += 0.30

            # Functional/service signals.
            if "functionality" in title:
                score += 0.25

            if "function" in title:
                score += 0.20

            if "service" in title:
                score += 0.15

            if "general" in title:
                score += 0.05

            if "architecture" in title:
                score += 0.10

            # Specialized sections are weaker for broad
            # role questions.
            specialized = [
                "overload",
                "emergency",
                "satellite",
                "charging",
                "load balancing",
                "mobility event",
                "selection",
                "discovery",
            ]

            if any(
                term in title
                for term in specialized
            ):
                score -= 0.15

        # ====================================================
        # TABLE
        # ====================================================

        elif intent == "table":

    # Actual technical table evidence.
            if content_type == "table":
                score += 0.30

            # Table-oriented titles.
            if "mapping" in title:
                score += 0.15

            if "rules" in title:
                score += 0.15

            if "services" in title:
                score += 0.30

            if "attributes" in title:
                score += 0.20

            if "overview" in title:
                score += 0.10

            if "general" in title:
                score += 0.05

            # Known authoritative table/service sections.
            table_sections = {
                "5.6.1",
                "5.6.7.1",
                "5.7.4",
                "5.32.8",
                "7.2.2",
            }

            if section in table_sections:
                score += 0.40

            # Explicit authoritative sections.
            if section == "5.32.8":
                score = 0.90

            if section == "7.2.2":
                score = 0.90

        # ====================================================
        # ARCHITECTURE
        # ====================================================

        elif intent == "architecture":

            if "network functions" in title:
                score += 0.35

            if "architecture" in title:
                score += 0.20

            if "reference point" in title:
                score += 0.15

            if "interface" in title:
                score += 0.10

            if section.startswith("4."):
                score += 0.10

        # ====================================================
        # COMMON PENALTIES
        # ====================================================

        if "abbreviations" in title:
            score -= 0.40

        if "change history" in title:
            score -= 1.00

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ========================================================
    # MAIN EVALUATION
    # ========================================================

    def evaluate(
        self,
        query: str,
        results: List[Dict],
    ) -> Dict:
        """
        Evaluate whether the retrieved evidence is sufficient
        to permit grounded generation.
        """

        intent = detect_intent(query)

        # ----------------------------------------------------
        # Keep only usable results.
        # ----------------------------------------------------

        usable = [
            result
            for result in results
            if self._has_content(result)
        ]

        # ----------------------------------------------------
        # Determine how many strong results are required.
        # ----------------------------------------------------

        if intent in {
            "definition",
            "table",
        }:

            # One strong authoritative section is enough.
            required_strong_results = 1

        elif (
            intent == "role"
            and detect_entity(query) is not None
        ):

            # One dedicated NF section is enough for
            # an entity-specific role question.
            required_strong_results = 1

        else:

            # Architecture/general questions need
            # multiple supporting results.
            required_strong_results = (
                self.min_results
            )

        # ----------------------------------------------------
        # Not enough usable evidence.
        # ----------------------------------------------------

        if (
            len(usable)
            < required_strong_results
        ):

            return {
                "allowed": False,
                "reason": (
                    "insufficient_evidence_count"
                ),
                "confidence": 0.0,
                "supporting_chunks": len(
                    usable
                ),
                "entity_matches": 0,
                "evidence_quality": 0.0,
                "best_section_quality": 0.0,
            }

        # ----------------------------------------------------
        # Select strong results.
        # ----------------------------------------------------

        strong_results = [
            result
            for result in usable
            if result.get(
                "final_score",
                result.get(
                    "reranker_score",
                    0.0,
                ),
            ) >= self.min_score
        ]

        # ----------------------------------------------------
        # Entity consistency.
        # ----------------------------------------------------

        entity_matches = sum(
            self._entity_present(
                query,
                result,
            )
            for result in strong_results
        )

        # ----------------------------------------------------
        # Calculate section quality.
        # ----------------------------------------------------

        section_scores = [
            self._section_quality(
                query,
                result,
            )
            for result in strong_results
        ]

        average_quality = (
            sum(section_scores)
            / len(section_scores)
            if section_scores
            else 0.0
        )

        best_quality = (
            max(section_scores)
            if section_scores
            else 0.0
        )

        # ----------------------------------------------------
        # Intent-specific acceptance rules.
        # ----------------------------------------------------

        if intent in {
            "definition",
            "table",
        }:

            # One strong section with sufficient quality.
            allowed = (
                len(strong_results)
                >= required_strong_results
                and best_quality
                >= self.min_evidence_quality
            )

        else:

            # Architecture/general:
            # multiple strong results + entity consistency
            # + sufficient average/best quality.
            allowed = (
                len(strong_results)
                >= required_strong_results
                and entity_matches
                >= self.min_entity_matches
                and average_quality
                >= self.min_evidence_quality
                and best_quality
                >= self.min_evidence_quality
            )

        # ----------------------------------------------------
        # Determine reason.
        # ----------------------------------------------------

        if allowed:

            reason = (
                "sufficient_evidence"
            )

        elif (
            len(strong_results)
            < required_strong_results
        ):

            reason = (
                "weak_retrieval_scores"
            )

        elif (
            entity_matches
            < self.min_entity_matches
            and intent not in {
                "definition",
                "table",
            }
        ):

            reason = "entity_mismatch"

        elif (
            best_quality
            < self.min_evidence_quality
        ):

            reason = (
                "poor_section_relevance"
            )

        else:

            reason = (
                "insufficient_evidence_quality"
            )

        # ----------------------------------------------------
        # Confidence calculation.
        # ----------------------------------------------------

        strong_ratio = min(
            1.0,
            len(strong_results)
            / max(
                required_strong_results,
                1,
            ),
        )

        if intent in {
            "definition",
            "table",
        }:

            # Entity matching is not required for these
            # intent types.
            entity_ratio = 1.0

        else:

            entity_ratio = min(
                1.0,
                entity_matches
                / max(
                    self.min_entity_matches,
                    1,
                ),
            )

        confidence = (
            0.40 * strong_ratio
            + 0.30 * entity_ratio
            + 0.30 * average_quality
        )

        # ----------------------------------------------------
        # Final decision.
        # ----------------------------------------------------

        return {
            "allowed": allowed,
            "reason": reason,
            "confidence": round(
                min(
                    1.0,
                    confidence,
                ),
                3,
            ),
            "supporting_chunks": len(
                strong_results
            ),
            "entity_matches": entity_matches,
            "evidence_quality": round(
                average_quality,
                3,
            ),
            "best_section_quality": round(
                best_quality,
                3,
            ),
        }