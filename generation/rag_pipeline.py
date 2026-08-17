from __future__ import annotations

from typing import Dict

from generation.claim_validator import validate_claims
from generation.completeness_validator import (
    validate_completeness,
)
from generation.evidence_gate import EvidenceGate
from generation.llm_client import LLMClient


class RAGPipeline:

    def __init__(
        self,
        retriever,
        llm_client: LLMClient,
        evidence_gate: EvidenceGate,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.evidence_gate = evidence_gate

    def run(
        self,
        query: str,
    ) -> Dict:

        # ====================================================
        # 1. Retrieve evidence
        # ====================================================

        results = self.retriever.retrieve(
            query=query,
            candidate_k=20,
            final_k=5,
        )

        # ====================================================
        # 2. Evidence gate
        # ====================================================

        decision = self.evidence_gate.evaluate(
            query=query,
            results=results,
        )

        # ====================================================
        # 3. Refuse when evidence is insufficient
        # ====================================================

        if not decision["allowed"]:

            return {
                "answer": (
                    "I do not have sufficient evidence "
                    "in the provided 3GPP documentation "
                    "to answer this question."
                ),
                "allowed": False,
                "reason": decision["reason"],
                "confidence": decision["confidence"],
                "sources": [],
                "claims": [],
                "claim_validation": {
                    "valid": False,
                    "reason": (
                        "not_run_evidence_gate_failed"
                    ),
                    "total_claims": 0,
                    "valid_claims": 0,
                    "invalid_claims": [],
                    "claims": [],
                },
                "completeness_validation": {
                    "required": False,
                    "valid": False,
                    "reason": (
                        "not_run_evidence_gate_failed"
                    ),
                    "expected_items": 0,
                    "covered_items": 0,
                    "coverage": 0.0,
                    "missing_items": [],
                },
            }

        # ====================================================
        # 4. Generate structured grounded response
        # ====================================================

        response = self.llm_client.answer(
            query=query,
            evidence=results,
        )

        # ====================================================
        # 5. Validate structured claims
        # ====================================================

        claim_check = validate_claims(
            response=response,
            evidence=results,
        )

        # ====================================================
        # 6. Refuse if claim validation fails
        # ====================================================

        if not claim_check["valid"]:

            return {
                "answer": (
                    "I could not produce a sufficiently "
                    "grounded answer from the retrieved "
                    "3GPP evidence."
                ),
                "allowed": False,
                "reason": "claim_validation_failed",
                "confidence": decision["confidence"],
                "sources": [],
                "claims": response.get(
                    "claims",
                    [],
                ),
                "claim_validation": claim_check,
                "completeness_validation": {
                    "required": False,
                    "valid": False,
                    "reason": (
                        "not_run_claim_validation_failed"
                    ),
                    "expected_items": 0,
                    "covered_items": 0,
                    "coverage": 0.0,
                    "missing_items": [],
                },
            }

        # ====================================================
        # 7. Validate completeness for list questions
        # ====================================================

        completeness_check = validate_completeness(
            query=query,
            response=response,
            evidence=results,
        )

        # ====================================================
        # 8. Refuse if completeness validation fails
        # ====================================================

        if not completeness_check["valid"]:

            return {
                "answer": (
                    "I could not produce a sufficiently "
                    "complete and grounded answer from "
                    "the retrieved 3GPP evidence."
                ),
                "allowed": False,
                "reason": (
                    "completeness_validation_failed"
                ),
                "confidence": decision["confidence"],
                "sources": [],
                "claims": response.get(
                    "claims",
                    [],
                ),
                "claim_validation": claim_check,
                "completeness_validation": (
                    completeness_check
                ),
            }

        # ====================================================
        # 9. Render final answer from validated claims
        # ====================================================

        rendered_claims = []

        for claim in response.get(
            "claims",
            [],
        ):

            claim_text = claim["text"].strip()

            section = str(
                claim["section"]
            ).strip()

            rendered_claims.append(
                f"{claim_text} "
                f"[TS 23.501, Section {section}]"
            )

        answer = "\n\n".join(
            rendered_claims
        )

        # ====================================================
        # 10. Build source metadata
        # ====================================================

        sources = []

        for result in results:

            sources.append(
                {
                    "specification": result[
                        "specification"
                    ],
                    "version": result[
                        "version"
                    ],
                    "release": result[
                        "release"
                    ],
                    "section": result[
                        "section"
                    ],
                    "section_title": result[
                        "section_title"
                    ],
                    "content_type": result[
                        "content_type"
                    ],
                    "score": result.get(
                        "final_score",
                        0.0,
                    ),
                }
            )

        # ====================================================
        # 11. Successful grounded response
        # ====================================================

        return {
            "answer": answer,
            "allowed": True,
            "reason": decision["reason"],
            "confidence": decision["confidence"],
            "sources": sources,
            "claims": response.get(
                "claims",
                [],
            ),
            "claim_validation": claim_check,
            "completeness_validation": (
                completeness_check
            ),
        }