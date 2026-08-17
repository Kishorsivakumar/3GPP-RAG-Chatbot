from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import (
    SectionAwareRetriever,
)
from retrieval.vector_store import VectorStore

from generation.evidence_gate import EvidenceGate
from generation.llm_client import LLMClient
from generation.rag_pipeline import RAGPipeline


QUESTIONS_PATH = Path(
    r"evaluation\questions.json"
)


# ============================================================
# Build complete RAG pipeline
# ============================================================

def build_pipeline() -> RAGPipeline:

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        embedding_model
    )

    vector_store.load()

    bm25_store = BM25Store()
    bm25_store.load()

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    reranker = Reranker()

    retriever = SectionAwareRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    evidence_gate = EvidenceGate()

    llm_client = LLMClient()

    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        evidence_gate=evidence_gate,
    )


# ============================================================
# Main evaluation
# ============================================================

def main():

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    pipeline = build_pipeline()

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    total = len(questions)

    generation_attempted = 0
    generation_success = 0

    expected_answerable_count = 0
    correct_answerability = 0

    claim_validation_passes = 0
    completeness_passes = 0

    api_failures = 0

    # --------------------------------------------------------
    # Category statistics
    # --------------------------------------------------------

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "generation_attempted": 0,
            "generation_success": 0,
            "answerability_correct": 0,
            "claim_validation_passes": 0,
            "completeness_passes": 0,
            "api_failures": 0,
        }
    )

    # ========================================================
    # Evaluate questions
    # ========================================================

    for item in questions:

        question = item["question"]

        expected_answerable = item[
            "answerable"
        ]

        category = item.get(
            "category",
            "unknown",
        )

        category_stats[
            category
        ]["total"] += 1

        print(
            "\n" + "=" * 90
        )

        print(
            f"QUESTION: {question}"
        )

        print(
            "=" * 90
        )

        print(
            f"ID                  : "
            f"{item.get('id', 'N/A')}"
        )

        print(
            f"Category            : "
            f"{category}"
        )

        print(
            f"Expected answerable : "
            f"{expected_answerable}"
        )

        # ----------------------------------------------------
        # Run complete pipeline
        # ----------------------------------------------------

        generation_attempted += 1

        category_stats[
            category
        ]["generation_attempted"] += 1

        try:

            result = pipeline.run(
                question
            )

        except Exception as exc:

            api_failures += 1

            category_stats[
                category
            ]["api_failures"] += 1

            print(
                "\nPIPELINE ERROR"
            )

            print(
                f"Type   : "
                f"{type(exc).__name__}"
            )

            print(
                f"Message: "
                f"{exc}"
            )

            print(
                "\nGeneration status: "
                "SKIPPED"
            )

            continue

        # ----------------------------------------------------
        # Basic result information
        # ----------------------------------------------------

        allowed = result.get(
            "allowed",
            False,
        )

        reason = result.get(
            "reason",
            "",
        )

        confidence = result.get(
            "confidence",
            0.0,
        )

        print(
            f"\nAllowed             : "
            f"{allowed}"
        )

        print(
            f"Reason              : "
            f"{reason}"
        )

        print(
            f"Confidence          : "
            f"{confidence:.3f}"
        )

        # ----------------------------------------------------
        # Answerability correctness
        # ----------------------------------------------------

        if allowed == expected_answerable:

            correct_answerability += 1

            category_stats[
                category
            ]["answerability_correct"] += 1

        if expected_answerable:

            expected_answerable_count += 1

        # ----------------------------------------------------
        # Successful grounded generation
        # ----------------------------------------------------

        if allowed:

            generation_success += 1

            category_stats[
                category
            ]["generation_success"] += 1

            print(
                "\nANSWER:"
            )

            print(
                result.get(
                    "answer",
                    "",
                )
            )

        else:

            print(
                "\nANSWER:"
            )

            print(
                result.get(
                    "answer",
                    "",
                )
            )

        # ----------------------------------------------------
        # Claim validation
        # ----------------------------------------------------

        claim_validation = result.get(
            "claim_validation",
            {},
        )

        claim_valid = claim_validation.get(
            "valid",
            False,
        )

        if claim_valid:

            claim_validation_passes += 1

            category_stats[
                category
            ]["claim_validation_passes"] += 1

        print(
            "\nCLAIM VALIDATION:"
        )

        print(
            claim_validation
        )

        # ----------------------------------------------------
        # Completeness validation
        # ----------------------------------------------------

        completeness_validation = result.get(
            "completeness_validation",
            {},
        )

        completeness_required = (
            completeness_validation.get(
                "required",
                False,
            )
        )

        completeness_valid = (
            completeness_validation.get(
                "valid",
                False,
            )
        )

        # For normal questions completeness is not required.
        # Treat those as passing the completeness dimension.
        if (
            not completeness_required
            and allowed
        ):
            completeness_pass = True

        else:
            completeness_pass = (
                completeness_valid
            )

        if completeness_pass:

            completeness_passes += 1

            category_stats[
                category
            ]["completeness_passes"] += 1

        print(
            "\nCOMPLETENESS VALIDATION:"
        )

        print(
            completeness_validation
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        print(
            "\nSOURCES:"
        )

        sources = result.get(
            "sources",
            [],
        )

        if not sources:

            print(
                "- None"
            )

        else:

            for source in sources:

                print(
                    f"- "
                    f"{source['specification']} "
                    f"V{source['version']} "
                    f"{source['release']} "
                    f"Section "
                    f"{source['section']} "
                    f"("
                    f"{source['section_title']}"
                    f")"
                )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "END-TO-END RAG EVALUATION"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Generation success
    # --------------------------------------------------------

    if generation_attempted:

        generation_success_rate = (
            generation_success
            / generation_attempted
        )

    else:

        generation_success_rate = 0.0

    print(
        f"Generation success : "
        f"{generation_success_rate:.2%}"
    )

    print(
        f"Generation attempts : "
        f"{generation_attempted}"
    )

    # --------------------------------------------------------
    # Answerability accuracy
    # --------------------------------------------------------

    answerability_accuracy = (
        correct_answerability
        / total
        if total
        else 0.0
    )

    print(
        f"Answerability acc.  : "
        f"{answerability_accuracy:.2%}"
    )

    # --------------------------------------------------------
    # Claim validation
    # --------------------------------------------------------

    claim_validation_rate = (
        claim_validation_passes
        / generation_attempted
        if generation_attempted
        else 0.0
    )

    print(
        f"Claim validation    : "
        f"{claim_validation_rate:.2%}"
    )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    completeness_rate = (
        completeness_passes
        / generation_attempted
        if generation_attempted
        else 0.0
    )

    print(
        f"Completeness pass   : "
        f"{completeness_rate:.2%}"
    )

    # --------------------------------------------------------
    # API failures
    # --------------------------------------------------------

    api_failure_rate = (
        api_failures
        / generation_attempted
        if generation_attempted
        else 0.0
    )

    print(
        f"API failure rate    : "
        f"{api_failure_rate:.2%}"
    )

    print(
        f"Questions evaluated : "
        f"{total}"
    )

    # ========================================================
    # Category summary
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "CATEGORY SUMMARY"
    )

    print(
        "=" * 90
    )

    for category in sorted(
        category_stats
    ):

        stats = category_stats[
            category
        ]

        category_total = stats[
            "total"
        ]

        generation_rate = (
            stats["generation_success"]
            / stats["generation_attempted"]
            if stats["generation_attempted"]
            else 0.0
        )

        answerability_rate = (
            stats["answerability_correct"]
            / category_total
            if category_total
            else 0.0
        )

        claim_rate = (
            stats["claim_validation_passes"]
            / stats["generation_attempted"]
            if stats["generation_attempted"]
            else 0.0
        )

        completeness_rate_category = (
            stats["completeness_passes"]
            / stats["generation_attempted"]
            if stats["generation_attempted"]
            else 0.0
        )

        failure_rate = (
            stats["api_failures"]
            / stats["generation_attempted"]
            if stats["generation_attempted"]
            else 0.0
        )

        print(
            f"{category:15s} "
            f"n={category_total:2d} | "
            f"generation={generation_rate:.2%} | "
            f"answerability={answerability_rate:.2%} | "
            f"claims={claim_rate:.2%} | "
            f"complete={completeness_rate_category:.2%} | "
            f"api_fail={failure_rate:.2%}"
        )


if __name__ == "__main__":
    main()