from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from generation.evidence_gate import EvidenceGate
from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import (
    SectionAwareRetriever,
)
from retrieval.vector_store import VectorStore


QUESTIONS_PATH = Path(
    r"evaluation\questions.json"
)


def build_retriever():
    """
    Build the complete retrieval pipeline.
    """

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

    return retriever


def section_matches(
    expected_section: str,
    retrieved_section: str,
) -> bool:
    """
    Determine whether a retrieved section satisfies
    an expected section.

    Exact match:
        6.2.2 == 6.2.2

    Parent/child match:
        6.2.6 == 6.2.6.1
        4.2.8.3 == 4.2.8.3.1

    This is useful for hierarchical 3GPP sections because
    the retrieved chunk may belong to a child subsection
    while still containing the requested subject matter.
    """

    expected = str(
        expected_section
    ).strip().lower()

    retrieved = str(
        retrieved_section
    ).strip().lower()

    if not expected or not retrieved:
        return False

    # Exact section.
    if retrieved == expected:
        return True

    # Child subsection.
    if retrieved.startswith(
        expected + "."
    ):
        return True

    return False


def retrieval_hit(
    expected_sections: set[str],
    retrieved_sections: set[str],
) -> bool:
    """
    Return True when at least one expected section is
    represented by one retrieved section.
    """

    return any(
        section_matches(
            expected_section,
            retrieved_section,
        )
        for expected_section in expected_sections
        for retrieved_section in retrieved_sections
    )


def main():

    # ========================================================
    # Load benchmark
    # ========================================================

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    # ========================================================
    # Build retriever and gate
    # ========================================================

    retriever = build_retriever()

    gate = EvidenceGate()

    # ========================================================
    # Overall statistics
    # ========================================================

    total = len(questions)

    retrieval_evaluated = 0
    retrieval_hits = 0

    gate_correct = 0

    # ========================================================
    # Category statistics
    # ========================================================

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "retrieval_evaluated": 0,
            "retrieval_hits": 0,
            "gate_correct": 0,
        }
    )

    # ========================================================
    # Evaluate every question
    # ========================================================

    for item in questions:

        question = item["question"]

        answerable = item[
            "answerable"
        ]

        category = item.get(
            "category",
            "unknown",
        )

        expected_sections = {
            str(section)
            for section in item.get(
                "expected_sections",
                [],
            )
        }

        category_stats[
            category
        ]["total"] += 1

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        results = retriever.retrieve(
            query=question,
            candidate_k=20,
            final_k=5,
        )

        retrieved_sections = {
            str(
                result["section"]
            )
            for result in results
        }

        # ----------------------------------------------------
        # Retrieval evaluation
        # ----------------------------------------------------

        if expected_sections:

            retrieval_evaluated += 1

            category_stats[
                category
            ]["retrieval_evaluated"] += 1

            hit = retrieval_hit(
                expected_sections,
                retrieved_sections,
            )

            if hit:

                retrieval_hits += 1

                category_stats[
                    category
                ]["retrieval_hits"] += 1

        else:

            # No expected section for an unanswerable
            # question.
            hit = None

        # ----------------------------------------------------
        # Evidence gate
        # ----------------------------------------------------

        decision = gate.evaluate(
            query=question,
            results=results,
        )

        predicted_answerable = (
            decision["allowed"]
        )

        gate_is_correct = (
            predicted_answerable
            == answerable
        )

        if gate_is_correct:

            gate_correct += 1

            category_stats[
                category
            ]["gate_correct"] += 1

        # ----------------------------------------------------
        # Print question result
        # ----------------------------------------------------

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
            f"{answerable}"
        )

        if expected_sections:

            print(
                f"Expected sections   : "
                f"{sorted(expected_sections)}"
            )

        else:

            print(
                "Expected sections   : "
                "not assigned"
            )

        print(
            f"Retrieved sections  : "
            f"{sorted(retrieved_sections)}"
        )

        print(
            f"Retrieval hit       : "
            f"{hit}"
        )

        print(
            f"Gate allowed        : "
            f"{predicted_answerable}"
        )

        print(
            f"Gate correct        : "
            f"{gate_is_correct}"
        )

        print(
            f"Gate reason         : "
            f"{decision['reason']}"
        )

        print(
            f"Gate confidence     : "
            f"{decision['confidence']:.3f}"
        )

        print("\nTop evidence:")

        for rank, result in enumerate(
            results[:3],
            start=1,
        ):

            print(
                f"[{rank}] "
                f"{result['section']} "
                f"{result['section_title']} "
                f"| score="
                f"{result.get('final_score', 0.0):.4f}"
            )

    # ========================================================
    # Overall summary
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 90
    )

    if retrieval_evaluated:

        retrieval_hit_rate = (
            retrieval_hits
            / retrieval_evaluated
        )

        print(
            f"Retrieval hit rate : "
            f"{retrieval_hit_rate:.2%}"
        )

    else:

        print(
            "Retrieval hit rate : N/A"
        )

    gate_accuracy = (
        gate_correct
        / total
        if total
        else 0.0
    )

    print(
        f"Gate accuracy      : "
        f"{gate_accuracy:.2%}"
    )

    print(
        f"Questions evaluated: "
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

        category_gate_accuracy = (
            stats["gate_correct"]
            / category_total
            if category_total
            else 0.0
        )

        if stats[
            "retrieval_evaluated"
        ]:

            category_retrieval_rate = (
                stats["retrieval_hits"]
                / stats[
                    "retrieval_evaluated"
                ]
            )

            retrieval_text = (
                f"{category_retrieval_rate:.2%}"
            )

        else:

            retrieval_text = "N/A"

        print(
            f"{category:15s} "
            f"n={category_total:2d} | "
            f"retrieval={retrieval_text:>6s} | "
            f"gate={category_gate_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()