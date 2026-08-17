from __future__ import annotations

from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import (
    SectionAwareRetriever,
)

from generation.evidence_gate import EvidenceGate


QUESTIONS = [
    "What is the role of the AMF in the 5G System?",
    "What network functions are part of the 5G System architecture?",
    "What is a PDU Session?",
    "What is the capital of France?",
]


def build_retriever():

    embedding_model = EmbeddingModel()

    vector_store = __import__(
        "retrieval.vector_store",
        fromlist=["VectorStore"],
    ).VectorStore(
        embedding_model
    )

    vector_store.load()

    bm25_store = BM25Store()
    bm25_store.load()

    hybrid = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    reranker = Reranker()

    return SectionAwareRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )


def main():

    retriever = build_retriever()

    gate = EvidenceGate(
        min_results=2,
        min_score=1.5,
        min_entity_matches=1,
    )

    for question in QUESTIONS:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = retriever.retrieve(
            query=question,
            candidate_k=20,
            final_k=5,
        )

        decision = gate.evaluate(
            query=question,
            results=results,
        )

        print("\nEVIDENCE DECISION")
        print(
            f"Allowed    : {decision['allowed']}"
        )
        print(
            f"Reason     : {decision['reason']}"
        )
        print(
            f"Confidence : "
            f"{decision['confidence']:.3f}"
        )
        print(
            f"Strong chunks       : "
            f"{decision['supporting_chunks']}"
        )

        print(
            f"Entity matches      : "
            f"{decision.get('entity_matches', 0)}"
        )

        print(
            f"Evidence quality    : "
            f"{decision.get('evidence_quality', 0.0):.3f}"
        )

        print(
            f"Best section quality: "
            f"{decision.get('best_section_quality', 0.0):.3f}"
        )

        print("\nTOP EVIDENCE")

        for rank, result in enumerate(
            results[:3],
            start=1,
        ):
            print(
                f"\n[{rank}] "
                f"{result['section']} "
                f"{result['section_title']}"
            )
            print(
                f"Score: "
                f"{result.get('final_score', 0.0):.4f}"
            )


if __name__ == "__main__":
    main()