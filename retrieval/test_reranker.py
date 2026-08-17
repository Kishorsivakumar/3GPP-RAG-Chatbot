from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.reranked_retriever import RerankedRetriever
from retrieval.vector_store import VectorStore


QUESTIONS = [
    "What is the role of the AMF in the 5G System?",
    "What network functions are part of the 5G System architecture?",
    "What is a PDU Session?",
    "What is the role of the SMF?",
]


def main():

    # ---------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        embedding_model
    )

    vector_store.load()

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    bm25_store = BM25Store()

    bm25_store.load()

    # ---------------------------------------------------------
    # Hybrid
    # ---------------------------------------------------------

    hybrid = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # ---------------------------------------------------------
    # Cross encoder
    # ---------------------------------------------------------

    reranker = Reranker()

    retriever = RerankedRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )

    # ---------------------------------------------------------
    # Test questions
    # ---------------------------------------------------------

    for question in QUESTIONS:

        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")
        print("=" * 90)

        results = retriever.retrieve(
            query=question,
            candidate_k=20,
            final_k=5,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"Final={result['final_score']:.4f} "
                f"Reranker={result['reranker_score']:.4f} "
                f"RRF={result['rrf_score']:.5f}"
            )

            print(
                f"Section: "
                f"{result['section']} "
                f"{result['section_title']}"
            )

            print(
                f"Type: "
                f"{result['content_type']}"
            )

            print(
                f"Content:\n"
                f"{result['content'][:700]}"
            )
            from retrieval.relevance import (
                detect_entity,
                detect_intent,
            )

            print(
                f"Intent: {detect_intent(question)}"
            )

            print(
                f"Entity: {detect_entity(question)}"
            )

if __name__ == "__main__":
    main()