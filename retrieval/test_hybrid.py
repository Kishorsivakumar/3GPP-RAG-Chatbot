from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore


QUESTIONS = [
    "What is the role of the AMF in the 5G System?",
    "What network functions are part of the 5G System architecture?",
    "What is a PDU Session?",
    "What is the role of the SMF?",
]


def main():

    # Dense
    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        embedding_model
    )

    vector_store.load()

    # BM25
    bm25_store = BM25Store()
    bm25_store.load()

    # Hybrid
    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    for question in QUESTIONS:

        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")
        print("=" * 90)

        results = retriever.retrieve(
            query=question,
            top_k_dense=20,
            top_k_bm25=20,
            final_k=10,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"RRF={result['rrf_score']:.5f} "
                f"DenseRank={result['dense_rank']} "
                f"BM25Rank={result['bm25_rank']}"
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
                f"{result['content'][:500]}"
            )


if __name__ == "__main__":
    main()