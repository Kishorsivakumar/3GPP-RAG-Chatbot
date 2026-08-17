from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import SectionAwareRetriever
from retrieval.vector_store import VectorStore

from generation.evidence_gate import EvidenceGate
from generation.llm_client import LLMClient
from generation.rag_pipeline import RAGPipeline


def build_pipeline():
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


def main():

    pipeline = build_pipeline()

    questions = [
        "What network functions are part of the 5G System architecture?",
        "What is a PDU Session?",
        "What is the capital of France?",
        "List all network functions specified in Section 4.2.2.",
    ]

    for question in questions:

        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")
        print("=" * 90)

        result = pipeline.run(
            question
        )

        print("\nANSWER:")
        print(result["answer"])

        print(
            f"\nAllowed: {result['allowed']}"
        )

        print(
            f"Reason: {result['reason']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print("\nCLAIM VALIDATION:")
        print(
            result.get(
                "claim_validation",
                {},
            )
        )

        print("\nCOMPLETENESS VALIDATION:")
        print(
            result.get(
                "completeness_validation",
                {},
            )
        )

        print("\nSOURCES:")

        sources = result.get(
            "sources",
            [],
        )

        if not sources:
            print("- None")

        for source in sources:
            print(
                f"- "
                f"{source['specification']} "
                f"V{source['version']} "
                f"{source['release']} "
                f"Section {source['section']} "
                f"({source['section_title']})"
            )


if __name__ == "__main__":
    main()