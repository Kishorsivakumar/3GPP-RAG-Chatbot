from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import SectionAwareRetriever
from retrieval.vector_store import VectorStore

from generation.completeness_validator import (
    validate_completeness,
)


def build_retriever():

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
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

    query = (
        "List all network functions "
        "specified in Section 4.2.2."
    )

    retriever = build_retriever()

    results = retriever.retrieve(
        query=query,
        candidate_k=20,
        final_k=5,
    )

    # Simulate the complete list directly from
    # the retrieved authoritative source.
    authoritative = [
        result
        for result in results
        if result["section"] == "4.2.2"
    ]

    if not authoritative:
        print(
            "ERROR: Section 4.2.2 was not retrieved."
        )
        return

    source_text = "\n".join(
        result["content"]
        for result in authoritative
    )

    from generation.completeness_validator import (
        extract_source_items,
    )

    items = extract_source_items(
        source_text
    )

    # Build a synthetic structured response
    # containing every extracted source item.
    response = {
        "answer": "Complete source list",
        "claims": [
            {
                "text": item,
                "section": "4.2.2",
            }
            for item in items
        ],
    }

    result = validate_completeness(
        query=query,
        response=response,
        evidence=results,
    )

    print("\n" + "=" * 80)
    print("COMPLETENESS TEST")
    print("=" * 80)

    print(
        f"Required       : {result['required']}"
    )

    print(
        f"Valid          : {result['valid']}"
    )

    print(
        f"Expected items : "
        f"{result['expected_items']}"
    )

    print(
        f"Covered items  : "
        f"{result['covered_items']}"
    )

    print(
        f"Coverage       : "
        f"{result['coverage']:.2%}"
    )

    print(
        f"Missing items  : "
        f"{len(result['missing_items'])}"
    )

    if result["missing_items"]:
        print("\nMissing:")
        for item in result["missing_items"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()