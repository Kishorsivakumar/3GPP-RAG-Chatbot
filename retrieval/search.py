from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import VectorStore


def main():

    model = EmbeddingModel()

    store = VectorStore(model)

    store.load()

    questions = [
        "What is the role of the AMF in the 5G System?",
        "What network functions are part of the 5G System architecture?",
        "What is a PDU Session?",
        "What is the role of the SMF?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = store.search(
            question,
            top_k=5,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"Score: {result['score']:.4f}"
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


if __name__ == "__main__":
    main()