from __future__ import annotations

import json
from pathlib import Path

from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import VectorStore


CHUNKS_PATH = Path(
    r"data\processed\chunks.json"
)


def main():
    print("=" * 60)
    print("BUILDING 3GPP FAISS INDEX")
    print("=" * 60)

    # Load chunks
    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    # Load embedding model
    model = EmbeddingModel()

    # Build vector store
    store = VectorStore(model)

    store.build(chunks)

    # Save
    store.save()

    print("\nFAISS index built successfully.")


if __name__ == "__main__":
    main()