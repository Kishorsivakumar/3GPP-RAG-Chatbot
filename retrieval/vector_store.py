from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from retrieval.embeddings import EmbeddingModel


CHUNKS_PATH = Path(
    r"data\processed\chunks.json"
)

INDEX_DIR = Path(
    r"indexes"
)

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"


class VectorStore:
    """
    FAISS vector store for the 3GPP RAG system.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks: List[Dict] = []

    def build(
        self,
        chunks: List[Dict],
    ) -> None:
        """
        Generate embeddings and build the FAISS index.
        """

        texts = [
            self._build_embedding_text(chunk)
            for chunk in chunks
        ]

        print(
            f"Generating embeddings for "
            f"{len(texts)} chunks..."
        )

        embeddings = (
            self.embedding_model.encode(texts)
        )

        dimension = embeddings.shape[1]

        print(
            f"Embedding dimension: {dimension}"
        )

        # Inner product works as cosine similarity
        # because embeddings are normalized.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        self.chunks = chunks

        print(
            f"FAISS index contains "
            f"{self.index.ntotal} vectors."
        )

    @staticmethod
    def _build_embedding_text(
        chunk: Dict,
    ) -> str:
        """
        Create the text actually embedded.

        Section metadata is included so that semantic
        retrieval can use the technical context.
        """

        return (
            f"Specification: "
            f"{chunk['specification']}\n"
            f"Release: {chunk['release']}\n"
            f"Version: {chunk['version']}\n"
            f"Section: {chunk['section']}\n"
            f"Section title: {chunk['section_title']}\n"
            f"Content type: {chunk['content_type']}\n\n"
            f"{chunk['content']}"
        )

    def save(self) -> None:
        """
        Save FAISS index and chunk metadata.
        """

        if self.index is None:
            raise RuntimeError(
                "Vector index has not been built."
            )

        INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(INDEX_PATH),
        )

        with METADATA_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"\nSaved FAISS index: {INDEX_PATH}"
        )

        print(
            f"Saved metadata: {METADATA_PATH}"
        )

    def load(self) -> None:
        """
        Load an existing FAISS index and metadata.
        """

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        with METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.chunks = json.load(file)

        if self.index.ntotal != len(
            self.chunks
        ):
            raise ValueError(
                "FAISS index size does not match "
                "metadata size."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search the vector store.
        """

        if self.index is None:
            raise RuntimeError(
                "Vector store is not loaded."
            )

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k,
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            result = dict(
                self.chunks[index]
            )

            result["score"] = float(
                score
            )

            results.append(
                result
            )

        return results